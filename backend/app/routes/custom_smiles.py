"""
Custom SMILES card endpoint.

Builds a single ResultCard for an arbitrary SMILES string by scoring it
against a caller-provided reference fingerprint (predicted FP or query FP)
without running a new retrieval pass.
"""
from __future__ import annotations

import asyncio
import logging
import math

import torch
from fastapi import APIRouter, HTTPException, status

from app.schemas import CustomSmilesCardRequest, CustomSmilesCardResponse, ResultCard

logger = logging.getLogger(__name__)
router = APIRouter()


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().view(-1)
    b = b.detach().float().view(-1)
    n = min(a.numel(), b.numel())
    if n == 0:
        return 0.0
    a, b = a[:n], b[:n]
    dot = float(torch.dot(a, b))
    denom = float(torch.linalg.norm(a)) * float(torch.linalg.norm(b))
    if denom < 1e-9:
        return 0.0
    val = dot / denom
    return max(0.0, min(1.0, val)) if math.isfinite(val) else 0.0


def _tanimoto(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().view(-1)
    b = b.detach().float().view(-1)
    n = min(a.numel(), b.numel())
    if n == 0:
        return 0.0
    a, b = a[:n], b[:n]
    dot = torch.dot(a, b)
    denom = a.pow(2).sum() + b.pow(2).sum() - dot
    if denom <= 0:
        return 0.0
    val = (dot / denom).item()
    return max(0.0, min(1.0, val)) if math.isfinite(val) else 0.0


@router.post(
    "/custom-smiles-card",
    response_model=CustomSmilesCardResponse,
    status_code=status.HTTP_200_OK,
)
async def custom_smiles_card(body: CustomSmilesCardRequest):
    """
    Score an arbitrary SMILES against a reference fingerprint and return a ResultCard.

    The reference_fp should be the predicted fingerprint (from /predict) or the
    query fingerprint (from /smiles-search) from the current session.
    """
    from app.manifest import resolve_model_id
    from app.registry import ensure_loaded
    from app.config import MOLECULE_IMG_SIZE
    from app.renderer import render_plain_svg, render_enhanced_svg

    mid, err = resolve_model_id(body.model_id)
    if err:
        raise HTTPException(status_code=err[0], detail=err[1])

    smiles = body.smiles.strip()

    session = ensure_loaded(mid)
    ref_tensor = torch.tensor(body.reference_fp, dtype=torch.float32)

    # Build the MARINA fingerprint for the custom SMILES.
    try:
        mfp = await asyncio.to_thread(session.fp_loader.build_mfp_for_smiles, smiles)
    except Exception as exc:
        logger.warning("build_mfp_for_smiles failed for %r: %s", smiles, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid SMILES or fingerprint generation failed",
        )

    if mfp is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid SMILES or fingerprint generation failed",
        )

    mfp_tensor = mfp.clone().detach().float() if isinstance(mfp, torch.Tensor) else torch.tensor(mfp, dtype=torch.float32)

    cosine_sim  = _cosine(ref_tensor, mfp_tensor)
    tanimoto_sim = _tanimoto(ref_tensor, mfp_tensor)

    # SVG rendering
    enhanced_svg = await asyncio.to_thread(
        render_enhanced_svg, smiles, ref_tensor, session.fp_loader, MOLECULE_IMG_SIZE
    )
    plain_svg = await asyncio.to_thread(render_plain_svg, smiles, MOLECULE_IMG_SIZE)

    # Active bit indices for the custom molecule's fingerprint
    fp_indices: list[int] = []
    try:
        nz = torch.nonzero(mfp_tensor > 0.5, as_tuple=False).squeeze(-1)
        fp_indices = nz.tolist() if nz.numel() > 0 else []
        if not isinstance(fp_indices, list):
            fp_indices = [fp_indices]
    except Exception as exc:
        logger.warning("Failed to extract FP indices for custom SMILES: %s", exc)

    # Exact mass via RDKit
    exact_mass: float | None = None
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            exact_mass = Descriptors.ExactMolWt(mol)
    except Exception:
        pass

    card = ResultCard(
        index=-1,
        smiles=smiles,
        similarity=cosine_sim,
        cosine_similarity=cosine_sim,
        tanimoto_similarity=tanimoto_sim,
        svg=enhanced_svg,
        plain_svg=plain_svg,
        name=None,
        primary_link=None,
        database_links={},
        retrieved_molecule_fp_indices=fp_indices,
        exact_mass=exact_mass,
    )

    return CustomSmilesCardResponse(result=card)
