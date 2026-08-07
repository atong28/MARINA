"""
POST /api/fingerprints/indices – compute entropy fingerprint bit indices for a SMILES string.
POST /api/fingerprints/explain – describe predicted bits against a candidate structure.
"""
import logging

from fastapi import APIRouter, HTTPException, status

from app.schemas import (
    BitExplainRequest, BitExplainResponse,
    BitHighlightRequest, BitHighlightResponse,
    FingerprintIndicesRequest, FingerprintIndicesResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/fingerprints/indices",
    response_model=FingerprintIndicesResponse,
    status_code=status.HTTP_200_OK,
)
async def fingerprint_indices(body: FingerprintIndicesRequest) -> FingerprintIndicesResponse:
    """
    Return the sorted list of active entropy-fingerprint bit indices for a SMILES string.
    Useful for visualising which substructure features are present in a molecule.
    """
    from app.manifest import resolve_model_id
    from app.registry import ensure_loaded
    import asyncio

    mid, err = resolve_model_id(body.model_id)
    if err:
        raise HTTPException(status_code=err[0], detail=err[1])

    try:
        session = await asyncio.to_thread(ensure_loaded, mid)
        indices = await asyncio.to_thread(session.fp_indices_for_smiles, body.smiles)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("fingerprint_indices error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Fingerprint computation failed.")

    if indices is None:
        raise HTTPException(
            status_code=422,
            detail="Failed to compute fingerprint indices for the provided SMILES.",
        )

    return FingerprintIndicesResponse(smiles=body.smiles, fp_indices=indices)


@router.post(
    "/fingerprints/explain",
    response_model=BitExplainResponse,
    status_code=status.HTTP_200_OK,
)
async def fingerprints_explain(body: BitExplainRequest) -> BitExplainResponse:
    """
    Describe the predicted fingerprint bits against a candidate structure: what
    substructure each bit encodes, how likely it is to be present, and where it
    sits in the candidate. Ordered by disagreement — confident bits the
    candidate lacks come first.
    """
    from app.manifest import resolve_model_id
    from app.registry import ensure_loaded
    from app.calibration import load_calibrator
    from app.bit_explain import explain_bits
    import asyncio

    mid, err = resolve_model_id(body.model_id)
    if err:
        raise HTTPException(status_code=err[0], detail=err[1])

    session = await asyncio.to_thread(ensure_loaded, mid)

    # A wrong length means the fingerprint came from a different model; scoring a
    # truncated prefix would silently mislabel every bit.
    expected_dim = getattr(session.fp_loader, "out_dim", None)
    if expected_dim is not None and len(body.pred_fp) != expected_dim:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"pred_fp has {len(body.pred_fp)} values but model {mid!r} "
                f"expects {expected_dim}. Re-run the search with this model selected."
            ),
        )

    model_root = getattr(session, "model_root", None)
    calibrator = load_calibrator(model_root) if model_root else None

    try:
        result = await asyncio.to_thread(
            explain_bits, session, body.smiles.strip(), body.pred_fp,
            body.limit, calibrator, body.include_fragment_svg,
        )
    except Exception as exc:
        logger.error("fingerprints_explain error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Bit explanation failed.")

    return BitExplainResponse(**result)


@router.post(
    "/fingerprints/highlight",
    response_model=BitHighlightResponse,
    status_code=status.HTTP_200_OK,
)
async def fingerprints_highlight(body: BitHighlightRequest) -> BitHighlightResponse:
    """
    Render a structure with one bit's environment picked out. Takes the atom and
    bond indices from /fingerprints/explain; needs no model, so it does not
    resolve or load one.
    """
    from app.config import MOLECULE_IMG_SIZE
    from app.renderer import render_bit_svg
    import asyncio

    svg = await asyncio.to_thread(
        render_bit_svg, body.smiles.strip(), body.atoms, body.bonds, MOLECULE_IMG_SIZE,
    )
    return BitHighlightResponse(smiles=body.smiles, svg=svg)
