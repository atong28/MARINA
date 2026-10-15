"""
POST /api/refine/search      – retrieve again after deleting a region of a candidate.
POST /api/refine/shared-core – the largest substructures the retrieved compounds share.

See app/refine.py for the fingerprint arithmetic.
"""
import asyncio
import logging
from collections import Counter

import torch
from fastapi import APIRouter, HTTPException, status

from app.schemas import (
    RefineRequest, RefineResponse, ResultCard,
    SharedCoreRequest, SharedCoreResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/refine/search", response_model=RefineResponse, status_code=status.HTTP_200_OK)
async def refine_search(body: RefineRequest) -> RefineResponse:
    """
    Subtract the selected regions from the query fingerprint and retrieve again.
    Runs in the main process: no model forward pass.
    """
    from app.manifest import resolve_model_id
    from app.registry import ensure_loaded
    from app.calibration import load_calibrator
    from app.config import MOLECULE_IMG_SIZE
    from app.result_builder import build_result_cards
    from app import refine

    mid, err = resolve_model_id(body.model_id)
    if err:
        raise HTTPException(status_code=err[0], detail=err[1])
    session = await asyncio.to_thread(ensure_loaded, mid)

    expected_dim = getattr(session.fp_loader, "out_dim", None)
    if expected_dim is not None and len(body.pred_fp) != expected_dim:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(f"pred_fp has {len(body.pred_fp)} values but model {mid!r} expects "
                    f"{expected_dim}. Re-run the search with this model selected."),
        )

    model_root = getattr(session, "model_root", None)
    calibrator = load_calibrator(model_root) if model_root else None

    def _run():
        ladders = refine._ladders(session)
        p = torch.tensor(body.pred_fp, dtype=torch.float32).clamp(0.0, 1.0)
        removed, summaries = Counter(), []
        for op in body.operations:
            counts = refine.selection_counts(session, op.smiles.strip(), op.atoms)
            removed.update(counts)
            summaries.append({
                "selection_smiles": refine.selection_smiles(op.smiles.strip(), op.atoms),
                "removed_occurrences": sum(counts.values()),
            })
        q = refine.shift_query(p, removed, ladders)
        scores = refine.score_bank(session, q, body.metric)
        n = min(body.k, scores.numel())
        top, idx = torch.topk(scores, n)
        pairs = [(int(i), float(min(max(s, 0.0), 1.0))) for s, i in zip(top.tolist(), idx.tolist())]
        cards = build_result_cards(session, pairs, q, img_size=MOLECULE_IMG_SIZE, max_cards=n)
        conf = (lambda v: calibrator(float(v))) if calibrator else float
        return {
            "results": cards,
            "query_fp": q.tolist(),
            "unexplained": refine.unexplained_bits(session, q, calibrator),
            "operations": summaries,
            "confident_before": sum(1 for v in p.tolist() if v >= 0.05 and conf(v) >= 0.5),
            "confident_after": sum(1 for v in q.tolist() if v >= 0.05 and conf(v) >= 0.5),
        }

    try:
        out = await asyncio.to_thread(_run)
    except refine.RefineUnsupported as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("refine_search error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Refined search failed.")

    return RefineResponse(
        results=[ResultCard(**c) for c in out["results"]],
        query_fp=out["query_fp"],
        unexplained=out["unexplained"],
        operations=out["operations"],
        confident_before=out["confident_before"],
        confident_after=out["confident_after"],
        calibrated=calibrator is not None,
        metric=body.metric,
        model_id=mid,
    )


@router.post("/refine/shared-core", response_model=SharedCoreResponse, status_code=status.HTTP_200_OK)
async def refine_shared_core(body: SharedCoreRequest) -> SharedCoreResponse:
    """
    Maximum common substructures of the retrieved compounds at several coverage
    levels (every compound, 80 %, 60 %, half), complete rings only, each mapped
    onto the target compound so the client can select it there. With `matches`,
    each is also located on every compound, with the geometry to draw it over
    the result cards.
    """
    from app import refine
    from app.config import MOLECULE_IMG_SIZE
    from app.renderer import depiction_geometry

    smiles = [s.strip() for s in body.smiles]
    try:
        n, cores = await asyncio.to_thread(
            refine.shared_cores, smiles, body.target.strip() if body.target else None, body.matches)
        geometries = None
        if body.matches:
            geometries = await asyncio.to_thread(
                lambda: [depiction_geometry(s, MOLECULE_IMG_SIZE) for s in smiles[:n]])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("refine_shared_core error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Shared-core search failed.")
    return SharedCoreResponse(molecules=n, cores=cores, geometries=geometries)
