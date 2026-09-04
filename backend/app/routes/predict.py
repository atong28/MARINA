"""
POST /api/predict – predict molecular structures from spectral data.
"""
import asyncio
import logging

import torch
from fastapi import APIRouter, HTTPException, Request, status

from app.schemas import PredictRequest, PredictResponse, ResultCard
from app.compute_pool import ComputeOverloadedError, ComputeTimeoutError
from app.stats import record_query

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
async def predict(request: Request, body: PredictRequest):
    """
    Run MARINA/SPECTRE inference on spectral data and return top-k candidate molecules.
    """
    from app.manifest import resolve_model_id
    from app.model_selection import select_model
    from app.registry import ensure_loaded
    from app.config import MAX_TOP_K, PREDICT_TIMEOUT_S, MOLECULE_IMG_SIZE
    from app.result_builder import build_result_cards
    from app.session import MWDataUnavailable, FormulaDataUnavailable

    if body.mw_min is not None and body.mw_max is not None and body.mw_min > body.mw_max:
        raise HTTPException(status_code=400, detail="mw_min cannot exceed mw_max")

    k = min(body.k, MAX_TOP_K)

    raw = {}
    if body.raw.hsqc:      raw["hsqc"]      = body.raw.hsqc
    if body.raw.h_nmr:     raw["h_nmr"]     = body.raw.h_nmr
    if body.raw.c_nmr:     raw["c_nmr"]     = body.raw.c_nmr
    if body.raw.mass_spec:     raw["mass_spec"]     = body.raw.mass_spec
    if body.raw.mass_spec_neg: raw["mass_spec_neg"] = body.raw.mass_spec_neg
    if body.raw.mw:            raw["mw"]            = body.raw.mw
    if body.raw.formula:       raw["formula"]       = body.raw.formula

    # A prediction needs at least one *spectral* modality; mw/formula are only
    # descriptors and carry no structure on their own (a bare MW retrieves nothing
    # meaningful). This is also why those descriptor-only combos are excluded from
    # the metrics matrix (docs/model-metrics-schema.md §7).
    _SPECTRAL = {"hsqc", "c_nmr", "h_nmr", "mass_spec", "mass_spec_neg"}
    if not (raw.keys() & _SPECTRAL):
        raise HTTPException(status_code=400, detail="At least one spectral input is required")

    # Model choice: an explicit model_id pins the checkpoint (back-compat / testing);
    # otherwise auto-select the best checkpoint for the modalities actually supplied.
    if body.model_id:
        mid, err = resolve_model_id(body.model_id)
        if err:
            raise HTTPException(status_code=err[0], detail=err[1])
        selection = None
    else:
        selection = select_model(raw.keys())
        mid = selection.model_id

    # {element: (min, max)} for the atom-count filter; picklable for the worker pool.
    formula_filter = ({c.element: (c.min, c.max) for c in body.formula_filter}
                      if body.formula_filter else None)

    try:
        pool = request.app.state.compute_pool

        if pool is not None:
            # Production path: delegate to a worker process.
            scores, global_idxs, pred_fp = await pool.run(
                "predict",
                {
                    "raw_inputs":     raw,
                    "k":              k,
                    "model_id":       mid,
                    "mw_min":         body.mw_min,
                    "mw_max":         body.mw_max,
                    "formula_filter": formula_filter,
                },
                timeout=PREDICT_TIMEOUT_S,
                request_id=body.request_id,
            )
        else:
            # Dev / no-pool path: run inference in a thread so the event loop stays free.
            from app.predictor import predict_from_raw
            scores, global_idxs, pred_fp = await asyncio.wait_for(
                asyncio.to_thread(
                    predict_from_raw,
                    raw_inputs=raw,
                    k=k,
                    model_id=mid,
                    mw_min=body.mw_min,
                    mw_max=body.mw_max,
                    formula_filter=formula_filter,
                ),
                timeout=PREDICT_TIMEOUT_S,
            )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Prediction timed out.")
    except ComputeOverloadedError:
        raise HTTPException(status_code=503, detail="Server is busy. Please retry shortly.")
    except ComputeTimeoutError:
        raise HTTPException(status_code=504, detail="Prediction timed out.")
    except (MWDataUnavailable, FormulaDataUnavailable) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("predict error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed.")

    # Both of these are heavy and synchronous — loading a cold model, and
    # rendering k molecules with RDKit — so they must not run on the event loop.
    session     = await asyncio.to_thread(ensure_loaded, mid)
    pred_tensor = torch.tensor(pred_fp, dtype=torch.float32)
    pairs       = list(zip(global_idxs, scores))
    cards       = await asyncio.to_thread(
        build_result_cards, session, pairs, pred_tensor,
        img_size=MOLECULE_IMG_SIZE, max_cards=k,
    )

    record_query("predict")

    result_cards = [ResultCard(**c) for c in cards]
    from app.manifest import get_model_info
    info = get_model_info(mid)
    return PredictResponse(
        results=result_cards,
        total_count=len(result_cards),
        offset=0,
        limit=len(result_cards),
        pred_fp=pred_fp,
        model_id=mid,
        model_display_name=(selection.display_name if selection
                            else (info.display_name or info.id) if info else mid),
        auto_selected=selection is not None,
    )
