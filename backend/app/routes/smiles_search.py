"""
POST /api/smiles-search – nearest-neighbour retrieval from a SMILES query.
"""
import logging

import torch
from fastapi import APIRouter, HTTPException, Request, status

from app.schemas import SmilesSearchRequest, SmilesSearchResponse, ResultCard
from app.stats import record_query

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/smiles-search", response_model=SmilesSearchResponse, status_code=status.HTTP_200_OK)
async def smiles_search(request: Request, body: SmilesSearchRequest):
    """
    Compute the MARINA fingerprint for a SMILES string and retrieve similar molecules.
    Runs in the main process (no model forward pass needed).
    """
    from app.manifest import resolve_model_id
    from app.registry import ensure_loaded
    from app.config import MAX_TOP_K, MOLECULE_IMG_SIZE
    from app.result_builder import build_result_cards

    mid, err = resolve_model_id(body.model_id)
    if err:
        raise HTTPException(status_code=err[0], detail=err[1])

    smiles = body.smiles.strip()
    if not smiles:
        raise HTTPException(status_code=400, detail="SMILES string cannot be empty")

    if body.mw_min is not None and body.mw_max is not None and body.mw_min > body.mw_max:
        raise HTTPException(status_code=400, detail="mw_min cannot exceed mw_max")

    k = min(body.k, MAX_TOP_K)

    try:
        import asyncio
        session = await asyncio.to_thread(ensure_loaded, mid)

        fp = await asyncio.to_thread(session.fp_loader.build_mfp_for_smiles, smiles)
        if fp is None:
            raise HTTPException(status_code=422, detail="Invalid SMILES or fingerprint generation failed")

        # build_mfp_for_smiles returns a torch.Tensor (from torch.from_numpy)
        query_tensor = fp.clone().detach().float() if isinstance(fp, torch.Tensor) else torch.tensor(fp, dtype=torch.float32)

        def _retrieve():
            rs, kept = session.get_filtered_rankingset(body.mw_min, body.mw_max)
            n = min(k, len(kept))
            if n == 0:
                return [], []
            sims, local_idxs = rs.retrieve_with_scores(query_tensor.unsqueeze(0), n=n)
            sim_list = sims.reshape(-1).tolist()
            idx_list = local_idxs.reshape(-1).tolist()
            global_idxs = [int(kept[li]) for li in idx_list]
            return [float(s) for s in sim_list], global_idxs

        scores, global_idxs = await asyncio.to_thread(_retrieve)
        pairs  = list(zip(global_idxs, scores))
        cards  = await asyncio.to_thread(
            build_result_cards, session, pairs, query_tensor,
            img_size=MOLECULE_IMG_SIZE, max_cards=k,
        )

        record_query("smiles_search", request)

        result_cards = [ResultCard(**c) for c in cards]
        return SmilesSearchResponse(
            results=result_cards,
            total_count=len(result_cards),
            offset=0,
            limit=len(result_cards),
            query_smiles=smiles,
            query_fp=query_tensor.tolist(),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("smiles_search error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="SMILES search failed.")
