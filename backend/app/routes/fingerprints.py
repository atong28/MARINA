"""
POST /api/fingerprints/indices – compute entropy fingerprint bit indices for a SMILES string.
"""
import logging

from fastapi import APIRouter, HTTPException, status

from app.schemas import FingerprintIndicesRequest, FingerprintIndicesResponse

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
        session = ensure_loaded(mid)
        indices = await asyncio.to_thread(session.fp_indices_for_smiles, body.smiles)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("fingerprint_indices error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    if indices is None:
        raise HTTPException(
            status_code=422,
            detail="Failed to compute fingerprint indices for the provided SMILES.",
        )

    return FingerprintIndicesResponse(smiles=body.smiles, fp_indices=indices)
