"""Model discovery endpoint."""
from fastapi import APIRouter, status

router = APIRouter()


@router.get("/models", status_code=status.HTTP_200_OK)
async def models():
    """List all available models and their load status."""
    from app.manifest import list_models, get_default_model_id
    from app.registry import is_loaded

    entries    = list_models()
    default_id = get_default_model_id()

    return {
        "default_model_id": default_id,
        "models": [
            {
                "id":           e.id,
                "type":         e.type,
                "root":         e.root_rel,
                "default":      e.default,
                "loaded":       is_loaded(e.id),
                "display_name": e.display_name or e.id,
            }
            for e in entries
        ],
    }
