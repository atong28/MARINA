"""
Model manifest: load and validate models.json, expose helpers to resolve model
roots and identify the default model.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

SUPPORTED_TYPES = frozenset({"marina", "spectre"})


@dataclass
class ModelEntry:
    id:           str
    root:         str           # absolute resolved path
    root_rel:     str           # as written in JSON (for API responses)
    type:         str
    default:      bool
    display_name: Optional[str] = None


_manifest: Optional[List[ModelEntry]] = None


def load_models_json(path: Optional[str] = None) -> List[ModelEntry]:
    """
    Parse models.json. Returns empty list on missing file or validation error.
    Exactly one entry must have default=true (unless the list has one entry,
    which is implicitly the default).
    """
    global _manifest

    from app.config import DATA_DIR, MODELS_JSON_PATH

    p = path if path is not None else MODELS_JSON_PATH
    if not os.path.isfile(p):
        logger.debug("models.json not found at %s", p)
        return []

    try:
        with open(p) as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to parse models.json: %s", exc)
        return []

    raw = data.get("models")
    if not isinstance(raw, list) or not raw:
        logger.warning("models.json: missing or empty 'models' array")
        return []

    entries: List[ModelEntry] = []
    seen: set[str] = set()
    default_count = 0

    for i, m in enumerate(raw):
        if not isinstance(m, dict):
            continue
        mid          = m.get("id")
        root_rel     = m.get("root")
        typ          = m.get("type")
        is_default   = bool(m.get("default", False))
        display_name = m.get("display_name") or None

        if not isinstance(mid, str) or not mid:
            logger.warning("models.json[%d]: invalid 'id'", i)
            continue
        if mid in seen:
            logger.warning("models.json: duplicate id %r", mid)
            continue
        seen.add(mid)

        if not isinstance(root_rel, str) or not root_rel:
            logger.warning("models.json: model %r missing 'root'", mid)
            continue

        root_abs = root_rel if os.path.isabs(root_rel) else os.path.normpath(
            os.path.join(DATA_DIR, root_rel)
        )

        # Checked here rather than at load time: preload re-raises, so a root
        # that does not exist would otherwise crash-loop the container with a
        # traceback pointing at the checkpoint loader instead of the manifest.
        if not os.path.isdir(root_abs):
            logger.warning(
                "models.json: model %r has root %r, which does not exist (%s)",
                mid, root_rel, root_abs,
            )
            continue

        if typ not in SUPPORTED_TYPES:
            logger.warning("models.json: model %r has unsupported type %r", mid, typ)
            continue

        if is_default:
            default_count += 1

        entries.append(ModelEntry(
            id=mid, root=root_abs, root_rel=root_rel,
            type=typ, default=is_default, display_name=display_name,
        ))

    if not entries:
        return []

    if default_count == 0:
        # Implicitly treat the first entry as default
        e = entries[0]
        entries[0] = ModelEntry(
            id=e.id, root=e.root, root_rel=e.root_rel,
            type=e.type, default=True, display_name=e.display_name,
        )
    elif default_count > 1:
        logger.warning("models.json: more than one entry has default=true; keeping first")
        patched, found = [], False
        for e in entries:
            if e.default and not found:
                found = True
                patched.append(e)
            elif e.default:
                patched.append(ModelEntry(
                    id=e.id, root=e.root, root_rel=e.root_rel,
                    type=e.type, default=False, display_name=e.display_name,
                ))
            else:
                patched.append(e)
        entries = patched

    _manifest = entries
    return entries


def _ensure_loaded() -> List[ModelEntry]:
    if _manifest is not None:
        return _manifest
    return load_models_json()


def list_models() -> List[ModelEntry]:
    return _ensure_loaded()


def get_model_info(model_id: str) -> Optional[ModelEntry]:
    return next((e for e in _ensure_loaded() if e.id == model_id), None)


def get_default_model_id() -> str:
    for e in _ensure_loaded():
        if e.default:
            return e.id
    from app.config import DEFAULT_MODEL_ID
    return DEFAULT_MODEL_ID


def resolve_model_id(model_id: Optional[str]) -> tuple[str, Optional[tuple[int, str]]]:
    """
    Resolve and validate a model_id from an API request.
    Returns (resolved_id, None) on success, or (id, (status_code, detail)) on error.
    """
    entries = _ensure_loaded()
    default = get_default_model_id()
    mid = model_id or default

    if not entries:
        return default, None          # no manifest → always use default, no error

    if get_model_info(mid) is None:
        return mid, (400, f"Unknown model_id {mid!r}")

    return mid, None
