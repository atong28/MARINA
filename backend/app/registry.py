"""
Model registry: maps model_id → ModelSession with optional LRU eviction.
The default model (and any explicitly pinned models) are never evicted.
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from app.session import ModelSession

logger = logging.getLogger(__name__)

_registry:      Dict[str, "ModelSession"] = {}
_registry_lock: threading.Lock           = threading.Lock()
_lru:           List[str]                = []   # oldest → newest
_pinned:        Set[str]                 = set()

# One lock per model id, so two callers racing for the same cold model do not
# each build a full copy. Guarded by _registry_lock.
_load_locks:    Dict[str, threading.Lock] = {}


def _load_lock_for(model_id: str) -> threading.Lock:
    with _registry_lock:
        lock = _load_locks.get(model_id)
        if lock is None:
            lock = _load_locks[model_id] = threading.Lock()
        return lock


def _touch(model_id: str) -> None:
    """Move model_id to the end of the LRU list (most recently used)."""
    if model_id in _lru:
        _lru.remove(model_id)
    _lru.append(model_id)


def _evict_if_needed() -> None:
    """Evict the least recently used unpinned model if at capacity."""
    from app.config import MAX_LOADED_MODELS
    if MAX_LOADED_MODELS <= 0 or len(_registry) < MAX_LOADED_MODELS:
        return
    for candidate in list(_lru):
        if candidate not in _pinned and candidate in _registry:
            del _registry[candidate]
            _lru.remove(candidate)
            logger.info("Evicted model %s (LRU, max_loaded=%d)", candidate, MAX_LOADED_MODELS)
            return


def get(model_id: str) -> Optional["ModelSession"]:
    """Return the loaded session for model_id, or None if not in registry."""
    with _registry_lock:
        session = _registry.get(model_id)
        if session is not None:
            _touch(model_id)
        return session


def is_loaded(model_id: str) -> bool:
    return get(model_id) is not None


def load(model_id: str, model_root: Optional[str] = None) -> "ModelSession":
    """
    Load a model by id. Uses model_root if given; otherwise resolves via manifest.

    Thread-safe: at most one load per model_id runs at a time. The per-id lock is
    held across the load itself — the registry lock is not, so other model ids
    stay servable while a slow load is in flight.
    """
    with _registry_lock:
        if model_id in _registry:
            _touch(model_id)
            return _registry[model_id]

    with _load_lock_for(model_id):
        # Another caller may have finished the load while we waited.
        with _registry_lock:
            if model_id in _registry:
                _touch(model_id)
                return _registry[model_id]
            _evict_if_needed()

        root = model_root
        model_type: Optional[str] = None
        if root is None:
            from app.manifest import get_model_info
            from app.config import MODEL_ROOT, DEFAULT_MODEL_ID
            info = get_model_info(model_id)
            if info is not None:
                root = info.root
                model_type = info.type
            elif model_id == DEFAULT_MODEL_ID:
                root = MODEL_ROOT
            else:
                raise RuntimeError(f"Unknown model_id {model_id!r} – not in models.json")

        from app.session import ModelSession
        logger.info("Loading model %s (type=%s) from %s", model_id, model_type or "marina", root)
        session = ModelSession.from_model_root(root, model_type=model_type)

        with _registry_lock:
            _registry[model_id] = session
            _touch(model_id)
        logger.info("Model %s loaded successfully", model_id)
        return session


def pin(model_id: str) -> None:
    """Mark model_id as pinned so LRU never evicts it."""
    with _registry_lock:
        _pinned.add(model_id)


def ensure_loaded(model_id: str, model_root: Optional[str] = None) -> "ModelSession":
    """Return existing session or load on demand."""
    session = get(model_id)
    if session is None:
        session = load(model_id, model_root=model_root)
    return session
