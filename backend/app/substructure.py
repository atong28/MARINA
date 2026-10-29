"""
Substructure search over a model's retrieval bank, for "require this region"
edits: every retrieved compound must contain the selected substructure.

An RDKit SubstructLibrary (pattern-fingerprint screen, then an exact match) over
the bank's 2-D SMILES. Building it for the 531,927-structure MARINA-DB bank takes
~40 s on 8 threads and ~1 GB, so it is built once per bank in a background thread
when the backend starts and shared by every model ranking against that bank (the
shifts-only fallback links to the flagship's metadata). A search then takes
0.1–2 s (benzene, the worst case, matches 299k compounds in 2.3 s).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)

BUILD_THREADS = max(1, min(8, (os.cpu_count() or 2) // 2))
SEARCH_THREADS = BUILD_THREADS
# How long a request waits for an index still being built before giving up.
WAIT_S = 120


class IndexNotReady(RuntimeError):
    """The bank's substructure index is still being built (or failed to build)."""


class _Index:
    def __init__(self) -> None:
        self.ready = threading.Event()
        self.lib = None
        self.error: Optional[str] = None


_indexes: Dict[str, _Index] = {}
_lock = threading.Lock()


def _key(session) -> str:
    """One index per bank: the metadata file's real path (the fallback model's is a symlink)."""
    return os.path.realpath(os.path.join(session.model_root, "metadata.json"))


def _build(index: _Index, session) -> None:
    from rdkit import RDLogger
    from rdkit.Chem import rdSubstructLibrary as SL

    t0 = time.time()
    try:
        RDLogger.DisableLog("rdApp.*")
        holder = SL.CachedTrustedSmilesMolHolder()
        n = len(session._metadata)
        for i in range(n):
            entry = session.get_entry(i) or {}
            holder.AddSmiles(entry.get("canonical_2d_smiles") or entry.get("smiles") or "")
        lib = SL.SubstructLibrary(holder)
        SL.AddPatterns(lib, numThreads=BUILD_THREADS)
        index.lib = lib
        logger.info("Substructure index: %d structures in %.0f s", n, time.time() - t0)
    except Exception as exc:                                    # pragma: no cover - logged, surfaced on use
        index.error = str(exc)
        logger.error("Substructure index build failed: %s", exc, exc_info=True)
    finally:
        index.ready.set()


def warm(session) -> None:
    """Start building the session's bank index in the background, once."""
    key = _key(session)
    with _lock:
        if key in _indexes:
            return
        index = _indexes[key] = _Index()
    threading.Thread(target=_build, args=(index, session), name="substruct-index", daemon=True).start()


def matches(session, query, wait_s: float = WAIT_S) -> Set[int]:
    """Bank rows that contain the query molecule (SMARTS or Mol)."""
    warm(session)
    index = _indexes[_key(session)]
    if not index.ready.wait(wait_s):
        raise IndexNotReady("The substructure index is still being built (about a minute after startup); "
                            "try again shortly.")
    if index.lib is None:
        raise IndexNotReady(f"The substructure index could not be built: {index.error}")
    return set(index.lib.GetMatches(query, numThreads=SEARCH_THREADS, maxResults=-1))
