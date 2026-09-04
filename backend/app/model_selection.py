"""
Dynamic model selection.

The website no longer asks the user to pick a checkpoint. Instead, for the exact
set of input modalities a request carries, we pick the checkpoint that scores
best on that combination — so an HSQC-only request is served by whichever model
ranks the true structure best from HSQC alone, and adding MS narrows the field to
models that accept MS at all.

Inputs to the decision:
  * each model's declared ``input_types`` (manifest, from params.json) — a model
    is only *eligible* for inputs it was trained to accept;
  * each model's ``metrics.json`` (schema: docs/model-metrics-schema.md) — the
    per-combination ``mean_cos`` we rank eligible models by.

If no model has a metric for the exact combination we fall back deterministically
(largest measured subset, then the manifest default), so selection degrades to
the old "default model" behaviour rather than failing when metrics are absent.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

METRICS_FILENAME = "metrics.json"

# Mirror of src/modules/core/const.INPUTS_CANONICAL_ORDER. Kept local so selection
# stays importable without pulling in the model library (and so tests need no src).
CANONICAL_ORDER = ["hsqc", "c_nmr", "h_nmr", "mass_spec", "mass_spec_neg", "mw", "formula"]

# Ordered (eval_set, split) preference; first present wins (docs §6):
#   benchmark/both (real-world, pooled) → simulated test.
SELECTION_PREFERENCE: Tuple[Tuple[str, str], ...] = (("benchmark", "both"), ("test", "test_all7pop"))

# The single metric selection ranks on. strict_top* are stored but unused today.
RANK_METRIC = "mean_cos"


ComboKey = Tuple[str, ...]


def combo_key(modalities: Iterable[str]) -> ComboKey:
    """Canonicalise a set of modalities into a hashable, order-independent key."""
    uniq = set(modalities)
    return tuple(sorted(
        uniq,
        key=lambda m: CANONICAL_ORDER.index(m) if m in CANONICAL_ORDER else len(CANONICAL_ORDER),
    ))


@dataclass
class Selection:
    model_id:     str
    display_name: Optional[str]
    auto_selected: bool
    reason:       str
    score:        Optional[float] = None


# ── metrics.json → {combo_key: mean_cos} cache ────────────────────────────────

_scores_cache: Dict[str, Tuple[float, Dict[ComboKey, float]]] = {}  # model_id → (mtime, scores)
_cache_lock = threading.Lock()


def _index_measurements(metrics: dict) -> Dict[ComboKey, float]:
    """Collapse a metrics.json into {combo_key: mean_cos} by the §6 preference."""
    by_combo: Dict[ComboKey, Dict[Tuple[str, str], float]] = {}
    for m in metrics.get("measurements", []):
        mods = m.get("modalities")
        if not isinstance(mods, list):
            continue
        score = (m.get("metrics") or {}).get(RANK_METRIC)
        if not isinstance(score, (int, float)):
            continue
        by_combo.setdefault(combo_key(mods), {})[(m.get("eval_set"), m.get("split"))] = float(score)

    out: Dict[ComboKey, float] = {}
    for key, sources in by_combo.items():
        for pref in SELECTION_PREFERENCE:
            if pref in sources:
                out[key] = sources[pref]
                break
    return out


def _combo_scores(model_id: str, model_root: str) -> Dict[ComboKey, float]:
    """Load & cache a model's {combo_key: mean_cos}, refreshing if metrics.json changed."""
    path = os.path.join(model_root, METRICS_FILENAME)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {}                                   # no metrics.json → no scored combos

    with _cache_lock:
        cached = _scores_cache.get(model_id)
        if cached is not None and cached[0] == mtime:
            return cached[1]

    try:
        with open(path) as fh:
            metrics = json.load(fh)
        scores = _index_measurements(metrics)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("metrics.json for %s unreadable: %s", model_id, exc)
        scores = {}

    with _cache_lock:
        _scores_cache[model_id] = (mtime, scores)
    return scores


def clear_cache() -> None:
    """Drop the metrics cache (tests, or after swapping a metrics.json)."""
    with _cache_lock:
        _scores_cache.clear()


# ── selection ─────────────────────────────────────────────────────────────────

def select_model(present: Iterable[str]) -> Selection:
    """Pick the best checkpoint for the modalities actually supplied."""
    from app.manifest import list_models, get_default_model_id, get_model_info

    present_set = set(present)
    present_key = combo_key(present_set)
    entries = list_models()
    default_id = get_default_model_id()

    def _display(mid: str) -> Optional[str]:
        info = get_model_info(mid)
        return (info.display_name or info.id) if info else None

    if not entries:
        return Selection(default_id, None, True, "no manifest")

    # Eligibility: a model must accept every supplied modality. Models that declare
    # no input_types (e.g. spectre, or a params.json without the field) can't be
    # checked, so they're excluded from scored selection but remain the fallback.
    eligible = [e for e in entries if e.input_types and present_set <= set(e.input_types)]

    # (1) exact-combo match, ranked by mean_cos (id as deterministic tiebreak).
    scored = []
    for e in eligible:
        score = _combo_scores(e.id, e.root).get(present_key)
        if score is not None:
            scored.append((score, e.id))
    if scored:
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        best_score, best_id = scored[0]
        return Selection(best_id, _display(best_id), True, "exact combo", best_score)

    # (2) fallback: largest measured subset of the requested inputs, then score.
    best: Optional[Tuple[int, float, str]] = None
    for e in eligible:
        for key, score in _combo_scores(e.id, e.root).items():
            if set(key) <= present_set:
                cand = (len(key), score, e.id)
                if best is None or cand > best:
                    best = cand
    if best is not None:
        return Selection(best[2], _display(best[2]), True, "subset fallback", best[1])

    # (3) fallback: an eligible model with no metrics, else the manifest default.
    if eligible:
        mid = eligible[0].id
        return Selection(mid, _display(mid), True, "eligible, no metrics")
    return Selection(default_id, _display(default_id), True, "fallback default")
