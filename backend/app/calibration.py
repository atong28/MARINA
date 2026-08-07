"""
Isotonic recalibration of predicted fingerprint bit probabilities.

The raw sigmoid is not a probability. Measured over the full MARINA1 test split
(analysis/bit-confidence), a bit predicted at 0.82 is actually present 0.70 of
the time, and one predicted at 0.90 is present 0.88. The error is two-sided:
the head is over-confident above ~0.45 and under-confident below it.

`calibration.json` in the model root carries a monotone lookup table fitted
there. Applying it brings every confidence bucket within 0.4 points of the
empirical presence rate. It is monotone by construction, so it changes the
number shown to a user and never the ordering of bits.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# model_root -> curve (or None when the model ships no calibration.json)
_cache: Dict[str, Optional["Calibrator"]] = {}


class Calibrator:
    def __init__(self, grid: List[float], curve: List[float], ckpt: str = "") -> None:
        self.grid = grid
        self.curve = curve
        self.ckpt = ckpt

    def __call__(self, p: float) -> float:
        """Piecewise-linear interpolation, clamped to the fitted range."""
        g, c = self.grid, self.curve
        if p <= g[0]:
            return c[0]
        if p >= g[-1]:
            return c[-1]
        # Binary search for the bracketing interval.
        lo, hi = 0, len(g) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if g[mid] <= p:
                lo = mid
            else:
                hi = mid
        span = g[hi] - g[lo]
        if span <= 0:
            return c[lo]
        return c[lo] + (c[hi] - c[lo]) * (p - g[lo]) / span


def load_calibrator(model_root: str) -> Optional[Calibrator]:
    """Return the model's calibrator, or None if it ships without one."""
    if model_root in _cache:
        return _cache[model_root]

    path = os.path.join(model_root, "calibration.json")
    cal: Optional[Calibrator] = None
    if os.path.isfile(path):
        try:
            with open(path) as f:
                d = json.load(f)
            grid, curve = list(d["grid"]), list(d["curve"])
            if len(grid) != len(curve) or len(grid) < 2:
                raise ValueError(f"grid/curve length mismatch: {len(grid)}/{len(curve)}")
            if any(b < a for a, b in zip(curve, curve[1:])):
                raise ValueError("curve is not monotone non-decreasing")
            cal = Calibrator(grid, curve, str(d.get("ckpt", "")))
            logger.info("Loaded bit calibration from %s (fitted on %s)", path, cal.ckpt)
        except Exception as exc:
            # A malformed curve must not take the endpoint down; raw values are
            # still usable, they are just over-confident.
            logger.warning("Ignoring unusable calibration at %s: %s", path, exc)
            cal = None
    else:
        logger.info("No calibration.json under %s; confidences will be raw.", model_root)

    _cache[model_root] = cal
    return cal
