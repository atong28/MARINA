"""Molecular-formula parsing and fixed-length count-vector encoding.

Shared by the precompute step (scripts/precompute_formula.py, which writes `formula_vec`
into index.pkl) and by training (the FFN formula encoder). Both must agree on element order,
so the vocabulary lives in one place: FORMULA_ELEMENTS in core.const.
"""
import re
from typing import Dict, List

from ..core.const import FORMULA_ELEMENTS

# A formula token is an element symbol (one uppercase + optional lowercase) followed by an
# optional count. Charge signs (+/-) and anything non-alphabetic are ignored, matching how
# RDKit's CalcMolFormula strings are consumed elsewhere.
_FORMULA_TOKEN = re.compile(r'([A-Z][a-z]?)(\d*)')

_ELEMENT_INDEX = {el: i for i, el in enumerate(FORMULA_ELEMENTS)}
_STAR_INDEX = _ELEMENT_INDEX['*']


def parse_formula(formula: str) -> Dict[str, int]:
    """Parse a Hill-notation formula string into element -> count."""
    counts: Dict[str, int] = {}
    for sym, num in _FORMULA_TOKEN.findall(formula):
        if not sym:
            continue
        counts[sym] = counts.get(sym, 0) + (int(num) if num else 1)
    return counts


def formula_to_vector(formula: str) -> List[int]:
    """Fixed-length count vector over FORMULA_ELEMENTS. Unlisted elements fold into '*'."""
    vec = [0] * len(FORMULA_ELEMENTS)
    for el, n in parse_formula(formula).items():
        vec[_ELEMENT_INDEX.get(el, _STAR_INDEX)] += n
    return vec
