"""Precompute a fixed-length molecular-formula count vector into MARINA-DB's index.pkl.

Adds `formula_vec` (list[int], length len(FORMULA_ELEMENTS)) to every entry, parsed from the
`formula` string already stored in the index. This is the zero-training-overhead path: the
vector rides in the in-RAM index exactly like `mw`, so no per-sample disk I/O is added.

Run from the repo root:  pixi run python3 scripts/precompute_formula.py
"""
import os
import pickle
import sys
import collections

# Allow `import src...` when run from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.core.const import DATASET_ROOT, FORMULA_ELEMENTS
from src.modules.data.formula import formula_to_vector, parse_formula

INDEX_PATH = os.path.join(DATASET_ROOT, 'index.pkl')


def main() -> None:
    with open(INDEX_PATH, 'rb') as f:
        data = pickle.load(f)
    print(f'Loaded {len(data)} entries from {INDEX_PATH}')

    star_idx = FORMULA_ELEMENTS.index('*')
    star_hits = collections.Counter()   # formulas that put mass into the '*' catch-all
    n_written = 0
    for _idx, entry in data.items():
        formula = entry['formula']
        vec = formula_to_vector(formula)
        entry['formula_vec'] = vec

        # Verify: every element parsed from the string is represented in the vector, and the
        # only way to land in '*' is an element outside FORMULA_ELEMENTS. Flag those.
        counts = parse_formula(formula)
        for el in counts:
            if el not in set(FORMULA_ELEMENTS):
                star_hits[el] += 1
        if vec[star_idx] > 0:
            # Reconstruct expected '*' total from unlisted elements only.
            expected_star = sum(n for el, n in counts.items() if el not in set(FORMULA_ELEMENTS))
            assert vec[star_idx] == expected_star, (formula, vec[star_idx], expected_star)
        # Heavy-atom / total-atom sanity: vector sum == sum of all parsed counts.
        assert sum(vec) == sum(counts.values()), (formula, sum(vec), sum(counts.values()))
        n_written += 1

    print(f'Computed formula_vec (dim={len(FORMULA_ELEMENTS)}) for {n_written} entries')
    if star_hits:
        print(f'WARNING: {sum(star_hits.values())} formulas used the "*" catch-all: {dict(star_hits)}')
    else:
        print('No formulas fell into the "*" catch-all (full element coverage).')

    # Spot-check a few known cases.
    checks = {
        'C20H24N2O2': {'C': 20, 'H': 24, 'N': 2, 'O': 2},
        'CH4': {'C': 1, 'H': 4},
        'C6H12O6': {'C': 6, 'H': 12, 'O': 6},
    }
    for formula, expected in checks.items():
        vec = formula_to_vector(formula)
        got = {FORMULA_ELEMENTS[i]: v for i, v in enumerate(vec) if v}
        assert got == expected, (formula, got, expected)
    print('Spot-checks passed:', list(checks))

    # Atomic replace: write to a temp file, then rename over the original.
    tmp_path = INDEX_PATH + '.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, INDEX_PATH)
    print(f'Wrote updated index to {INDEX_PATH}')

    # Confirm round-trip.
    with open(INDEX_PATH, 'rb') as f:
        reloaded = pickle.load(f)
    first = next(iter(reloaded.values()))
    assert 'formula_vec' in first and len(first['formula_vec']) == len(FORMULA_ELEMENTS)
    print('Round-trip OK. Sample entry formula_vec:', first['formula'], '->',
          {FORMULA_ELEMENTS[i]: v for i, v in enumerate(first['formula_vec']) if v})


if __name__ == '__main__':
    main()
