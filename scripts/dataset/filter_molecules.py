"""
Filter smiles_dict.json before dataset construction.

Default filters (all enabled unless overridden):
  --mw-min 100     exclude MW < 100 (fragments, trivial molecules)
  --mw-max 1000    exclude MW > 1000 (beyond model scope)
  --require-c-or-h exclude molecules with no carbon and no hydrogen atoms

Run after process_smiles.py (i.e. after download_data.sh) and before
generate_dataset.py. Overwrites smiles_dict.json in place by default.

Usage:
    python scripts/dataset/filter_molecules.py
    python scripts/dataset/filter_molecules.py --mw-min 50 --no-mw-max
    python scripts/dataset/filter_molecules.py --dry-run
"""
import argparse
import json
import os

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm

DEFAULT_PATH = "data/cleaned/smiles_dict.json"


def _filter_reason(mol, mw_min: float | None, mw_max: float | None, require_c_or_h: bool) -> str | None:
    """Return a failure reason string, or None if the molecule passes all filters."""
    mw = rdMolDescriptors.CalcExactMolWt(mol)
    if mw_min is not None and mw < mw_min:
        return f"mw_too_small ({mw:.1f} < {mw_min})"
    if mw_max is not None and mw > mw_max:
        return f"mw_too_large ({mw:.1f} > {mw_max})"
    if require_c_or_h:
        atomic_nums = {atom.GetAtomicNum() for atom in Chem.AddHs(mol).GetAtoms()}
        if 6 not in atomic_nums and 1 not in atomic_nums:
            return "no_c_or_h"
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mw-min", type=float, default=100.0,
                        help="Exclude molecules with MW below this value (default: 100)")
    parser.add_argument("--mw-max", type=float, default=1000.0,
                        help="Exclude molecules with MW above this value (default: 1000)")
    parser.add_argument("--no-mw-min", action="store_true",
                        help="Disable the MW lower bound filter")
    parser.add_argument("--no-mw-max", action="store_true",
                        help="Disable the MW upper bound filter")
    parser.add_argument("--no-require-c-or-h", action="store_true",
                        help="Disable the C/H atom filter")
    parser.add_argument("--input", default=DEFAULT_PATH,
                        help=f"Input smiles_dict.json (default: {DEFAULT_PATH})")
    parser.add_argument("--output", default=None,
                        help="Output path (default: overwrite input)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print filter statistics without writing output")
    args = parser.parse_args()

    mw_min = None if args.no_mw_min else args.mw_min
    mw_max = None if args.no_mw_max else args.mw_max
    require_c_or_h = not args.no_require_c_or_h
    out_path = args.output or args.input

    with open(args.input) as f:
        smiles_dict = json.load(f)

    kept: dict[str, dict] = {}
    reason_counts: dict[str, int] = {}

    for smiles, entry in tqdm(smiles_dict.items(), desc="Filtering"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            reason_counts["invalid_smiles"] = reason_counts.get("invalid_smiles", 0) + 1
            continue
        reason = _filter_reason(mol, mw_min, mw_max, require_c_or_h)
        if reason is None:
            kept[smiles] = entry
        else:
            key = reason.split(" ")[0]  # strip the parenthetical detail for counting
            reason_counts[key] = reason_counts.get(key, 0) + 1

    n_total = len(smiles_dict)
    n_removed = n_total - len(kept)
    print(f"Total:   {n_total:>10,}")
    print(f"Kept:    {len(kept):>10,}")
    print(f"Removed: {n_removed:>10,}")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count:,}")

    if args.dry_run:
        print("(dry run — no file written)")
    else:
        with open(out_path, "w") as f:
            json.dump(kept, f)
        print(f"Written to {out_path}")
