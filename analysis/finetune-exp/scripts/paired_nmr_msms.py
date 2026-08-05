"""Which MARINA1 molecules have experimental NMR *and* experimental MS/MS?

Joins the HSQC provenance classification (marina1_hsqc_classify.py) and the MS/MS
union (union_coverage.py) against the canonicalized external NMR sets produced by
the domain-compare run. All sets share MARINA's double-pass, stereo-stripped
canonicalization, so exact SMILES joins are valid across the two analysis dirs.

Writes results/paired_nmr_msms.json and results/both_experimental_nmr_msms.parquet.
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "results"
DOMAIN = HERE.parent / "domain-compare/smiles"


def load_smiles(name):
    return set((DOMAIN / name).read_text().split("\n")) - {""}


hsqc = pd.read_parquet(RESULTS / "marina1_hsqc_classified.parquet")
msms = pd.read_parquet(RESULTS / "union_matched_molecules.parquet")

marina1 = set(hsqc.smiles)
jeol = set(hsqc[hsqc.provenance == "experimental_jeol"].smiles)
ms = set(msms.smiles)
negative = set(msms[msms.has_negative].smiles)

# External experimental NMR, restricted to MARINA1's chemical space.
# Both are 1D 13C/1H only -- neither carries HSQC.
nmrshiftdb2 = load_smiles("nmrshiftdb2.txt") & marina1
nmrexp = load_smiles("nmrexp.txt") & marina1

union_nmr = jeol | nmrshiftdb2 | nmrexp
split = hsqc.set_index("smiles").split


def enrichment(subset):
    """Observed/expected co-occurrence with MS/MS under independence."""
    expected = len(subset) * len(ms) / len(marina1)
    return len(subset & ms) / expected if expected else float("nan")


out = {
    "marina1_hsqc_molecules": len(marina1),
    "experimental_nmr": {
        "jeol_hsqc": len(jeol),
        "nmrshiftdb2_1d": len(nmrshiftdb2),
        "nmrexp_1d": len(nmrexp),
        "union": len(union_nmr),
        "new_vs_jeol": len(union_nmr - jeol),
        "pairwise": {
            "jeol_nmrshiftdb2": len(jeol & nmrshiftdb2),
            "jeol_nmrexp": len(jeol & nmrexp),
            "nmrshiftdb2_nmrexp": len(nmrshiftdb2 & nmrexp),
        },
    },
    "experimental_msms_union": len(ms),
    "paired": {
        name: {
            "n": len(s & ms),
            "frac_of_set": len(s & ms) / len(s),
            "enrichment_vs_independence": enrichment(s),
        }
        for name, s in [
            ("jeol_hsqc", jeol),
            ("nmrshiftdb2_1d", nmrshiftdb2),
            ("nmrexp_1d", nmrexp),
            ("union_any_experimental_nmr", union_nmr),
        ]
    },
}

both = union_nmr & ms
out["paired"]["union_any_experimental_nmr"]["by_split"] = (
    split.loc[sorted(both)].value_counts().to_dict()
)
out["paired"]["union_any_experimental_nmr"]["negative_mode"] = len(both & negative)
out["paired"]["jeol_hsqc"]["by_split"] = (
    split.loc[sorted(jeol & ms)].value_counts().to_dict()
)

(RESULTS / "paired_nmr_msms.json").write_text(json.dumps(out, indent=2))

hsqc[hsqc.smiles.isin(jeol & ms)].merge(msms, on="smiles").to_parquet(
    RESULTS / "both_experimental_nmr_msms.parquet", index=False
)

print(json.dumps(out, indent=2))
