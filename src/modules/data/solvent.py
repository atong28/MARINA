"""
Solvent-offset ("global jitter") augmentation for NMR inputs; training split only.

Real spectra carry a per-spectrum global shift, most of it solvent: every 13C coordinate moves by
the same dC and every 1H coordinate by the same dH. MARINA-DB's Mnova spectra are all one
convention (CDCl3), while the journal benchmark mixes solvents. Per training molecule this draws
a target solvent from the benchmark's empirical mix and a correlated (dC, dH) from that solvent's
offset distribution, relative to Mnova, measured on 486 NP-FIDBench compounds
(wiki MARINA/experiments/marina-global-shift-offsets.md). Spectra that are already experimental
(CH-NMR-NP in MARINA-DB-OPEN) are moved from their own solvent: its mean offset is subtracted.
The model never sees the solvent, so inference is solvent-blind.
"""
import math
import os

import pyarrow.parquet as pq
import torch

# solvent -> (13C mean, 13C robust sd, 1H mean, 1H robust sd), ppm, relative to Mnova.
# "other" = the pooled distribution, used for solvents outside the four measured ones.
OFFSETS = {
    "CDCl3":   (-0.02, 0.56, -0.018, 0.062),
    "DMSO-d6": (-0.48, 0.40, -0.076, 0.079),
    "CD3OD":   (+0.70, 0.68, -0.021, 0.078),
    "C5D5N":   (+1.00, 1.36, +0.121, 0.095),
    "other":   (0.00, 0.85, 0.000, 0.090),
}
TARGET_MIX = {"CDCl3": 0.38, "DMSO-d6": 0.31, "CD3OD": 0.24, "C5D5N": 0.04, "other": 0.03}
CORR_CH = 0.37   # per-compound correlation of the 13C and 1H offsets

NMR_MODALITIES = ("hsqc", "c_nmr", "h_nmr")


# lowercase name fragments -> solvent; the four measured ones map to OFFSETS keys
SOLVENT_NAMES = {
    "CDCl3": ("cdcl3",), "DMSO-d6": ("dmso",), "CD3OD": ("cd3od", "cd3oh", "ch3od", "methanol"),
    "C5D5N": ("c5d5n", "pyridine"), "D2O": ("d2o",), "acetone": ("acetone",), "C6D6": ("c6d6",),
    "CD3CN": ("cd3cn",), "CD2Cl2": ("cd2cl2",), "DMF": ("dmf",), "THF": ("thf",),
}


def solvent_class(solvent):
    """Free-text solvent (e.g. CH-NMR-NP's 'DMSO-d6', 'CD3OD at 35C', 'pyridine-d5') -> an OFFSETS key.
    Two or more named solvents (mixtures such as 'CDCl3-CD3OD (9:1)') and unmeasured solvents -> 'other';
    additives (TFA, ND3) are ignored; missing -> None (treated as the Mnova reference)."""
    s = (solvent or "").strip().lower()
    if not s:
        return None
    found = {name for name, keys in SOLVENT_NAMES.items() if any(k in s for k in keys)}
    if len(found) == 1 and (name := found.pop()) in OFFSETS:
        return name
    return "other"


class SolventJitter:
    """Draws one per-modality (dC, dH) shift per call and applies it to a loaded input dict."""

    def __init__(self, p, source_mean):
        """p: probability a training sample is re-solvented.
        source_mean: {idx: {modality: (mean dC, mean dH)}} of each spectrum's current solvent;
        absent idx / modality = the Mnova reference (0, 0)."""
        self.p = p
        self.source_mean = source_mean
        self.targets = list(TARGET_MIX)
        self.target_probs = torch.tensor([TARGET_MIX[t] for t in self.targets])

    @classmethod
    def from_dataset(cls, root, idxs, p):
        """Source solvents from `root/nmr_sources.parquet` (MARINA-DB-OPEN): 'mnova' spectra are the
        reference; 'chnmr' spectra use their record's `chnmr_solvent`. Without the file every
        spectrum is treated as the reference."""
        path = os.path.join(root, "nmr_sources.parquet")
        if not os.path.isfile(path):
            return cls(p, {})
        cols = pq.read_schema(path).names
        t = pq.read_table(path).to_pydict()
        if "chnmr" in set(t["hsqc"]) | set(t["c_nmr"]) | set(t["h_nmr"]) and "chnmr_solvent" not in cols:
            raise ValueError(f"{path} has CH-NMR-NP spectra but no chnmr_solvent column; rebuild the dataset")
        idxs = set(idxs)
        source_mean = {}
        for i, row_idx in enumerate(t["idx"]):
            if row_idx not in idxs:
                continue
            cls_ = solvent_class(t["chnmr_solvent"][i])
            if cls_ is None:
                continue
            # unmeasured / mixed solvents: mean offset unknown, treated as the reference
            mean = (OFFSETS[cls_][0], OFFSETS[cls_][2]) if cls_ != "other" else (0.0, 0.0)
            per_mod = {m: mean for m in NMR_MODALITIES if t[m][i] == "chnmr"}
            if per_mod:
                source_mean[row_idx] = per_mod
        return cls(p, source_mean)

    def _draw_target(self):
        target = self.targets[int(torch.multinomial(self.target_probs, 1))]
        mu_c, sd_c, mu_h, sd_h = OFFSETS[target]
        z1, z2 = torch.randn(2).tolist()
        d_c = mu_c + sd_c * z1
        d_h = mu_h + sd_h * (CORR_CH * z1 + math.sqrt(1 - CORR_CH ** 2) * z2)
        return d_c, d_h

    def apply(self, idx, inputs):
        if self.p <= 0 or float(torch.rand(())) >= self.p:
            return inputs
        d_c, d_h = self._draw_target()
        src = self.source_mean.get(idx, {})
        out = dict(inputs)
        for mod in NMR_MODALITIES:
            if mod not in out:
                continue
            s_c, s_h = src.get(mod, (0.0, 0.0))
            x = out[mod].clone()
            if mod == "hsqc":
                x[:, 0] += d_c - s_c
                x[:, 1] += d_h - s_h
            elif mod == "c_nmr":
                x += d_c - s_c
            else:
                x += d_h - s_h
            out[mod] = x
        return out
