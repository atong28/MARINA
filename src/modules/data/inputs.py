# inputs.py
import json
import math
import os
from pathlib import Path
from typing import Iterable, Dict, Optional
import numpy as np
import pickle
import torch
import torch.nn.functional as F

from ..core.const import INPUT_TYPES
from .fp_loader import FPLoader
from .arrow_store import open_tensor_store


def normalize_mass_spec(mass_spec: torch.Tensor, floor: float = 0.01, top_k: int = 100) -> torch.Tensor:
    """Base-peak normalize an MS/MS spectrum so every input reaches the model on
    the same [0, 1] intensity scale regardless of its source. Matches
    analysis/finetune-exp/scripts/parse_iceberg.py: intensity /= max(intensity),
    drop below 1% of the base peak, keep the top-100, sort by m/z. `mass_spec`
    is (N, >=2) with column 1 the intensity; the column width is preserved."""
    if mass_spec.ndim != 2 or mass_spec.shape[0] == 0:
        return mass_spec
    top = mass_spec[:, 1].max()
    if top <= 0:
        return mass_spec
    mass_spec = mass_spec.clone()
    mass_spec[:, 1] = mass_spec[:, 1] / top
    mass_spec = mass_spec[mass_spec[:, 1] >= floor]
    if mass_spec.shape[0] > top_k:
        mass_spec = mass_spec[torch.topk(mass_spec[:, 1], top_k).indices]
    return mass_spec[torch.argsort(mass_spec[:, 0])]

class SpectralInputLoader:
    '''
    Represents the MARINA input data types.

    - HSQC NMR ('hsqc')
    - H NMR ('h_nmr')
    - C NMR ('c_nmr')
    - MS/MS ('mass_spec')
    - Molecular Weight ('mw')
    '''
    def __init__(self, root: str, data_dict: dict, split: Optional[str] = None, dtype=torch.float32):
        '''
        In index.pkl, it is stored idx: data_dict pairs. Feed this in for initialization.
        We read from Arrow shards under {root}/arrow/{split}/{MODALITY}.parquet.
        '''
        self.root = root
        self.data_dict = data_dict
        self.dtype = dtype

        self.split = split  # 'train'|'val'|'test'

        # Arrow layout: {root}/arrow/{split}/{mod}.parquet
        self._arrow = {}
        arrow_base = os.path.join(self.root, "arrow")
        if self.split is None:
            raise ValueError("SpectralInputLoader requires split for Arrow discovery.")
        arrow_split_dir = os.path.join(arrow_base, self.split)
        if not os.path.isdir(arrow_split_dir):
            raise FileNotFoundError(f"Arrow split directory not found: {arrow_split_dir}")
        for mod in ("HSQC_NMR", "H_NMR", "C_NMR", "MassSpec", "MassSpecNeg", "HMBC_NMR", "COSY_NMR"):
            path = os.path.join(arrow_split_dir, f"{mod}.parquet")
            if os.path.isfile(path):
                self._arrow[mod] = open_tensor_store(path)

    # ---- public API ----
    def load(self, idx, input_types: Iterable[INPUT_TYPES], jittering: float = 0.0, augmenter=None) -> Dict[str, torch.Tensor]:
        '''
        Load spectral inputs from Arrow shards.
        Returns dict of requested input types and their data.
        '''
        data_inputs = {}
        for input_type in input_types:
            data_inputs.update(getattr(self, f'_load_{input_type}')(idx, jittering))
        if augmenter is not None:
            for mod in ('hsqc', 'c_nmr', 'h_nmr'):
                if mod in data_inputs:
                    data_inputs[mod] = augmenter.augment(data_inputs[mod], mod)
        return data_inputs

    # ---- helpers ----
    def _get_tensor(self, idx: int, modality_dir: str) -> torch.Tensor:
        """
        Read a tensor from Arrow shard (required).
        `modality_dir` is e.g. 'HSQC_NMR', 'H_NMR', 'C_NMR', 'MassSpec'.
        """
        if modality_dir not in self._arrow:
            raise FileNotFoundError(f"Missing Arrow shard for modality {modality_dir}")
        t = self._arrow[modality_dir].get_tensor(idx)
        return t.to(dtype=self.dtype)

    # ---- individual modality loaders ----
    def _load_hsqc(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        hsqc = self._get_tensor(idx, 'HSQC_NMR')
        if jittering > 0:
            hsqc[:,0] = hsqc[:,0] + torch.randn_like(hsqc[:,0]) * jittering
            hsqc[:,1] = hsqc[:,1] + torch.randn_like(hsqc[:,1]) * jittering * 0.1
        return {'hsqc': hsqc}

    def _load_c_nmr(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def _load_h_nmr(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def _load_mass_spec(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def _load_mass_spec_neg(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def _load_mw(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def _load_hmbc(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def _load_cosy(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

# MARINA2.0 ceiling 2D shards (scripts/marina_db/build_2d.py):
#   HMBC_NMR row = [dC, dH, ptype, ctype, n_bonds, |J|]   COSY_NMR row = [dHa, dHb, cls, ptype_a, ptype_b, exch, |J|]
# The model sees only the first two columns; the rest drive the calibrated training-time dropout
# (wiki/experiments/marina-experiments/hmbc-cosy-dropout-calibration.md).
PTYPE_NAMES = ('CH3', 'CH2', 'CH', 'arom', 'olef', 'exch')
CTYPE_NAMES = ('protonated', 'quaternary', 'carbonyl')
DROPOUT_2D_PARAMS_DEFAULT = Path(__file__).resolve().parents[3] / 'analysis' / 'hmbc-cosy-calibration' / 'results' / 'marina2_dropout_params.json'


class MARINAInputLoader(SpectralInputLoader):
    # ---- MARINA2.0 2D modalities -------------------------------------------------------
    dropout_2d: bool = False
    hmbc_max_peaks: int = 200
    cosy_max_peaks: int = 120

    def configure_2d(self, args, split: str) -> None:
        '''Load the calibrated dropout table; dropout is applied on the train split only.'''
        self.hmbc_max_peaks = int(getattr(args, 'hmbc_max_peaks', 200))
        self.cosy_max_peaks = int(getattr(args, 'cosy_max_peaks', 120))
        self.dropout_2d = bool(getattr(args, 'dropout_2d', True)) and split == 'train'
        if not self.dropout_2d:
            return
        path = getattr(args, 'dropout_2d_params', '') or str(DROPOUT_2D_PARAMS_DEFAULT)
        with open(path) as f:
            p = json.load(f)
        self._P_hmbc = torch.tensor([[float(p['P_hmbc'][pt][ct]) for ct in CTYPE_NAMES] for pt in PTYPE_NAMES[:-1]],
                                    dtype=torch.float32)                      # (5 carbon-bound ptypes, 3 ctypes)
        self._keep_exch = (float(p['keep_exch']['aprotic']), float(p['keep_exch']['protic']))
        self._cosy_keep = torch.tensor([float(p['cosy']['3J']), float(p['cosy']['gem'])], dtype=torch.float32)
        mm = p['molecule_multiplier']
        self._mult = (float(mm['mu']), float(mm['sigma']), float(mm['m_min']), float(mm['m_max']))

    def _molecule_draws(self):
        mu, sigma, lo, hi = self._mult
        m = math.exp(mu + sigma * torch.randn(1).item())
        m = min(max(m, lo), hi)
        protic = torch.rand(1).item() < 0.5                 # solvent class is unknown for simulated data
        exch_visible = torch.rand(1).item() < self._keep_exch[1 if protic else 0]
        return m, exch_visible

    @staticmethod
    def _apply_keep(t: torch.Tensor, keep_p: torch.Tensor, exch: torch.Tensor, exch_visible: bool) -> torch.Tensor:
        keep = torch.rand(t.shape[0]) < keep_p.clamp(0.0, 1.0)
        keep[exch] = exch_visible                            # exchangeable protons: all-or-nothing per molecule
        if not bool(keep.any()):                             # never hand the model an empty modality
            cand = torch.nonzero(~exch).flatten()
            pick = cand[torch.randint(len(cand), (1,))] if len(cand) else torch.randint(t.shape[0], (1,))
            keep[pick] = True
        return t[keep]

    def _cap(self, xy: torch.Tensor, max_rows: int) -> torch.Tensor:
        if xy.shape[0] <= max_rows:
            return xy
        if self.dropout_2d:
            return xy[torch.randperm(xy.shape[0])[:max_rows]]
        return xy[:max_rows]                                 # deterministic for val/test (rows are sorted by dC/dH)

    def _load_hmbc(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        t = self._get_tensor(idx, 'HMBC_NMR')                # (N, 6)
        if self.dropout_2d and t.shape[0] > 0:
            m, exch_visible = self._molecule_draws()
            ptype = t[:, 2].long()
            ctype = t[:, 3].long()
            exch = ptype == len(PTYPE_NAMES) - 1
            keep_p = torch.zeros(t.shape[0])
            keep_p[~exch] = m * self._P_hmbc[ptype[~exch], ctype[~exch]]
            t = self._apply_keep(t, keep_p, exch, exch_visible)
        xy = t[:, :2].clone()
        if jittering > 0:
            xy[:, 0] = xy[:, 0] + torch.randn_like(xy[:, 0]) * jittering
            xy[:, 1] = xy[:, 1] + torch.randn_like(xy[:, 1]) * jittering * 0.1
        return {'hmbc': self._cap(xy, self.hmbc_max_peaks)}

    def _load_cosy(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        t = self._get_tensor(idx, 'COSY_NMR')                # (N, 7), one row per unordered pair
        if self.dropout_2d and t.shape[0] > 0:
            m, exch_visible = self._molecule_draws()
            cls = t[:, 2].long()
            exch = t[:, 5] > 0.5
            keep_p = m * self._cosy_keep[cls]
            t = self._apply_keep(t, keep_p, exch, exch_visible)
        xy = self._cap(t[:, :2].clone(), max(1, self.cosy_max_peaks // 2))
        xy = torch.cat([xy, xy[:, [1, 0]]], dim=0)           # symmetric spectrum: both orderings
        if jittering > 0:
            xy = xy + torch.randn_like(xy) * jittering * 0.1
        return {'cosy': xy}

    # ---- existing modalities ------------------------------------------------------------
    def _load_mw(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        mw = torch.tensor(self.data_dict[idx]['mw'], dtype=self.dtype)
        mw = mw.view(1, 1)
        return {'mw': mw}

    def _load_formula(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        # Precomputed element-count vector, read straight from the in-RAM index (like mw):
        # no disk I/O. Shape (1, n_elements) so it flows through collate as a single token.
        formula = torch.tensor(self.data_dict[idx]['formula_vec'], dtype=self.dtype)
        return {'formula': formula.view(1, -1)}

    def _load_mass_spec(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        mass_spec = self._get_tensor(idx, 'MassSpec')
        mass_spec = normalize_mass_spec(mass_spec)
        if jittering > 0:
            noise = torch.zeros_like(mass_spec)
            noise[:, 0].copy_(torch.randn_like(mass_spec[:, 0]) * mass_spec[:, 0] / 100_000)
            noise[:, 1].copy_(torch.randn_like(mass_spec[:, 1]) * mass_spec[:, 1] / 10)
            mass_spec = mass_spec + noise
        return {'mass_spec': mass_spec}

    def _load_mass_spec_neg(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        mass_spec = self._get_tensor(idx, 'MassSpecNeg')
        mass_spec = normalize_mass_spec(mass_spec)
        if jittering > 0:
            noise = torch.zeros_like(mass_spec)
            noise[:, 0].copy_(torch.randn_like(mass_spec[:, 0]) * mass_spec[:, 0] / 100_000)
            noise[:, 1].copy_(torch.randn_like(mass_spec[:, 1]) * mass_spec[:, 1] / 10)
            mass_spec = mass_spec + noise
        return {'mass_spec_neg': mass_spec}

    def _load_c_nmr(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        c_nmr = self._get_tensor(idx, 'C_NMR')
        c_nmr = c_nmr.view(-1,1)                   # (N,1)
        if jittering > 0:
            c_nmr = c_nmr + torch.randn_like(c_nmr) * jittering
        return {'c_nmr': c_nmr}

    def _load_h_nmr(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        h_nmr = self._get_tensor(idx, 'H_NMR')
        h_nmr = h_nmr.view(-1,1)                    # (N,1)
        if jittering > 0:
            h_nmr = h_nmr + torch.randn_like(h_nmr) * jittering * 0.1
        return {'h_nmr': h_nmr}

class SPECTREInputLoader(SpectralInputLoader):
    def _load_mw(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        return {'mw': torch.tensor(self.data_dict[idx]['mw'], dtype=self.dtype)}

    def _load_mass_spec(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        mass_spec = self._get_tensor(idx, 'MassSpec')
        mass_spec = normalize_mass_spec(mass_spec)
        mass_spec = F.pad(mass_spec, (0,1), "constant", 0)
        if jittering > 0:
            noise = torch.zeros_like(mass_spec)
            noise[:, 0].copy_(torch.randn_like(mass_spec[:, 0]) * mass_spec[:, 0] / 100_000)
            noise[:, 1].copy_(torch.randn_like(mass_spec[:, 1]) * mass_spec[:, 1] / 10)
            mass_spec = mass_spec + noise
        return {'mass_spec': mass_spec}

    def _load_mass_spec_neg(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        mass_spec = self._get_tensor(idx, 'MassSpecNeg')
        mass_spec = normalize_mass_spec(mass_spec)
        mass_spec = F.pad(mass_spec, (0,1), "constant", 0)
        if jittering > 0:
            noise = torch.zeros_like(mass_spec)
            noise[:, 0].copy_(torch.randn_like(mass_spec[:, 0]) * mass_spec[:, 0] / 100_000)
            noise[:, 1].copy_(torch.randn_like(mass_spec[:, 1]) * mass_spec[:, 1] / 10)
            mass_spec = mass_spec + noise
        return {'mass_spec_neg': mass_spec}

    def _load_c_nmr(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        c_nmr = self._get_tensor(idx, 'C_NMR')
        c_nmr = c_nmr.view(-1,1)                   # (N,1)
        c_nmr = F.pad(c_nmr, (0,2), "constant", 0) # -> (N,3)
        if jittering > 0:
            c_nmr = c_nmr + torch.randn_like(c_nmr) * jittering
        return {'c_nmr': c_nmr}

    def _load_h_nmr(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        h_nmr = self._get_tensor(idx, 'H_NMR')
        h_nmr = h_nmr.view(-1,1)                    # (N,1)
        h_nmr = F.pad(h_nmr, (1,1), "constant", 0)  # -> (N,3)
        if jittering > 0:
            h_nmr = h_nmr + torch.randn_like(h_nmr) * jittering * 0.1
        return {'h_nmr': h_nmr}
    
class MFInputLoader:
    '''
    The Morgan Fingerprint groundtruth loader.
    '''
    def __init__(self, fp_loader: FPLoader):
        self.fp_loader = fp_loader

    def load(self, idx: int) -> torch.Tensor:
        return self.fp_loader.build_mfp(idx)
