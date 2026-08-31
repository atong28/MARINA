"""CPU smoke test for the formula encoder: data plumbing + forward pass with masking.

Avoids loading the large Arrow spectral shards by testing _load_formula/_load_mw on real
index entries (these read only the in-RAM index) and running the model forward on a synthetic
batch with correct per-modality shapes, including one dropped-formula sample.
"""
import os
import pickle
import torch

from src.modules.core.const import DATASET_ROOT, FORMULA_ELEMENTS
from src.modules.marina.args import MARINAArgs
from src.modules.marina.model import MARINA
from src.modules.data.inputs import MARINAInputLoader
from src.modules.marina.dataset import collate

torch.manual_seed(0)

# --- 1. Real data path: _load_formula reads formula_vec from the index (no Arrow needed) ---
with open(os.path.join(DATASET_ROOT, 'index.pkl'), 'rb') as f:
    data = pickle.load(f)
some = dict(list(data.items())[:4])
loader = MARINAInputLoader(DATASET_ROOT, some, split='train')
for idx, entry in some.items():
    out = loader._load_formula(idx)
    fv = out['formula']
    assert fv.shape == (1, len(FORMULA_ELEMENTS)), fv.shape
    recon = {FORMULA_ELEMENTS[i]: int(v) for i, v in enumerate(fv[0]) if v}
    print(f'idx {idx}: {entry["formula"]:>14} -> {recon}')

# --- 2. collate with a present and a dropped (absent) formula ---
present = ({'mw': loader._load_mw(0)['mw'], 'formula': loader._load_formula(0)['formula']},
           torch.zeros(4))
dropped = ({'mw': loader._load_mw(1)['mw']}, torch.zeros(4))  # no formula key -> dropped
bi, _ = collate([present, dropped])
print('collated formula shape:', bi['formula'].shape,
      '| row0 nonzero:', int(bi['formula'][0].abs().sum() > 0),
      '| row1 nonzero:', int(bi['formula'][1].abs().sum() > 0))
assert bi['formula'].shape == (2, 1, len(FORMULA_ELEMENTS))
assert bi['formula'][1].abs().sum() == 0  # dropped sample is all-zero

# --- 3. Full forward on a synthetic batch (correct shapes), one dropped formula ---
B = 3
batch = {
    'hsqc':          torch.randn(B, 5, 3),
    'c_nmr':         torch.randn(B, 4, 1),
    'h_nmr':         torch.randn(B, 4, 1),
    'mass_spec':     torch.randn(B, 6, 2),
    'mass_spec_neg': torch.randn(B, 6, 2),
    'mw':            torch.randn(B, 1, 1).abs(),
    'formula':       torch.randint(0, 30, (B, 1, len(FORMULA_ELEMENTS))).float(),
}
batch['formula'][1] = 0.0  # simulate a dropped-formula sample mid-batch

class _StubFP:
    pass

args = MARINAArgs()
assert 'formula' in args.input_types
model = MARINA(args, _StubFP())
model.eval()
with torch.no_grad():
    out = model(batch)
print('forward out:', tuple(out.shape), '| finite:', bool(torch.isfinite(out).all()))
assert out.shape == (B, args.out_dim)
assert torch.isfinite(out).all()
assert model.enc_formula is not None

# Sanity: with formula entirely absent from the batch, forward still works (branch skipped).
batch_no_formula = {k: v for k, v in batch.items() if k != 'formula'}
with torch.no_grad():
    out2 = model(batch_no_formula)
assert out2.shape == (B, args.out_dim)
print('forward (no formula key) out:', tuple(out2.shape))
print('ALL SMOKE CHECKS PASSED')
