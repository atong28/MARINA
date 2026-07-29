#!/usr/bin/env python3
"""
Does MARINA's early cross-attention reading matter?

Per-layer attribution shows almost nothing read before block ~10 survives into
the final CLS token, and that blocks 0-2 write a near-constant feedforward vector
large enough to overwrite whatever preceded it. That is consistent with two very
different stories: either the early reads are genuinely discarded, or they matter
but reach the output through the feedforward path, which a linear decomposition
cannot follow.

This settles it directly. Suppress attention in a contiguous set of blocks -- the
block still runs its feedforward and LayerNorms, it just reads nothing -- and
measure what happens to fingerprint cosine. Sweeping the prefix shows where
reading first becomes load-bearing; sweeping the suffix gives the comparison.

Suppression is done with a forward hook on each block's MultiheadAttention, so
nothing in the model changes and the training path is untouched.

Usage:
    DATASET_ROOT=/path/to/dataset pixi run python scripts/analysis/early_attn_ablation.py \
        --ckpt .../best.ckpt --params .../params.json --limit 256
"""
import argparse
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import fields as dc_fields

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.modules.marina import MARINA, MARINAArgs
from src.modules.marina.dataset import MARINADataset, collate
from src.modules.data.fp_loader import make_fp_loader
from src.modules.core.const import DATASET_ROOT

do_cos = torch.nn.CosineSimilarity(dim=1)


@contextmanager
def attention_suppressed(model, blocks):
    """Zero the attention output of `blocks`, leaving their feedforward intact."""
    def hook(_module, _inputs, output):
        return (torch.zeros_like(output[0]), output[1])

    handles = [model.cross_blocks[i].attn.register_forward_hook(hook)
               for i in blocks]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def mean_cosine(model, batches, device):
    total, n = 0.0, 0
    for batch_inputs, fps in batches:
        logits = model({k: v.to(device) for k, v in batch_inputs.items()})
        total += do_cos(fps.to(device), (logits >= 0).float()).sum().item()
        n += fps.size(0)
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--params', required=True)
    ap.add_argument('--limit', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--out', default='scripts/analysis/results/early_attn_ablation.json')
    a = ap.parse_args()

    with open(a.params) as f:
        params = json.load(f)
    valid = {f.name for f in dc_fields(MARINAArgs)}
    args = MARINAArgs(**{k: v for k, v in params.items() if k in valid})

    fp_loader = make_fp_loader(args.fp_type, entropy_out_dim=args.out_dim,
                               retrieval_path=os.path.join(DATASET_ROOT, 'retrieval.pkl'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MARINA(args, fp_loader)
    model.load_state_dict(torch.load(a.ckpt, map_location='cpu')['state_dict'])
    model.eval().to(device)

    ds = MARINADataset(args, fp_loader, split='test')
    if 0 < a.limit < len(ds):
        g = torch.Generator().manual_seed(a.seed)
        ds = Subset(ds, torch.randperm(len(ds), generator=g)[:a.limit].tolist())
    # materialise once: every condition must see identical molecules, and the
    # sweep re-reads the same data many times
    batches = list(DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                              collate_fn=collate, num_workers=a.num_workers))

    n_layers = len(model.cross_blocks)
    base = mean_cosine(model, batches, device)
    print(f'device={device} n={sum(f.size(0) for _, f in batches)} '
          f'layers={n_layers}\nbaseline cos = {base:.4f}\n', flush=True)

    prefix, suffix = {}, {}
    print('suppress attention in the FIRST k blocks:', flush=True)
    for k in [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16]:
        with attention_suppressed(model, range(k)):
            c = mean_cosine(model, batches, device)
        prefix[k] = c
        print(f'  k={k:<3} blocks 0-{k-1:<2} cos={c:.4f}  drop={base - c:+.4f}', flush=True)

    print('\nsuppress attention in the LAST k blocks:', flush=True)
    for k in [1, 2, 3, 4, 6, 8]:
        with attention_suppressed(model, range(n_layers - k, n_layers)):
            c = mean_cosine(model, batches, device)
        suffix[k] = c
        print(f'  k={k:<3} blocks {n_layers-k}-{n_layers-1} cos={c:.4f}  '
              f'drop={base - c:+.4f}', flush=True)

    result = {
        'ckpt': a.ckpt, 'n_molecules': sum(f.size(0) for _, f in batches),
        'seed': a.seed, 'n_layers': n_layers, 'baseline_cos': base,
        'suppress_first_k': {str(k): {'cos': v, 'drop': base - v} for k, v in prefix.items()},
        'suppress_last_k': {str(k): {'cos': v, 'drop': base - v} for k, v in suffix.items()},
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
