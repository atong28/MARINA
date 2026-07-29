#!/usr/bin/env python3
"""
Measure how much each input modality contributes to MARINA's final CLS token.

Two measurements over the *same* molecules, so they can be compared per-sample:

  attribution  MARINA's cross-attention stack reads from a memory that never
               updates across blocks, so the final CLS token decomposes exactly
               into additive per-source terms (MARINA.forward_contributions).
               Projecting each term onto the final CLS direction gives a signed
               budget whose parts sum to 1.

  ablation     Re-run the same molecules with one modality withheld and measure
               the drop in fingerprint cosine.

The two are not the same quantity and neither is "importance" on its own.
Attribution measures usage: of what the CLS reads, how much comes from each
modality with everything present. Ablation measures irreplaceability: how much
worse the output gets once the model is free to compensate. MARINA trains with
50% dropout on every modality (core/const.py DROP_PERCENTAGE), so it is
explicitly optimised to route around any single missing input -- which means
ablation systematically understates modalities whose information is carried
redundantly elsewhere. Read the two columns together; the gap between them is
itself a redundancy signal.

Ablation happens at the dataloader level. Dropping a modality from `input_types`
would delete its encoder and break the strict load_state_dict -- see the note in
scripts/benchmark/eval_no_msms.py.

Usage:
    DATASET_ROOT=/path/to/dataset pixi run python scripts/analysis/modality_contribution.py \
        --ckpt /root/gurusmart/Moonshot/Checkpoints/MARINA/final1/best.ckpt \
        --params /root/gurusmart/Moonshot/Checkpoints/MARINA/final1/params.json \
        --limit 2048 \
        --out scripts/analysis/modality_contribution.json
"""
import argparse
import json
import os
import sys
from dataclasses import fields as dc_fields

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.modules.marina import MARINA, MARINAArgs
from src.modules.marina.dataset import MARINADataset, collate
from src.modules.data.fp_loader import make_fp_loader
from src.modules.core.const import DATASET_ROOT

do_cos = torch.nn.CosineSimilarity(dim=1)

# the mutually-overlapping NMR block, tested as a group against its parts
NMR_GROUP = ['hsqc', 'c_nmr', 'h_nmr']


def build_args(params: dict, ckpt: str) -> MARINAArgs:
    valid = {f.name for f in dc_fields(MARINAArgs)}
    kw = {k: v for k, v in params.items() if k in valid}
    kw.update(train=False, test=True, benchmark=False, load_from_checkpoint=ckpt)
    return MARINAArgs(**kw)


def make_loader(args, fp_loader, input_types, limit, seed, batch_size, num_workers):
    ds = MARINADataset(args, fp_loader, split='test',
                       override_input_types=list(input_types))
    if limit is not None and 0 < limit < len(ds):
        # Same seed and same split length under every condition, so all conditions
        # see the same molecules and the passes stay paired. Sampling rather than
        # taking the head, which would inherit any ordering in the split.
        g = torch.Generator().manual_seed(seed)
        ds = Subset(ds, torch.randperm(len(ds), generator=g)[:limit].tolist())
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      collate_fn=collate, num_workers=num_workers)


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def run_attribution(model, loader, device):
    """Signed projection of every contribution bucket onto the final CLS direction.

    Buckets sum to ||cls_final||, so the returned shares sum to 1.
    """
    totals, token_counts, n = {}, {}, 0
    inj, sur = {}, {}
    for batch_inputs, _ in loader:
        batch_inputs = to_device(batch_inputs, device)
        cls_final, contribs, injection, survival = model.forward_contributions(
            batch_inputs, per_layer=True)
        norm = cls_final.norm(dim=-1, keepdim=True)
        u = cls_final / norm
        for name, c in contribs.items():
            share = ((c * u).sum(-1, keepdim=True) / norm).squeeze(-1)
            totals[name] = totals.get(name, 0.0) + share.sum().item()
        for k in injection:
            inj[k] = inj.get(k, 0.0) + injection[k].sum().item()
            sur[k] = sur.get(k, 0.0) + survival[k].sum().item()
        for m, x in batch_inputs.items():
            live = (x.abs().sum(-1) != 0).sum().item()
            token_counts[m] = token_counts.get(m, 0) + live
        n += cls_final.size(0)
    per_layer = {
        f'{name}|{li}': {
            'injection': inj[(name, li)] / n,
            'survival': sur[(name, li)] / n,
            # what one unit of injected norm is worth at the output; rises with
            # depth if later LayerNorms are diluting the earlier writes
            'retention': sur[(name, li)] / inj[(name, li)] if inj[(name, li)] else float('nan'),
        }
        for (name, li) in inj
    }
    return ({k: v / n for k, v in totals.items()},
            {k: v / n for k, v in token_counts.items()}, per_layer, n)


@torch.no_grad()
def run_cosine(model, loader, device):
    """Mean fingerprint cosine, matching the `cos` metric in core/metrics.py."""
    total, n = 0.0, 0
    for batch_inputs, fps in loader:
        batch_inputs = to_device(batch_inputs, device)
        fps = fps.to(device)
        logits = model(batch_inputs)
        pred = (logits >= 0).float()
        total += do_cos(fps, pred).sum().item()
        n += fps.size(0)
    return total / n


def spearman(a, b):
    def rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--params', required=True)
    ap.add_argument('--limit', type=int, default=2048,
                    help='random subsample of the test split; <=0 for the whole split')
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0,
                    help='selects the subsample when --limit is set')
    ap.add_argument('--out', default='scripts/analysis/modality_contribution.json')
    a = ap.parse_args()

    with open(a.params) as f:
        params = json.load(f)
    args = build_args(params, a.ckpt)
    batch_size = a.batch_size or args.batch_size

    fp_loader = make_fp_loader(
        args.fp_type,
        entropy_out_dim=args.out_dim,
        retrieval_path=os.path.join(DATASET_ROOT, 'retrieval.pkl'),
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MARINA(args, fp_loader)
    model.load_state_dict(torch.load(a.ckpt, map_location='cpu')['state_dict'])
    model.eval().to(device)

    modalities = list(args.input_types)
    print(f'device={device} modalities={modalities} limit={a.limit}', flush=True)

    loader = make_loader(args, fp_loader, modalities, a.limit, a.seed, batch_size, a.num_workers)
    shares, tokens, per_layer, n = run_attribution(model, loader, device)
    baseline_cos = run_cosine(model, loader, device)
    print(f'n={n} all_inputs cos={baseline_cos:.4f} '
          f'share_sum={sum(shares.values()):.6f}', flush=True)

    # Each modality alone, then the three NMR modalities together. If they really
    # are substitutes for one another, removing all three should cost far more
    # than the sum of removing them one at a time.
    nmr = [m for m in NMR_GROUP if m in modalities]
    removals = [[m] for m in modalities]
    if len(nmr) > 1:
        removals.append(nmr)

    ablation = {}
    for group in removals:
        kept = [t for t in modalities if t not in group]
        loo = make_loader(args, fp_loader, kept, a.limit, a.seed, batch_size, a.num_workers)
        cos = run_cosine(model, loo, device)
        key = '+'.join(group)
        ablation[key] = {'removed': group, 'cos_without': cos,
                         'drop': baseline_cos - cos}
        print(f'  without {key:<22} cos={cos:.4f}  drop={baseline_cos - cos:+.4f}', flush=True)

    peak_total = sum(shares[m] for m in modalities)
    attribution = {}
    for m in modalities:
        # `share` is the fraction of the whole CLS budget (which includes the
        # unattributable ffn term); `share_peaks_only` renormalises over the five
        # modalities. Per-token is derived from the latter so every reported
        # column sits on the same normalisation.
        peaks_only = shares[m] / peak_total if peak_total else float('nan')
        attribution[m] = {
            'share': shares[m],
            'share_peaks_only': peaks_only,
            'mod_token_share': shares[f'{m}:token'],
            'mean_live_tokens': tokens.get(m, 0.0),
            'share_per_token': peaks_only / tokens[m] if tokens.get(m) else float('nan'),
        }

    order = [attribution[m]['share_peaks_only'] for m in modalities]
    drops = [ablation[m]['drop'] for m in modalities]

    redundancy = None
    nmr_key = '+'.join(nmr)
    if nmr_key in ablation:
        additive = sum(ablation[m]['drop'] for m in nmr)
        joint = ablation[nmr_key]['drop']
        redundancy = {
            'group': nmr,
            'sum_of_individual_drops': additive,
            'joint_drop': joint,
            # >1 means the group is redundant: losing all of it hurts more than
            # the parts suggest, because each one was covering for the others.
            # ~1 means the members carry independent information.
            'superadditivity': joint / additive if additive else float('nan'),
        }

    result = {
        'ckpt': a.ckpt,
        'n_molecules': n,
        'seed': a.seed,
        'modalities': modalities,
        'all_inputs_cos': baseline_cos,
        'budget_sums_to': sum(shares.values()),
        'attribution': attribution,
        'unattributed': {k: shares[k] for k in ('ffn', 'cls_init', 'bias')},
        'ablation': ablation,
        'agreement': {'spearman_share_vs_drop': spearman(order, drops)},
        'redundancy': redundancy,
        'per_layer': per_layer,
    }

    with open(a.out, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'\n{"modality":<12}{"attr share":>12}{"per token":>12}{"modtok":>10}{"abl drop":>10}')
    for m in modalities:
        at = attribution[m]
        print(f'{m:<12}{at["share_peaks_only"]:>12.4f}{at["share_per_token"]:>12.2e}'
              f'{at["mod_token_share"]:>10.4f}{ablation[m]["drop"]:>+10.4f}')
    print(f'\nunattributed: ' + '  '.join(
        f'{k}={shares[k]:+.4f}' for k in ('ffn', 'cls_init', 'bias')))
    print(f'spearman(attr share, ablation drop) = '
          f'{result["agreement"]["spearman_share_vs_drop"]:.3f}')

    n_layers = len(model.cross_blocks)
    print(f'\nper-layer survival (x1e3), summing across a row = that row\'s share:')
    print('layer'.ljust(12) + ''.join(f'{li:>7}' for li in range(n_layers)))
    for m in modalities + ['ffn']:
        row = [per_layer[f'{m}|{li}']['survival'] * 1e3 for li in range(n_layers)]
        print(f'{m:<12}' + ''.join(f'{v:>7.1f}' for v in row))
    print(f'\nper-layer retention (survival/injection, x1e3):')
    print('layer'.ljust(12) + ''.join(f'{li:>7}' for li in range(n_layers)))
    for m in modalities:
        row = [per_layer[f'{m}|{li}']['retention'] * 1e3 for li in range(n_layers)]
        print(f'{m:<12}' + ''.join(f'{v:>7.2f}' for v in row))
    if redundancy:
        print(f'\n{"+".join(redundancy["group"])} removed together:'
              f'  joint drop {redundancy["joint_drop"]:+.4f}'
              f'  vs sum of parts {redundancy["sum_of_individual_drops"]:+.4f}'
              f'  -> {redundancy["superadditivity"]:.2f}x')
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
