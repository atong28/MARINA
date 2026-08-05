#!/usr/bin/env python3
"""Compare per-layer injection/survival between the trained 8-layer arms and the
16-layer reference. Survival is normalised within each run so the two depths are
on the same scale."""
import json
import os
import sys

# Resolve the JSONs next to this file, so cwd does not matter.
HERE = os.path.dirname(os.path.abspath(__file__))

MODS = ['hsqc', 'c_nmr', 'h_nmr', 'mw', 'mass_spec']


def load(path):
    with open(path) as f:
        d = json.load(f)
    n = 1 + max(int(k.split('|')[1]) for k in d['per_layer'])
    return d, n


def survival_by_layer(d, n, buckets):
    return [sum(d['per_layer'][f'{m}|{li}']['survival'] for m in buckets)
            for li in range(n)]


def injection_by_layer(d, n, buckets):
    return [sum(d['per_layer'][f'{m}|{li}']['injection'] for m in buckets)
            for li in range(n)]


runs = [('layers8-s0', os.path.join(HERE, 'layers8_s0_n2048.json')),
        ('layers8-s1', os.path.join(HERE, 'layers8_s1_n2048.json')),
        ('layers16-final1', os.path.join(HERE, 'layers16_final1_n256.json'))]

print('=== Modality survival per block, as % of that run\'s total modality survival ===')
for label, path in runs:
    d, n = load(path)
    sur = survival_by_layer(d, n, MODS)
    tot = sum(sur)
    print(f'\n{label} ({n} layers, total modality survival {tot:.4f})')
    print('  block  ' + ''.join(f'{li:>7}' for li in range(n)))
    print('  %tot   ' + ''.join(f'{100*v/tot:>7.1f}' for v in sur))

print('\n=== Concentration summary ===')
print(f'{"run":<18}{"last blk %":>12}{"last 2 %":>10}{"first half %":>14}'
      f'{"blocks >1%":>12}{"blocks >5%":>12}')
for label, path in runs:
    d, n = load(path)
    sur = survival_by_layer(d, n, MODS)
    tot = sum(sur)
    frac = [100 * v / tot for v in sur]
    print(f'{label:<18}{frac[-1]:>12.1f}{sum(frac[-2:]):>10.1f}'
          f'{sum(frac[:n//2]):>14.2f}{sum(1 for v in frac if v > 1):>12}'
          f'{sum(1 for v in frac if v > 5):>12}')

print('\n=== Injection per block (mean norm written), modalities only ===')
for label, path in runs:
    d, n = load(path)
    inj = injection_by_layer(d, n, MODS)
    print(f'{label:<18}' + ''.join(f'{v:>7.2f}' for v in inj))

print('\n=== Retention (survival/injection) x1e3, modalities only ===')
for label, path in runs:
    d, n = load(path)
    sur = survival_by_layer(d, n, MODS)
    inj = injection_by_layer(d, n, MODS)
    print(f'{label:<18}' + ''.join(f'{1e3*s/i if i else 0:>7.2f}'
                                   for s, i in zip(sur, inj)))

print('\n=== FFN survival per block, % of run total FFN survival ===')
for label, path in runs:
    d, n = load(path)
    sur = survival_by_layer(d, n, ['ffn'])
    tot = sum(sur)
    print(f'{label:<18}' + ''.join(f'{100*v/tot:>7.1f}' for v in sur))

print('\n=== Headline aggregates ===')
for label, path in runs:
    d, n = load(path)
    print(f'{label:<18} cos={d["all_inputs_cos"]:.4f}  n={d["n_molecules"]}  '
          f'ffn share={d["unattributed"]["ffn"]:.4f}  '
          f'NMR superadditivity={d["redundancy"]["superadditivity"]:.2f}x')
