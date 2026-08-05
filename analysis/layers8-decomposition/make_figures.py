#!/usr/bin/env python3
r"""
Per-layer injection/survival figures for the trained 8-layer arms of the layers
sweep. Same measures and same visual grammar as the CLS-decomposition writeup's
fig-layer-profile, so the two are readable side by side.

    cd <marina-repo> && pixi run python3 analysis/layers8-decomposition/make_figures.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))

# dataviz reference palette, light mode -- identical to the cls-decomposition figures.
BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS = '#e1e0d9', '#c3c2b7'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 8,
    'axes.edgecolor': AXIS,
    'axes.labelcolor': INK2,
    'axes.linewidth': 0.8,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'xtick.labelcolor': INK2,
    'ytick.labelcolor': INK2,
    'text.color': INK,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'legend.frameon': False,
})

MODS = ['hsqc', 'c_nmr', 'mass_spec', 'h_nmr', 'mw']
PRETTY = {'hsqc': 'HSQC', 'c_nmr': r'$^{13}$C NMR', 'mass_spec': 'MS',
          'h_nmr': r'$^{1}$H NMR', 'mw': 'MW'}


def load(fname):
    with open(f'{OUT}/{fname}') as f:
        d = json.load(f)
    d['n_layers'] = 1 + max(int(k.split('|')[1]) for k in d['per_layer'])
    return d


def series(d, m, field):
    return np.array([d['per_layer'][f'{m}|{i}'][field] for i in range(d['n_layers'])])


def bare(ax, axis='x'):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis=axis, color=GRID, lw=0.6, ls='-')
    ax.set_axisbelow(True)


s0 = load('layers8_s0_n2048.json')
s1 = load('layers8_s1_n2048.json')
ref = load('layers16_final1_n256.json')
NL = s0['n_layers']
ARMS = [('layers8-s0', s0, BLUE, '-'), ('layers8-s1', s1, ORANGE, '-')]

# --------------------------------------------------------------------------
# Figure 1: injection vs survival by depth, 8-layer arms. Small multiples for
# the same reason as the 16-layer version -- five modalities in one panel would
# need five hues. Here hue carries seed instead, which is only two levels, and
# the two seeds tracking each other is itself the thing to see.
# --------------------------------------------------------------------------
fig, axes = plt.subplots(len(MODS), 2, figsize=(6.3, 6.2), sharex=True)
# Survival is a share of one common budget, so its column shares a scale.
# Injection is a raw activation norm with no cross-modality meaning, so each row
# gets its own.
smax = max(series(d, m, 'survival').max() for m in MODS for _, d, _, _ in ARMS) * 1.12
xs = np.arange(NL)

for r, m in enumerate(MODS):
    imax = max(series(d, m, 'injection').max() for _, d, _, _ in ARMS) * 1.18
    for c, (field, vmax) in enumerate([('injection', imax), ('survival', smax)]):
        ax = axes[r, c]
        for label, d, color, ls in ARMS:
            ax.plot(xs, series(d, m, field), color=color, lw=1.6, ls=ls,
                    solid_joinstyle='round', marker='o', ms=3.2, mec='white',
                    mew=0.8, label=label if (r == 0 and c == 0) else None)
        ax.set_ylim(0, vmax)
        ax.set_xlim(-0.6, NL - 0.4)
        bare(ax, axis='y')
        ax.tick_params(labelsize=7)
        if r == 0:
            ax.set_title('Injection  $\\|c_{k,\\ell}\\|$' if c == 0
                         else 'Survival  (share reaching the output)',
                         fontsize=8.5, color=INK, loc='left', pad=8)
        if c == 0:
            ax.set_ylabel(PRETTY[m], fontsize=8.5, color=INK, rotation=0,
                          ha='right', va='center', labelpad=10)
        if r == len(MODS) - 1:
            ax.set_xlabel('cross-attention block $\\ell$')
            ax.set_xticks(range(NL))

axes[0, 0].legend(loc='upper right', fontsize=7.5, labelcolor=INK2)
fig.tight_layout()
fig.savefig(f'{OUT}/fig-layer-profile-8L.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/fig-layer-profile-8L.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# --------------------------------------------------------------------------
# Figure 2: the depth comparison. Survival is plotted against distance from the
# output rather than block index, because that is the axis on which the two
# depths are comparable at all -- block 7 of 8 and block 15 of 16 are the same
# position in the residual stream, block 0 of 8 and block 0 of 16 are not.
# Normalised within each run so a 2048-molecule run and a 256-molecule run of
# different models sit on one scale. Hue carries depth; the two 8-layer seeds
# share a hue and are separated by line style and marker, since seed is not the
# contrast -- that they lie on top of each other is the point.
#
# Log y: the decay spans four orders of magnitude, and on a linear axis
# everything below the last two blocks is pinned to zero and unreadable. On log
# the geometric decay is a straight line, so "same slope, different length" is
# legible directly. Points at or below zero (the 16-layer run's far tail, which
# is float noise around zero) have no log position and are dropped.
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.3, 2.8))
MAXOFF, FLOOR = 7, 4e-4

for label, d, color, ls, mk in [('layers8-s0', s0, BLUE, '-', 'o'),
                                ('layers8-s1', s1, BLUE, (0, (3, 2)), 's'),
                                ('layers16 (final1)', ref, ORANGE, '-', 'o')]:
    sur = sum(series(d, m, 'survival') for m in MODS)
    pct = (100 * sur / sur.sum())[::-1]  # index = blocks from output
    off = np.arange(len(pct))
    keep = (off <= MAXOFF) & (pct > FLOOR)
    ax.plot(off[keep], pct[keep], color=color, lw=1.6, ls=ls, marker=mk, ms=4.0,
            mec='white', mew=1.0, solid_joinstyle='round', label=label)

ax.set_yscale('log')
ax.set_xlabel('blocks from the output  (0 = final cross-attention block)')
ax.set_ylabel('% of run\'s total\nmodality survival')
ax.set_xlim(-0.4, MAXOFF + 0.4)
ax.set_ylim(FLOOR, 200)
ax.set_xticks(range(MAXOFF + 1))
ax.set_yticks([0.001, 0.01, 0.1, 1, 10, 100],
              ['0.001', '0.01', '0.1', '1', '10', '100'])
ax.legend(loc='upper right', fontsize=7.5, labelcolor=INK2)
bare(ax, axis='y')
fig.tight_layout()
fig.savefig(f'{OUT}/fig-depth-alignment.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/fig-depth-alignment.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# --------------------------------------------------------------------------
# Figure 3: attention-depth suppression, 8 layers against 16. Both series inside
# a panel are the same measure in the same units, so they share an axis; the two
# depths get separate panels because their x-axes are different lengths and
# overlaying them would imply block k of 8 and block k of 16 are the same place.
# Shared y so the panels are read against each other.
# --------------------------------------------------------------------------
sup = {k: json.load(open(f'{OUT}/{v}')) for k, v in
       [('s0', 'layers8_s0_suppression_n2048.json'),
        ('s1', 'layers8_s1_suppression_n2048.json'),
        ('16L', 'layers16_suppression_n512.json')]}

fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True)
panels = [(axes[0], '8 layers', [('s0', '-'), ('s1', (0, (3, 2)))]),
          (axes[1], '16 layers (final1)', [('16L', '-')])]

for ax, title, arms in panels:
    for run, ls in arms:
        d = sup[run]
        for key, color in [('suppress_first_k', BLUE), ('suppress_last_k', ORANGE)]:
            ks = sorted(int(k) for k in d[key])
            ax.plot(ks, [d[key][str(k)]['cos'] for k in ks], color=color, lw=1.6,
                    ls=ls, marker='o', ms=3.8, mec='white', mew=1.0,
                    solid_joinstyle='round')
        ax.axhline(d['baseline_cos'], color=MUTED, lw=0.8)
    nl = sup[arms[0][0]]['n_layers']
    ax.set_xlim(0, nl + 0.6)
    ax.set_xticks([1, 2, 4, 6, 8] if nl == 8 else [1, 4, 8, 12, 16])
    ax.set_title(title, fontsize=8.5, color=INK, loc='left', pad=8)
    ax.set_xlabel('blocks suppressed, $k$')
    bare(ax, axis='y')

axes[0].set_ylim(0, 1.02)
axes[0].set_ylabel('fingerprint cosine')
# Hue carries direction, so the key goes in one panel only.
axes[0].plot([], [], color=BLUE, lw=1.6, label='first $k$ blocks')
axes[0].plot([], [], color=ORANGE, lw=1.6, label='last $k$ blocks')
axes[0].legend(loc='lower left', fontsize=7.5, labelcolor=INK2)
fig.tight_layout()
fig.savefig(f'{OUT}/fig-depth-ablation-8L.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/fig-depth-ablation-8L.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# --------------------------------------------------------------------------
# Numbers, printed so nothing in the writeup is hand-transcribed.
# --------------------------------------------------------------------------
print('%%% suppression: drop vs k %%%')
for run in ('s0', 's1', '16L'):
    d = sup[run]
    for key in ('suppress_first_k', 'suppress_last_k'):
        ks = sorted(int(k) for k in d[key])
        print(f'  {run:<4} {key[9:]:<8}' +
              ''.join(f'{d[key][str(k)]["drop"]:>8.4f}' for k in ks))
print()

print('%%% per-modality peak injection block and survival, 8-layer arms %%%')
for label, d, _, _ in ARMS:
    print(f'  {label}  (n={d["n_molecules"]}, cos={d["all_inputs_cos"]:.4f})')
    for m in MODS:
        inj, sur = series(d, m, 'injection'), series(d, m, 'survival')
        print(f'    {m:<10} argmax inj = block {int(inj.argmax())} ({inj.max():.1f}), '
              f'inj at {NL-1} = {inj[-1]:.1f}, sur at {NL-1} = {sur[-1]:.4f}, '
              f'sum sur = {sur.sum():.4f}')

print('\n%%% survival by distance from output, % of run total %%%')
for label, d in [('layers8-s0', s0), ('layers8-s1', s1), ('layers16', ref)]:
    sur = sum(series(d, m, 'survival') for m in MODS)
    pct = (100 * sur / sur.sum())[::-1]
    print(f'  {label:<12}' + ''.join(f'{v:>7.1f}' for v in pct[:8]))

print('\nwrote 2 PDFs (+PNG previews) to', OUT)
