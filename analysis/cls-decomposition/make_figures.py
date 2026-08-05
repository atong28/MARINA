#!/usr/bin/env python3
r"""
Figures for the CLS-decomposition writeup. Reads the result JSONs produced by
scripts/analysis/modality_contribution.py and early_attn_ablation.py, emits
vector PDFs for \includegraphics, and prints the LaTeX table bodies so no
number in the document is hand-transcribed.

    cd <marina-repo> && pixi run python3 analysis/cls-decomposition/make_figures.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))          # this directory
ANALYSIS = os.path.dirname(OUT)                            # <repo>/analysis
MARINA_ROOT = os.path.dirname(ANALYSIS)                    # <repo>

# The measurement JSONs are archived in inputs/ so this directory is
# self-contained. The generator that produced them (scripts/analysis/
# modality_contribution.py) lives on the modality-attribution branch, so the
# live results dir is only present when that branch is checked out.
RESULTS = os.path.join(OUT, 'inputs')
if not os.path.isdir(RESULTS):
    RESULTS = os.path.join(MARINA_ROOT, 'scripts', 'analysis', 'results')

# dataviz reference palette, light mode. Slots 1 and 2; charts with one series
# per panel use slot 1 only, so hue never carries identity on its own.
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

seeds = [json.load(open(f'{RESULTS}/full_seed{s}.json')) for s in (0, 1, 2)]
perlayer = json.load(open(f'{RESULTS}/perlayer_seed1_n256.json'))
depth = json.load(open(f'{RESULTS}/early_attn_ablation.json'))
NL = depth['n_layers']


def mean(fn):
    return {m: float(np.mean([fn(d, m) for d in seeds])) for m in MODS}


def bare(ax, axis='x'):
    """Recessive chrome: hairline grid on the value axis only, no top/right rules."""
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis=axis, color=GRID, lw=0.6, ls='-')
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------
# Figure 1: usage vs irreplaceability. Two measures on wildly different scales,
# so two panels rather than one chart with two x-axes. Category order is fixed
# by attribution in both panels, which is what makes the rank swap visible.
# --------------------------------------------------------------------------
attr = mean(lambda d, m: d['attribution'][m]['share_peaks_only'])
abl = mean(lambda d, m: d['ablation'][m]['drop'])

fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5))
y = np.arange(len(MODS))[::-1]

for ax, vals, title, fmt, pad in [
    (axes[0], attr, 'Attribution — usage', '{:.3f}', 0.012),
    (axes[1], abl, 'Ablation — irreplaceability', '{:.4f}', 0.00035),
]:
    v = [vals[m] for m in MODS]
    ax.barh(y, v, height=0.55, color=BLUE)
    for yi, vi in zip(y, v):
        ax.text(vi + pad, yi, fmt.format(vi), va='center', ha='left',
                fontsize=7.5, color=INK2)
    ax.set_yticks(y, [PRETTY[m] for m in MODS])
    ax.set_title(title, fontsize=8.5, color=INK, loc='left', pad=8)
    ax.set_xlim(0, max(v) * 1.28)
    bare(ax)

axes[0].set_xlabel('share of modality read budget')
axes[1].set_xlabel('fingerprint cosine lost when removed')
axes[1].set_xticks([0, 0.005, 0.010, 0.015], ['0.000', '0.005', '0.010', '0.015'])
fig.tight_layout()
fig.savefig(f'{OUT}/fig-usage-vs-irreplaceability.pdf', bbox_inches='tight')
plt.close(fig)

# --------------------------------------------------------------------------
# Figure 2: superadditivity of the NMR block, per seed.
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.3, 2.1))
x = np.arange(3)
parts = [d['redundancy']['sum_of_individual_drops'] for d in seeds]
joint = [d['redundancy']['joint_drop'] for d in seeds]
ratio = [d['redundancy']['superadditivity'] for d in seeds]

ax.bar(x - 0.14, parts, width=0.22, color=BLUE, label='sum of individual drops')
ax.bar(x + 0.14, joint, width=0.22, color=ORANGE, label='joint drop (all three removed)')
for xi, p, j, r in zip(x, parts, joint, ratio):
    ax.text(xi - 0.14, p + 0.006, f'{p:.4f}', ha='center', fontsize=7.5, color=INK2)
    ax.text(xi + 0.14, j + 0.006, f'{j:.4f}', ha='center', fontsize=7.5, color=INK2)
    ax.text(xi, j + 0.032, f'{r:.1f}×', ha='center', fontsize=9, color=INK)

ax.set_xticks(x, [f'seed {s}' for s in (0, 1, 2)])
ax.set_ylabel('fingerprint cosine lost')
ax.set_ylim(0, max(joint) * 1.55)
ax.legend(loc='upper right', fontsize=7.5, labelcolor=INK2)
bare(ax, axis='y')
fig.tight_layout()
fig.savefig(f'{OUT}/fig-superadditivity.pdf', bbox_inches='tight')
plt.close(fig)

# --------------------------------------------------------------------------
# Figure 3: injection vs survival by depth. Five modalities would need five
# hues in one panel, which fails the all-pairs CVD floors, so this facets into
# small multiples: one series per panel, identity from the row label. Shared
# y-scale down each column keeps the modalities comparable.
# --------------------------------------------------------------------------
inj = {m: np.array([perlayer['per_layer'][f'{m}|{i}']['injection'] for i in range(NL)])
       for m in MODS}
sur = {m: np.array([perlayer['per_layer'][f'{m}|{i}']['survival'] for i in range(NL)])
       for m in MODS}

fig, axes = plt.subplots(len(MODS), 2, figsize=(6.3, 6.2), sharex=True)
# Survival is a share of one common budget, so its column shares a scale.
# Injection is a raw activation norm with no cross-modality meaning, so each
# row gets its own -- a shared scale there would flatten every row but MS.
smax = max(v.max() for v in sur.values()) * 1.12
xs = np.arange(NL)

for r, m in enumerate(MODS):
    for c, (vals, vmax) in enumerate([(inj[m], inj[m].max() * 1.18), (sur[m], smax)]):
        ax = axes[r, c]
        ax.plot(xs, vals, color=BLUE, lw=1.8, solid_joinstyle='round',
                marker='o', ms=3.4, mec='white', mew=0.8)
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
            ax.set_xticks([0, 3, 6, 9, 12, 15])

fig.tight_layout()
fig.savefig(f'{OUT}/fig-layer-profile.pdf', bbox_inches='tight')
plt.close(fig)

# --------------------------------------------------------------------------
# Figure 4: attention-depth suppression. Both series are the same measure in
# the same units, so they share one axis.
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.3, 2.6))
base = depth['baseline_cos']

for key, color, label in [('suppress_first_k', BLUE, 'first $k$ blocks suppressed'),
                          ('suppress_last_k', ORANGE, 'last $k$ blocks suppressed')]:
    ks = sorted(int(k) for k in depth[key])
    cs = [depth[key][str(k)]['cos'] for k in ks]
    ax.plot(ks, cs, color=color, lw=1.8, marker='o', ms=4.2, mec='white', mew=1.0,
            solid_joinstyle='round', label=label)

ax.axhline(base, color=MUTED, lw=0.8, ls='-')
ax.text(16.1, base, f'baseline {base:.4f}', va='center', ha='left',
        fontsize=7.5, color=MUTED)
ax.set_xlabel('number of blocks with cross-attention suppressed, $k$')
ax.set_ylabel('fingerprint cosine')
ax.set_xlim(0, 16.6)
ax.set_ylim(0, 1.02)
ax.set_xticks([1, 2, 4, 6, 8, 10, 12, 14, 16])
ax.legend(loc='lower left', fontsize=7.5, labelcolor=INK2)
bare(ax, axis='y')
fig.tight_layout()
fig.savefig(f'{OUT}/fig-depth-ablation.pdf', bbox_inches='tight')
plt.close(fig)

# --------------------------------------------------------------------------
# Figure 5: the fingerprint target's own redundancy, from the companion
# fp-redundancy analysis. Panel (a) is a dot plot, not bars -- the four
# quantities span three orders of magnitude, and bars on a log axis misstate
# proportion. Panel (b) is a single-series histogram.
# --------------------------------------------------------------------------
# Archived alongside the other inputs so this directory is self-contained on the
# modality-attribution branch, where the fp-redundancy sibling does not exist.
FPR = os.path.join(OUT, 'inputs', 'bit_mi.npz')
if not os.path.exists(FPR):
    FPR = os.path.join(ANALYSIS, 'fp-redundancy', 'results', 'bit_mi.npz')
fpr = np.load(FPR)
red = fpr['redundancy']

fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5))

ladder = [('Nominal width', 16384), ('$\\sum_i H_i$', fpr['H'].sum()),
          ('Joint entropy (bound)', 147.4), ('Needed to index 519K', 19.0)]
y = np.arange(len(ladder))[::-1]
ax = axes[0]
ax.plot([v for _, v in ladder], y, 'o', color=BLUE, ms=6, mec='white', mew=1.2,
        linestyle='none')
for yi, (_, v) in zip(y, ladder):
    ax.annotate(f'{v:,.1f}'.replace('.0', '') if v >= 100 else f'{v:.1f}',
                (v, yi), textcoords='offset points', xytext=(0, 7),
                ha='center', fontsize=7.5, color=INK2)
ax.set_xscale('log')
ax.set_yticks(y, [lab for lab, _ in ladder])
ax.set_xlim(8, 60000)
ax.set_ylim(-0.6, len(ladder) - 0.3)
ax.set_xlabel('bits (log scale)')
ax.set_title('Realised capacity of the 16,384-bit FP', fontsize=8.5,
             color=INK, loc='left', pad=8)
bare(ax)

ax = axes[1]
ax.hist(red, bins=60, color=BLUE)
med = float(np.median(red))
ax.axvline(med, color=MUTED, lw=0.9)
ax.annotate(f'median {med:.2f}', (med, ax.get_ylim()[1] * 0.92),
            textcoords='offset points', xytext=(-5, 0), ha='right',
            fontsize=7.5, color=INK2)
ax.set_xlabel(r'per-bit redundancy  $\max_j I(i;j)\,/\,H_i$')
ax.set_ylabel('bits')
ax.set_xlim(0, 1)
ax.set_title('Redundancy per bit', fontsize=8.5, color=INK, loc='left', pad=8)
bare(ax, axis='y')

fig.tight_layout()
fig.savefig(f'{OUT}/fig-fp-redundancy.pdf', bbox_inches='tight')
plt.close(fig)

print('\n%%% FP redundancy summary %%%')
print(f'  bits={red.size}  sum_H={fpr["H"].sum():.1f}  median_red={med:.4f} '
      f'mean_red={red.mean():.4f}')
print(f'  >0.9: {(red > 0.9).sum()}   >0.99: {(red > 0.99).sum()}')
print(f'  P(i|j)=1 for some j: {(fpr["max_cond"] >= 1.0).sum()} bits')
g = fpr['dup_group_size']
print(f'  exact-duplicate bits: {(g > 1).sum()} in '
      f'{len(set(fpr["dup_root"][g > 1].tolist()))} groups')
print(f'  median count={np.median(fpr["counts"]):.0f} '
      f'(presence {np.median(fpr["counts"]) / 518901:.4f})')
for t in ('0.5', '0.7', '0.9', '0.99'):
    print(f'  mean partners@{t} = {fpr[f"n_partners_{t}"].mean():.2f}')

# --------------------------------------------------------------------------
# LaTeX table bodies -- printed so the document never carries a typed number.
# --------------------------------------------------------------------------
print('\n%%% TABLE 1: attribution + ablation, per seed %%%')
for m in MODS:
    po = [d['attribution'][m]['share_peaks_only'] for d in seeds]
    sh = [d['attribution'][m]['share'] for d in seeds]
    mt = [d['attribution'][m]['mod_token_share'] for d in seeds]
    dr = [d['ablation'][m]['drop'] for d in seeds]
    tok = seeds[0]['attribution'][m]['mean_live_tokens']
    print(f'{PRETTY[m]} & {np.mean(sh):.4f} & {np.mean(po):.4f} & '
          f'{np.mean(mt):+.4f} & {tok:.1f} & '
          + ' / '.join(f'{v:.4f}' for v in dr) + r' \\')

print('\n%%% TABLE 2: unattributed buckets %%%')
for k in ('ffn', 'bias', 'cls_init'):
    v = [d['unattributed'][k] for d in seeds]
    print(f'{k} & ' + ' & '.join(f'{x:.5f}' if abs(x) > 1e-6 else f'{x:.2e}'
                                 for x in v) + f' & {np.mean(v):.5f}' + r' \\')

print('\n%%% TABLE 3: superadditivity %%%')
for i, d in enumerate(seeds):
    r = d['redundancy']
    print(f"seed {i} & {d['all_inputs_cos']:.4f} & {r['sum_of_individual_drops']:.4f} & "
          f"{r['joint_drop']:.4f} & {r['superadditivity']:.2f} & "
          f"{d['agreement']['spearman_share_vs_drop']:.1f}" + r' \\')

print('\n%%% TABLE 4: per-layer survival x1e3 (rows = modality, cols = block) %%%')
for m in MODS:
    print(f'{PRETTY[m]} & ' + ' & '.join(f'{v*1e3:.1f}' for v in sur[m][8:]) + r' \\')
print('\n%%% TABLE 4b: per-layer injection (blocks 8-15) %%%')
for m in MODS:
    print(f'{PRETTY[m]} & ' + ' & '.join(f'{v:.1f}' for v in inj[m][8:]) + r' \\')
print('\n%%% peak injection block per modality %%%')
for m in MODS:
    print(f'  {m}: argmax inj = block {int(inj[m].argmax())} ({inj[m].max():.1f}), '
          f'inj at 15 = {inj[m][15]:.1f}, sur at 15 = {sur[m][15]:.4f}, '
          f'sum sur = {sur[m].sum():.4f}')
ffn_inj = np.array([perlayer['per_layer'][f'ffn|{i}']['injection'] for i in range(NL)])
ffn_sur = np.array([perlayer['per_layer'][f'ffn|{i}']['survival'] for i in range(NL)])
print(f'  ffn: max inj = {ffn_inj.max():.0f} at block {int(ffn_inj.argmax())}, '
      f'sum sur = {ffn_sur.sum():.4f}')
print(f"\nper-layer file: n={perlayer['n_molecules']}, cos={perlayer['all_inputs_cos']:.4f}")
print(f"depth file: n={depth['n_molecules']}, baseline={base:.4f}")
print('\nwrote 4 PDFs to', OUT)
