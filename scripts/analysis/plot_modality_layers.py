#!/usr/bin/env python3
"""
Plot where MARINA reads each modality across the cross-attention stack, against
how much of each read survives into the final CLS token.

Reads the `per_layer` block written by modality_contribution.py.

Usage:
    python scripts/analysis/plot_modality_layers.py \
        --json scripts/analysis/results/full_seed1.json \
        --out scripts/analysis/results/modality_layers.png
"""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# categorical slots 1-5, fixed order, assigned to modalities by identity (not rank)
SERIES = {
    'hsqc':      ('#2a78d6', 'HSQC'),
    'c_nmr':     ('#008300', '¹³C NMR'),
    'h_nmr':     ('#e87ba4', '¹H NMR'),
    'mass_spec': ('#eda100', 'MS/MS'),
    'mw':        ('#1baf7a', 'MW'),
}
SURFACE, INK, INK2, MUTED, GRID, AXIS = (
    '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7')


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    d = json.load(open(a.json))
    pl = d['per_layer']
    n = 1 + max(int(k.split('|')[1]) for k in pl)
    xs = list(range(n))

    fig = plt.figure(figsize=(9.5, 10.5), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.62], hspace=0.42)
    ax1, ax2, ax3 = (fig.add_subplot(g) for g in gs)

    # --- panel 1: where the model reads -----------------------------------
    for m, (c, label) in SERIES.items():
        ys = [pl[f'{m}|{i}']['injection'] for i in xs]
        ax1.plot(xs, ys, color=c, linewidth=2, marker='o', markersize=4,
                 markeredgecolor=SURFACE, markeredgewidth=1, zorder=3, label=label)
        pk = max(xs, key=lambda i: ys[i])                 # label at each peak
        ax1.annotate(label, (pk, ys[pk]), textcoords='offset points',
                     xytext=(0, 9), ha='center', fontsize=9, color=INK2, zorder=4)
    ax1.set_title('Where the model reads each modality',
                  fontsize=13, color=INK, loc='left', pad=14, fontweight='bold')
    ax1.set_ylabel('injection  ‖c‖ at read time', fontsize=10, color=INK2)

    # --- panel 2: what survives to the output ------------------------------
    for m, (c, label) in SERIES.items():
        ys = [pl[f'{m}|{i}']['survival'] * 1e3 for i in xs]
        ax2.plot(xs, ys, color=c, linewidth=2, marker='o', markersize=4,
                 markeredgecolor=SURFACE, markeredgewidth=1, zorder=3, label=label)
        ax2.annotate(label, (xs[-1], ys[-1]), textcoords='offset points',
                     xytext=(7, 0), ha='left', va='center',
                     fontsize=9, color=INK2, zorder=4)
    ax2.set_title('What reaches the final CLS token',
                  fontsize=13, color=INK, loc='left', pad=14, fontweight='bold')
    ax2.set_ylabel('survival  ⟨c, ĉls⟩ / ‖cls‖   (×10⁻³)', fontsize=10, color=INK2)
    ax2.set_xlim(-0.5, n + 1.6)

    # --- panel 3: the mechanism -------------------------------------------
    ff = [pl[f'ffn|{i}']['injection'] for i in xs]
    ax3.plot(xs, ff, color=INK2, linewidth=2, marker='o', markersize=4,
             markeredgecolor=SURFACE, markeredgewidth=1, zorder=3)
    ax3.axhline(28, color=AXIS, linewidth=1.5, linestyle='--', zorder=2)
    ax3.annotate('residual stream  ≈28', (6, 28), textcoords='offset points',
                 xytext=(0, 8), ha='center', fontsize=9, color=MUTED)
    ax3.set_yscale('log')
    ax3.set_title('Why: feedforward writes dwarf the stream in early blocks',
                  fontsize=13, color=INK, loc='left', pad=14, fontweight='bold')
    ax3.set_ylabel('‖ff_out‖  (log)', fontsize=10, color=INK2)

    for ax in (ax1, ax2, ax3):
        style(ax)
        ax.set_xticks(xs)
    ax3.set_xlabel('cross-attention block', fontsize=10, color=INK2)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=5, fontsize=9,
               labelcolor=INK2, loc='lower center',
               bbox_to_anchor=(0.5, 0.038), handlelength=1.6)

    fig.text(0.5, 0.012,
             f"MARINA seed 1 · {d['n_molecules']:,}-molecule sample of the test "
             "split · summing a survival curve reproduces that modality's "
             'reported share',
             ha='center', fontsize=9, color=MUTED)

    fig.subplots_adjust(bottom=0.12)
    fig.savefig(a.out, dpi=160, facecolor=SURFACE)
    print(f'wrote {a.out}')

    print(f"\n{'layer':<7}" + ''.join(f'{SERIES[m][1]:>10}' for m in SERIES))
    for i in xs:
        print(f'{i:<7}' + ''.join(
            f"{pl[f'{m}|{i}']['survival'] * 1e3:>10.2f}" for m in SERIES))


if __name__ == '__main__':
    main()
