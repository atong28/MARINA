import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

BENCHMARK_ROOT = os.environ.get('BENCHMARK_ROOT', '/home/user/atong/Benchmark')
BENCHMARKS_DIR = os.path.join(BENCHMARK_ROOT, 'benchmarks')
PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'wiki', 'natural-products-chemistry', 'marina-benchmark')
os.makedirs(PLOT_DIR, exist_ok=True)

SEEDS = [0, 1, 2]
COLORS = ['#2196F3', '#FF9800', '#4CAF50']
BENCHMARKS = {
    'NP-MRD': 'benchmark_results',
    'Journal': 'benchmark_journal_results',
}


def load_all(suffix):
    return [
        pickle.load(open(os.path.join(BENCHMARKS_DIR, f'marina-final-run-seed-{s}_{suffix}.pkl'), 'rb'))
        for s in SEEDS
    ]


def cos_sims(data):
    return [v['predictions']['cosine_sim'].item() for v in data.values()]


def topk_hits(data):
    n = len(data)
    return [
        sum(v['predictions']['dereplication_topk'][k] for v in data.values()) / n * 100
        for k in range(1, 11)
    ]


# --- Plot 1: Cosine similarity histograms ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
bins = np.linspace(0, 1, 25)

for ax, (label, suffix) in zip(axes, BENCHMARKS.items()):
    datasets = load_all(suffix)
    for seed, data, color in zip(SEEDS, datasets, COLORS):
        sims = cos_sims(data)
        mean_val = np.mean(sims)
        ax.hist(sims, bins=bins, alpha=0.5, color=color, label=f'Seed {seed} (μ={mean_val:.3f})')
        ax.axvline(mean_val, color=color, linestyle='--', linewidth=1)
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Count')
    ax.set_title(f'{label} Benchmark')
    ax.legend(fontsize=8)

fig.suptitle('Distribution of Predicted Fingerprint Cosine Similarity', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'cosine_similarity_hist.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved cosine_similarity_hist.png')


# --- Plot 2: Top-K dereplication curves ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for ax, (label, suffix) in zip(axes, BENCHMARKS.items()):
    datasets = load_all(suffix)
    n = len(datasets[0])
    all_topk = np.array([topk_hits(d) for d in datasets])
    mean_topk = all_topk.mean(axis=0)
    std_topk = all_topk.std(axis=0)
    ks = range(1, 11)

    for seed, hits, color in zip(SEEDS, all_topk, COLORS):
        ax.plot(ks, hits, color=color, alpha=0.4, linewidth=1)
        ax.scatter(ks, hits, color=color, s=20, alpha=0.6)

    ax.plot(ks, mean_topk, color='black', linewidth=2, label='Mean')
    ax.fill_between(ks, mean_topk - std_topk, mean_topk + std_topk, color='black', alpha=0.1)
    ax.axhline(100, color='gray', linestyle=':', linewidth=1, label='Theoretical max')

    for seed, color in zip(SEEDS, COLORS):
        ax.plot([], [], color=color, label=f'Seed {seed}')

    ax.set_xlabel('Top-K')
    ax.set_ylabel('Dereplication Rate (%)')
    ax.set_title(f'{label} Benchmark (n={n})')
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=8)

fig.suptitle('Top-K Dereplication Rate', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'topk_dereplication.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved topk_dereplication.png')
