#!/usr/bin/env python3
"""Figures + machine-readable stats for the bit-calibration report.

Recomputes everything 10_bit_calibration.py prints, emits vector PDFs for
report/bit-calibration.tex and a stats.json so the LaTeX tables cannot drift
from the numbers.

    pixi run python scripts/11_report_figures.py
"""
import json
import os

import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "report", "figs")
os.makedirs(FIGS, exist_ok=True)

NBINS, BOOT, SEED = 16, 2000, 0
CI, CD = "#0072B2", "#D55E00"          # colourblind-safe pair (implied / independent)
CG = "#009E73"

plt.rcParams.update({
    "figure.dpi": 140, "savefig.bbox": "tight", "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

# ---------------------------------------------------------------- load
d = np.load(os.path.join(RES, "preds.npz"), allow_pickle=True)
P = d["probs"].astype(np.float32)
n, K = P.shape
Y = np.zeros((n, K), dtype=np.float32)
idx, ptr = d["fp_idx"], d["fp_ptr"]
for i in range(n):
    Y[i, idx[ptr[i]:ptr[i + 1]]] = 1.0
mi = np.load(os.path.join(RES, "bit_mi.npz"))
implied = mi["max_cond"] >= 0.999999
counts_rs = mi["counts"]

S = {"n_mol": n, "K": K, "ckpt": str(d["ckpt"]),
     "true_bits_per_mol": float(Y.sum(1).mean()),
     "pred_mass_per_mol": float(P.sum(1).mean()),
     "n_implied": int(implied.sum())}
cos = (P * Y).sum(1) / (np.linalg.norm(P, axis=1) * np.linalg.norm(Y, axis=1))
S["cos_mean"], S["cos_median"] = float(cos.mean()), float(np.median(cos))


def prf(cols):
    y, q = Y[:, cols], P[:, cols]
    pred = q >= 0.5
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    pr = tp / (tp + fp) if tp + fp else float("nan")
    rc = tp / (tp + fn) if tp + fn else float("nan")
    return dict(prec=pr, rec=rc, f1=2 * pr * rc / (pr + rc) if tp else float("nan"),
                tp=tp, fp=fp, fn=fn, npos=tp + fn)


def per_bit_auc(cols):
    a = []
    for j in cols:
        y = Y[:, j]; k = int(y.sum())
        if k in (0, n):
            continue
        r = rankdata(P[:, j])
        a.append((r[y == 1].sum() - k * (k + 1) / 2) / (k * (n - k)))
    return float(np.mean(a)) if a else float("nan")


# ---------------------------------------------------------------- per-bin
rng = np.random.default_rng(SEED)
bins = np.array_split(np.arange(K), NBINS)
rows = []
for b in bins:
    y, q = Y[:, b], P[:, b]
    qs, ys = q.sum(1), y.sum(1)
    s = rng.integers(0, n, (BOOT, n))
    r = qs[s].sum(1) / np.maximum(ys[s].sum(1), 1)
    lo, hi = np.percentile(r, [2.5, 97.5])
    m = prf(b)
    m.update(lo=int(b[0]), hi=int(b[-1]), center=float(b.mean()),
             true_rate=float(y.mean()), ratio=float(qs.sum() / ys.sum()),
             ci_lo=float(lo), ci_hi=float(hi), auc=per_bit_auc(b),
             pct_implied=float(implied[b].mean()))
    rows.append(m)
S["bins"] = rows

ok = Y.sum(0) >= 5
rel = (P.mean(0)[ok] - Y.mean(0)[ok]) / Y.mean(0)[ok]
rho, pv = spearmanr(np.arange(K)[ok], rel)
S["trend"] = dict(nbits=int(ok.sum()), rho=float(rho), p=float(pv),
                  mean=float(rel.mean()), median=float(np.median(rel)))

# ---------------------------------------------------------------- stratified
oct_rows = []
for b in np.array_split(np.arange(K), 8):
    e = prf(b[implied[b]]) if implied[b].any() else None
    i_ = prf(b[~implied[b]]) if (~implied[b]).any() else None
    oct_rows.append(dict(lo=int(b[0]), hi=int(b[-1]), center=float(b.mean()),
                         pct_implied=float(implied[b].mean()),
                         impl=e, indep=i_))
S["octiles"] = oct_rows

# ---------------------------------------------------------------- reliability
edges = [0.5, 0.7, 0.9, 0.99, 0.999, 1.0001]
groups = {"0--1023 (common)": np.arange(0, 1024),
          "1024--8191": np.arange(1024, 8192),
          "8192--16383 (rarest)": np.arange(8192, K),
          "8192--16383 independent": np.arange(8192, K)[~implied[8192:K]]}
S["reliability"] = {}
for name, cols in groups.items():
    q, y = P[:, cols].ravel(), Y[:, cols].ravel()
    pts = []
    for i in range(len(edges) - 1):
        s = (q >= edges[i]) & (q < edges[i + 1])
        if s.sum():
            pts.append(dict(lo=edges[i], hi=edges[i + 1], n=int(s.sum()),
                            mean_pred=float(q[s].mean()), emp=float(y[s].mean())))
    S["reliability"][name] = dict(nbits=len(cols), pts=pts,
                                  miss=float(((y == 1) & (q < 0.5)).sum() / y.sum()))

# ---------------------------------------------------------------- implication direction
X = sp.csr_matrix((np.ones(len(np.load(os.path.join(RES, 'rankingset.npz'))['indices']), np.float32),
                   np.load(os.path.join(RES, "rankingset.npz"))["indices"],
                   np.load(os.path.join(RES, "rankingset.npz"))["indptr"]),
                  shape=tuple(np.load(os.path.join(RES, "rankingset.npz"))["shape"])).tocsc()
rk = np.load(os.path.join(RES, "rankingset.npz"), allow_pickle=True)
frags, radii = rk["frags"], rk["radii"]
samp = np.random.default_rng(0).choice(np.nonzero(implied)[0], 300, replace=False)
n11 = (X[:, samp].T @ X).toarray()
cond = n11 / np.maximum(counts_rs[None, :], 1)
for k_, i_ in enumerate(samp):
    cond[k_, i_] = -1
jj = cond.argmax(1)
good = cond[np.arange(len(samp)), jj] >= 0.999999
si, sj = samp[good], jj[good]
S["implication"] = dict(
    n=int(good.sum()),
    pct_antecedent_later=float((sj > si).mean()),
    pct_antecedent_bigger_radius=float((radii[sj] > radii[si]).mean()),
    med_count_consequent=float(np.median(counts_rs[si])),
    med_count_antecedent=float(np.median(counts_rs[sj])),
    examples=[dict(i=int(a), j=int(b), ri=int(radii[a]), rj=int(radii[b]),
                   ci=int(counts_rs[a]), cj=int(counts_rs[b]),
                   fi=str(frags[a]), fj=str(frags[b]))
              for a, b in list(zip(si, sj))[:6]])

# ================================================================ FIGURES
c = [r["center"] for r in rows]

# Fig 1 — the bit ordering itself
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
ax[0].semilogy(np.arange(K), counts_rs / X.shape[0], lw=0.6, color=CI)
ax[0].set_xlabel("bit index"); ax[0].set_ylabel("presence rate (rankingset)")
ax[0].set_title("(a) Bits are ordered by decreasing frequency", fontsize=9)
w = 512
fr = [implied[i:i + w].mean() for i in range(0, K, w)]
ax[1].plot(np.arange(0, K, w) + w / 2, fr, lw=1.2, color=CD)
ax[1].set_ylim(0, 1.02); ax[1].set_xlabel("bit index")
ax[1].set_ylabel("fraction implied")
ax[1].set_title("(b) Redundancy concentrates at low indices", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIGS}/fig1_ordering.pdf"); plt.close(fig)

# Fig 2 — mass bias
fig, ax = plt.subplots(figsize=(4.4, 3.0))
ax.axhline(1.0, color="k", lw=0.8, ls="--")
pt = np.array([r["ratio"] for r in rows])
ax.errorbar(c, pt, yerr=[pt - [r["ci_lo"] for r in rows], [r["ci_hi"] for r in rows] - pt],
            fmt="o-", ms=4, lw=1.2, capsize=3, color=CI)
ax.set_xlabel("bit index (rarer $\\rightarrow$)")
ax.set_ylabel("predicted mass / true mass")
ax.set_title("Mass bias: flat, uniformly $\\approx$3% low", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIGS}/fig2_massbias.pdf"); plt.close(fig)

# Fig 3 — P/R/F1 overall + stratified
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
ax[0].plot(c, [r["prec"] for r in rows], "o-", ms=3.5, color=CI, label="precision")
ax[0].plot(c, [r["rec"] for r in rows], "s-", ms=3.5, color=CD, label="recall")
ax[0].plot(c, [r["f1"] for r in rows], "^-", ms=3.5, color=CG, label="F1")
ax[0].set_ylim(0.90, 1.0); ax[0].legend(fontsize=7.5, ncol=3)
ax[0].set_xlabel("bit index (rarer $\\rightarrow$)"); ax[0].set_ylabel("micro score @ 0.5")
ax[0].set_title("(a) All index variation lives in recall", fontsize=9)
oc = [r["center"] for r in oct_rows]
ax[1].plot(oc, [r["impl"]["f1"] for r in oct_rows], "o-", ms=4, color=CI, label="implied, F1")
ax[1].plot(oc, [r["indep"]["f1"] for r in oct_rows], "s-", ms=4, color=CD, label="independent, F1")
ax[1].plot(oc, [r["impl"]["rec"] for r in oct_rows], "o--", ms=3, lw=1, alpha=0.45, color=CI, label="implied, recall")
ax[1].plot(oc, [r["indep"]["rec"] for r in oct_rows], "s--", ms=3, lw=1, alpha=0.45, color=CD, label="independent, recall")
ax[1].set_ylim(0.85, 1.0); ax[1].legend(fontsize=7, ncol=2)
ax[1].set_xlabel("bit index (rarer $\\rightarrow$)"); ax[1].set_ylabel("micro score @ 0.5")
ax[1].set_title("(b) Flat within each redundancy stratum", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIGS}/fig3_prf.pdf"); plt.close(fig)

# Fig 4 — reliability + error asymmetry
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
ax[0].plot([0.5, 1], [0.5, 1], "k--", lw=0.8, label="perfect")
for (name, rr), col, mk in zip(S["reliability"].items(), [CI, CD, CG, "#CC79A7"], "os^D"):
    ax[0].plot([p["mean_pred"] for p in rr["pts"]], [p["emp"] for p in rr["pts"]],
               mk + "-", ms=3.5, lw=1.1, color=col, label=name.replace("--", "-"))
ax[0].set_xlabel("mean predicted probability"); ax[0].set_ylabel("empirical rate")
ax[0].legend(fontsize=6.5); ax[0].set_title("(a) Identical over-confidence at every index", fontsize=9)
wd = [(r["hi"] - r["lo"]) for r in rows]
ax[1].bar(np.array(c) - 300, [r["fn"] for r in rows], 600, color=CD, label="false negatives")
ax[1].bar(np.array(c) + 300, [r["fp"] for r in rows], 600, color=CI, label="false positives")
ax[1].set_yscale("log"); ax[1].legend(fontsize=7)
ax[1].set_xlabel("bit index (rarer $\\rightarrow$)"); ax[1].set_ylabel("count @ 0.5 (log)")
ax[1].set_title("(b) FN $>$ FP in every bin", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIGS}/fig4_reliability.pdf"); plt.close(fig)

# Fig 5 — sharpness
fig, ax = plt.subplots(figsize=(4.4, 2.8))
q = P.ravel()
be = np.concatenate([[0], np.logspace(-5, 0, 40)])
ax.hist(q, bins=be, color=CI, log=True)
ax.set_xscale("symlog", linthresh=1e-5)
ax.set_xlabel("predicted probability"); ax.set_ylabel("count (log)")
ax.set_title("Predictions are near-binary", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIGS}/fig5_sharpness.pdf"); plt.close(fig)

with open(os.path.join(HERE, "report", "stats.json"), "w") as f:
    json.dump(S, f, indent=1)
print(f"wrote 5 figures to {FIGS} and report/stats.json")
print(json.dumps({k: S[k] for k in ("n_mol", "K", "n_implied", "cos_mean", "trend", "implication")},
                 indent=1, default=str)[:900])
