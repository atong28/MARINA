# fp-metric

Does MARINA's fingerprint error come from predicting the *wrong chemistry*, or from
predicting the *right* substructure in the wrong Morgan context — and if the latter, does
crediting same-substructure columns improve retrieval?

Isolated analysis, torch-free. Consumes exports made with MARINA's own env, following
`analysis/fp-redundancy/`. Reuses that study's cached predictions
(`../fp-redundancy/results/preds.npz`, 3,000 MARINA1 test molecules under
`Checkpoints/MARINA/final1`) and its rankingset export — no GPU pass needed.

## Scripts

| script | env | what |
|---|---|---|
| `00_export_groups.py` | MARINA (`pixi run python3` from repo root, needs RDKit) | groups the 16,384 Morgan columns by substructure; enumerates which substructures each test molecule actually contains |
| `01_error_decomposition.py` | this dir | where the fingerprint error comes from |
| `02_merge_kernel.py` | this dir | whether the off-diagonal metric `M = GGᵀ` improves ranks |

## Grouping

Two Morgan columns share a group iff they describe the same substructure. The deployed
16k spans **7,479 distinct fragment SMILES**; `CC(C)C` alone occupies 124 columns.

Radius-0 features all serialize to `frag_smiles == ''` (`MolFragmentToSmiles` on an empty
bond environment), so keying naively on that field would merge unrelated atom-type bits
into one bucket. Only 52 of 16,384 columns are radius-0 here, and two variants bracket
the choice:

- `strict` — radius-0 columns are singleton groups (7,530 groups). Conservative: can only
  understate within-group error.
- `atom` — radius-0 columns grouped by atom symbol (7,488 groups). Aggressive.

They disagree over columns carrying 5.66% of total |error|, so every headline number
below is a tight range rather than a point estimate.

## 1. Where the error comes from

**Exact L1 decomposition, threshold-free.** For a group `G` with residuals `r_i = p_i − x_i`:
`total_G = Σ|r_i|`, `between_G = |Σ r_i|` (wrong amount of this chemistry),
`within_G = total_G − between_G` (right amount, wrong slots). Non-negative by the triangle
inequality, zero iff no residual in the group flips sign. `within` is exactly the error
annihilated by the merge projection `Gᵀ`, so it upper-bounds what any same-substructure
kernel can recover.

| | strict | atom |
|---|---|---|
| total L1 / molecule | 4.256 | 4.256 |
| between (wrong chemistry) | 3.318 — 77.97% | 3.249 — 76.35% |
| **within (wrong slot)** | **0.938 — 22.03%** | **1.007 — 23.65%** |

Heavily skewed: per-molecule within-fraction has median 0.08 but p90 0.42. Slot confusion
is concentrated in a minority of molecules.

**Hard decisions at top-k** (k = the molecule's true on-bit count; calibration-free, no
threshold). 5,691 false positives over 3,000 molecules (1.9/molecule):

| false-positive class | strict | atom |
|---|---|---|
| **A1** in-vocab slot error — fragment present, a sibling column carries it in the target | **54.07%** | **60.55%** |
| **A2** target-coding artifact — fragment present, no sibling on: the entropy selection dropped the context this molecule uses | 6.57% | 6.57% |
| **B** chemistry error — fragment genuinely absent | 39.36% | 32.88% |

False negatives (5,691, forced equal by top-k): 40–44% slot-swapped (a sibling column was
predicted), 56–60% missed chemistry.

`sib_not_present` is 0 — no false positive had a sibling on in the target while the RDKit
enumeration said the fragment was absent, cross-validating the presence enumeration
against the FragIdx targets.

The L1 view (22%) and the top-k view (54–61%) disagree because L1 is dominated by diffuse
low-probability mass across 16,384 columns, which lands in `between`, whereas top-k looks
only at the model's confident decisions.

> **Takeaway.** Most of what the model gets confidently wrong is *not* wrong chemistry —
> 54–61% of false positives name a substructure the molecule really has, just in the wrong
> Morgan slot. A further 6.6% are cases where the model is right and the *label* is wrong.
> Only ~1/3 are genuine chemistry errors.

## 2. Whether crediting it helps retrieval

`M = GGᵀ` is the off-diagonal metric implied by "same-substructure columns should not be
orthogonal" — outside the diagonal family that `../fp-redundancy/03_reweight_sweep.py`
already ruled out. Same rank protocol (`rank = #{rows with sim ≥ sim(query, truth)} − 1`),
same tie tolerance, 3,000 queries against the 518,901-molecule rankingset. Baseline top-1
0.7587 reproduces the fp-redundancy figure exactly.

| scheme | top-1 | top-5 | mean rank | Δtop-1 (95% CI) | fixed | broke |
|---|---|---|---|---|---|---|
| baseline (plain cosine) | 0.7587 | 0.9193 | 21.1 | — | — | — |
| strict-sum | 0.7393 | 0.9077 | 36.8 | −0.0193 [−0.0287, −0.0103] | 72 | 130 |
| strict-or | 0.6640 | 0.8723 | 63.2 | −0.0947 [−0.1073, −0.0827] | 55 | 339 |
| atom-sum | 0.7330 | 0.8990 | 190.0 | −0.0257 [−0.0353, −0.0163] | 70 | 147 |
| atom-or | 0.4693 | 0.6160 | 127.2 | −0.2893 [−0.3067, −0.2723] | 30 | 898 |

Every kernel loses, all CIs exclude zero on a paired bootstrap over queries (10,000
resamples). The `fixed`/`broke` columns show why: merging really does repair 72 queries
that slot confusion had cost — the effect from §1 is real — but it breaks ~1.8× more,
because the duplicate columns were carrying discriminative signal that the projection
destroys.

`sum` beats `or` by a wide margin (−0.019 vs −0.095), so binarizing after merge throws
away real information.

> **Takeaway.** Slot confusion is real but *not* what's costing retrieval. Crediting
> same-substructure columns fixes 72 queries and breaks 130. The redundant columns earn
> their keep as discriminators even though they are near-duplicates as features — the
> off-diagonal metric family is now ruled out alongside the diagonal one.

## What this does and does not say about the substructure fingerprint

It does **not** predict that the substructure FP will fail, and the distinction matters:

- This kernel **shrinks** the representation, 16,384 → 7,530 dims. The substructure FP
  **keeps** 16,384 dims, all distinct substructures — 2.19× more distinct chemistry at the
  same width, not less.
- It is test-time surgery on a model trained with a uniform-weight loss over the Morgan
  vocabulary. The substructure model never learns the duplicate structure in the first
  place.
- A2 (6.6% of false positives) is label noise that `Gᵀ` cannot fix but the substructure FP
  can, since it keys on the fragment alone.

The `sum` ≫ `or` gap is the one direct warning: the substructure branch uses binary (OR)
targets, and counts clearly carry information here.

> **Overall takeaway.** The fingerprint wastes over half its confident errors on slot
> confusion, but repairing that at the metric level costs more than it recovers. Both the
> diagonal and off-diagonal metric families are now exhausted; the remaining moves are on
> the representation side — retrain on distinct chemistry (the substructure branch), and
> use counts rather than binary.
