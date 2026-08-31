# Hand-off: build & benchmark reference fingerprints for `tab:fp_quality`

**Goal.** Produce intrinsic-quality numbers (specificity + MCES agreement) for the
*Count Your Bits* reference fingerprints so the paper table `tab:fp_quality` can be filled.
Targets, each as its own `rankingset.pt`:

| Fingerprint | Widths to build |
|---|---|
| ECFP4 (Morgan, radius 2) | 2048 **and** 16384 |
| Atom Pair | 2048 **and** 16384 |
| MAP4 | 2048 **and** 16384 |
| Biosynfoni | fixed 39 (single) |

That is **7 rankingsets**. The MARINA fingerprints (multiplicity, sherlock, substructure)
are handled separately by the full-MARINA-DB recount — do **not** rebuild them here, but
your reference FPs MUST use the SAME retrieval file (below) so row order and N align.

## Fixed inputs

- **Retrieval set (NEW — use this one):** `MARINA/data/cleaned/retrieval.pkl`
  - `dict{int: {'smiles': str}}`, **N = 531,087**, contiguous integer keys `0..N-1`.
  - This is the row order every `rankingset.pt` must follow. All the analysis scripts key
    masses/pairs off this file directly, so alignment is automatic as long as every FP is
    built from it in key order.
- **Analysis dir:** `MARINA/analysis/fp-quality/` (scripts below live in `scripts/`).
- **Compute:** Nautilus CPU jobs (namespace `guru-research`, PVC `atong-spectre`, image
  `gitlab-registry.nrp-nautilus.io/a8tong/smart-moonshot/pixi-cuda:12.8.1`). Follow the
  `nrp-nautilus` skill for the job YAML. These are CPU-only (rdkit + multiprocessing); ask
  for ~16 CPU / 32Gi. No GPU.

## The rankingset contract (all FPs must match this)

Copy the output format of `scripts/build_ecfp4.py` exactly:
- `torch.sparse_csr_tensor(crow, col, vals, size=(N, D))`, saved to
  `<out_dir>/<NAME>/rankingset.pt`.
- Row `i` = molecule at `retrieval.pkl[i]`; `col` = the **on-bit indices** for that row;
  `vals` = `1/sqrt(nnz)` per row (so cosine == the deployed retrieval metric).
- Binary on-bit sets only (the analysis is Tanimoto/column-set based). For a **count** FP,
  binarize (nonzero → on-bit) before writing.

## Step 1 — build the 7 rankingsets

**ECFP4 (already scripted).** `scripts/build_ecfp4.py` is parametrized — just run it twice:
```
python scripts/build_ecfp4.py --retrieval data/cleaned/retrieval.pkl \
    --out_dir <OUT> --radius 2 --nbits 2048  --name ECFP4_2048  --workers 16
python scripts/build_ecfp4.py --retrieval data/cleaned/retrieval.pkl \
    --out_dir <OUT> --radius 2 --nbits 16384 --name ECFP4_16384 --workers 16
```

**Atom Pair (RDKit-native, write `scripts/build_atompair.py`).** Identical to
`build_ecfp4.py` but swap the generator in `_init`:
```python
from rdkit.Chem import rdFingerprintGenerator
_GEN = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=nbits)   # binary folded
```
`_GEN.GetFingerprint(m).GetOnBits()` gives the folded on-bits directly → same CSR contract.
Build at `--nbits 2048` and `--nbits 16384`. (Cleanest: generalize `build_ecfp4.py` to take
`--fp-type {morgan,atompair}` rather than copy-paste.)

**MAP4 (needs `chemap`; write `scripts/build_map4.py`).** MAP4 is natively a *MinHash* FP,
NOT a folded bit vector — so "MAP4 at 2048/16384 bits" means **folding the MAP4 shingle set**
(the atom-pair substructure shingles) into an N-bit binary vector, exactly the folded variant
Count Your Bits benchmarks. Do this via **chemap** (`pip install chemap`, the paper's own
library — Huber & Pollmann, J. Cheminform. 2026), which exposes MAP4 in folded binary form:
- Verify chemap's API first (it's not installed locally — inspect it in the pod/session).
  You want the **folded, binary** MAP4 at a given `fpSize`/`n_bits`, then take its on-bits.
- If chemap doesn't fold MAP4 to an arbitrary width, fall back to the `map4` package
  (`reymond-group/map4`): enumerate its shingle set per molecule and fold manually —
  `bit = hash(shingle) % nbits`, set-union the bits. Do NOT use MAP4's MinHash vector with
  set-Tanimoto; MinHash Jaccard ≠ column-set Tanimoto, which would break the metric.
- Build at 2048 and 16384.

**Biosynfoni (fixed dictionary; write `scripts/build_biosynfoni.py`).** Fixed **39**
substructure keys — there is NO folding, so a single width. Use the `biosynfoni` package
(or chemap's Biosynfoni), compute the per-molecule count vector, **binarize** (nonzero → on),
and write the 39-column CSR. `D = 39`.

**Sanity per build:** print `N`, `D`, and on-bits mean/min/max (build_ecfp4.py already does).
Confirm `N == 531087` and `D` matches the intended width.

## Step 2 — specificity (Exp 2, the `Collision % / Bad>200Da % / Largest tie` columns)

`scripts/exp2_specificity.py` already takes many FPs at once. Point it at all 7 rankingsets:
```
python scripts/exp2_specificity.py --retrieval data/cleaned/retrieval.pkl \
    --fp ECFP4_2048=<OUT>/ECFP4_2048/rankingset.pt \
    --fp ECFP4_16384=<OUT>/ECFP4_16384/rankingset.pt \
    --fp AtomPair_2048=<OUT>/AtomPair_2048/rankingset.pt \
    --fp AtomPair_16384=<OUT>/AtomPair_16384/rankingset.pt \
    --fp MAP4_2048=<OUT>/MAP4_2048/rankingset.pt \
    --fp MAP4_16384=<OUT>/MAP4_16384/rankingset.pt \
    --fp Biosynfoni=<OUT>/Biosynfoni/rankingset.pt \
    --out results/exp2_reference.json
```
Reports `collision_pct`, `bad_collision_pct` (>200 Da), `largest_group`, plus `D` and the
mass-bin breakdown — one entry per FP. (On-bits/mol for the table = the build-log mean, or
add it to the JSON.)

## Step 3 — MCES agreement (Exp 1, the `MCES ρ` column)

The RASCAL ground truth depends only on the pair set (which comes from `retrieval.pkl` +
seed), NOT on the fingerprint. So:
- **If a `exp1_pairs_*.parquet` already exists for THIS retrieval.pkl** (same file, same
  `--seed 0`, `--n-pool 5000`, `--n-pairs 100000`): just add the reference FP Tanimoto
  columns with `scripts/exp1_add_fps.py` (no MCES re-run):
  ```
  python scripts/exp1_add_fps.py --pairs results/exp1_pairs_<new>.parquet \
      --out results/exp1_pairs_ref.parquet \
      --fp ECFP4_2048=<OUT>/ECFP4_2048/rankingset.pt --fp ... (all 7)
  ```
- **If no pairs parquet exists for the new set yet** (likely — the new retrieval.pkl changes
  the pool): run `scripts/exp1_mces.py` ONCE over the new set (this is the expensive
  ~hours-long RASCAL pass), passing all 7 reference FPs (and, ideally, coordinate with the
  MARINA-FP recount so MCES is computed once for everyone):
  ```
  python scripts/exp1_mces.py --retrieval data/cleaned/retrieval.pkl \
      --n-pool 5000 --n-pairs 100000 --workers 16 --seed 0 --timeout 60 \
      --fp ECFP4_2048=... --fp ... (all 7) --out results/exp1_pairs_ref.parquet
  ```
  Keep `returnEmptyMCES=True` (already in the script — it's the v2 bias fix; do not revert).
- Then compute Spearman ρ + bootstrap CIs:
  ```
  python scripts/exp1_analyze.py --pairs results/exp1_pairs_ref.parquet --ref sherlock
  ```
  (`--ref` just sets which FP the Δρ is measured against; sherlock if its column is present,
  else any MARINA FP — or drop the Δ and report raw ρ.)

## Deliverables back to me

For each of the 7 FPs: **on-bits/mol, collision %, bad-collision (>200 Da) %, largest tie,
MCES ρ (+ CI)**. Drop them into `PaperWriting/figures/fp-quality/fp_quality.tex` (rows are
already stubbed with `\na`; replace the placeholders and `\best{}` the winning value per
column). JSON artifacts: `analysis/fp-quality/results/exp2_reference.json` and
`exp1_pairs_ref.parquet` (+ the analyze stdout).

## Addendum — FCFP9 — DONE 2026-08-26 (job `atong-fpq-build-fcfp9`, Complete)

Built + benchmarked. Results (`exp2_fcfp.json`, `exp1_spearman_fcfp.json` on PVC `$R`):

| FCFP9 | on-bits/mol | collision % | bad >200 Da % | largest tie | MCES ρ (95% CI) |
|---|---|---|---|---|---|
| 2048  | 112.8 | 6.28 | 0.20 | 248 | 0.432 [0.427, 0.438] |
| 16384 | 116.8 | 6.22 | 0.20 | 248 | 0.550 [0.545, 0.555] |

**FCFP9@16384 ρ=0.550 sits above sherlock (0.517) but below all three MARINA entropy FPs
(substructure 0.580, cap5 0.612, uncapped 0.654)** — Count-your-bits' own MCES champion does
NOT top our benchmark, so the "wins meaningful similarity" claim survives the head-to-head.
Table rows in `fp_quality.tex` are filled. Original build recipe below (for reproduction).



FCFP9 = feature-class (functional/pharmacophore) Morgan at **radius 9** — the row with the
highest Spearman vs RascalMCES in Huber & Pollmann. Already wired into `build_ecfp4.py`
(`--fp-type fcfp`, verified: uses `GetMorganFeatureAtomInvGen`). Build at both widths, same
as the other circular FPs:
```
python scripts/build_ecfp4.py --retrieval data/cleaned/retrieval.pkl --out_dir <OUT> \
    --fp-type fcfp --radius 9 --nbits 2048  --name FCFP9_2048  --workers 16
python scripts/build_ecfp4.py --retrieval data/cleaned/retrieval.pkl --out_dir <OUT> \
    --fp-type fcfp --radius 9 --nbits 16384 --name FCFP9_16384 --workers 16
```
Then fold into the existing results (no MCES re-run — same pair set):
- **Exp 2:** add `--fp FCFP9_2048=... --fp FCFP9_16384=...` to `exp2_specificity.py`
  (append to `results/exp2_reference.json` or a new json).
- **Exp 1:** `python scripts/exp1_add_fps.py --pairs <existing exp1_pairs*.parquet> --out
  results/exp1_pairs_fcfp.parquet --fp FCFP9_2048=... --fp FCFP9_16384=...` then
  `exp1_analyze.py --pairs results/exp1_pairs_fcfp.parquet --ref sherlock`.

The two `FCFP9` rows in `fp_quality.tex` (2048 / 16384) are stubbed with `\na` — fill them.
Naming caveat: "9" is the **radius** (Count Your Bits sweeps r3–r9), not a Pipeline-Pilot
diameter. If the paper's FCFP9 turns out to be a specific fold width, keep the one that matches.

## Gotchas

- **Same retrieval.pkl for every FP** or the rows misalign — `data/cleaned/retrieval.pkl`,
  N=531,087. Don't reuse an old MARINA1 (518,901) rankingset in the same table.
- **Width is a confound, that's the point.** The 2048-vs-16384 pair per FP isolates fold
  width from design; report both, don't average.
- **MAP4 must be the FOLDED binary variant**, not MinHash — otherwise Tanimoto is invalid
  (see Step 1).
- **Biosynfoni is fixed 39-bit** — a single row, no width sweep; its high collision rate is
  expected and is the intended NP counter-example.
- chemap / map4 / biosynfoni are **not** in the base image — `pip install` them in the job
  (or bake a layer); verify imports before the long run.
- Nautilus etiquette per the `nrp-nautilus` skill: CPU job, no GPU request, clean up the
  Job after the JSON lands on the PVC.
```
