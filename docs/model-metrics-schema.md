# Model metrics schema (`metrics.json`) — for dynamic model selection

**Purpose.** The MARINA website will drop its manual model dropdown and instead
**auto-select the best checkpoint for whatever inputs the user actually provides.**
To do that, every deployed checkpoint ships a `metrics.json` describing how well
that checkpoint ranks the true structure **per input-modality combination**, on
whichever evaluation set is available (real-world benchmark preferred, simulated
test otherwise).

This document is the contract. Another session will write a **new, standalone
eval script** that produces one `metrics.json` per checkpoint in exactly this
format. The backend selector consumes it. Nothing here depends on
`eval_grapefruit.py` — it is only referenced because it already computes the same
underlying numbers (`src/modules/benchmark.py::_rank_conventions`,
`src/modules/core/ranker.py::dot_prod_rank`), so you can reuse those definitions.

---

## 1. File location

One file per checkpoint, colocated with `params.json` / `calibration.json`:

```
checkpoints/<model_dir>/metrics.json
```

This mirrors the existing per-checkpoint sidecar convention (`calibration.json`,
`mw_index.json`, …). Adding a model to the site = drop its dir containing a
`metrics.json`; the selector aggregates across every model listed in
`checkpoints/models.json`. No central metrics file.

---

## 2. Modality vocabulary & the combo key

Use the **canonical modality keys** exactly as the model consumes them
(`src/modules/core/const.py::INPUT_MAP`):

```
hsqc, c_nmr, h_nmr, mass_spec, mass_spec_neg, mw, formula
```

Whenever you list modalities, order them by `INPUTS_CANONICAL_ORDER`:

```
["hsqc", "c_nmr", "h_nmr", "mass_spec", "mass_spec_neg", "mw", "formula"]
```

A measurement's `modalities` array is the **exact set of inputs fed to the model**
for that measurement (inputs the model did not receive are masked out). The
backend derives a lookup key by sorting `modalities` into canonical order and
joining with `+` (e.g. `["h_nmr","hsqc"] → "hsqc+h_nmr"`). You do **not** write
the key; you write `modalities` and the backend canonicalizes.

---

## 3. Metric definitions (must match these exactly)

For a set of evaluated molecules, rank the true fingerprint against the
checkpoint's own `rankingset.pt` (cosine). `rank` is **0-based** (rank 0 = the
true structure came first). Top-k accuracy uses the **strict** convention — ties
broken against you (the true item is placed last among equal-scoring items), so
the number is a lower bound.

Store exactly these fields per measurement:

| field         | definition                                                        |
|---------------|-------------------------------------------------------------------|
| `n`           | number of molecules evaluated for this measurement                |
| `mean_cos`    | mean cosine similarity between predicted and true fingerprint     |
| `strict_top1` | 100 × fraction with `strict_rank < 1` (i.e. exact top-1), **percent** |
| `strict_top5` | 100 × fraction with `strict_rank < 5`                              |
| `strict_top10`| 100 × fraction with `strict_rank < 10`                             |

Top-k are **percentages in [0, 100]**. `mean_cos` is in [0, 1]. Use `null` for a
metric that could not be computed (never `NaN` — JSON has no NaN).

**`mean_cos` is the metric the selector ranks on** (§6). `strict_top{1,5,10}` are
stored so we can switch the ranking metric later without re-running evals; the
tie-aware convention is intentionally dropped to keep the file lean.

---

## 4. Evaluation sets (`eval_set`)

Each measurement declares where it came from:

- `"benchmark"` — **real-world** experimental spectra (the journal set,
  `benchmark-journal.pkl`). Preferred for selection when available.
- `"test"` — **simulated** spectra from the dataset's held-out test split.
  Used when no benchmark measurement exists for that combo.

`split` further identifies the slice:
- benchmark: emit **three** measurements per combo — `"val"`, `"test"`, and
  `"both"`. `"both"` is the two journal splits **pooled** (metrics recomputed over
  the union of val+test molecules; `n = n_val + n_test`, equivalently the
  n-weighted average of the per-split rates). `val`/`test` are stored for
  reference; **the selector uses `both`.**
- test (simulated): `"test_all7pop"` — the fixed population of test molecules that
  have *all* modalities present, so every combo is scored on the same molecules
  and combos are directly comparable. **Simulated side has no validation split** —
  `test` only.

So per combo you emit either three benchmark rows (`val`, `test`, `both`) or one
simulated row (`test_all7pop`) — see §7. The selector resolves a combo to a
single measurement via the ordered preference in §6.

---

## 5. File format

```jsonc
{
  "schema_version": "1.0",
  "model_id": "marina_unifcard_full_s0",   // must match the id in models.json
  "input_types": ["hsqc","c_nmr","h_nmr","mass_spec","mass_spec_neg","mw","formula"],
  "generated_at": "2026-09-04T18:00:00Z",   // ISO-8601 UTC
  "eval_provenance": {                        // free-form, for humans; not parsed
    "ckpt": "last.ckpt",
    "ckpt_epoch": 804,
    "dataset": "MARINA-DB",
    "rankingset": "RankingEntropyMultiplicityUncapped/rankingset.pt",
    "test_population": "test split, molecules with all 7 modalities present",
    "benchmark_source": "benchmark-journal.pkl",
    "eval_script": "<path/to/the/new/script>",
    "notes": ""
  },
  "measurements": [
    // benchmark-coverable combo (no MS): three rows — val, test, both. Selector uses `both`.
    { "modalities": ["hsqc","mw"], "eval_set": "benchmark", "split": "val",  "n": 148,
      "metrics": { "mean_cos": 0.751, "strict_top1": 12.8, "strict_top5": 30.4, "strict_top10": 39.9 } },
    { "modalities": ["hsqc","mw"], "eval_set": "benchmark", "split": "test", "n": 152,
      "metrics": { "mean_cos": 0.744, "strict_top1": 12.5, "strict_top5": 29.6, "strict_top10": 38.8 } },
    { "modalities": ["hsqc","mw"], "eval_set": "benchmark", "split": "both", "n": 300,
      "metrics": { "mean_cos": 0.747, "strict_top1": 12.7, "strict_top5": 30.0, "strict_top10": 39.3 } },
    // MS-containing combo the journal can't cover: one simulated row, no validation.
    { "modalities": ["hsqc","mass_spec"], "eval_set": "test", "split": "test_all7pop", "n": 1287,
      "metrics": { "mean_cos": 0.858, "strict_top1": 27.0, "strict_top5": 51.2, "strict_top10": 61.8 } }
    // ... benchmark: val+test+both for every journal-coverable subset;
    //     test: test_all7pop for every remaining (MS-containing) subset.
  ]
}
```

**All numbers above are SAMPLE placeholders** — real values come from the eval run.

---

## 6. How the backend selects a model (so you know what the numbers drive)

Given the set `S` of modalities the user actually supplied:

1. **Eligibility** — keep only models whose `input_types ⊇ S`. A model can never
   be chosen for an input it was not trained to accept. (E.g. an NMR-only model
   is ineligible the moment the user adds MS.)
2. **Lookup** — for each eligible model, find the measurement whose `modalities`
   equals `S` exactly (canonicalized), choosing the `(eval_set, split)` by this
   **ordered preference** (first match wins):
   1. `benchmark` / `both`   ← real-world, pooled val+test (primary)
   2. `test` / `test_all7pop` ← simulated (used for MS combos the journal can't cover)

   By construction (§7) exactly one of these exists per combo. The stored
   benchmark `val` / `test` rows are for reference and are **not** consulted for
   selection. (This ordered list is the one place to extend if we later want,
   say, `benchmark`/`test` to outrank simulated.)
3. **Rank** — order eligible models by **`mean_cos`** of the chosen measurement
   (higher wins). This is the sole ranking metric. `strict_top{1,5,10}` are stored
   but not used for selection today; the ranking metric is a single backend
   constant, so we can switch to a top-k metric later without re-running evals.
4. **Fallbacks** (deterministic, in order):
   a. No eligible model has an exact-`S` measurement → for each eligible model use
      its measurement on the **largest measured subset of `S`** (most overlap); if
      still tied, the model's `all_inputs` measurement.
   b. No usable measurement anywhere → fall back to the **manifest default model**
      (`models.json` `default: true`).

**Implication for the producer:** the more combos you measure, the better the
selection. At minimum, measure the combos the UI can actually emit (see §7). Any
combo you omit degrades to a fallback rather than an exact choice — not an error.

---

## 7. Which combos to measure — **all of them**

Measure **every subset** of the model's `input_types` that carries at least one
**spectral** modality. Spectral = `{hsqc, c_nmr, h_nmr, mass_spec, mass_spec_neg}`;
non-spectral descriptors = `{mw, formula}`. Excluded as useless: the empty set and
the descriptor-only combos `{mw}`, `{formula}`, `{mw, formula}` — the website
requires at least one spectral input (`/predict` rejects a descriptor-only request),
so those are never queried. For the full 7-modality model that leaves
**2⁷ − 1 − 3 = 124 combos**. A reduced model has fewer (e.g. NMR-only
`[hsqc,c_nmr,h_nmr]` → 2³ − 1 = 7). Enumerate programmatically from `input_types`,
dropping any subset with no spectral modality; do not hand-pick a "common" set.

**Prefer real-world; simulate only the gaps.** Each combo gets exactly one
source — benchmark if the real data can supply it, simulated test otherwise:

- **`benchmark` (real-world journal)** — for **every** subset the journal data can
  actually supply, emit **three rows** (`val`, `test`, `both`; see §4). A combo is
  benchmark-coverable iff each of its modalities is present in the journal. In
  practice the journal carries NMR + `mw`, and `formula` is derived from the known
  structure — i.e. the coverable modalities are `{hsqc, c_nmr, h_nmr, mw, formula}`.
  The journal has **no MS channels**, so any combo containing `mass_spec` or
  `mass_spec_neg` is *not* benchmark-coverable. Determine the coverable modality set
  from the data, don't hardcode it.
- **`test` (simulated)** — **only** for the subsets the benchmark cannot cover
  (i.e. every combo that includes `mass_spec` and/or `mass_spec_neg`), one row each
  (`test_all7pop`, no validation). Score them on one fixed population of test-split
  molecules that have all modalities present, masking inputs down per combo, so
  those combos are mutually comparable.

Do **not** emit a `test` measurement for a combo the benchmark already covers.
Every measured (spectral-bearing) combo resolves to exactly one selection source —
`benchmark/both` or `simulated/test` — so every input the UI can produce has one
exact-combo `mean_cos` and the fallbacks in §6 rarely fire.

---

## 8. Producer checklist

- [ ] `model_id` matches `checkpoints/models.json`.
- [ ] Every `modalities` array is a subset of `input_types`, canonical keys only.
- [ ] `mean_cos` in [0,1]; `strict_top{1,5,10}` percentages in [0,100]; missing → `null`.
- [ ] Each measurement stores `mean_cos` + `strict_top1/5/10` (no `tie_*`).
- [ ] `benchmark` rows `val`+`test`+`both` for every journal-coverable subset (no MS channels).
- [ ] `both` = pooled val+test (`n = n_val + n_test`), the row the selector uses.
- [ ] `test` (`test_all7pop`, no validation) for every remaining subset (those containing MS) — and only those.
- [ ] Combos are the 124 spectral-bearing subsets (exclude empty, `{mw}`, `{formula}`, `{mw,formula}`).
- [ ] Every measured combo resolves to exactly one of `benchmark/both` or `simulated/test`.
- [ ] Written to `checkpoints/<model_dir>/metrics.json`, valid JSON (no NaN).
