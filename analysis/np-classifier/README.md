# np-classifier — NPClassifier annotations for the MARINA1 retrieval set

Every one of the 518,901 retrieval molecules classified by
[NPClassifier](https://npclassifier.gnps2.org/) into pathway / superclass / class, plus
the glycoside flag.

Two consumers:

- **Ground truth for fingerprint similarity work.** NPClassifier labels are derived from
  the structure but not from any Morgan/substructure enumeration, so they are an
  *independent* referee for "are these two molecules chemically alike?" — which
  ECFP-Tanimoto is not, since it shares a feature basis with the vocabularies under test.
- **The website.** Class annotations render on each result card and in the expanded
  compound view.

**Read-only on `Datasets/`.** `results/provenance.json` records the md5 of the
`retrieval.pkl` these records are keyed against; the indices are only meaningful for that
exact file.

## Environment

Runs in MARINA's own pixi env — it needs `httpx`, `pandas` and `pyarrow`, all of which are
already there, and nothing else. No torch, no RDKit.

```
cd /home/user/atong/MARINA
pixi run python3 analysis/np-classifier/scripts/01_fetch.py --concurrency 32   # ~45 min
pixi run python3 analysis/np-classifier/scripts/02_export.py                   #  ~2 min
pixi run python3 analysis/np-classifier/scripts/03_summary.py                  #  <1 min
```

## The fetch

`01_fetch.py` is append-only and resumable: each answered index lands in
`results/npclassifier.jsonl` as it arrives, and a re-run reads that file and asks only for
what is missing. Killing it costs at most the in-flight requests.

Concurrency 32 sustains ~190 requests/s. Raising it further did not help in testing —
the service appears to throttle — so it is not worth the extra load on a community-run
server. Requests carry a `User-Agent` identifying the project, so a 519k-request pass is
attributable rather than anonymous.

A `500` is how the service reports a structure it cannot parse, but transient overload
looks identical, so each SMILES gets 4 attempts with backoff before being recorded as an
error. Errors stay in the JSONL and are counted by `02_export.py`; deleting those lines
and re-running `01_fetch.py` retries exactly those molecules.

**An empty result is not an error.** NPClassifier declines to classify some structures and
returns empty tiers with HTTP 200. `02_export.py` and `03_summary.py` both report that
count separately, because it bounds any analysis that leans on these labels as truth.

## Artifacts

| file | consumer | notes |
|---|---|---|
| `results/npclassifier.jsonl` | — | raw append log, one record per molecule (not tracked) |
| `results/npclassifier.parquet` | analysis | one row per molecule, list columns per tier |
| `results/npclassifier.json` | **backend** | keyed by `str(global_idx)`, label strings interned |
| `results/coverage.json`, `summary.md` | — | coverage, per-tier distributions |

The JSON interns its label strings into a shared table rather than repeating them per
molecule — there are only a few hundred distinct labels across 519k molecules, so spelling
them out inflates the file several-fold for no added information.

## Deploying to the website

The backend reads `npclassifier.json` from each **model root** — the directory holding
`best.ckpt`, `params.json`, `metadata.json` — and keys it exactly like `metadata.json`.
Copy it next to `metadata.json` for every served model:

```
cp results/npclassifier.json <model_root>/npclassifier.json
```

The file is optional. Absent, `session.get_npclassifier()` returns `None`, result cards
carry `npclassifier: null`, and the UI renders as it did before.

**The index space is the constraint.** These records are keyed on positions in
`Datasets/MARINA1/retrieval.pkl` (md5 `796bef710eb7042192a3f6039d11b72b`). A model root
whose `retrieval.pkl` differs would attach the wrong annotation to every card, silently.
Check the md5 matches `results/provenance.json` before copying. The retrieval set is
byte-identical across MARINA1–4, so this holds for those; it is not automatic for anything
else.

## Related

- [`sfp-report/`](../sfp-report/) — what the deployed fingerprint is distributed like.
- [`fp-metric/`](../fp-metric/) — where fingerprint error comes from.
