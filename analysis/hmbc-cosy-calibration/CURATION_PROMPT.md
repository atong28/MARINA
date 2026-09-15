# Hand-curated HMBC / COSY / HSQC peak lists for MARINA2.0 dropout calibration

You are curating ground-truth 2D NMR peak lists from raw NP-MRD deposits. The lists calibrate
how MARINA2.0 drops peaks from its "ceiling" pseudo-predicted HMBC/COSY during training
(design: `~/Workspace/wiki/models/marina2.0-hmbc-cosy.md`, section "Decisions"). What matters
downstream is, per compound: **which of the topologically possible correlations are actually
visible in the spectrum**, so the calibration can learn observation rates by bond count,
coupling size, solvent and proton type. That makes two things load-bearing:

1. **Recall of real peaks.** Every visible cross-peak goes in, including weak ones to
   quaternary carbons (1–5 % of the tallest peak is normal for the most valuable HMBC peaks).
2. **No invented peaks.** The structure is known and you should use it to *interpret* what
   you see (which column is which proton, whether a blob is a real peak or t1 noise), but a
   peak is recorded only when there is a contour at that position in the image. Never add a
   correlation because the structure predicts it. Record what is observable, not what should be.

## Where things are

- Deposits: `~/Workspace/AgentBench/npmrd/compounds/<NPID>/` — `benchmark_meta.json`
  (solvent, MHz, SMILES, DOIs, experiment → zip), `structure.smiles`, `fids/*.zip`.
- Helper (run from `~/Workspace`, master pixi env has nmrglue/rdkit/matplotlib):

  ```bash
  cd ~/Workspace
  H=MARINA/analysis/hmbc-cosy-calibration/spectra.py
  pixi run python3 $H queue --n 60                        # what to do next, stratified by solvent; 'done' = already curated
  pixi run python3 $H info NP0350544                      # metadata + list of processable spectra
  pixi run python3 $H pick NP0350544 HMBC --snr 3         # automatic picker: peaks with f2_ppm (1H), f1_ppm (13C), rel_intensity, flags
  pixi run python3 $H plot NP0350544 HMBC /tmp/cur/hmbc_full.png --snr 3
  pixi run python3 $H plot NP0350544 HMBC /tmp/cur/hmbc_z1.png --f2 0.5 3.0 --f1 10 80 --snr 3   # zoom: --f2 = 1H range, --f1 = 13C range
  ```

  Experiments are named `HSQC`, `HMBC`, `COSY` (also `13C`, `1H`, `DEPT` if present). Picked
  peaks are circled in the plot. `--snr` lowers the picker threshold (default 5); use 2–3 for the
  HMBC pass so weak quaternary correlations are circled, then judge each one in the image.
- The picker's known failure modes and the column-by-column HMBC method are in
  `~/Workspace/AgentBench/harness/prompts/peaks.md` (sections "How to work" and "What the
  automatic picker gets wrong"). Read it once before the first compound.
- Some deposits are non-uniformly sampled without a schedule; the processor refuses them with a
  note. Record the compound as `status: "unprocessable"` with the note and move on.

## Method per compound (budget ~15–25 minutes; do not rush the HMBC)

1. `info` → note solvent, MHz, formula, SMILES. Draw the structure mentally (or with RDKit) and
   write down the expected proton environments; count CH3/CH2/CH/C.
2. **HSQC** first: `pick` + full plot. One entry per protonated carbon (two for a CH₂ with
   distinct protons). This gives you the proton ↔ carbon map used to read the other two.
3. **COSY**: full plot, then zooms of crowded regions. Record every cross-peak visible on both
   sides of the diagonal, once per unordered pair (`f2_ppm` ≤ `f1_ppm` is not required; just
   don't list both mirror images). Diagonal, t1 streaks, water/solvent rows are excluded.
4. **HMBC, column by column**: for each proton (from HSQC + exchangeables in the 1H spectrum),
   zoom that 1H column over the full 13C range and list every carbon it correlates to. Weak
   contours at a real proton × real carbon intersection are kept. Members of a t1 streak, the
   ¹J doublets straddling an HSQC position (the picker flags `one_bond_artifact`), and ridge
   junk near the solvent carbon are dropped. Do not skip columns.
5. **Assignments (strongly wanted).** For each recorded HMBC/COSY peak give the atom indices of
   the correlated nuclei when you are confident: `assign: [h_atom, c_atom]` for HMBC and
   `assign: [h_atom_a, h_atom_b]` for COSY, where an atom is the **0-based RDKit heavy-atom
   index in `benchmark_meta.json`'s `smiles` string parsed as-is** (`Chem.MolFromSmiles(smiles)`;
   a proton is referred to by the heavy atom it sits on; for diastereotopic CH₂ add `"a"`/`"b"`
   suffixes as strings, e.g. `["7a", 12]`). Leave `assign` out when unsure; never guess.
   Assignments turn shift-matching into exact per-pair statistics, which is what the
   calibration wants most.
6. Write the JSON (schema below), then verify it loads and the counts are sane
   (HSQC ≈ protonated carbons, every HMBC proton position appears in HSQC or is marked
   exchangeable).

## Output: one file per compound

`~/Workspace/MARINA/analysis/hmbc-cosy-calibration/curated/<NPID>.json`

```json
{
  "npid": "NP0350544",
  "smiles": "<benchmark_meta.json smiles, verbatim>",
  "formula": "C30H47NO6",
  "solvent": "CDCl3",
  "frequency_mhz": 600.2,
  "status": "ok",
  "curated_by": "claude session <date>",
  "spectra": {
    "HSQC": {"pulse_program": "hsqcedetgpsisp2.3", "snr_used": 5,
             "peaks": [{"f2_ppm": 3.64, "f1_ppm": 61.2, "assign": [13, 13], "sign": -1, "confidence": "high"}]},
    "COSY": {"pulse_program": "cosygpppqf", "snr_used": 3,
             "peaks": [{"f2_ppm": 3.64, "f1_ppm": 1.52, "assign": [13, 14], "confidence": "high"}]},
    "HMBC": {"pulse_program": "hmbcgplpndqf", "snr_used": 3,
             "peaks": [{"f2_ppm": 3.64, "f1_ppm": 172.1, "assign": [13, 18], "rel_intensity": 0.03, "confidence": "medium"},
                       {"f2_ppm": 5.31, "f1_ppm": 172.1, "assign": ["OH", 18], "exchangeable": true, "confidence": "low"}]}
  },
  "exchangeable_protons_seen": [{"ppm": 5.31, "type": "OH", "assign": 9}],
  "excluded": {"HMBC": "12 picker peaks dropped: t1 streaks under the 0.88 methyl; 3 one-bond artifacts at 27.9/1.24",
               "COSY": "water row at 1.56"},
  "notes": "13C axis unreferenced by ~0.4 ppm in HMBC relative to HSQC; used HSQC as reference."
}
```

Conventions: `f2_ppm` is always the ¹H axis; for HMBC/HSQC `f1_ppm` is ¹³C, for COSY it is the
second ¹H. `sign` only for edited HSQC when the processor reports `signs_reliable: true`.
`confidence` ∈ high / medium / low — low means you saw a contour but could not rule out noise;
include it, that is what the field is for. `rel_intensity` from the picker when the peak was
picked; omit when you added it by eye. Copy `pulse_program` from the picker record.

## Order of work and stopping

- Follow `queue --n 60` first (solvent-stratified, seeded, stable across sessions), then extend
  with `--n 200`. Rare solvents (D₂O, acetone, CD₂Cl₂) are exhausted early on purpose.
- After every compound, append one line to `curated/LEDGER.md`:
  `| NPID | solvent | HSQC n | COSY n | HMBC n | assigned % | minutes | remarks |`.
- Every ~10 compounds, re-read this file and `peaks.md` — long sessions drift toward accepting
  the picker's list unedited, which defeats the purpose.
- Skip a compound (status `skipped`, with the reason) when the structure and the spectra clearly
  disagree (wrong deposit, mixture) — say so in the ledger; those are useful too.
