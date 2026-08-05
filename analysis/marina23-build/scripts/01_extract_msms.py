"""Extract experimental positive-mode MS/MS peak lists for MARINA1 molecules.

Sources: GNPS (4.5 GB MGF), MassBank (MSP), MassSpecGym (TSV). Negative mode is
parsed and counted but deliberately NOT emitted -- it is a separate modality slot
that the current schema cannot hold, and is out of scope for this build.

One spectrum per molecule. Rather than merging across collision energies (which
produces a spectrum that never physically existed), the single best real spectrum
is selected: curated sources first, then most peaks. MARINA1's own MS/MS is one
ICEBERG spectrum at a fixed 20 eV, so one-real-spectrum matches the contract.

Output peaks are normalized to MARINA1's convention, measured from its tensors:
base peak = 100, floor = 1, top 100 peaks, sorted by m/z.

Writes results/msms_experimental.pkl : {canonical_smiles: [[mz, intensity], ...]}
"""
import csv
import os
import pickle
import sys
from pathlib import Path

from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")
csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("MARINA_DATA_ROOT", "/home/user/atong"))
OUT = ROOT / "results"
FINETUNE_RAW = ROOT.parent / "finetune-exp" / "raw"
DOMAIN_RAW = ROOT.parent / "domain-compare" / "raw"
GNPS = FINETUNE_RAW / "gnps/ALL_GNPS.mgf"
MASSBANK = FINETUNE_RAW / "massbank/MassBank_NISTformat.msp"
MSGYM = DOMAIN_RAW / "massspecgym/MassSpecGym.tsv"
MARINA1_INDEX = DATA_ROOT / "Datasets/MARINA1/index.pkl"

MAX_PEAKS = 100
INTENSITY_FLOOR = 1.0

# Lower tier wins. Curated libraries first, then the GNPS bulk.
TIER_CURATED, TIER_BENCHMARK, TIER_BULK = 0, 1, 2

_canon_cache = {}


def canonicalize(smiles):
    """MARINA's rule: canonicalize twice, drop stereochemistry. Cached."""
    if smiles in _canon_cache:
        return _canon_cache[smiles]
    out = None
    if isinstance(smiles, str) and smiles.strip():
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            once = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            mol = Chem.MolFromSmiles(once)
            if mol is not None:
                out = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    _canon_cache[smiles] = out
    return out


def normalize(peaks):
    """-> MARINA1 convention: base peak 100, floor 1, top-100, m/z sorted."""
    peaks = [(mz, inten) for mz, inten in peaks if inten > 0]
    if not peaks:
        return None
    top = max(i for _, i in peaks)
    scaled = [(mz, i / top * 100.0) for mz, i in peaks]
    kept = [p for p in scaled if p[1] >= INTENSITY_FLOOR]
    if not kept:
        return None
    kept.sort(key=lambda p: -p[1])
    kept = kept[:MAX_PEAKS]
    kept.sort(key=lambda p: p[0])
    return [[round(mz, 5), round(i, 5)] for mz, i in kept]


index = pickle.load(open(MARINA1_INDEX, "rb"))
targets = {v["smiles"] for v in index.values()}
print(f"MARINA1 target SMILES: {len(targets):,}", flush=True)

# {smiles: (tier, -n_peaks, peaks)} -- keep the best seen so far
best = {}
stats = {"pos": 0, "neg": 0, "no_mode": 0, "matched": 0}


def offer(smiles_raw, peaks, tier):
    canon = canonicalize(smiles_raw)
    if canon is None or canon not in targets:
        return
    stats["matched"] += 1
    key = (tier, -len(peaks))
    if canon not in best or key < best[canon][0]:
        best[canon] = (key, peaks)


# ------------------------------------------------------------------ GNPS MGF
def gnps_records(path):
    rec, peaks = {}, []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line == "BEGIN IONS":
                rec, peaks = {}, []
            elif line == "END IONS":
                yield rec, peaks
                rec, peaks = {}, []
            elif "=" in line and line[:1].isalpha():
                k, _, v = line.partition("=")
                rec[k.upper()] = v.strip()
            elif line and (line[0].isdigit() or line[0] in "-."):
                bits = line.split()
                if len(bits) >= 2:
                    try:
                        peaks.append((float(bits[0]), float(bits[1])))
                    except ValueError:
                        pass


print("parsing GNPS...", flush=True)
for rec, peaks in tqdm(gnps_records(GNPS), unit=" spec"):
    try:
        if float(rec.get("MSLEVEL", 0)) < 2:
            continue
    except ValueError:
        continue
    mode = rec.get("IONMODE", "").strip().lower()
    if mode.startswith("pos"):
        stats["pos"] += 1
    elif mode.startswith("neg"):
        stats["neg"] += 1
        continue
    else:
        stats["no_mode"] += 1
        continue
    if not peaks:
        continue
    gold = rec.get("LIBRARYQUALITY", "").strip() == "1"
    offer(rec.get("SMILES", ""), peaks, TIER_CURATED if gold else TIER_BULK)
print(f"  GNPS: pos {stats['pos']:,} neg {stats['neg']:,} unlabelled {stats['no_mode']:,}",
      flush=True)

# -------------------------------------------------------------- MassBank MSP
print("parsing MassBank...", flush=True)
mb_pos = mb_neg = 0
with open(MASSBANK, "r", encoding="utf-8", errors="replace") as fh:
    smiles = mode = None
    peaks, in_peaks = [], False
    for line in fh:
        line = line.rstrip("\n")
        if not line.strip():
            if smiles and mode == "positive" and peaks:
                offer(smiles, peaks, TIER_CURATED)
            smiles = mode = None
            peaks, in_peaks = [], False
            continue
        if in_peaks:
            bits = line.split()
            if len(bits) >= 2:
                try:
                    peaks.append((float(bits[0]), float(bits[1])))
                except ValueError:
                    pass
            continue
        if line.startswith("SMILES:"):
            smiles = line.split(":", 1)[1].strip()
        elif line.startswith("Ion_mode:"):
            mode = line.split(":", 1)[1].strip().lower()
            if mode.startswith("p"):
                mb_pos += 1
            else:
                mb_neg += 1
        elif line.startswith("Num Peaks:"):
            in_peaks = True
    if smiles and mode == "positive" and peaks:
        offer(smiles, peaks, TIER_CURATED)
print(f"  MassBank: pos {mb_pos:,} neg {mb_neg:,}", flush=True)

# ------------------------------------------------------------ MassSpecGym TSV
print("parsing MassSpecGym...", flush=True)
n_gym = 0
with open(MSGYM, newline="") as fh:
    for row in tqdm(csv.DictReader(fh, delimiter="\t"), unit=" spec"):
        adduct = (row.get("adduct") or "").strip()
        if not adduct.endswith("+"):
            continue
        try:
            mzs = [float(x) for x in row["mzs"].split(",") if x]
            ints = [float(x) for x in row["intensities"].split(",") if x]
        except (ValueError, KeyError):
            continue
        if not mzs or len(mzs) != len(ints):
            continue
        n_gym += 1
        offer(row.get("smiles", ""), list(zip(mzs, ints)), TIER_BENCHMARK)
print(f"  MassSpecGym: positive spectra {n_gym:,}", flush=True)

# ------------------------------------------------------------------- finalize
out = {}
for smiles, (_, peaks) in best.items():
    norm = normalize(peaks)
    if norm:
        out[smiles] = norm

OUT.mkdir(exist_ok=True)
with open(OUT / "msms_experimental.pkl", "wb") as fh:
    pickle.dump(out, fh)

npk = [len(v) for v in out.values()]
print(f"\nmolecules with experimental positive MS/MS: {len(out):,}")
print(f"  peaks/spectrum: min {min(npk)} median {sorted(npk)[len(npk)//2]} max {max(npk)}")
print(f"wrote {OUT/'msms_experimental.pkl'}")
