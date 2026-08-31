#!/usr/bin/env python3
"""
Modality-ablation retrieval eval on the MARINA-DB test split (paper tables 1 & 2).

For each model (name|params.json|ckpt) and each modality subset, evaluates on the
per-subset test set -- all test molecules carrying that subset's modalities, via
MARINADataset(override_input_types=subset) (requires == subset). This is MARINA's
native additional_test_types semantics; rows have different n and are NOT forced
onto a common molecule set (per the "full data per subset" decision).

Per subset it reports three metric families over the top-10 retrievals:
  - rank@1/5/10        : the exact query structure ranks in the top-k (tie-aware, D6)
  - derep_top1/5/10    : a near-identical structure (cosine of the target sparse FP
                         to a retrieved bank row > 0.99) appears in the top-k
  - annot_top1/5/10    : a structural analog appears in the top-k -- any retrieved
                         candidate whose ECFP4 (Morgan r=2, 2048) cosine to the gold
                         SMILES is >= 0.8. This is SPECTRE's "structure annotation"
                         metric (infer_perf_annotation.py): cosine of two binary
                         Morgan vectors == |A&B| / sqrt(|A|*|B|), threshold >= 0.8,
                         any-of-top-k.

Bank            = DATA_DATASET/<fp_type>/rankingset.pt (per-model fp_type).
Candidate SMILES= load_smiles_index(RETRIEVAL_PKL), in bank-row order.

IMPORTANT staging (the model's own vocabulary/radius, or you get silent garbage):
  - FP_RADIUS must match the model build (multuncap = 6, NOT the default 10).
  - DATASET_ROOT/<fp_type>/{rankingset.pt,bitinfo_to_idx.pkl} and
    DATASET_ROOT/arrow/<split>/<FragIdxMultiplicityUncapped.parquet> and
    DATASET_ROOT/count_multiplicity_uncapped_under_radius_6.pkl must be the model's.
  - Sanity: all_inputs / hsqc / c_nmr / h_nmr / mass_spec rank@1/5/10 must match the
    run's stored test_result.pkl (same subsets, same split). --validate does this.

Usage:
  FP_RADIUS=6 DATASET_ROOT=/workspace PYTHONPATH=. pixi run python3 \
      scripts/marina_db/eval/eval_modality_ablation.py \
      --models models_ablation.txt --out ablation.json [--validate]
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))   # scripts/marina_db  (config, lib)
sys.path.insert(0, str(_HERE.parents[3]))   # repo root          (src)

import argparse
import json
import os
import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from config import DATA_DATASET, RETRIEVAL_PKL
from lib.eval_loop import build_args, to_device

from src.modules import MARINA
from src.modules.marina.dataset import MARINADataset, collate
from src.modules.data.fp_loader import make_fp_loader
from src.modules.data.fp_utils import load_smiles_index
from src.modules.core.ranker import RankingSet

# Modality subsets for tables 1 & 2. None => the model's own input_types (all inputs;
# requires all modalities present). Keys are the on-disk / json result names.
SUBSETS = {
    "all_inputs":        None,
    "hsqc_c_nmr_h_nmr_mw": ["hsqc", "c_nmr", "h_nmr", "mw"],   # T1: ME-HSQC,13C,1H,MW
    "hsqc_c_nmr":        ["hsqc", "c_nmr"],                     # T1: ME-HSQC,13C
    "hsqc_h_nmr":        ["hsqc", "h_nmr"],                     # T1: ME-HSQC,1H
    "hsqc_c_nmr_h_nmr":  ["hsqc", "c_nmr", "h_nmr"],           # T2: NMR
    "hsqc":              ["hsqc"],                              # T1/T2: HSQC
    "c_nmr":             ["c_nmr"],                             # T2: 13C
    "h_nmr":             ["h_nmr"],                             # T2: 1H
    "mass_spec":         ["mass_spec"],                         # T2: MS/MS (+)
}

_MFPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def onbits(smiles):
    """ECFP4 (r=2, 2048) on-bit index set for `smiles`, or None if unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return set(_MFPGEN.GetFingerprint(mol).GetOnBits())


def bit_cosine(a, b):
    """Cosine of two binary Morgan vectors == |A&B| / sqrt(|A|*|B|). SPECTRE's metric."""
    if not a or not b:
        return 0.0
    return len(a & b) / (len(a) * len(b)) ** 0.5


@torch.no_grad()
def eval_subset(args, model, fp_loader, smiles_index, device, override,
                batch_size, num_workers, annot_thresh=0.8, topn=10):
    """One modality subset -> aggregate dereplication (strict + tie-aware) + annotation.

    Dereplication = the query's OWN fingerprint ranks in top-k (exact-FP identity). We
    report two tie conventions (>0.99 cosine is deliberately NOT used: multiplicity FPs
    have ~150 on-bits, so distinct structures routinely exceed 0.99):
      - strict    : ties (bank rows with sim == cos(pred,truth)) count against the truth
                    (pessimistic; == the training test-loop / test_result.pkl rank@k)
      - tie-aware : fingerprint-identical twins (sim tied with the truth) do NOT count
                    against it (decision D6; optimistic)
    Annotation = any top-k retrieved candidate has ECFP4 (r=2,2048) cosine >= 0.8 to the
    gold SMILES (SPECTRE's structure-annotation metric), any-of-top-k.
    """
    ds = MARINADataset(args, fp_loader, split="test", override_input_types=override)
    gold_smiles = [entry["smiles"] for _, entry in ds.data]     # loader order (shuffle=False)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                        collate_fn=collate, shuffle=False)

    cand_bits = {}          # bank idx -> ECFP4 on-bit set (cache; candidates repeat)
    recs = []
    pos = 0
    for batch in tqdm(loader, desc=f"{override or 'all_inputs'}", leave=False):
        inputs, truth = to_device(batch[0], device), to_device(batch[1], device)
        # Models were trained/tested in bf16-mixed; forward under the same autocast so
        # predictions (hence rank) match the run's stored test_result.pkl. Ranking is fp32.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
            pred = model(batch=inputs)
            pred = pred[0] if isinstance(pred, (tuple, list)) else pred
        pred = torch.sigmoid(pred.float())
        b = pred.size(0)

        # One similarity matmul against the whole bank; derive both rank conventions and
        # the top-k retrieval from it. sims[i,q] = cos(pred_q, bank_i) (bank pre-normalized).
        sims = model.ranker._sims(pred)                          # (N, b)
        qn = F.normalize(pred, dim=1, p=2.0)
        tn = F.normalize(truth, dim=1, p=2.0)
        thresh = (qn * tn).sum(1).unsqueeze(0)                   # (1, b) = cos(pred, truth)
        close = torch.isclose(sims, thresh)
        strict = ((sims >= thresh) | close).sum(0) - 1          # ties+self count, drop self
        tie = ((sims > thresh) & ~close).sum(0)                 # strictly better than truth
        strict, tie = strict.tolist(), tie.tolist()

        topk_idx = torch.topk(sims, k=topn, dim=0).indices.t().tolist()   # b x topn

        for r in range(b):
            g = onbits(gold_smiles[pos + r])
            ann = []
            for j in topk_idx[r]:
                if j not in cand_bits:
                    cand_bits[j] = onbits(smiles_index[int(j)])
                ann.append(g is not None and bit_cosine(g, cand_bits[j]) >= annot_thresh)
            recs.append({"strict": strict[r], "tie": tie[r], "ann": ann})
        pos += b

    return summarise(recs)


def summarise(recs):
    n = len(recs)
    out = {"n": n}
    if not n:
        return out
    for k in (1, 5, 10):
        out[f"derep_strict_top{k}"] = 100.0 * sum(r["strict"] < k for r in recs) / n
        out[f"derep_tie_top{k}"] = 100.0 * sum(r["tie"] < k for r in recs) / n
        out[f"annot_top{k}"] = 100.0 * sum(any(r["ann"][:k]) for r in recs) / n
    return out


def eval_model(params_path, ckpt_path, device, batch_size, num_workers):
    params = json.load(open(params_path))
    args = build_args(params, ckpt_path)
    fp_loader = make_fp_loader(args.fp_type, entropy_out_dim=args.out_dim,
                               retrieval_path=str(RETRIEVAL_PKL))
    model = MARINA(args, fp_loader)
    # The current model always instantiates a negative-MS encoder (enc_ms_neg); the
    # multuncap checkpoints were trained without it (input_types has no mass_spec_neg).
    # Load non-strict, but assert the ONLY gap is that unused encoder -- nothing else may
    # silently mismatch. enc_ms_neg is never fed here, so its random init is inert.
    missing, unexpected = model.load_state_dict(
        torch.load(ckpt_path, map_location="cpu")["state_dict"], strict=False)
    bad = [k for k in missing if not k.startswith("enc_ms_neg")]
    assert not bad, f"unexpected MISSING keys (not enc_ms_neg): {bad}"
    assert not unexpected, f"unexpected keys in checkpoint: {unexpected}"
    model.eval()
    model.ranker = RankingSet(
        store=torch.load(os.path.join(DATA_DATASET, args.fp_type, "rankingset.pt"),
                         map_location="cpu"))
    model = model.to(device)

    smiles_index = load_smiles_index(str(RETRIEVAL_PKL))
    n_bank = model.ranker.data.size(0)
    assert len(smiles_index) == n_bank, \
        f"bank/smiles_index mismatch: {n_bank} rows vs {len(smiles_index)} smiles"

    res = {"fp_type": args.fp_type, "ckpt": ckpt_path, "fp_radius": fp_loader.max_radius}
    for name, override in SUBSETS.items():
        res[name] = eval_subset(args, model, fp_loader, smiles_index, device,
                                override, batch_size, num_workers)
        print(f"  {name}: {res[name]}", flush=True)

    del model, fp_loader
    torch.cuda.empty_cache()
    return res


def validate_against_test_result(res, ckpt_path):
    """Cross-check STRICT dereplication vs the run's stored test_result.pkl rank@k.

    test_result.pkl is a list of metric dicts (keys like 'test/mean_rank_1/<subset>');
    the training test loop's rank@k is the strict convention, so our derep_strict_top-k
    should match it (up to the ES-checkpoint vs last.ckpt difference). Never raises.
    """
    try:
        trp = os.path.join(os.path.dirname(ckpt_path), "test_result.pkl")
        if not os.path.isfile(trp):
            print(f"[validate] no test_result.pkl beside {ckpt_path}", flush=True)
            return
        tr = pickle.load(open(trp, "rb"))
        merged = {k: v for rec in (tr if isinstance(tr, list) else [tr]) for k, v in rec.items()}
        print("[validate] strict derep vs test_result.pkl rank@k (frac->%):", flush=True)
        for name in ("all_inputs", "hsqc", "c_nmr", "h_nmr", "mass_spec"):
            if name not in res:
                continue
            got = {k: round(res[name].get(f"derep_strict_top{k}", float("nan")), 2) for k in (1, 5, 10)}
            tgt = {}
            for k in (1, 5, 10):
                key = f"test/mean_rank_{k}/{name}"
                if key in merged:
                    v = merged[key]
                    tgt[k] = round(v * 100 if v <= 1 else v, 2)
            print(f"  {name}: strict={got}  test_result={tgt}", flush=True)
    except Exception as e:
        print(f"[validate] skipped ({e!r})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="name|params.json|ckpt per line")
    ap.add_argument("--out", required=True, help="results json (resumed if it exists)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--validate", action="store_true",
                    help="print a rank@k cross-check vs each run's test_result.pkl")
    ap.add_argument("--no_ms_norm", action="store_true",
                    help="disable base-peak normalize_mass_spec (feed raw MS). Required for "
                         "checkpoints trained before 42ebe9f (2026-08-25), e.g. multuncap: "
                         "the model saw un-normalized MS, so the new normalization corrupts it.")
    ap.add_argument("--subsets", default=None,
                    help="comma-separated subset keys to run (default: all). Use to re-run "
                         "only the MS-affected subsets (all_inputs,mass_spec).")
    a = ap.parse_args()

    if a.no_ms_norm:
        # Revert _load_mass_spec's normalization to the pre-42ebe9f identity behaviour so the
        # MS representation matches what these checkpoints were trained on.
        import src.modules.data.inputs as _inp
        _inp.normalize_mass_spec = lambda ms, *args, **kw: ms
        print("[cfg] normalize_mass_spec DISABLED (raw MS, pre-2026-08-25 checkpoints)", flush=True)
    if a.subsets:
        keep = set(a.subsets.split(","))
        for k in list(SUBSETS):
            if k not in keep:
                del SUBSETS[k]
        print(f"[cfg] running subsets: {list(SUBSETS)}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  FP_RADIUS={os.environ.get('FP_RADIUS', '10 (default!)')}", flush=True)

    rows = [ln.strip() for ln in open(a.models) if ln.strip() and not ln.startswith("#")]
    results = json.load(open(a.out)) if os.path.exists(a.out) else {}

    for row in tqdm(rows, desc="scoring"):
        name, params_path, ckpt_path = row.split("|")
        if name in results:
            print(f"skip {name} (done)", flush=True)
            continue
        print(f"=== {name}", flush=True)
        try:
            results[name] = eval_model(params_path, ckpt_path, device,
                                       a.batch_size, a.num_workers)
            if a.validate:
                validate_against_test_result(results[name], ckpt_path)
        except Exception as e:
            results[name] = {"error": repr(e)}
            print(f"ERROR {name}: {e!r}", flush=True)
        json.dump(results, open(a.out, "w"), indent=2)

    print(f"done -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
