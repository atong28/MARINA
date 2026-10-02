#!/usr/bin/env python3
"""Single-atom multiplicity-bit accuracy: NMR vs NMR+Formula on the journal benchmark.

The multiplicity fingerprint's radius-0 features are single heavy atoms -- their fragment
is a bare element token ("C", "c" aromatic, "N", "O", "Cl", ...). A molecule with n atoms
of a token lights the cumulative buckets ("C",1)..("C",n), so these columns encode
per-element counts -- exactly what the molecular formula supplies directly. This script
isolates those columns and measures how well the model reconstructs them under:

    nmr          = {hsqc, c_nmr, h_nmr}
    nmr_formula  = {hsqc, c_nmr, h_nmr, formula}   (formula from SMILES; MW & MS/MS withheld)

For each split (val/test) and combo it reports, and the (nmr_formula - nmr) delta:
  bit           : mean # single-atom bits wrong per molecule (sigmoid>=0.5 vs gold),
                  overall single-atom bit accuracy.
  element-token : per token (C, c, N, ...) predicted count (# lit buckets) vs gold count;
                  mean tokens wrong per molecule + per-token exact-count accuracy.
  element-total : per element summing aromatic+aliphatic+charged tokens (C+c, N+n+..,
                  O+o+..) -- the quantity the formula determines directly.

Reuses the benchmark helpers so the model/inputs are built exactly as in the main eval.
Runs on CPU automatically when no GPU is present. Never modifies benchmark-journal.pkl.

Usage (from the MARINA repo root, with DATASET_ROOT/BENCHMARK_ROOT set):
    python scripts/benchmark/single_atom_bit_eval.py --ckpt RUN/best.ckpt
    python scripts/benchmark/single_atom_bit_eval.py --ckpt RUN/best.ckpt --params RUN/params.json \
        --splits val test --out eval_out/single_atom.json
"""
import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from dataclasses import fields as dc_fields

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")  # radius-0 aromatic single-atom frags trip sanitize warnings

from src.modules import MARINA, MARINAArgs, MARINADataModule
from src.modules.data.fp_loader import make_fp_loader
from src.modules.benchmark import filter_data, formula_vec_from_smiles, _to_device
from src.modules.core.const import DATASET_ROOT, BENCHMARK_ROOT

NMR = ["hsqc", "c_nmr", "h_nmr"]
COMBOS = {"nmr": NMR, "nmr_formula": NMR + ["formula"]}


def single_atom_columns(fp_loader):
    """Map col -> (token, bucket_k) for every radius-0 single-heavy-atom feature, plus a
    token -> element-symbol map (so C/c/[O-] fold to their element for the total rollup)."""
    cols, token_elem = {}, {}
    for col, feat in fp_loader.fp_index_to_bitinfo_map.items():
        frag, k = feat if isinstance(feat, tuple) and len(feat) == 2 else (feat, None)
        if not isinstance(frag, str) or frag == "":
            continue
        mol = Chem.MolFromSmiles(frag, sanitize=False)
        if mol is None or mol.GetNumAtoms() != 1 or mol.GetNumBonds() != 0:
            continue
        cols[col] = (frag, k)
        token_elem[frag] = mol.GetAtomWithIdx(0).GetSymbol()
    return cols, token_elem


def build_args(params_path, ckpt, fp_type):
    with open(params_path) as f:
        params = json.load(f)
    valid = {f.name for f in dc_fields(MARINAArgs)}
    kw = {k: v for k, v in params.items() if k in valid}
    kw.update(train=False, test=False, benchmark=True,
              load_from_checkpoint=ckpt, experiment_name="single_atom_bit_eval")
    if fp_type:
        kw["fp_type"] = fp_type
    return MARINAArgs(**kw)


@torch.no_grad()
def predict_bits(model, data_module, entry, combo_mods, dev):
    """Binarized predicted fingerprint (sigmoid>=0.5) for one entry under a modality combo,
    or None if no spectral modality survives the restriction."""
    raw = entry["input"]
    if "formula" in combo_mods:
        raw = {**raw, "formula": formula_vec_from_smiles(entry["smiles"])}
    clean = filter_data(raw, combo_mods)
    if not any(k not in ("mw", "formula") for k in clean):
        return None
    inputs = data_module.format_inference_data(clean)
    inputs = _to_device(inputs, dev)
    output = model(**inputs)
    return (output[0] >= 0.0).float().cpu()  # logits>=0 == sigmoid>=0.5


def score_entry(pred_bits, gold_bits, cols, token_elem):
    """Per-entry stats over the single-atom columns."""
    col_ids = list(cols)
    p = pred_bits[col_ids]
    g = gold_bits[col_ids]
    bit_wrong = int((p != g).sum().item())

    # counts per token / element = # lit buckets (gold is contiguous 1..n so == true count)
    p_tok, g_tok = defaultdict(int), defaultdict(int)
    p_el, g_el = defaultdict(int), defaultdict(int)
    for col, (frag, _k) in cols.items():
        el = token_elem[frag]
        pb, gb = int(pred_bits[col].item() >= 0.5), int(gold_bits[col].item() >= 0.5)
        p_tok[frag] += pb; g_tok[frag] += gb
        p_el[el] += pb; g_el[el] += gb

    # per-token / per-element exact-count correctness, restricted to present (gold>0) --
    # this is a recall view (the per-element breakdown table).
    tok_correct = {t: (p_tok[t] == g_tok[t]) for t in g_tok if g_tok[t] > 0}
    el_correct = {e: (p_el[e] == g_el[e]) for e in g_el if g_el[e] > 0}
    # phantom-penalizing view: score over the UNION of present-or-predicted elements, so a
    # false positive on an absent element (gold 0, pred>0) counts as an error, while
    # absent-and-not-predicted elements are not scored (no trivial true-negative inflation).
    union = set(e for e in g_el if g_el[e] > 0) | set(e for e in p_el if p_el[e] > 0)
    el_union_correct = {e: (p_el[e] == g_el[e]) for e in union}
    return {
        "bit_wrong": bit_wrong,
        "tok_correct": tok_correct, "tok_abserr": {t: abs(p_tok[t] - g_tok[t]) for t in g_tok if g_tok[t] > 0},
        "el_correct": el_correct, "el_abserr": {e: abs(p_el[e] - g_el[e]) for e in g_el if g_el[e] > 0},
        "el_union_correct": el_union_correct,
    }


def aggregate(recs, n_single_cols):
    """Aggregate per-entry stats into the reported metrics."""
    n = len(recs)
    if n == 0:
        return {"n": 0}
    total_wrong = sum(r["bit_wrong"] for r in recs)
    # element rollups: accuracy over (molecule, element-present) pairs
    def rollup(key_c, key_e):
        per = defaultdict(lambda: [0, 0, 0])  # key -> [correct, total, abserr_sum]
        mols_all_correct = 0
        for r in recs:
            cc = r[key_c]
            if cc and all(cc.values()):
                mols_all_correct += 1
            ae = r[key_e]
            for k, ok in cc.items():
                per[k][0] += int(ok); per[k][1] += 1; per[k][2] += ae[k]
        per_key = {k: {"acc_pct": 100.0 * c / t, "mean_abserr": s / t, "n": t}
                   for k, (c, t, s) in sorted(per.items())}
        tot = sum(t for _, t, _ in per.values())
        cor = sum(c for c, _, _ in per.values())
        return {"overall_acc_pct": 100.0 * cor / tot if tot else 0.0,
                "mol_all_correct_pct": 100.0 * mols_all_correct / n,
                "per_key": per_key}
    # phantom-penalizing element totals (union of present-or-predicted): raw counts so the
    # combined (val+test) pool is exact downstream. overall = micro-avg over union pairs;
    # mol_all = molecules whose full per-element count vector matches gold exactly.
    u_correct = sum(int(v) for r in recs for v in r["el_union_correct"].values())
    u_total = sum(len(r["el_union_correct"]) for r in recs)
    u_mol = sum(1 for r in recs if r["el_union_correct"] and all(r["el_union_correct"].values()))
    return {
        "n": n,
        "bit": {"mean_wrong_per_mol": total_wrong / n,
                "accuracy_pct": 100.0 * (1 - total_wrong / (n * n_single_cols)),
                "n_single_cols": n_single_cols},
        "element_token": rollup("tok_correct", "tok_abserr"),
        "element_total": rollup("el_correct", "el_abserr"),
        "element_total_strict": {
            "overall_acc_pct": 100.0 * u_correct / u_total if u_total else 0.0,
            "mol_all_correct_pct": 100.0 * u_mol / n,
            "correct": u_correct, "total": u_total, "mol_all": u_mol},
    }


def print_report(results):
    for split, combos in results.items():
        print(f"\n{'='*72}\nSPLIT: {split}\n{'='*72}")
        for combo, m in combos.items():
            if m.get("n", 0) == 0:
                print(f"[{combo}] no entries"); continue
            b = m["bit"]
            print(f"\n[{combo}]  n={m['n']}")
            print(f"  bit          : {b['mean_wrong_per_mol']:.3f} wrong/mol   "
                  f"acc={b['accuracy_pct']:.2f}%  (over {b['n_single_cols']} single-atom cols)")
            for name in ("element_token", "element_total", "element_total_strict"):
                r = m[name]
                print(f"  {name:20}: overall_acc={r['overall_acc_pct']:.2f}%  "
                      f"mol_all_correct={r['mol_all_correct_pct']:.2f}%")
            r = m["element_total"]["per_key"]
            row = "  per-element  : " + "  ".join(
                f"{k}={v['acc_pct']:.0f}%(n{v['n']})" for k, v in r.items())
            print(row)
        if "nmr" in combos and "nmr_formula" in combos and combos["nmr"].get("n"):
            a, c = combos["nmr"], combos["nmr_formula"]
            print(f"\n  DELTA (nmr_formula - nmr):")
            print(f"    bit wrong/mol : {c['bit']['mean_wrong_per_mol'] - a['bit']['mean_wrong_per_mol']:+.3f}")
            print(f"    bit acc pp    : {c['bit']['accuracy_pct'] - a['bit']['accuracy_pct']:+.2f}")
            print(f"    el-total acc  : {c['element_total']['overall_acc_pct'] - a['element_total']['overall_acc_pct']:+.2f} pp")
            print(f"    el-total full : {c['element_total']['mol_all_correct_pct'] - a['element_total']['mol_all_correct_pct']:+.2f} pp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--params", default=None, help="default: params.json beside --ckpt")
    ap.add_argument("--fp_type", default=None, help="override params.json fp_type")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args()

    params_path = args.params or os.path.join(os.path.dirname(args.ckpt), "params.json")
    a = build_args(params_path, args.ckpt, args.fp_type)
    if "formula" not in a.input_types:
        print(f"WARNING: model input_types={a.input_types} has no 'formula' encoder; "
              f"the nmr_formula combo will be meaningless for this checkpoint.")

    fp_loader = make_fp_loader(a.fp_type, entropy_out_dim=a.out_dim)
    model = MARINA(a, fp_loader)
    data_module = MARINADataModule(a, fp_loader)
    # Load weights directly (map_location handles a CUDA-saved ckpt on a CPU box) and skip
    # setup_ranker -- per-bit accuracy needs only the forward pass, not the retrieval bank.
    sd = torch.load(args.ckpt, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    unexpected = [k for k in unexpected if not k.startswith("ranker.")]  # ranker intentionally absent
    if missing or unexpected:
        print(f"load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected (non-ranker)")
        if missing:
            print("  missing:", missing[:10])
        if unexpected:
            print("  unexpected:", unexpected[:10])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()

    cols, token_elem = single_atom_columns(fp_loader)
    print(f"fp_type={a.fp_type}  single-atom columns={len(cols)}  device={dev}")
    if not cols:
        raise SystemExit(
            f"fp_type={a.fp_type} exposes no single-atom (radius-0 element) columns -- its "
            f"radius-0 fragments collapse to ''. Use a multiplicity fp_type "
            f"(e.g. RankingEntropyMultiplicityUncapped).")

    journal = pickle.load(open(os.path.join(BENCHMARK_ROOT, "benchmark-journal.pkl"), "rb"))
    gold_cache = {}
    results = {}
    for split in args.splits:
        split_data = {k: v for k, v in journal.items() if v.get("split") == split}
        results[split] = {}
        for combo, mods in COMBOS.items():
            recs = []
            for entry in split_data.values():
                pred = predict_bits(model, data_module, entry, mods, dev)
                if pred is None:
                    continue
                smi = entry["smiles"]
                if smi not in gold_cache:
                    gold_cache[smi] = fp_loader.build_mfp_for_smiles(smi)
                recs.append(score_entry(pred, gold_cache[smi], cols, token_elem))
            results[split][combo] = aggregate(recs, len(cols))
            print(f"  done {split}/{combo}: n={results[split][combo]['n']}")

    print_report(results)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"ckpt": args.ckpt, "fp_type": a.fp_type,
                       "n_single_cols": len(cols), "results": results}, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
