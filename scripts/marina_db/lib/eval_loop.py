"""
Shared inference/scoring primitives for the marina_db eval stage.

These are the functions that were copy-pasted verbatim across the four legacy
scorers (eval_comprehensive.py, eval_all_models.py, score_journal.py,
compare_benchmarks.py). One clean home now.

eval_split runs the per-entry forward pass once and returns BOTH retrieval
metrics, because they answer different questions:
  - dereplication top-k: a near-identical structure (cosine of sparse FP > 0.99)
    appears within the top-k retrievals.
  - rank@k: the *exact* query structure ranks in the top-k, via
    RankingSet.batched_rank. Only well-defined when the gold row is in the bank
    (hence the augmented rankingset).

Ranking is tie-aware by default (decision D6): a fingerprint-identical twin does
not count against the truth. Pass strict=True to also emit the pessimistic
(legacy) rank@k for reproducing previously published numbers.
"""
import sys
from dataclasses import fields as dc_fields
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))   # scripts/marina_db  (config)
sys.path.insert(0, str(_HERE.parents[3]))   # repo root          (src)

from src.modules import MARINAArgs, SPECTREArgs
from src.modules.benchmark import filter_data

ARGS_CLASSES = {"MARINA": MARINAArgs, "SPECTRE": SPECTREArgs}


def cos(a, b):
    """Cosine similarity of two 1-D tensors as a python float."""
    return (torch.dot(a, b) / (torch.norm(a) * torch.norm(b))).item()


def to_device(obj, device):
    """Recursively move tensors inside dicts/lists/tuples to `device`."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


def build_args(params: dict, ckpt: str):
    """Rebuild the model args from a saved params.json, forced into benchmark mode.

    Dispatches on params['project_name'] (MARINA/SPECTRE), defaulting to MARINA.
    """
    ArgsCls = ARGS_CLASSES[params.get("project_name", "MARINA")]
    valid = {f.name for f in dc_fields(ArgsCls)}
    kw = {k: v for k, v in params.items() if k in valid}
    kw.update(train=False, test=False, benchmark=True, load_from_checkpoint=ckpt)
    return ArgsCls(**kw)


@torch.no_grad()
def eval_split(entries, model, data_module, fp_loader, restrictions, device,
               smiles_key="smiles", strict=False, derep_thresh=0.99, topn=10,
               return_details=False):
    """Score a list of benchmark entries; one forward pass per entry.

    Returns a list of per-entry records (caller buckets by split + summarises):
        {"split", "cos", "rank", ["rank_strict"], "derep": [bool * topn], "entry"}

    `rank` is tie-aware (D6). With strict=True, `rank_strict` adds the pessimistic
    count. Derep bools flag near-identical hits (cosine of sparse FP > derep_thresh)
    among the top-`topn` retrievals.

    With return_details=True each record also carries the per-compound retrieval
    detail that the derep bools are thresholded from:
        "idxs": [int * topn]        retrieved rankingset row indices
        "derep_cos": [float * topn] cosine of query sparse FP to each retrieved row
    (default off, so score.py's record shape is unchanged.)
    """
    records = []
    for entry in entries:
        inputs = to_device(
            data_module.format_inference_data(filter_data(entry["input"], restrictions)),
            device)
        pred = torch.sigmoid(model(**inputs)[0])
        sfp = fp_loader.build_mfp_for_smiles(entry[smiles_key]).to(device)
        sfp = sfp / torch.norm(sfp)

        idxs = model.ranker.retrieve_idx(pred, topn).tolist()
        derep_cos = [cos(sfp, model.ranker.data[i].to_dense().float().to(device))
                     for i in idxs]
        derep = [c > derep_thresh for c in derep_cos]

        rec = {
            "split": entry.get("split"),
            "cos": cos(pred, sfp),
            "rank": int(model.ranker.batched_rank(
                pred.unsqueeze(0), sfp.unsqueeze(0), tie_aware=True)[0]),
            "derep": derep,
            "entry": entry,
        }
        if strict:
            rec["rank_strict"] = int(model.ranker.batched_rank(
                pred.unsqueeze(0), sfp.unsqueeze(0), tie_aware=False)[0])
        if return_details:
            rec["idxs"] = idxs
            rec["derep_cos"] = derep_cos
        records.append(rec)
    return records


def summarise(records, strict=False):
    """Aggregate per-entry records into rank@1/5/10, derep top-1/5/10, mean_cos."""
    n = len(records)
    out = {"n": n}
    if not n:
        return out
    out["mean_cos"] = sum(r["cos"] for r in records) / n
    for k in (1, 5, 10):
        out[f"rank_{k}"] = 100.0 * sum(r["rank"] < k for r in records) / n
    if strict and "rank_strict" in records[0]:
        for k in (1, 5, 10):
            out[f"rank_{k}_strict"] = 100.0 * sum(r["rank_strict"] < k for r in records) / n
    if records[0].get("derep") is not None:
        for k in (1, 5, 10):
            out[f"derep_top{k}"] = 100.0 * sum(any(r["derep"][:k]) for r in records) / n
    return out
