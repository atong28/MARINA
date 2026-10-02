#!/usr/bin/env python3
"""Write paper/checkpoints/<experiment>.json — one manifest per checkpoint used by a paper table.

Inputs (produced while building the package, 2026-10-01):
  --pvc   dump from the Nautilus PVC:  "@@CKPT <exp> <bytes> <sha256>  <relpath>" / "@@PARAMS <exp>" <params.json> "@@END"
  --local lines "<bytes> <sha256>  <path>" (sha256sum of local copies); params.json read beside each ckpt
  --registry wiki active/experiment-registry.md (training commit + cluster per experiment)
Re-verify a staged work dir against the manifests with paper/tools/verify_ckpts.py.
"""
import argparse
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "checkpoints")

TABLES = {
    "marina-db-open-chnmr-uniqmult-formula": ["results_main (flagship)", "spectre_comparison (flagship)", "sim_exp_gap"],
    "marina-db-uniqmult-formula": ["fp_comparison"],
    "marina-db-sherlock-formula": ["fp_comparison"],
    "marina-db-uncapped-formula": ["fp_comparison (appendix row)", "results_training_regime (All inputs)"],
    "marina-db-cap5-formula": ["fp_comparison (appendix row)"],
    "marina-deltaai-substructure": ["fp_comparison (appendix row)"],
    "marina-db-uncapped-nmr": ["results_training_regime (NMR+MW)"],
    "marina-db-uncapped-noform": ["results_training_regime (NMR+MS/MS+MW)"],
}
DATASET = {"marina-db-open": "MARINA-DB (CH-NMR-NP-first; disk Datasets/MARINA-DB-OPEN, zip sha256 2780459f...)"}


def family(exp):
    return re.sub(r"-s\d+$", "", exp)


def registry(path):
    reg = {}
    for line in open(path):
        m = re.match(r"\|\s*([a-z0-9.-]+-s\d)\s*\|\s*([^|]+?)\s*\|\s*`([0-9a-f]{7,40})`", line)
        if m:
            reg[m.group(1)] = {"cluster": m.group(2), "training_commit": m.group(3)}
    return reg


def write(exp, ckpt_file, nbytes, sha, params, locations, reg, extra=None):
    fam = family(exp)
    epoch = re.search(r"epoch=(\d+)", ckpt_file)
    man = {
        "experiment": exp,
        "seed": int(exp.rsplit("-s", 1)[1]),
        "model": "MARINA",
        "used_in": TABLES.get(fam, []),
        "training_dataset": DATASET["marina-db-open"] if fam.startswith("marina-db-open") else
                            "MARINA-DB-PRIVATE (disk Datasets/MARINA-DB)",
        "fp_type": params.get("fp_type"),
        "input_types": params.get("input_types"),
        **reg.get(exp, {}),
        "checkpoint": {"file": os.path.basename(ckpt_file), "epoch": int(epoch.group(1)) if epoch else None,
                       "bytes": int(nbytes), "sha256": sha,
                       "selection": "early-stopping best (max val/mean_cos); the only epoch_*.ckpt kept in the run dir"},
        "locations": locations,
        "params": params,
    }
    if extra:
        man.update(extra)
    os.makedirs(OUT, exist_ok=True)
    json.dump(man, open(os.path.join(OUT, f"{exp}.json"), "w"), indent=2)
    print(f"{exp:<45} ep{man['checkpoint']['epoch']} {sha[:12]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pvc", required=True)
    ap.add_argument("--local", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--results", default=os.path.expanduser("~/Workspace/MARINA/results"))
    a = ap.parse_args()
    reg = registry(a.registry)
    local = {}
    for line in open(a.local):
        nbytes, sha, path = line.split(maxsplit=2)
        local[os.path.realpath(os.path.join(a.results, path.strip()))] = (nbytes, sha)

    # PVC checkpoints (the copy every Nautilus eval Job reads)
    text = open(a.pvc).read()
    for m in re.finditer(r"@@CKPT (\S+) (\d+) ([0-9a-f]{64})\s+(\S+)\n@@PARAMS \S+\n(.*?)\n@@END", text, re.S):
        exp, nbytes, sha, rel, params = m.groups()
        locs = {"nautilus_pvc": f"atong-spectre:/root/gurusmart/Moonshot/results/{rel}"}
        for p, (lb, ls) in local.items():
            if p.endswith(os.path.basename(rel)) and f"/{exp}/" in p and ls == sha:
                locs["anthony3"] = p
        write(exp, rel, nbytes, sha, json.loads(params), locs, reg)

    # local copies of cluster runs (DeltaAI / Anvil) on anthony3
    remote = {"marina-db-open-chnmr-uniqmult-formula": ("deltaai", "/projects/bibx/atong1/runs/results"),
              "marina-deltaai-substructure": ("deltaai", "/projects/bibx/atong1/runs/results")}
    for exp in [f"{fam}-s{s}" for fam in remote for s in range(3)]:
        if not glob.glob(os.path.join(a.results, exp, "*", "epoch_*.ckpt")):
            print(f"{exp:<45} (no local copy yet; skipped)")
            continue
        ck = glob.glob(os.path.join(a.results, exp, "*", "epoch_*.ckpt"))
        assert len(ck) == 1, (exp, ck)
        ck = os.path.realpath(ck[0])
        nbytes, sha = local[ck]
        params = json.load(open(os.path.join(os.path.dirname(ck), "params.json")))
        site, cluster_dir = remote[family(exp)]
        locs = {"anthony3": ck, site: f"{cluster_dir}/{exp}/{os.path.basename(os.path.dirname(ck))}/{os.path.basename(ck)}"}
        extra = {"note": "early-stopped at epoch 144 after poor training; a finished run"} if exp.endswith("substructure-s2") else None
        write(exp, ck, nbytes, sha, params, locs, reg, extra)

    # SPECTRE deployed model
    sp = os.path.realpath(os.path.expanduser("~/Deployments/SPECTRE-web/checkpoints/spectre_flexible/best.ckpt"))
    nbytes, sha = next(v for p, v in local.items() if p == sp)
    params = json.load(open(os.path.join(os.path.dirname(sp), "params.json")))
    man = {"experiment": "spectre-deployed", "model": "SPECTRE (released website model)",
           "used_in": ["spectre_comparison"], "fp_type": "RankingEntropy (Sherlock, 16,384 bits)",
           "eval_flags": "--project_name SPECTRE --fp_type RankingEntropy --legacy_spectre",
           "checkpoint": {"file": "best.ckpt", "bytes": int(nbytes), "sha256": sha},
           "locations": {"anthony3": sp}, "params": params}
    json.dump(man, open(os.path.join(OUT, "spectre-deployed.json"), "w"), indent=2)
    print(f"{'spectre-deployed':<45} {sha[:12]}")


if __name__ == "__main__":
    main()
