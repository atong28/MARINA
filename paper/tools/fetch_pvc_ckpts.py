#!/usr/bin/env python3
"""Stream the checkpoints that live only on the Nautilus PVC into a work dir, then verify them.

Run on the GPU box (e.g. grapefruit). Each file is streamed through a host with kubectl access (default `vm3`):
    ssh vm3 kubectl exec -n guru-research <pod> -- cat <pvc path>   >  W/ckpt/<exp>/<ts>/<file>
so nothing is stored on that host. params.json is written from the manifest. Checkpoints already present with the
right sha256 are skipped. <pod> must be a running pod that mounts the atong-spectre PVC at /root/gurusmart.

  python3 paper/tools/fetch_pvc_ckpts.py --work $W --pod <pod> [--via vm3] [experiment ...]
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = "atong-spectre:/root/gurusmart/"


def sha256(path, buf=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--pod", required=True)
    ap.add_argument("--via", default="vm3", help="ssh host with kubectl access; '' = run kubectl locally")
    ap.add_argument("--namespace", default="guru-research")
    ap.add_argument("experiments", nargs="*")
    a = ap.parse_args()
    bad = 0
    for man_path in sorted(glob.glob(os.path.join(HERE, "..", "checkpoints", "*.json"))):
        man = json.load(open(man_path))
        exp, loc = man["experiment"], man.get("locations", {}).get("nautilus_pvc")
        if not loc or (a.experiments and exp not in a.experiments):
            continue
        rel = loc[len(PREFIX):]                       # Moonshot/results/<exp>/<ts>/<file>
        ts = os.path.basename(os.path.dirname(rel))
        dest_dir = os.path.join(a.work, "ckpt", exp, ts)
        dest = os.path.join(dest_dir, man["checkpoint"]["file"])
        os.makedirs(dest_dir, exist_ok=True)
        json.dump(man["params"], open(os.path.join(dest_dir, "params.json"), "w"), indent=2)
        want = man["checkpoint"]["sha256"]
        if os.path.exists(dest) and os.path.getsize(dest) == man["checkpoint"]["bytes"] and sha256(dest) == want:
            print(f"have     {exp}")
            continue
        print(f"fetch    {exp}  ({man['checkpoint']['bytes'] / 1e9:.2f} GB)", flush=True)
        kube = f"kubectl exec -n {a.namespace} {a.pod} -- cat '/root/gurusmart/{rel}'"
        cmd = ["ssh", a.via, kube] if a.via else ["bash", "-c", kube]   # --via '' = kubectl on this host
        with open(dest + ".part", "wb") as f:
            rc = subprocess.run(cmd, stdout=f).returncode
        ok = rc == 0 and sha256(dest + ".part") == want
        if ok:
            os.replace(dest + ".part", dest)
        bad += not ok
        print(f"{'ok' if ok else 'FAILED':<8} {exp}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
