#!/usr/bin/env python3
"""Check a staged work dir's checkpoints against paper/checkpoints/*.json (size + sha256).

Usage: python paper/tools/verify_ckpts.py <W>/ckpt [experiment ...]
"""
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(path, buf=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def main():
    root, only = sys.argv[1], set(sys.argv[2:])
    bad = 0
    for man_path in sorted(glob.glob(os.path.join(HERE, "..", "checkpoints", "*.json"))):
        man = json.load(open(man_path))
        exp = man["experiment"]
        if only and exp not in only:
            continue
        found = glob.glob(os.path.join(root, exp, "**", man["checkpoint"]["file"]), recursive=True)
        if not found:
            print(f"absent   {exp}")
            continue
        ok = (os.path.getsize(found[0]) == man["checkpoint"]["bytes"]
              and sha256(found[0]) == man["checkpoint"]["sha256"])
        bad += not ok
        print(f"{'ok' if ok else 'MISMATCH':<8} {exp}  {found[0]}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
