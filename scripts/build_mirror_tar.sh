#!/usr/bin/env bash
# Build the Snapshots/Dataset/MARINA-DB.tar mirror artifact from the exact file set in the
# distributable MARINA-DB.zip (so the Drive mirror == the training artifact, incl. the
# formula-updated index.pkl). Then checksum it and record it in MANIFEST.sha256.
set -euo pipefail

DS_ROOT=/home/atong/Workspace/MARINA/data/dataset
ZIP=/home/atong/Workspace/Datasets/MARINA-DB.zip
OUT_DIR=/home/atong/Workspace/Snapshots/Dataset
TAR="$OUT_DIR/MARINA-DB.tar"
MANIFEST="$OUT_DIR/MANIFEST.sha256"

# Exact file list from the zip (files only, no directory entries).
FILELIST=$(mktemp)
unzip -Z1 "$ZIP" | grep -v '/$' > "$FILELIST"
echo "==> taring $(wc -l < "$FILELIST") files from $DS_ROOT"

# Sanity: every listed file must exist under the dataset root.
missing=0
while IFS= read -r f; do
    [ -f "$DS_ROOT/$f" ] || { echo "  MISSING $DS_ROOT/$f" >&2; missing=1; }
done < "$FILELIST"
[ "$missing" -eq 0 ] || { echo "aborting: files missing" >&2; exit 1; }

tar -cf "$TAR" -C "$DS_ROOT" -T "$FILELIST"
rm -f "$FILELIST"
echo "==> wrote $TAR ($(du -h "$TAR" | cut -f1))"

echo "==> sha256"
SUM=$(sha256sum "$TAR" | cut -d' ' -f1)
# Keep other entries (e.g. MARINA1.tar), replace/add MARINA-DB.tar's line.
tmp=$(mktemp)
grep -v '  MARINA-DB.tar$' "$MANIFEST" 2>/dev/null > "$tmp" || true
echo "$SUM  MARINA-DB.tar" >> "$tmp"
mv "$tmp" "$MANIFEST"
echo "==> MANIFEST.sha256 now:"
cat "$MANIFEST"

echo "==> verify index.pkl inside the tar carries formula_vec"
python3 - "$TAR" <<'PY'
import sys, tarfile, pickle, io
tar = tarfile.open(sys.argv[1])
member = tar.getmember('index.pkl')
data = pickle.load(io.BytesIO(tar.extractfile(member).read()))
first = next(iter(data.values()))
assert 'formula_vec' in first and len(first['formula_vec']) == 40, 'formula_vec missing!'
print('  OK: index.pkl in tar has formula_vec; sample', first['formula'], 'sum', sum(first['formula_vec']))
PY
echo "DONE"
