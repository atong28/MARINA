"""Replace index.pkl inside MARINA-DB.zip without recompressing the other entries.

The archive is entirely STORED (no compression), so every other entry is stream-copied raw.
Writes to a temp file and atomically renames, so the original is untouched on failure.
"""
import os
import shutil
import zipfile

SRC = '/home/atong/Workspace/Datasets/MARINA-DB.zip'
NEW_INDEX = '/home/atong/Workspace/MARINA/data/dataset/index.pkl'
TMP = SRC + '.new'

with zipfile.ZipFile(SRC, 'r') as zin, zipfile.ZipFile(TMP, 'w', allowZip64=True) as zout:
    for item in zin.infolist():
        if item.filename == 'index.pkl':
            continue
        zi = zipfile.ZipInfo(item.filename, date_time=item.date_time)
        zi.compress_type = zipfile.ZIP_STORED
        zi.external_attr = item.external_attr
        zi.internal_attr = item.internal_attr
        with zin.open(item) as sf, zout.open(zi, 'w') as df:
            shutil.copyfileobj(sf, df, length=16 * 1024 * 1024)
    # Append the updated index.pkl (STORED, to match the rest).
    zi = zipfile.ZipInfo('index.pkl')
    zi.compress_type = zipfile.ZIP_STORED
    with open(NEW_INDEX, 'rb') as f, zout.open(zi, 'w') as df:
        shutil.copyfileobj(f, df, length=16 * 1024 * 1024)

os.replace(TMP, SRC)
print('Rewrote', SRC)

# Verify: index.pkl present with the new size and carries formula_vec.
import pickle, io
with zipfile.ZipFile(SRC, 'r') as z:
    names = z.namelist()
    assert 'index.pkl' in names
    info = z.getinfo('index.pkl')
    print('index.pkl in zip:', info.file_size, 'bytes;', len(names), 'total entries')
    data = pickle.loads(z.read('index.pkl'))
    first = next(iter(data.values()))
    assert 'formula_vec' in first and len(first['formula_vec']) == 40, 'formula_vec missing/wrong'
    print('formula_vec present in zip index. sample:', first['formula'],
          'sum(vec)=', sum(first['formula_vec']))
print('OK')
