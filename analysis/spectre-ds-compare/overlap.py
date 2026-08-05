import os, pickle, json
from multiprocessing import Pool
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("MARINA_DATA_ROOT", "/home/user/atong")

def canon(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    s = Chem.MolToSmiles(m, isomericSmiles=False, canonical=True)
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, isomericSmiles=False, canonical=True) if m else None

def cset(S, pool):
    return {x for x in pool.imap_unordered(canon, S, chunksize=500) if x}

if __name__ == '__main__':
    E = os.path.join(ROOT, 'extract')
    spec = set()
    for ds in ['SMILES_dataset', 'OneD_Only_Dataset']:
        for sp in ['train', 'val', 'test']:
            spec |= set(pickle.load(open(f'{E}/{ds}/{sp}/SMILES/index.pkl', 'rb')).values())
    mar = pickle.load(open(os.path.join(DATA_ROOT, 'Datasets/MARINA1/index.pkl'), 'rb'))
    mar_s = {v['smiles'] for v in mar.values()}
    ret = pickle.load(open(os.path.join(DATA_ROOT, 'Datasets/MARINA1/retrieval.pkl'), 'rb'))
    ret_s = {v['smiles'] for v in ret.values()}

    with Pool(16) as pool:
        cs = cset(spec, pool)
        cm = cset(mar_s, pool)
        cr = cset(ret_s, pool)

    r = dict(
        spectre_raw_unique=len(spec), spectre_canon=len(cs),
        marina_index_raw=len(mar_s), marina_index_canon=len(cm),
        marina_retrieval_raw=len(ret_s), marina_retrieval_canon=len(cr),
        index_subset_of_retrieval=bool(cm <= cr),
        spectre_in_marina_index=len(cs & cm),
        spectre_in_marina_retrieval=len(cs & cr),
        spectre_not_in_marina_retrieval=len(cs - cr),
    )
    print(json.dumps(r, indent=2))
    json.dump(r, open(os.path.join(ROOT, 'overlap.json'), 'w'), indent=2)
