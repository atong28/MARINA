import os, pickle, json, collections
from multiprocessing import Pool
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
RDLogger.DisableLog('rdApp.*')
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("MARINA_DATA_ROOT", "/home/user/atong")

def canon(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    s = Chem.MolToSmiles(m, isomericSmiles=False, canonical=True)
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, isomericSmiles=False, canonical=True) if m else None

def classify(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return 'invalid_smiles'
    if '.' in s: return 'salt_or_mixture'
    mw = rdMolDescriptors.CalcExactMolWt(m)
    if mw > 1000: return 'mw_gt_1000'
    if mw < 100: return 'mw_lt_100'
    nums = {a.GetAtomicNum() for a in Chem.AddHs(m).GetAtoms()}
    if 6 not in nums and 1 not in nums: return 'no_c_or_h'
    return 'other'

if __name__ == '__main__':
    E = os.path.join(ROOT, 'extract_full')
    spec = set()
    for ds in ['SMILES_dataset', 'OneD_Only_Dataset']:
        for sp in ['train', 'val', 'test']:
            spec |= set(pickle.load(open(f'{E}/{ds}/{sp}/SMILES/index.pkl', 'rb')).values())
    ret = pickle.load(open(os.path.join(DATA_ROOT, 'Datasets/MARINA1/retrieval.pkl'), 'rb'))
    ret_s = {v['smiles'] for v in ret.values()}

    with Pool(16) as pool:
        cs = {x for x in pool.imap_unordered(canon, spec, chunksize=500) if x}
        cr = {x for x in pool.imap_unordered(canon, ret_s, chunksize=500) if x}
        missing = sorted(cs - cr)
        cls = pool.map(classify, missing, chunksize=50)
        reasons = collections.Counter(cls)
        other = [m for m, c in zip(missing, cls) if c == 'other']
        # cache canonical sets so future diffs are cheap
        pickle.dump({'spectre': cs, 'marina_retrieval': cr}, open(os.path.join(ROOT, 'canon_sets.pkl'),'wb'))

    # raw-form salt check on the pre-canonical SPECTRE SMILES
    raw_salts = sum(1 for s in spec if '.' in s)
    ret_salts = sum(1 for s in ret_s if '.' in s)

    out = dict(
        spectre_canon=len(cs), marina_retrieval_canon=len(cr),
        missing=len(missing), missing_reasons=dict(reasons.most_common()),
        spectre_raw_with_dot=raw_salts, marina_retrieval_with_dot=ret_salts,
        missing_examples=missing[:15], other_molecules=other,
    )
    print(json.dumps(out, indent=2))
    json.dump(out, open(os.path.join(ROOT, 'retrieval_diff.json'), 'w'), indent=2)
