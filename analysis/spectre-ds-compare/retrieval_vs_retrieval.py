import os, pickle, json, collections
from multiprocessing import Pool
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
RDLogger.DisableLog('rdApp.*')
ROOT = os.path.dirname(os.path.abspath(__file__))

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
    rows = [l.rstrip('\n').split('\t') for l in open(f'{ROOT}/spectre_retrieval.tsv')]
    raw = [r[0] for r in rows]
    src = {r[0]: r[1] for r in rows}
    marina = pickle.load(open(f'{ROOT}/canon_sets.pkl', 'rb'))['marina_retrieval']

    with Pool(16) as pool:
        cmap = dict(zip(raw, pool.map(canon, raw, chunksize=500)))
    cs = {c for c in cmap.values() if c}
    # keep a source tag per canonical molecule (first raw that maps to it)
    csrc = {}
    for r, c in cmap.items():
        if c and c not in csrc: csrc[c] = src[r]

    only_spectre = sorted(cs - marina)
    only_marina  = marina - cs
    with Pool(16) as pool:
        reasons = collections.Counter(pool.map(classify, only_spectre, chunksize=200))

    out = dict(
        spectre_raw=len(raw), spectre_raw_unique=len(set(raw)), spectre_canon=len(cs),
        spectre_canon_collapse=len(set(raw)) - len(cs),
        marina_canon=len(marina),
        shared=len(cs & marina),
        only_spectre=len(only_spectre), only_marina=len(only_marina),
        only_spectre_reasons=dict(reasons.most_common()),
        only_spectre_sources=dict(collections.Counter(csrc[m] for m in only_spectre).most_common()),
        spectre_raw_with_dot=sum(1 for s in raw if '.' in s),
    )
    print(json.dumps(out, indent=2))
    json.dump(out, open(f'{ROOT}/retrieval_vs_retrieval.json', 'w'), indent=2)
    pickle.dump({'spectre_retrieval': cs, 'source': csrc}, open(f'{ROOT}/spectre_retrieval_canon.pkl', 'wb'))
