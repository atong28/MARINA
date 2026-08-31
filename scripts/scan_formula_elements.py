import pickle, re, collections
from pathlib import Path

idx_path = Path('data/dataset/index.pkl')
with open(idx_path, 'rb') as f:
    data = pickle.load(f)
print('entries:', len(data))

tok = re.compile(r'([A-Z][a-z]?)(\d*)')
elem_count = collections.Counter()   # how many molecules contain element
elem_max = collections.Counter()     # max count of element in any molecule
missing_formula = 0
sample = None
for k, v in data.items():
    f = v.get('formula')
    if not f:
        missing_formula += 1
        continue
    if sample is None:
        sample = (k, v.get('smiles'), f)
    for sym, num in tok.findall(f):
        if not sym:
            continue
        n = int(num) if num else 1
        elem_count[sym] += 1
        if n > elem_max[sym]:
            elem_max[sym] = n

print('missing formula:', missing_formula)
print('sample entry:', sample)
print('num distinct elements:', len(elem_count))
print()
N = len(data)
print('%-6s %8s %8s %9s' % ('elem', '#mols', '%mols', 'maxcount'))
for el, c in elem_count.most_common():
    print('%-6s %8d %7.3f%% %9d' % (el, c, 100 * c / N, elem_max[el]))
