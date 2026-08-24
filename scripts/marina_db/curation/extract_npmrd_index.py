#!/usr/bin/env python3
"""Flatten the NP-MRD NP0300001-NP0350000 dump into a compact index for candidate selection.

Writes npmrd_index.json: accession -> {name, smiles, formula, mw, creation_date,
update_date, refs (list of reference_text), dois, pmids}.
"""
import json
import re

SRC = '/home/user/atong/Benchmark/npmrd_json/npmrd_natural_products_NP0300001_NP0350000.json'
OUT = '/home/user/atong/Benchmark/npmrd_index.json'

DOI_RE = re.compile(r'\b(10\.\d{4,9}/[^\s"<>,;)\]]+)')
PMID_RE = re.compile(r'\bPMID:?\s*(\d{6,9})\b', re.I)


def ref_texts(gr):
    """general_references -> list of reference_text strings (schema varies)."""
    if not gr:
        return []
    ref = gr.get('reference')
    if not ref:
        return []
    if isinstance(ref, dict):
        ref = [ref]
    out = []
    for r in ref:
        t = r.get('reference_text')
        if t:
            out.append(t)
    return out


with open(SRC) as f:
    data = json.load(f)
records = data['np_mrd']['natural_product']
print(f'loaded {len(records)} records')

index = {}
for r in records:
    texts = ref_texts(r.get('general_references'))
    joined = ' '.join(texts)
    dois = sorted({d.rstrip('.') for d in DOI_RE.findall(joined)})
    pmids = sorted(set(PMID_RE.findall(joined)))
    index[r['accession']] = {
        'name': r.get('name'),
        'smiles': r.get('smiles'),
        'formula': r.get('chemical_formula'),
        'mw': r.get('average_molecular_weight'),
        'creation_date': r.get('creation_date'),
        'update_date': r.get('update_date'),
        'refs': texts,
        'dois': dois,
        'pmids': pmids,
    }

with open(OUT, 'w') as f:
    json.dump(index, f)
print(f'wrote {len(index)} -> {OUT}')

n_refs = sum(1 for v in index.values() if v['refs'])
n_lotus_only = sum(1 for v in index.values()
                   if v['refs'] and all('lotus' in t.lower() for t in v['refs']))
n_doi = sum(1 for v in index.values() if v['dois'])
print(f'with refs: {n_refs}  lotus-only: {n_lotus_only}  with DOI: {n_doi}')
