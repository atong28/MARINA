#!/usr/bin/env python3
"""Write the manual download queue: one NP-MRD link per source paper.

Excludes papers whose PDF is already in papers/ (downloaded for an adjacent NP ID
during the original 154-compound run). Each row links to the lowest-numbered
un-extracted NP ID on that paper, so each source is opened exactly once.
"""
import json
import os

BASE = '/home/user/atong/Benchmark'
papers = json.load(open(os.path.join(BASE, 'new_candidates.json')))['papers']
old = json.load(open(os.path.join(BASE, 'npmrd_scraped_refs.json')))

# DOI -> True when a PDF for that paper already sits in papers/
have = set()
for npid, refs in old.items():
    if refs and refs[0].get('doi') and os.path.exists(os.path.join(BASE, 'papers', f'{npid}.pdf')):
        have.add(refs[0]['doi'].lower())

todo = [p for p in papers if not (p['ref_type'] == 'doi' and p['ref'] in have)]
todo.sort(key=lambda p: p['order'])          # keep the seed-0 shuffle
for i, p in enumerate(todo, 1):
    p['n'] = i

with open(os.path.join(BASE, 'npmrd_links.txt'), 'w') as f:
    for p in todo:
        f.write(f"https://np-mrd.org/natural_products/{p['npids'][0]}\n")

with open(os.path.join(BASE, 'npmrd_links.md'), 'w') as f:
    f.write(f'# Papers to download - {len(todo)} sources, '
            f'{sum(p["n_compounds"] for p in todo)} compounds (shuffled, seed 0)\n\n')
    f.write('| # | NP-MRD (first of series) | N | Year | Access | NMR? | Title |\n'
            '|--:|---|--:|--:|---|---|---|\n')
    for p in todo:
        acc = p['npids'][0]
        title = (p.get('title') or p.get('ref_text') or '').replace('|', '/')
        title = title[:70] + ('...' if len(title) > 70 else '')
        f.write(f"| {p['n']} | [{acc}](https://np-mrd.org/natural_products/{acc}) | "
                f"{p['n_compounds']} | {p.get('year', '')} | {p.get('access', '')} | "
                f"{p.get('nmr_evidence', '')} | {title} |\n")

json.dump(todo, open(os.path.join(BASE, 'npmrd_links.json'), 'w'), indent=1)
print(f'{len(papers)} papers - {len(papers) - len(todo)} already on disk = {len(todo)} to download')
print(f'covering {sum(p["n_compounds"] for p in todo)} compounds')
