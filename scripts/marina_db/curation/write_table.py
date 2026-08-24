#!/usr/bin/env python3
"""Render the shuffled paper queue as new_candidates.csv and new_candidates.md."""
import csv
import json
import os

BASE = '/home/user/atong/Benchmark'
CAND = os.path.join(BASE, 'new_candidates.json')

papers = json.load(open(CAND))['papers']
papers.sort(key=lambda p: p['order'])

with open(os.path.join(BASE, 'new_candidates.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['order', 'url', 'ref', 'title', 'journal', 'year', 'oa_pdf',
                'n_compounds', 'npids', 'names', 'npmrd_spectra', 'pdf_target',
                'ref_text'])
    for p in papers:
        w.writerow([p['order'], p['url'], p['ref'], p.get('title', ''),
                    p.get('journal', ''), p.get('year', ''),
                    p.get('oa_pdf_url') or '', p['n_compounds'],
                    ' '.join(p['npids']), ' | '.join(p['names']),
                    ' '.join(p['spectra']), f"papers/{p['npids'][0]}.pdf",
                    p.get('ref_text', '')])

with open(os.path.join(BASE, 'new_candidates.md'), 'w') as f:
    f.write(f'# Extraction queue - {len(papers)} papers, '
            f'{sum(p["n_compounds"] for p in papers)} compounds (shuffled, seed 0)\n\n')
    f.write('| # | Paper | Title | Year | OA | N | NP IDs |\n')
    f.write('|--:|---|---|--:|:-:|--:|---|\n')
    for p in papers:
        title = (p.get('title') or p.get('ref_text') or '').replace('|', '/')
        title = title[:70] + ('...' if len(title) > 70 else '')
        f.write(f"| {p['order']} | [{p['ref']}]({p['url']}) | {title} | "
                f"{p.get('year', '')} | {'Y' if p.get('oa_pdf_url') else 'N'} | "
                f"{p['n_compounds']} | {' '.join(p['npids'])} |\n")

print(f'wrote new_candidates.csv and new_candidates.md ({len(papers)} papers)')
