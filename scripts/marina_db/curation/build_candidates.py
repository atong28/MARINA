#!/usr/bin/env python3
"""Group un-extracted NP-MRD compounds by source paper, shuffle, and write the work queue.

Reads the NP-Card scrape (scrape_npmrd.py output), drops anything already in
Benchmark/filtered/, groups the rest by source reference (many adjacent NP IDs
share one paper), shuffles the papers, and writes new_candidates.json.

Usage: build_candidates.py <scrape_json> [<scrape_json> ...]
"""
import json
import os
import random
import sys

BASE = '/home/user/atong/Benchmark'
FILTERED = os.path.join(BASE, 'filtered')
OUT = os.path.join(BASE, 'new_candidates.json')
SEED = 0


def main():
    scrapes = sys.argv[1:] or ['/tmp/window_scrape.json']
    cards = {}
    for p in scrapes:
        cards.update(json.load(open(p)))

    # "Done" means extracted, not merely set up: setup_batch.py creates the
    # filtered/ dir when the PDF lands, well before any CSV exists, and those
    # compounds must stay in the queue.
    done = {d for d in os.listdir(FILTERED)
            if any(os.path.exists(os.path.join(FILTERED, d, f))
                   for f in ('1H.csv', '13C.csv', 'HSQC.csv'))}

    # Group by source reference: DOI when present, else PMID.
    groups = {}
    no_ref = []
    for acc in sorted(cards):
        card = cards[acc]
        if acc in done or 'error' in card:
            continue
        dois = card.get('dois') or []
        pmids = card.get('pmids') or []
        if dois:
            key = ('doi', dois[0].lower())
        elif pmids:
            key = ('pmid', pmids[0])
        else:
            no_ref.append(acc)
            continue
        g = groups.setdefault(key, {'npids': [], 'names': [], 'spectra': set(),
                                    'ref_text': ''})
        g['npids'].append(acc)
        g['names'].append(card.get('name') or '')
        g['spectra'].update(card.get('exp_spectra') or [])
        if not g['ref_text']:
            field = key[0]
            for r in card.get('refs') or []:
                if (r.get(field) or '').lower() == key[1]:
                    g['ref_text'] = r.get('text') or ''
                    break

    papers = []
    for (kind, val), g in groups.items():
        papers.append({
            'ref_type': kind,
            'ref': val,
            'url': (f'https://doi.org/{val}' if kind == 'doi'
                    else f'https://pubmed.ncbi.nlm.nih.gov/{val}/'),
            'ref_text': g['ref_text'],
            'npids': g['npids'],
            'names': g['names'],
            'n_compounds': len(g['npids']),
            'spectra': sorted(g['spectra']),
        })

    # Shuffle papers, not compounds, so a paper is downloaded and mined once.
    random.Random(SEED).shuffle(papers)

    # Widening the scrape later must not renumber papers already being worked
    # through: keep the established order and append anything new after it.
    prior = {}
    if os.path.exists(OUT):
        prior = {p['ref']: p['order'] for p in json.load(open(OUT))['papers']}
    papers.sort(key=lambda p: (prior.get(p['ref'], len(prior) + 1),))
    for i, p in enumerate(papers, 1):
        p['order'] = i
    if prior:
        kept = sum(1 for p in papers if p['ref'] in prior)
        print(f'preserved queue order for {kept} papers already listed')

    json.dump({'papers': papers, 'no_ref': no_ref}, open(OUT, 'w'), indent=1)

    n_comp = sum(p['n_compounds'] for p in papers)
    print(f'cards scanned:      {len(cards)}')
    print(f'already extracted:  {len(done)}')
    print(f'candidates w/ ref:  {n_comp} compounds -> {len(papers)} unique papers')
    print(f'candidates no ref:  {len(no_ref)} compounds (need PubMed name search)')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
