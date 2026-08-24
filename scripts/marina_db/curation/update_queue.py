#!/usr/bin/env python3
"""Re-order the manual download queue against what is actually on disk.

Derives a status for every queued paper, then rewrites npmrd_links.txt / .md / .csv
and npmrd_links_priority.txt so the top of the list is only work that is still worth
opening. Idempotent - re-run it after any extraction batch.

status:
  extracted      at least one of the paper's NP IDs has a finished HSQC.csv
  staged         filtered/ dir and PDF exist, extraction not finished yet
  no-nmr-data    tried by hand, no usable assignment tables found
  pending        not yet attempted

extracted and staged papers are dropped from the download queue (the PDF is already
in hand). no-nmr-data papers are kept but sorted to the bottom - they are not deleted,
because "checked, nothing there" is worth recording.

A paper counts as no-nmr-data when either:
  - assess_evidence.py graded it nmr_evidence == 'none' (a hand check), or
  - it sits above the high-water mark (the deepest queue position already worked)
    but was never staged - i.e. it was passed over while working down the list.
"""
import csv
import json
import os

BASE = '/home/user/atong/Benchmark'
LINKS = os.path.join(BASE, 'npmrd_links.json')
FILTERED = os.path.join(BASE, 'filtered')
PAPERS = os.path.join(BASE, 'papers')

RANK = {'pending': 0, 'no-nmr-data': 1}


def _nonempty_hsqc(path):
    """A finished HSQC.csv must have at least one data row, not just the header.

    A header-only HSQC.csv (e.g. a 1H-only paper with no 13C/HSQC) contributes no
    correlations, so it does not count as an extraction.
    """
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return sum(1 for line in f if line.strip()) > 1


def disk_status(p):
    """extracted / no-nmr-data / staged / None, from what is actually in filtered/.

    An extraction_log.txt with no HSQC.csv beside it means the paper was read and
    found to have no usable assignment table - that is a finished negative result,
    not work still pending.
    """
    staged = attempted = False
    for npid in p['npids']:
        d = os.path.join(FILTERED, npid)
        if _nonempty_hsqc(os.path.join(d, 'HSQC.csv')):
            return 'extracted'
        if os.path.isdir(d):
            staged = True
            if os.path.exists(os.path.join(d, 'extraction_log.txt')):
                attempted = True
    if attempted:
        return 'no-nmr-data'
    return 'staged' if staged else None


def main():
    papers = json.load(open(LINKS))
    papers.sort(key=lambda p: p['order'])

    for p in papers:
        p['status'] = disk_status(p) or 'pending'

    # The high-water mark is the end of the CONTIGUOUS front of worked papers at
    # the top of the queue - not simply the deepest one worked. A paper pulled out
    # of turn from far down the queue (e.g. because its PDF arrived with another
    # paper's upload) must not drag the mark with it and strand everything above it.
    # The front ends after GAP consecutive untouched papers.
    GAP = 8
    highwater = run = 0
    for p in papers:
        if p['status'] == 'pending':
            run += 1
            if run > GAP:
                break
        else:
            run = 0
            highwater = p['order']

    for p in papers:
        if p['status'] != 'pending':
            continue
        if p['nmr_evidence'] == 'none' or p['order'] <= highwater:
            p['status'] = 'no-nmr-data'

    queue = [p for p in papers if p['status'] in RANK]
    queue.sort(key=lambda p: (RANK[p['status']], p['order']))
    for i, p in enumerate(queue, 1):
        p['n'] = i

    with open(os.path.join(BASE, 'npmrd_links.txt'), 'w') as f:
        for p in queue:
            f.write(f"https://np-mrd.org/natural_products/{p['npids'][0]}\n")

    prio = [p for p in queue if p['status'] == 'pending'
            and p['access'] == 'oa-open-in-browser'
            and p['nmr_evidence'] in ('likely', 'confirmed')]
    # ACS (10.1021) and MDPI (10.3390) publish per-atom NMR assignment tables most
    # consistently, so float them to the top of the priority list. sort() is stable,
    # so the seed-0 shuffle order is preserved within each tier.
    def pub_tier(p):
        doi = (p.get('ref') or '').lower()
        return 0 if doi.startswith(('10.1021', '10.3390')) else 1
    prio.sort(key=pub_tier)
    with open(os.path.join(BASE, 'npmrd_links_priority.txt'), 'w') as f:
        for p in prio:
            f.write(f"https://np-mrd.org/natural_products/{p['npids'][0]}\n")

    done = [p for p in papers if p['status'] not in RANK]
    with open(os.path.join(BASE, 'npmrd_links.md'), 'w') as f:
        f.write(f'# Papers to download - {len(queue)} sources, '
                f'{sum(p["n_compounds"] for p in queue)} compounds\n\n'
                f'Queue order: {sum(1 for p in queue if p["status"] == "pending")} pending '
                f'first (seed-0 shuffle), then '
                f'{sum(1 for p in queue if p["status"] == "no-nmr-data")} no-nmr-data at the '
                f'bottom. {len(done)} extracted/staged papers are listed separately below.\n\n')
        f.write('| # | NP-MRD (first of series) | N | Year | Access | NMR? | Status | Title |\n'
                '|--:|---|--:|--:|---|---|---|---|\n')
        for p in queue:
            acc = p['npids'][0]
            title = (p.get('title') or p.get('ref_text') or '').replace('|', '/')
            title = title[:70] + ('...' if len(title) > 70 else '')
            f.write(f"| {p['n']} | [{acc}](https://np-mrd.org/natural_products/{acc}) | "
                    f"{p['n_compounds']} | {p.get('year', '')} | {p['access']} | "
                    f"{p['nmr_evidence']} | {p['status']} | {title} |\n")
        f.write(f'\n## Off the queue - {len(done)} papers already in hand\n\n')
        f.write('| NP-MRD | N | Status | Title |\n|---|--:|---|---|\n')
        for p in sorted(done, key=lambda p: p['order']):
            acc = p['npids'][0]
            title = (p.get('title') or p.get('ref_text') or '').replace('|', '/')[:70]
            f.write(f"| [{acc}](https://np-mrd.org/natural_products/{acc}) | "
                    f"{p['n_compounds']} | {p['status']} | {title} |\n")

    with open(os.path.join(BASE, 'npmrd_links.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n', 'npmrd_url', 'doi', 'access', 'nmr_evidence', 'status',
                    'n_compounds', 'year', 'title', 'npids', 'pdf_target'])
        for p in queue + sorted(done, key=lambda p: p['order']):
            acc = p['npids'][0]
            w.writerow([p.get('n', ''), f'https://np-mrd.org/natural_products/{acc}',
                        p['ref'], p['access'], p['nmr_evidence'], p['status'],
                        p['n_compounds'], p.get('year', ''), p.get('title', ''),
                        ' '.join(p['npids']), f'papers/{acc}.pdf'])

    json.dump(papers, open(LINKS, 'w'), indent=1)

    c = {}
    for p in papers:
        c[p['status']] = c.get(p['status'], 0) + 1
    print(f'high-water mark: queue position {highwater}')
    for k in ('pending', 'no-nmr-data', 'staged', 'extracted'):
        n = c.get(k, 0)
        comp = sum(p['n_compounds'] for p in papers if p['status'] == k)
        print(f'  {k:<12} {n:>4} papers  {comp:>5} compounds')
    print(f'\ndownload queue: {len(queue)} papers '
          f'({sum(1 for p in queue if p["status"] == "pending")} pending, '
          f'{sum(1 for p in queue if p["status"] == "no-nmr-data")} parked at bottom)')
    print(f'priority (open-in-browser, NMR likely): {len(prio)}')


if __name__ == '__main__':
    main()
