#!/usr/bin/env python3
"""Prepare the next papers in the shuffled queue for extraction.

For each paper in new_candidates.json order:
  - write papers/<NPID>.txt (link/metadata file) for every NP ID on that paper
  - download the open-access PDF to papers/<primary NPID>.pdf when Unpaywall has one
  - once a PDF exists, create filtered/<NPID>/ with paper.pdf symlinks + info.txt

Papers with no OA PDF are listed at the end as "needs manual download" — drop the
PDF at the printed path and re-run to wire up the filtered/ dirs.

Usage: setup_batch.py [--from N] [--to N] [--no-download]
"""
import json
import os
import shutil
import sys

import requests

BASE = '/home/user/atong/Benchmark'
PAPERS = os.path.join(BASE, 'papers')
FILTERED = os.path.join(BASE, 'filtered')
CAND = os.path.join(BASE, 'new_candidates.json')
HEADERS = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0',
           'Accept': 'application/pdf,*/*'}


def arg(flag, default):
    return int(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def write_txt(npid, name, paper):
    lines = [f'Compound: {name}\n',
             f'NP-MRD: https://np-mrd.org/natural_products/{npid}\n']
    if paper.get('title'):
        lines.append(f'Title: {paper["title"]}\n')
    if paper.get('journal'):
        lines.append(f'Journal: {paper["journal"]}\n')
    if paper.get('year'):
        lines.append(f'Year: {paper["year"]}\n')
    if paper['ref_type'] == 'doi':
        lines.append(f'DOI: {paper["ref"]}\n')
        lines.append(f'URL: https://doi.org/{paper["ref"]}\n')
    else:
        lines.append(f'PubMed: {paper["url"]}\n')
    if paper.get('oa_pdf_url'):
        lines.append(f'OA_PDF_URL: {paper["oa_pdf_url"]}\n')
    lines.append(f'Compounds on this paper: {", ".join(paper["npids"])}\n')
    with open(os.path.join(PAPERS, f'{npid}.txt'), 'w') as f:
        f.writelines(lines)


def download(url, out_path):
    try:
        r = requests.get(url, headers=HEADERS, timeout=60, allow_redirects=True)
        if r.status_code == 200 and len(r.content) > 10000 and b'%PDF' in r.content[:10]:
            with open(out_path, 'wb') as f:
                f.write(r.content)
            return True
    except Exception:
        pass
    return False


def supp_files(primary):
    """Supplementary data may be .pdf, .docx or .doc depending on the publisher."""
    return [f for f in os.listdir(PAPERS)
            if f.startswith(f'{primary}-supp.') and not f.endswith('.txt')]


def link_group(paper, primary):
    """Create filtered/<NPID>/ with symlinks to the shared PDF + info.txt."""
    made = []
    links = [(f'{primary}.pdf', 'paper.pdf')]
    for s in supp_files(primary):
        links.append((s, 'paper-supp' + s[len(primary) + 5:]))
    for npid in paper['npids']:
        d = os.path.join(FILTERED, npid)
        os.makedirs(d, exist_ok=True)
        for srcname, linkname in links:
            src = os.path.join(PAPERS, srcname)
            dst = os.path.join(d, linkname)
            if os.path.exists(src) and not os.path.lexists(dst):
                os.symlink(src, dst)
        shutil.copyfile(os.path.join(PAPERS, f'{npid}.txt'),
                        os.path.join(d, 'info.txt'))
        made.append(npid)
    return made


def main():
    data = json.load(open(CAND))
    papers = data['papers']
    if '--npids' in sys.argv:
        want = set(sys.argv[sys.argv.index('--npids') + 1].split(','))
        batch = [p for p in papers if p['npids'][0] in want]
    else:
        lo, hi = arg('--from', 1), arg('--to', len(papers))
        batch = [p for p in papers if lo <= p['order'] <= hi]
    do_download = '--no-download' not in sys.argv

    ready, needs_pdf = [], []
    for p in batch:
        primary = p['npids'][0]
        for npid, name in zip(p['npids'], p['names']):
            write_txt(npid, name, p)

        pdf_path = os.path.join(PAPERS, f'{primary}.pdf')
        if not os.path.exists(pdf_path) and do_download and p.get('oa_pdf_url'):
            if download(p['oa_pdf_url'], pdf_path):
                print(f'  [{p["order"]:>4}] downloaded {primary}.pdf  ({p["ref"]})')

        if os.path.exists(pdf_path):
            link_group(p, primary)
            ready.append(p)
        else:
            needs_pdf.append(p)

    print(f'\nbatch: {len(batch)} papers, '
          f'{sum(len(p["npids"]) for p in batch)} compounds')
    print(f'  ready to extract: {len(ready)} papers '
          f'({sum(len(p["npids"]) for p in ready)} compounds)')
    print(f'  needs manual PDF: {len(needs_pdf)} papers '
          f'({sum(len(p["npids"]) for p in needs_pdf)} compounds)')
    if needs_pdf:
        print('\nDrop PDFs here (name them after the primary NP ID):')
        for p in needs_pdf:
            print(f'  [{p["order"]:>4}] papers/{p["npids"][0]}.pdf   {p["url"]}')
            print(f'         {(p.get("title") or "?")[:95]}')


if __name__ == '__main__':
    main()
