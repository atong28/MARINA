#!/usr/bin/env python3
"""Try to fetch each queued paper's open-access PDF and check it for NMR tables.

Writes access/NMR status back into npmrd_links.json so the queue table can tell
you which papers still need to be opened by hand.

Status values:
  downloaded      PDF is now in papers/ and contains what look like shift tables
  downloaded-nonmr  PDF fetched but no assignment tables found
  paywalled       no open-access copy; needs a manual download
  failed          an OA link existed but the fetch did not yield a usable PDF
"""
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import requests

BASE = '/home/user/atong/Benchmark'
PAPERS = os.path.join(BASE, 'papers')
LINKS = os.path.join(BASE, 'npmrd_links.json')
HEADERS = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0',
           'Accept': 'application/pdf,*/*'}

# Papers already handled by hand or currently being extracted - do not touch.
SKIP = {'NP0332392', 'NP0332400', 'NP0332842', 'NP0332851',
        'NP0333043', 'NP0333621', 'NP0333860'}

NMR_RE = re.compile(r'\b(1H|13C|¹H|¹³C)[\s-]*NMR\b', re.I)
SHIFT_RE = re.compile(r'\b\d{1,3}\.\d{1,2}\b')
lock = threading.Lock()


def pdf_has_nmr_tables(path):
    """Heuristic: an assignment table means many shift-like numbers near NMR text."""
    try:
        txt = subprocess.run(['pdftotext', '-layout', path, '-'],
                             capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return None, 0, 0
    mentions = len(NMR_RE.findall(txt))
    shifts = len(SHIFT_RE.findall(txt))
    return (mentions >= 2 and shifts >= 80), mentions, shifts


def download(url, out_path):
    try:
        r = requests.get(url, headers=HEADERS, timeout=90, allow_redirects=True)
        if r.status_code == 200 and len(r.content) > 10000 and b'%PDF' in r.content[:1024]:
            with open(out_path, 'wb') as f:
                f.write(r.content)
            return True
    except Exception:
        pass
    return False


def main():
    papers = json.load(open(LINKS))
    todo = [p for p in papers if p['npids'][0] not in SKIP and 'access' not in p]
    print(f'{len(papers)} papers, {len(todo)} to try')
    done = [0]

    def work(p):
        primary = p['npids'][0]
        path = os.path.join(PAPERS, f'{primary}.pdf')
        if os.path.exists(path):
            ok, m, s = pdf_has_nmr_tables(path)
            p['access'] = 'downloaded' if ok else 'downloaded-nonmr'
        elif p.get('oa_pdf_url'):
            if download(p['oa_pdf_url'], path):
                ok, m, s = pdf_has_nmr_tables(path)
                p['access'] = 'downloaded' if ok else 'downloaded-nonmr'
                p['nmr_mentions'], p['shift_tokens'] = m, s
            else:
                p['access'] = 'failed'
        else:
            p['access'] = 'paywalled'
        with lock:
            done[0] += 1
            if done[0] % 25 == 0:
                json.dump(papers, open(LINKS, 'w'), indent=1)
                print(f'  {done[0]}/{len(todo)}', flush=True)

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(work, todo))
    json.dump(papers, open(LINKS, 'w'), indent=1)

    c = {}
    for p in papers:
        c[p.get('access', 'skipped')] = c.get(p.get('access', 'skipped'), 0) + 1
    print('\n' + '\n'.join(f'  {k:18} {v}' for k, v in sorted(c.items())))


if __name__ == '__main__':
    main()
