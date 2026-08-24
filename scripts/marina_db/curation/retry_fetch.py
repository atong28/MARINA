#!/usr/bin/env python3
"""Second pass: retry recoverable OA hosts and re-classify NMR content.

Two fixes over fetch_and_check.py:
  * pdftotext splits superscripts, so "13C NMR" often comes out as "13 C NMR".
    The first-pass regex required them adjacent and under-counted badly.
  * PMC and MDPI need the article page visited first (PMC exposes the real PDF
    via citation_pdf_url; MDPI rejects requests without a Referer).
"""
import json
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

import requests

BASE = '/home/user/atong/Benchmark'
PAPERS = os.path.join(BASE, 'papers')
LINKS = os.path.join(BASE, 'npmrd_links.json')
UA = ('Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0')
HEADERS = {'User-Agent': UA, 'Accept': 'application/pdf,text/html,*/*'}

# Allow whitespace where pdftotext breaks a superscript off its isotope number.
NMR_RE = re.compile(r'(?:\b1\s*H|\b13\s*C|¹\s*H|¹³\s*C)\s*[-–]?\s*NMR', re.I)
SHIFT_RE = re.compile(r'\b\d{1,3}\.\d{1,2}\b')
lock = threading.Lock()


def classify(path):
    try:
        txt = subprocess.run(['pdftotext', '-layout', path, '-'],
                             capture_output=True, text=True, timeout=180).stdout
    except Exception:
        return 'downloaded-nonmr', 0, 0
    m = len(NMR_RE.findall(txt))
    s = len(SHIFT_RE.findall(txt))
    return ('downloaded' if (m >= 2 and s >= 80) else 'downloaded-nonmr'), m, s


def fetch_pmc(url, session):
    m = re.search(r'PMC(\d+)', url)
    if not m:
        return None
    page = session.get(f'https://pmc.ncbi.nlm.nih.gov/articles/PMC{m.group(1)}/',
                       headers=HEADERS, timeout=45)
    if page.status_code != 200:
        return None
    pdf = re.search(r'citation_pdf_url"?\s+content="([^"]+)"', page.text)
    return pdf.group(1) if pdf else None


def fetch(p, session):
    """Return True if a usable PDF landed on disk."""
    url = p.get('oa_pdf_url')
    if not url:
        return False
    out = os.path.join(PAPERS, f"{p['npids'][0]}.pdf")
    tries = [url]
    ref = None
    if 'pmc.ncbi.nlm.nih.gov' in url or '/pmc/' in url:
        real = fetch_pmc(url, session)
        if real:
            tries.insert(0, real)
    if 'mdpi.com' in url:
        ref = url.split('/pdf')[0]
        try:                               # MDPI sets cookies on the article page
            session.get(ref, headers=HEADERS, timeout=45)
        except Exception:
            pass
    for u in tries:
        h = dict(HEADERS)
        if ref:
            h['Referer'] = ref
        try:
            r = session.get(u, headers=h, timeout=90, allow_redirects=True)
            if r.status_code == 200 and len(r.content) > 10000 and b'%PDF' in r.content[:1024]:
                with open(out, 'wb') as f:
                    f.write(r.content)
                return True
        except Exception:
            pass
    return False


def main():
    papers = json.load(open(LINKS))
    # Re-classify everything already on disk, and retry the failures.
    todo = [p for p in papers
            if p.get('access') in ('failed', 'downloaded', 'downloaded-nonmr')]
    print(f'{len(todo)} to retry/re-classify')
    done = [0]

    def work(p):
        out = os.path.join(PAPERS, f"{p['npids'][0]}.pdf")
        if not os.path.exists(out) and p.get('access') == 'failed':
            session = requests.Session()
            if not fetch(p, session):
                p['access'] = 'paywalled-or-blocked'
                with lock:
                    done[0] += 1
                return
        if os.path.exists(out):
            p['access'], p['nmr_mentions'], p['shift_tokens'] = classify(out)
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
    print('\n' + '\n'.join(f'  {k:22} {v}' for k, v in sorted(c.items())))


if __name__ == '__main__':
    main()
