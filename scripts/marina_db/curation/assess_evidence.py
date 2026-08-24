#!/usr/bin/env python3
"""Grade each queued paper on how likely it is to contain usable NMR assignment data,
and normalise the access status. Regenerates npmrd_links.md / .csv.

access:
  have-pdf        PDF is in papers/ (downloaded or uploaded)
  oa-open-in-browser  open access exists but the host blocks scripted fetches
                      (PMC interstitial, MDPI/ACS Cloudflare) - free in a browser
  needs-purchase  no open-access copy found by Unpaywall

nmr_evidence:
  confirmed   shift tables seen in the PDF, or verified by hand
  likely      NP-MRD holds deposited experimental 1H+13C for one of its compounds,
              and/or the abstract describes spectroscopic structure elucidation
  unlikely    no deposited 1H+13C and nothing in the title/abstract suggesting NMR
  none        established by hand that there is no usable NMR data
"""
import html
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import requests

BASE = '/home/user/atong/Benchmark'
LINKS = os.path.join(BASE, 'npmrd_links.json')
ABS_CACHE = os.path.join(BASE, 'abstract_cache.json')
EMAIL = 'atong28.usa@gmail.com'

NMR_WORDS = re.compile(
    r'\bNMR\b|nuclear magnetic resonance|HSQC|HMBC|\bCOSY\b|ROESY|NOESY|'
    r'spectroscopic (?:data|analysis|method)|1D and 2D|chemical shift', re.I)
ISOLATION = re.compile(
    r'\bnew\b|novel|undescribed|isolat|structure elucidat|characteri[sz]ation of|'
    r'previously unreported|constituents|metabolites from', re.I)

# Set by hand from the user's manual check of the first 10 queue entries.
MANUAL = {
    'NP0332392': ('have-pdf', 'confirmed'), 'NP0332400': ('have-pdf', 'confirmed'),
    'NP0332851': ('have-pdf', 'confirmed'), 'NP0333043': ('have-pdf', 'confirmed'),
    'NP0332842': ('have-pdf', 'unlikely'),  'NP0333621': ('have-pdf', 'unlikely'),
    'NP0333860': ('have-pdf', 'none'),
    'NP0333355': ('needs-purchase', 'none'), 'NP0331206': ('needs-purchase', 'none'),
    'NP0332354': ('needs-purchase', 'none'),
}
lock = threading.Lock()


def get_abstract(doi, cache, session):
    if doi in cache:
        return cache[doi]
    a = ''
    try:
        r = session.get(f'https://api.crossref.org/works/{doi}?mailto={EMAIL}',
                        headers={'User-Agent': f'benchmark/1.0 (mailto:{EMAIL})'},
                        timeout=25)
        if r.status_code == 200:
            raw = r.json().get('message', {}).get('abstract') or ''
            a = re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', raw))).strip()
    except Exception:
        pass
    with lock:
        cache[doi] = a
    return a


def main():
    papers = json.load(open(LINKS))
    cache = json.load(open(ABS_CACHE)) if os.path.exists(ABS_CACHE) else {}
    session = requests.Session()

    need = [p for p in papers if p['ref_type'] == 'doi' and p['ref'] not in cache]
    print(f'fetching {len(need)} abstracts')
    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(lambda p: get_abstract(p['ref'], cache, session), need))
    json.dump(cache, open(ABS_CACHE, 'w'))

    for p in papers:
        primary = p['npids'][0]
        abstract = cache.get(p['ref'], '')
        p['abstract_mentions_nmr'] = bool(NMR_WORDS.search(abstract)) if abstract else None

        if primary in MANUAL:
            p['access'], p['nmr_evidence'] = MANUAL[primary]
            continue

        a = p.get('access', '')
        if a == 'downloaded':
            p['access'], p['nmr_evidence'] = 'have-pdf', 'confirmed'
            continue
        p['access'] = {'downloaded-nonmr': 'have-pdf',
                       'paywalled-or-blocked': 'oa-open-in-browser',
                       'paywalled': 'needs-purchase'}.get(a, a)

        has_hc = {'1H', '13C'} <= set(p['spectra'])
        text = (p.get('title') or '') + ' ' + abstract
        if a == 'downloaded-nonmr':
            p['nmr_evidence'] = 'unlikely'          # we read it; no tables found
        elif has_hc and (NMR_WORDS.search(text) or ISOLATION.search(text)):
            p['nmr_evidence'] = 'likely'
        elif has_hc:
            p['nmr_evidence'] = 'likely'
        else:
            p['nmr_evidence'] = 'unlikely'

    json.dump(papers, open(LINKS, 'w'), indent=1)

    def tally(key):
        c = {}
        for p in papers:
            c[p.get(key)] = c.get(p.get(key), 0) + 1
        return dict(sorted(c.items(), key=lambda kv: -kv[1]))
    print('\naccess:      ', tally('access'))
    print('nmr_evidence:', tally('nmr_evidence'))


if __name__ == '__main__':
    main()
