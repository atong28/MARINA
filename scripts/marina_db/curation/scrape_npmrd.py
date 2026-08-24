#!/usr/bin/env python3
"""Scrape np-mrd.org NP-Cards for source DOIs and experimental spectrum types.

Usage:
  scrape_npmrd.py <accessions_file> <out_json> [--workers N]

<accessions_file> is one NP accession per line. Results are written incrementally
so the scrape can be resumed: existing entries in <out_json> are skipped.
"""
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

URL = 'https://np-mrd.org/natural_products/{}'
HEADERS = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) benchmark-curation/1.0'}
# NP-MRD mints its own DOIs under this prefix; those are not literature refs.
NPMRD_DOI_PREFIX = '10.57994/'
# Parens are legal in DOIs (e.g. 10.1016/S1875-5364(22)60185-7), so match greedily
# and strip only trailing punctuation that can't belong to the identifier.
DOI_RE = re.compile(r'(?:doi\.org/|DOI:\s*)(10\.\d{4,9}/[^\s"\'<>]+)', re.I)
PMID_RE = re.compile(r'(?:PMID|PubMed):?\s*(\d{6,9})', re.I)
# Database provenance entries, not literature describing the compound.
DB_REF_RE = re.compile(r'^(LOTUS database|COCONUT|NPAtlas|KNApSAcK|Wikipedia)\b|'
                       r'LOTUS initiative for open knowledge management', re.I)
SPEC_TYPES = ['1H', '13C', 'HSQC', 'HMBC', 'COSY', 'NOESY', 'TOCSY', 'DEPT', 'JRES']

lock = threading.Lock()


def clean_doi(d):
    """Trim HTML/prose punctuation a greedy DOI match picks up at the end."""
    d = re.split(r'&(?:nbsp|amp|quot|lt|gt)', d)[0]
    while d:
        if d[-1] in '.,;:':
            d = d[:-1]
        elif d[-1] == ')' and d.count(')') > d.count('('):
            d = d[:-1]
        elif d[-1] == ']' and d.count(']') > d.count('['):
            d = d[:-1]
        else:
            break
    return d


def parse_refs(html):
    """Ordered literature references from the References section only.

    Scanning the whole page picks up DOIs that aren't references at all - most
    notably the LOTUS-initiative paper, which sits in a tooltip in the Links
    section and would otherwise look like the primary reference on ~8% of cards.
    """
    i = html.find('id="references"')
    if i == -1:
        return []
    m = re.search(r'<ol[^>]*cite-this-references[^>]*>(.*?)</ol>', html[i:], re.S)
    if not m:
        return []
    refs = []
    for item in re.findall(r'<li[^>]*>(.*?)</li>', m.group(1), re.S):
        text = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', item)).strip()
        text = re.sub(r'\[\s*(PubMed|PMID)[^\]]*\]', '', text, flags=re.I).strip()
        if not text or DB_REF_RE.search(text):
            continue
        doi = next((clean_doi(d) for d in DOI_RE.findall(item)
                    if not clean_doi(d).startswith(NPMRD_DOI_PREFIX)), None)
        pmid = PMID_RE.search(item)
        refs.append({'text': text, 'doi': doi, 'pmid': pmid.group(1) if pmid else None})
    return refs


def parse_card(html):
    """Pull name, SMILES, source DOIs/PMIDs and experimental spectrum types off an NP-Card."""
    name = None
    m = re.search(r'Showing NP-Card for (.+?) \(NP\d+\)', re.sub(r'<[^>]+>', ' ', html))
    if m:
        name = m.group(1).strip()

    smiles = None
    i = html.find('id="structure-text-smiles"')
    if i != -1:
        txt = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', html[i:i + 1500]))
        # Anchor on "Download" rather than the compound name - IUPAC names in the
        # modal title contain nested parentheses.
        m = re.search(r'(\S+)\s+Download', txt)
        if m:
            smiles = m.group(1)

    refs = parse_refs(html)
    dois = [r['doi'] for r in refs if r['doi']]
    pmids = [r['pmid'] for r in refs if r['pmid']]
    if not dois and not pmids:
        # Cards with an empty References section often still name the source in
        # the description, e.g. "... was first documented in 2022 (PMID: 35500784)".
        desc = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', html))
        m = re.search(r'first documented in \d{4}\s*\(PMID:\s*(\d{6,9})\)', desc, re.I)
        if m:
            pmids = [m.group(1)]

    # The "Experimental Spectra" table lists deposited spectra; predicted spectra
    # live in a separate section further down the card.
    start = html.find('id="experimental_spectra"')
    end = html.find('id="chemical_shift_spectra"')
    section = html[start:end] if start != -1 and end > start else ''
    section_txt = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', section))
    exp = []
    if 'experimental' in section_txt.lower():
        for row in re.split(r'View Spectrum', section_txt):
            if 'experimental' not in row.lower():
                continue
            r = row.replace(' ', '')
            for t in SPEC_TYPES:
                if t in r and t not in exp:
                    exp.append(t)
    return {'name': name, 'smiles': smiles,
            'dois': dois, 'pmids': pmids, 'refs': refs, 'exp_spectra': exp}


def fetch(acc, session):
    for attempt in range(3):
        try:
            resp = session.get(URL.format(acc), headers=HEADERS, timeout=30)
            if resp.status_code == 404:
                return {'error': '404'}
            if resp.status_code == 200:
                return parse_card(resp.text)
            time.sleep(2 * (attempt + 1))
        except Exception as e:
            if attempt == 2:
                return {'error': str(e)[:100]}
            time.sleep(2 * (attempt + 1))
    return {'error': 'failed'}


def main():
    acc_file, out_json = sys.argv[1], sys.argv[2]
    workers = 4
    if '--workers' in sys.argv:
        workers = int(sys.argv[sys.argv.index('--workers') + 1])

    accs = [l.strip() for l in open(acc_file) if l.strip()]
    results = {}
    if os.path.exists(out_json):
        results = json.load(open(out_json))
    todo = [a for a in accs if a not in results]
    print(f'{len(accs)} accessions, {len(results)} cached, {len(todo)} to fetch')

    session = requests.Session()
    done = [0]

    def work(acc):
        r = fetch(acc, session)
        with lock:
            results[acc] = r
            done[0] += 1
            if done[0] % 100 == 0:
                json.dump(results, open(out_json, 'w'))
                print(f'  {done[0]}/{len(todo)}', flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))

    json.dump(results, open(out_json, 'w'))
    ok = sum(1 for v in results.values() if 'error' not in v)
    with_doi = sum(1 for v in results.values() if v.get('dois'))
    with_exp = sum(1 for v in results.values() if v.get('exp_spectra'))
    print(f'done: {len(results)} total, {ok} ok, {with_doi} with DOI, {with_exp} with experimental spectra')


if __name__ == '__main__':
    main()
