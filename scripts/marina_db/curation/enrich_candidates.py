#!/usr/bin/env python3
"""Add title/journal/year and open-access status to new_candidates.json.

Titles come from CrossRef (DOIs) or PubMed (PMID-only refs); OA PDF links come
from Unpaywall. Results are cached in paper_meta_cache.json so re-runs are cheap.
"""
import html
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import requests

BASE = '/home/user/atong/Benchmark'
CAND = os.path.join(BASE, 'new_candidates.json')
CACHE = os.path.join(BASE, 'paper_meta_cache.json')
EMAIL = 'atong28.usa@gmail.com'
HEADERS = {'User-Agent': f'benchmark-curation/1.0 (mailto:{EMAIL})'}

lock = threading.Lock()


def clean_text(s):
    """Publisher metadata carries markup and hard line breaks; both wreck the table."""
    if not s:
        return s
    s = re.sub(r'<[^>]+>', '', s)
    s = html.unescape(s)
    return re.sub(r'\s+', ' ', s).strip()


def crossref(doi, session):
    """CrossRef throttles bursts, so back off and retry rather than losing the title."""
    for attempt in range(4):
        try:
            r = session.get(f'https://api.crossref.org/works/{quote(doi, safe="/")}'
                            f'?mailto={EMAIL}', headers=HEADERS, timeout=25)
            if r.status_code == 404:
                return {}
            if r.status_code == 200:
                m = r.json()['message']
                return {
                    'title': (m.get('title') or [''])[0],
                    'journal': (m.get('container-title') or [''])[0],
                    'year': str((m.get('issued', {}).get('date-parts') or [['']])[0][0] or ''),
                }
        except Exception:
            pass
        time.sleep(2 ** attempt)
    return {}


def pubmed_esummary(pmid, session):
    try:
        r = session.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi',
                        params={'db': 'pubmed', 'id': pmid, 'retmode': 'json',
                                'email': EMAIL},
                        headers=HEADERS, timeout=20)
        d = r.json()['result'][str(pmid)]
        return {'title': d.get('title', ''), 'journal': d.get('fulljournalname', ''),
                'year': (d.get('pubdate', '') or '')[:4],
                'doi': next((x['value'] for x in d.get('articleids', [])
                             if x.get('idtype') == 'doi'), '')}
    except Exception:
        return {}


def unpaywall(doi, session):
    try:
        r = session.get(f'https://api.unpaywall.org/v2/{doi}?email={EMAIL}',
                        headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return {}
        d = r.json()
        pdf_url = None
        best = d.get('best_oa_location') or {}
        pdf_url = best.get('url_for_pdf')
        if not pdf_url:
            for loc in d.get('oa_locations') or []:
                if loc.get('url_for_pdf'):
                    pdf_url = loc['url_for_pdf']
                    break
        return {'is_oa': bool(d.get('is_oa')), 'oa_pdf_url': pdf_url}
    except Exception:
        return {}


def main():
    data = json.load(open(CAND))
    papers = data['papers']
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    todo = [p for p in papers if p['ref'] not in cache]
    print(f'{len(papers)} papers, {len(cache)} cached, {len(todo)} to look up')

    session = requests.Session()
    done = [0]

    def work(p):
        ref = p['ref']
        if p['ref_type'] == 'doi':
            meta = crossref(ref, session)
            meta.update(unpaywall(ref, session))
        else:
            meta = pubmed_esummary(ref, session)
            if meta.get('doi'):
                meta.update(unpaywall(meta['doi'], session))
        with lock:
            cache[ref] = meta
            done[0] += 1
            if done[0] % 100 == 0:
                json.dump(cache, open(CACHE, 'w'))
                print(f'  {done[0]}/{len(todo)}', flush=True)

    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(work, todo))
    json.dump(cache, open(CACHE, 'w'))

    for p in papers:
        meta = {k: v for k, v in cache.get(p['ref'], {}).items() if k != 'doi'}
        for k in ('title', 'journal'):
            if k in meta:
                meta[k] = clean_text(meta[k])
        p.update(meta)
        p['ref_text'] = clean_text(p.get('ref_text'))
    json.dump(data, open(CAND, 'w'), indent=1)

    n_title = sum(1 for p in papers if p.get('title'))
    n_oa = sum(1 for p in papers if p.get('oa_pdf_url'))
    print(f'titles resolved: {n_title}/{len(papers)}')
    print(f'open-access PDF available: {n_oa}/{len(papers)} '
          f'(paywalled / manual: {len(papers) - n_oa})')


if __name__ == '__main__':
    main()
