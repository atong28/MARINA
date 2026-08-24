#!/usr/bin/env python3.10
"""
Create/update .txt files for all benchmark NP IDs using NP-MRD scraped DOIs.
For each DOI, look up paper details via PubMed (and CrossRef fallback).
Try to download open-access PDFs via Unpaywall.
"""

import json
import os
import re
import requests
import time
from Bio import Entrez

Entrez.email = 'atong28.usa@gmail.com'
PAPERS_DIR = '/home/user/atong/Benchmark/papers'
os.makedirs(PAPERS_DIR, exist_ok=True)

# Load scraped NP-MRD refs
with open('/home/user/atong/Benchmark/npmrd_scraped_refs.json') as f:
    npmrd_refs = json.load(f)

# Load NP-MRD metadata (names, MWs)
with open('/home/user/atong/Benchmark/npmrd_json/npmrd_natural_products_NP0300001_NP0350000.json') as f:
    data = json.load(f)
records = data['np_mrd']['natural_product']
rec_by_id = {r['accession']: r for r in records}

# Load benchmark NP IDs
with open('/home/user/atong/Benchmark/benchmark_npids.json') as f:
    npid_list = list(set(json.load(f).values()))

npid_to_name = {n: rec_by_id[n]['name'] for n in npid_list if n in rec_by_id}

# Papers not in NP-MRD refs - manual overrides
MANUAL_DOIS = {
    # Anthoteibinenes A-E (A, C, D, E = NP0332683, NP0332685, NP0332686, NP0332687)
    'NP0332683': '10.1021/acs.orglett.4c02549',
    'NP0332685': '10.1021/acs.orglett.4c02549',
    'NP0332686': '10.1021/acs.orglett.4c02549',
    'NP0332687': '10.1021/acs.orglett.4c02549',
    # Anthoteibinenes F-Q
    'NP0332688': '10.3390/md23010044',
    'NP0332689': '10.3390/md23010044',
    'NP0332690': '10.3390/md23010044',
    'NP0332691': '10.3390/md23010044',
    'NP0332692': '10.3390/md23010044',
    'NP0332693': '10.3390/md23010044',
    'NP0332694': '10.3390/md23010044',
    'NP0332696': '10.3390/md23010044',
    'NP0332697': '10.3390/md23010044',
    'NP0332698': '10.3390/md23010044',
    'NP0332699': '10.3390/md23010044',
    # Cavomycin A-C (not in NP-MRD refs yet - best guess)
    # 'NP0332525': None,  # will be None
    # 5,11-dihydroxy-3(12)-cyclotaxane (not found)
    # 'NP0332442': None,
}

def get_paper_details_by_doi(doi):
    """Look up paper details using DOI via PubMed esearch, then CrossRef fallback."""
    # Try PubMed
    try:
        handle = Entrez.esearch(db='pubmed', term=f'{doi}[Location ID]', retmax=3)
        record = Entrez.read(handle)
        handle.close()
        if record['IdList']:
            pmid = record['IdList'][0]
            handle = Entrez.efetch(db='pubmed', id=pmid, rettype='xml', retmode='xml')
            records = Entrez.read(handle)
            handle.close()
            article = records['PubmedArticle'][0]
            citation = article['MedlineCitation']
            art = citation['Article']
            title = str(art.get('ArticleTitle', ''))
            journal = str(art.get('Journal', {}).get('Title', ''))
            year = ''
            try:
                year = str(art.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {}).get('Year', ''))
            except: pass
            # Check if open access via PubMed
            pmcid = None
            for aid in article.get('PubmedData', {}).get('ArticleIdList', []):
                if str(aid.attributes.get('IdType', '')) == 'pmc':
                    pmcid = str(aid)
            return {'title': title, 'journal': journal, 'year': year, 'doi': doi,
                    'pmid': pmid, 'pmcid': pmcid}
    except Exception as e:
        pass

    # Fallback: CrossRef
    try:
        resp = requests.get(
            f'https://api.crossref.org/works/{doi}?mailto=atong28.usa@gmail.com',
            timeout=15)
        if resp.status_code == 200:
            item = resp.json()['message']
            title = item.get('title', [''])[0]
            journal = item.get('container-title', [''])[0]
            year = str(item.get('published', {}).get('date-parts', [['']])[0][0])
            return {'title': title, 'journal': journal, 'year': year, 'doi': doi,
                    'pmid': None, 'pmcid': None}
    except Exception as e:
        pass

    return None

def try_unpaywall(doi, out_path):
    """Check Unpaywall for OA PDF. Returns (pdf_url, is_oa)."""
    try:
        resp = requests.get(
            f'https://api.unpaywall.org/v2/{doi}?email=atong28.usa@gmail.com',
            timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            is_oa = data.get('is_oa', False)
            best_oa = data.get('best_oa_location', {})
            pdf_url = None
            if best_oa:
                pdf_url = best_oa.get('url_for_pdf')
            if not pdf_url:
                for loc in data.get('oa_locations', []):
                    if loc.get('url_for_pdf'):
                        pdf_url = loc['url_for_pdf']
                        break
            return pdf_url, is_oa
    except Exception:
        pass
    return None, False

def try_download_pdf(doi, pdf_url, out_path):
    """Try to download PDF. Returns True if successful."""
    if not pdf_url:
        return False
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0',
            'Accept': 'application/pdf,*/*',
        }
        # Try PMC first if we have PMC URL pattern
        if 'pmc.ncbi.nlm.nih.gov' in pdf_url or 'ncbi.nlm.nih.gov/pmc' in pdf_url:
            # Get the actual PDF filename from PMC page
            pmc_id_match = re.search(r'PMC(\d+)', pdf_url)
            if pmc_id_match:
                pmc_page_url = f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmc_id_match.group(1)}/"
                page_resp = requests.get(pmc_page_url, headers=headers, timeout=15)
                if page_resp.status_code == 200:
                    pdf_match = re.search(r'citation_pdf_url.*?content="([^"]+)"', page_resp.text)
                    if pdf_match:
                        pdf_url = pdf_match.group(1)

        resp = requests.get(pdf_url, headers=headers, timeout=60, allow_redirects=True)
        if resp.status_code == 200 and len(resp.content) > 10000:
            # Check for PDF magic bytes
            content_start = resp.content[:10]
            if b'%PDF' in content_start or b'\x25\x50\x44\x46' in content_start:
                with open(out_path, 'wb') as f:
                    f.write(resp.content)
                return True
    except Exception:
        pass
    return False

def write_txt(npid, paper, oa_url=None, note=None):
    """Write .txt file with paper metadata."""
    name = npid_to_name.get(npid, npid)
    lines = [f'Compound: {name}\n',
             f'NP-MRD: https://np-mrd.org/natural_products/{npid}\n']
    if note:
        lines.append(f'Note: {note}\n')
    if paper:
        if paper.get('title'):
            lines.append(f'Title: {paper["title"]}\n')
        if paper.get('journal'):
            lines.append(f'Journal: {paper["journal"]}\n')
        if paper.get('year'):
            lines.append(f'Year: {paper["year"]}\n')
        if paper.get('doi'):
            lines.append(f'DOI: {paper["doi"]}\n')
            lines.append(f'URL: https://doi.org/{paper["doi"]}\n')
        if paper.get('pmid'):
            lines.append(f'PubMed: https://pubmed.ncbi.nlm.nih.gov/{paper["pmid"]}/\n')
        if paper.get('pmcid'):
            lines.append(f'PMC: https://pmc.ncbi.nlm.nih.gov/articles/{paper["pmcid"]}/\n')
        if oa_url:
            lines.append(f'OA_PDF_URL: {oa_url}\n')
    else:
        lines.append('Note: No source paper found; check NP-MRD page above\n')
    txt_path = os.path.join(PAPERS_DIR, f'{npid}.txt')
    with open(txt_path, 'w') as f:
        f.writelines(lines)

# ---- Process all 154 NP IDs ----
# Cache paper details by DOI to avoid redundant API calls
doi_cache = {}

print("Processing all NP IDs...")
pdf_count = 0
txt_count = 0

for npid in sorted(npid_list):
    # Determine DOI
    refs = npmrd_refs.get(npid, [])
    if refs:
        doi = refs[0]['doi']  # Use first (primary) reference
    elif npid in MANUAL_DOIS:
        doi = MANUAL_DOIS[npid]
    else:
        doi = None

    # Get paper details
    paper = None
    if doi:
        if doi in doi_cache:
            paper = doi_cache[doi]
        else:
            paper = get_paper_details_by_doi(doi)
            doi_cache[doi] = paper
            time.sleep(0.4)  # Rate limit

    # Check if already have a PDF
    pdf_path = os.path.join(PAPERS_DIR, f'{npid}.pdf')
    txt_path = os.path.join(PAPERS_DIR, f'{npid}.txt')

    if os.path.exists(pdf_path):
        print(f'{npid}: Already has PDF, skipping')
        continue

    # Try Unpaywall for OA PDF
    downloaded = False
    oa_url = None
    if doi:
        pdf_url, is_oa = try_unpaywall(doi, pdf_path)
        if pdf_url:
            oa_url = pdf_url
        if is_oa and pdf_url:
            downloaded = try_download_pdf(doi, pdf_url, pdf_path)
            time.sleep(1)

    if downloaded:
        print(f'{npid}: Downloaded PDF from {oa_url[:60]}')
        pdf_count += 1
        # Also write .txt as companion
    else:
        note = None
        if not doi:
            note = 'No DOI found in NP-MRD or manual search'
        write_txt(npid, paper, oa_url=oa_url if not downloaded else None, note=note)
        txt_count += 1
        status = f'DOI: {doi}' if doi else 'NO DOI'
        print(f'{npid}: .txt written ({status})')

print(f"\nDone! PDFs: {pdf_count}, TXT files: {txt_count}")
print(f"Total files: {pdf_count + txt_count}")
