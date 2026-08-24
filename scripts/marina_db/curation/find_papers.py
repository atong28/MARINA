#!/usr/bin/env python3.10
"""
Search for original papers for benchmark compounds, download open-access PDFs,
and create .txt files with links for paywalled papers.
"""

import json
import time
import os
import re
import requests
from Bio import Entrez

Entrez.email = 'atong28.usa@gmail.com'
PAPERS_DIR = '/home/user/atong/Benchmark/papers'
os.makedirs(PAPERS_DIR, exist_ok=True)

# ---- Load benchmark and NP-MRD metadata ----
with open('/home/user/atong/Benchmark/benchmark_npids.json') as f:
    npid_index = json.load(f)
# bm-like mapping: index -> npid
bm_npids = list(set(npid_index.values()))

with open('/home/user/atong/Benchmark/npmrd_json/npmrd_natural_products_NP0300001_NP0350000.json') as f:
    data = json.load(f)
records = data['np_mrd']['natural_product']
rec_by_id = {r['accession']: r for r in records}

# Build mapping: npid -> compound name
npid_to_name = {}
for npid in bm_npids:
    if npid in rec_by_id:
        npid_to_name[npid] = rec_by_id[npid]['name']

# ---- Compound-to-search-term mapping ----
# Group related compounds to avoid redundant searches.
# Key: tuple of NP IDs that share a paper; Value: PubMed search term
SEARCH_GROUPS = {
    # Anthoteibinenes (deep-sea coral Anthothela)
    ('NP0332683','NP0332685','NP0332686','NP0332687','NP0332688','NP0332689',
     'NP0332690','NP0332691','NP0332692','NP0332693','NP0332694','NP0332696',
     'NP0332697','NP0332698','NP0332699'): 'Anthoteibinene',
    # Cipacinesarasins
    ('NP0331267','NP0331268','NP0331269','NP0331270','NP0331271','NP0331272','NP0331273'): 'Cipacinerasin',
    # Oxandrastins / Andrastin F
    ('NP0331323','NP0331324','NP0331325','NP0331326'): 'Oxandrastin',
    # Isoaustalide F
    ('NP0331327',): 'Isoaustalide',
    # Symphyocladin
    ('NP0331338',): 'Symphyocladin',
    # Cytochalasin H1, H2
    ('NP0331343','NP0331345'): 'Aspochalasin',
    # Pseudonochelin
    ('NP0331364',): 'Pseudonochelin',
    # Cryptoporic Acid T and E pentamethyl ester
    ('NP0331367','NP0331371'): 'Cryptoporic Acid',
    # Sterpuric acid derivatives
    ('NP0331376','NP0331377','NP0331378'): 'Sterpuric acid',
    # Brevipolide G
    ('NP0331381',): 'Brevipolide',
    # Tongalide
    ('NP0331384','NP0331385'): 'Tongalide',
    # Glenthmycin H (and Glenthol A/B)
    ('NP0331390','NP0333203','NP0333204'): 'Glenthmycin',
    # 3-angeloyloxy compound
    ('NP0332281',): '3-angeloyloxy-5-isobutanoyloxy-7-hydroxycarvotacetone',
    # Asterripeptides
    ('NP0332299','NP0332302','NP0332305'): 'Asterripeptide',
    # Asterresin/Terreuside/Giluterrin
    ('NP0332300','NP0332301','NP0332303','NP0332304'): 'Asterresin',
    # Tetrapetalone
    ('NP0332306','NP0332307','NP0332308','NP0332310'): 'Tetrapetalone',
    # Neolignans (all share a single paper series)
    ('NP0332315','NP0332316','NP0332317','NP0332318','NP0332320','NP0332321',
     'NP0332322','NP0332323','NP0332325','NP0332326'): 'hexahydro neolignan isolation',
    # Eugeniinalines
    ('NP0332329','NP0332331','NP0332332','NP0332333','NP0332335','NP0332336','NP0332337'): 'Eugeniinaline',
    # Phellinhart
    ('NP0332374','NP0332376'): 'Phellinhart',
    # Taxane derivative
    ('NP0332442',): '5,11-dihydroxy-3(12)-cyclotaxane',
    # Withanolides (epoxy)
    ('NP0332460','NP0332461'): 'epoxy withania withanolide isolation',
    # Jugiones
    ('NP0332499','NP0332501','NP0332504'): 'Jugione',
    # Cavomycins
    ('NP0332525','NP0332526','NP0332527'): 'Cavomycin',
    # Prisconnatanones
    ('NP0332532','NP0332538','NP0332539','NP0332540'): 'Prisconnatanone',
    # Bromo ether
    ('NP0332543',): '1,5-dibromo-2-(2,4-dibromophenoxy)-3-methoxybenzene',
    # Goniothalesdiol
    ('NP0332568',): '3-epi-goniothalesdiol',
    # Armillaribin desmethyl
    ('NP0332590',): "5'-O-desmethylarmillaribin",
    # Noducyclamides
    ('NP0332608','NP0332609'): 'Noducyclamide',
    # Levinoids
    ('NP0332793','NP0332795','NP0332796'): 'Levinoid',
    # Halichondamide
    ('NP0332808',): 'Halichondamide',
    # Pullenvalenes
    ('NP0332809','NP0332810'): 'Pullenvalene',
    # Armillarine linoleate
    ('NP0332873',): 'Armillarine linoleate',
    # Armillaridin / melleolide derivatives
    ('NP0332875','NP0332878'): 'Armillaridin',
    # HCDN / cannabidiol derivatives
    ('NP0332903','NP0332904','NP0332905','NP0332906'): 'hydroxylated cannabidiol HCDN isolation',
    # Talarolides
    ('NP0333100','NP0333101'): 'Talarolide',
    # Stevisalioside
    ('NP0333111',): 'Stevisalioside',
    # Sinulatone / Sinulalide
    ('NP0333133','NP0333135'): 'Sinulatone Sinulalide',
    # Sacrone
    ('NP0333136',): 'Sacrone',
    # Microthecaline
    ('NP0333042',): 'Microthecaline',
    # Wulfenioidin
    ('NP0333096',): 'Wulfenioidin',
    # Millefolactons
    ('NP0333147','NP0333148','NP0333149','NP0333150'): 'Millefolacton',
    # Raufiayunesins
    ('NP0333151','NP0333152'): 'Raufiayunesin',
    # Cannabifolins
    ('NP0333162','NP0333163'): 'Cannabifolin isolation natural product',
    # 1,2-di-(4-hydroxybenzoyl)-β-glucopyranose
    ('NP0333164',): '1,2-di-(4-hydroxybenzoyl) glucopyranose natural product',
    # Murragatin
    ('NP0333170',): 'Murragatin coumarin isolation',
    # Phyllolactones
    ('NP0333173','NP0333174','NP0333175','NP0333176'): 'Phyllolactone',
    # Corymbotin
    ('NP0333185',): 'Corymbotin',
    # Uvarialeptone
    ('NP0333242',): 'Uvarialeptone',
    # Oblarotenoids
    ('NP0333455','NP0333456'): 'Oblarotenoid',
    # Lipostrigaibols
    ('NP0333518','NP0333520'): 'Lipostrigaibol',
    # Strigaibols
    ('NP0333522','NP0333524','NP0333525','NP0333526','NP0333528'): 'Strigaibol',
    # Ecdysteroid derivatives
    ('NP0333562','NP0333563'): '24-hydroxymuristerone ecdysteroid',
    # Biphenanthrene
    ('NP0333565',): 'tetramethoxybiphenanthrene tetraol isolation',
    # Salamandamide
    ('NP0333622',): 'Salamandamide',
    # Tumonolide
    ('NP0333702',): 'Tumonolide',
    # Narcissidine / Zephyranine
    ('NP0333727','NP0333728','NP0333729'): '6-oxonarcissidine zephyranine isolation',
    # Octacyclin
    ('NP0333732',): 'Octacyclin',
    # Glycerol monostearate (from NP-MRD, likely LOTUS/metabolomics)
    ('NP0333770',): 'glycerol 1-monostearate natural product',
    # Lupanone derivative
    ('NP0333772',): '17alpha-29-epoxy-28-norlupan lupane triterpene',
    # Spiro compounds (withanolide-related dimer)
    ('NP0332472','NP0332473'): 'Withania withanolide dimer spiro isolation',
    # Phascolosomines
    ('NP0333010','NP0333012','NP0333013'): 'Phascolosomine guanidine isolation',
    # Retinestatin
    ('NP0332499',): 'Retinestatin polyene macrolide',
}

# ---- Helper functions ----

def pubmed_search(term, n=5):
    """Search PubMed and return up to n PMIDs."""
    try:
        handle = Entrez.esearch(db='pubmed', term=f'"{term}"[Title/Abstract]', retmax=n)
        record = Entrez.read(handle)
        handle.close()
        return record['IdList']
    except Exception as e:
        print(f"  PubMed search error for '{term}': {e}")
        return []

def get_paper_details(pmid):
    """Fetch title, journal, DOI for a PMID."""
    try:
        handle = Entrez.efetch(db='pubmed', id=pmid, rettype='xml', retmode='xml')
        records = Entrez.read(handle)
        handle.close()
        article = records['PubmedArticle'][0]
        citation = article['MedlineCitation']
        art = citation['Article']
        title = str(art.get('ArticleTitle', ''))
        journal = str(art.get('Journal', {}).get('Title', ''))
        doi = ''
        for loc in art.get('ELocationID', []):
            if loc.attributes.get('EIdType') == 'doi':
                doi = str(loc)
                break
        if not doi:
            for aid in article.get('PubmedData', {}).get('ArticleIdList', []):
                if aid.attributes.get('IdType') == 'doi':
                    doi = str(aid)
                    break
        year = ''
        try:
            year = str(citation.get('Article', {}).get('Journal', {}).get('JournalIssue', {}).get('PubDate', {}).get('Year', ''))
        except Exception:
            pass
        return {'pmid': pmid, 'title': title, 'journal': journal, 'doi': doi, 'year': year}
    except Exception as e:
        print(f"  Error fetching PMID {pmid}: {e}")
        return None

NP_JOURNAL_KEYWORDS = [
    'nat prod', 'natural product', 'org lett', 'organic letters',
    'j org chem', 'journal organic chemistry', 'phytochem', 'phytochemistry',
    'marine drug', 'tetrahedron', 'eur j org', 'chem commun',
    'bioorg', 'planta med', 'fitoterapia', 'molecules', 'acs omega',
    'journal chemical information', 'rsc adv', 'chin chem lett',
    'nat chem biol', 'angew chem', 'jacs', 'journal american chemical',
    'chem biol', 'isolation', 'alkaloid', 'terpene', 'polyketide',
]

def is_np_paper(details):
    """Heuristic: is this paper about natural product isolation?"""
    if not details:
        return False
    journal_lower = details['journal'].lower()
    title_lower = details['title'].lower()
    for kw in NP_JOURNAL_KEYWORDS:
        if kw in journal_lower or kw in title_lower:
            return True
    return False

def try_unpaywall(doi, npid, out_path):
    """Try to get open-access PDF via Unpaywall. Returns True if downloaded."""
    if not doi:
        return False
    url = f"https://api.unpaywall.org/v2/{doi}?email=atong28.usa@gmail.com"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return False
        data = resp.json()
        pdf_url = None
        best_oa = data.get('best_oa_location')
        if best_oa:
            pdf_url = best_oa.get('url_for_pdf') or best_oa.get('url')
        if not pdf_url:
            for loc in data.get('oa_locations', []):
                if loc.get('url_for_pdf'):
                    pdf_url = loc['url_for_pdf']
                    break
        if pdf_url:
            print(f"  Downloading OA PDF from: {pdf_url[:80]}")
            pdf_resp = requests.get(pdf_url, timeout=60, headers={'User-Agent': 'Mozilla/5.0'})
            if pdf_resp.status_code == 200 and b'%PDF' in pdf_resp.content[:10]:
                with open(out_path, 'wb') as f:
                    f.write(pdf_resp.content)
                print(f"  Saved: {os.path.basename(out_path)} ({len(pdf_resp.content)//1024} KB)")
                return True
        return False
    except Exception as e:
        print(f"  Unpaywall error for {doi}: {e}")
        return False

def write_link_file(npid, doi, title, journal, year, pmid=''):
    """Write a .txt file with paper metadata and link."""
    txt_path = os.path.join(PAPERS_DIR, f'{npid}.txt')
    lines = [f'Compound: {npid_to_name.get(npid, npid)}\n']
    if title:
        lines.append(f'Title: {title}\n')
    if journal:
        lines.append(f'Journal: {journal}\n')
    if year:
        lines.append(f'Year: {year}\n')
    if doi:
        lines.append(f'DOI: {doi}\n')
        lines.append(f'URL: https://doi.org/{doi}\n')
    if pmid:
        lines.append(f'PubMed: https://pubmed.ncbi.nlm.nih.gov/{pmid}/\n')
    if not doi and not pmid:
        lines.append('Note: No paper found in PubMed search\n')
    with open(txt_path, 'w') as f:
        f.writelines(lines)

# ---- Main processing ----

results_log = {}
processed_npids = set()

print(f"\nProcessing {len(SEARCH_GROUPS)} compound groups...")
print("=" * 70)

for npid_tuple, search_term in SEARCH_GROUPS.items():
    # Skip if all NP IDs in this group are already processed
    relevant_npids = [n for n in npid_tuple if n in npid_to_name]
    if not relevant_npids:
        continue

    group_name = search_term
    print(f"\nGroup: {group_name}")
    print(f"  NP IDs: {', '.join(relevant_npids)}")

    # Check if already done
    all_done = all(
        os.path.exists(os.path.join(PAPERS_DIR, f'{n}.pdf')) or
        os.path.exists(os.path.join(PAPERS_DIR, f'{n}.txt'))
        for n in relevant_npids
    )
    if all_done:
        print("  Already processed, skipping.")
        processed_npids.update(relevant_npids)
        continue

    # Search PubMed
    pmids = pubmed_search(search_term, n=10)
    time.sleep(0.4)

    best_paper = None
    if pmids:
        # Fetch details for each result, pick best NP paper
        for pmid in pmids[:5]:
            details = get_paper_details(pmid)
            time.sleep(0.3)
            if details and is_np_paper(details):
                best_paper = details
                break
        # Fallback: use first result
        if not best_paper and pmids:
            best_paper = get_paper_details(pmids[0])
            time.sleep(0.3)

    if best_paper:
        doi = best_paper['doi']
        print(f"  Paper: {best_paper['title'][:70]}")
        print(f"  DOI: {doi}  Journal: {best_paper['journal'][:40]}")

        # Try to download PDF via Unpaywall for each NP ID in group
        # (all IDs share the same paper, but we save per NP ID)
        pdf_downloaded = False
        primary_npid = relevant_npids[0]
        pdf_path = os.path.join(PAPERS_DIR, f'{primary_npid}.pdf')

        if not os.path.exists(pdf_path):
            pdf_downloaded = try_unpaywall(doi, primary_npid, pdf_path)
            time.sleep(1)

        # For all NP IDs in group: create PDF symlink or .txt
        for npid in relevant_npids:
            pdf_path_np = os.path.join(PAPERS_DIR, f'{npid}.pdf')
            txt_path_np = os.path.join(PAPERS_DIR, f'{npid}.txt')

            if npid == primary_npid:
                if not pdf_downloaded and not os.path.exists(pdf_path_np):
                    write_link_file(npid, doi, best_paper['title'],
                                    best_paper['journal'], best_paper['year'],
                                    best_paper['pmid'])
            else:
                # For secondary NP IDs in group, point to same paper
                if pdf_downloaded and not os.path.exists(pdf_path_np):
                    # Copy rather than symlink for portability
                    import shutil
                    shutil.copy2(os.path.join(PAPERS_DIR, f'{primary_npid}.pdf'), pdf_path_np)
                elif not pdf_downloaded and not os.path.exists(txt_path_np) and not os.path.exists(pdf_path_np):
                    write_link_file(npid, doi, best_paper['title'],
                                    best_paper['journal'], best_paper['year'],
                                    best_paper['pmid'])

        results_log[tuple(relevant_npids)] = {
            'search_term': search_term,
            'paper': best_paper,
            'pdf_downloaded': pdf_downloaded or os.path.exists(pdf_path),
        }
    else:
        print(f"  No paper found for: {search_term}")
        for npid in relevant_npids:
            txt_path = os.path.join(PAPERS_DIR, f'{npid}.txt')
            if not os.path.exists(txt_path) and not os.path.exists(os.path.join(PAPERS_DIR, f'{npid}.pdf')):
                write_link_file(npid, '', '', '', '', '')

    processed_npids.update(relevant_npids)

# Handle any remaining NP IDs not in groups
all_npids = set(bm_npids)
remaining = all_npids - processed_npids
if remaining:
    print(f"\n\nRemaining unprocessed NP IDs: {len(remaining)}")
    for npid in sorted(remaining):
        name = npid_to_name.get(npid, '')
        print(f"  {npid}: {name}")
        # Search individually
        if name:
            pmids = pubmed_search(name, n=5)
            time.sleep(0.4)
            paper = None
            for pmid in pmids[:3]:
                details = get_paper_details(pmid)
                time.sleep(0.3)
                if details:
                    paper = details
                    if is_np_paper(details):
                        break
            if paper:
                doi = paper['doi']
                print(f"    Found: {paper['title'][:60]}  DOI: {doi}")
                pdf_path = os.path.join(PAPERS_DIR, f'{npid}.pdf')
                if not os.path.exists(pdf_path):
                    downloaded = try_unpaywall(doi, npid, pdf_path)
                    if not downloaded:
                        write_link_file(npid, doi, paper['title'],
                                        paper['journal'], paper['year'], paper['pmid'])
            else:
                write_link_file(npid, '', '', '', '')

# ---- Summary ----
print("\n\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
pdf_files = [f for f in os.listdir(PAPERS_DIR) if f.endswith('.pdf')]
txt_files = [f for f in os.listdir(PAPERS_DIR) if f.endswith('.txt')]
print(f"PDFs downloaded: {len(pdf_files)}")
print(f"Link files (.txt): {len(txt_files)}")
print(f"Total NP IDs covered: {len(pdf_files) + len(txt_files)} / {len(all_npids)}")

# List uncovered
covered_ids = {f.split('.')[0] for f in os.listdir(PAPERS_DIR)}
uncovered = all_npids - covered_ids
if uncovered:
    print(f"\nUncovered NP IDs ({len(uncovered)}):")
    for n in sorted(uncovered):
        print(f"  {n}: {npid_to_name.get(n, '')}")
