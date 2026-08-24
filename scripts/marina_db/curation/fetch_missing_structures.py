#!/usr/bin/env python3
"""
Fetch SMILES for benchmark NPIDs absent from the local NP-MRD dump.

npmrd_json/ holds a 40,900-record scrape of the NP0300001-NP0350000 range, which is
partial -- 14 of the 66 newly extracted compounds are not in it (the accessions are
sparse: NP0333227 and NP0333229 are present, 226/228/230 are not). This pulls those
records from np-mrd.org directly and caches them, so build_benchmark.py has a SMILES
for every compound it assembles.

Writes: missing_structures.json  {npid: {smiles, name, mono_mass, formula}}
"""
import json
import os
import sys
import time

import requests
from xml.etree import ElementTree

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'missing_structures.json')
URL = 'https://np-mrd.org/natural_products/{}.xml'


def wanted_npids():
    """NPIDs with complete CSVs that no local source can supply a structure for."""
    dump = os.path.join(HERE, 'npmrd_json',
                        'npmrd_natural_products_NP0300001_NP0350000.json')
    known = {r['accession']
             for r in json.load(open(dump))['np_mrd']['natural_product']}
    known |= set(json.load(open(os.path.join(HERE, 'npmrd_index.json'))))

    filtered = os.path.join(HERE, 'filtered')
    out = []
    for npid in sorted(os.listdir(filtered)):
        d = os.path.join(filtered, npid)
        if not npid.startswith('NP') or not os.path.isdir(d):
            continue
        if not all(os.path.exists(os.path.join(d, f))
                   for f in ('13C.csv', '1H.csv', 'HSQC.csv')):
            continue
        if npid not in known:
            out.append(npid)
    return out


def parse(xml_text):
    """Pull the fields build_benchmark.py needs out of an NP-MRD record."""
    root = ElementTree.fromstring(xml_text)

    def text(tag):
        # NP-MRD serves these without a namespace on the inner elements.
        el = root.find(tag)
        return el.text.strip() if el is not None and el.text else None

    return {
        'smiles': text('smiles'),
        'name': text('name'),
        'mono_mass': text('monisotopic_molecular_weight'),
        'formula': text('chemical_formula'),
    }


def main():
    npids = wanted_npids()
    print(f'{len(npids)} NPIDs need fetching: {", ".join(npids)}')

    cache = json.load(open(OUT)) if os.path.exists(OUT) else {}
    session = requests.Session()
    session.headers['User-Agent'] = 'MARINA-benchmark-build/1.0'

    failed = []
    for npid in npids:
        if npid in cache and cache[npid].get('smiles'):
            print(f'  {npid}  cached')
            continue
        try:
            r = session.get(URL.format(npid), timeout=30)
            r.raise_for_status()
            rec = parse(r.text)
        except Exception as exc:                       # noqa: BLE001
            print(f'  {npid}  FAILED: {exc}')
            failed.append(npid)
            continue
        if not rec['smiles']:
            print(f'  {npid}  FAILED: record has no SMILES')
            failed.append(npid)
            continue
        cache[npid] = rec
        print(f'  {npid}  {rec["smiles"][:60]}  ({rec["name"]})')
        time.sleep(1.0)                                # be polite to NP-MRD

    with open(OUT, 'w') as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    print(f'\n{len(cache)} records -> {OUT}')
    if failed:
        print(f'{len(failed)} FAILED: {", ".join(failed)}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
