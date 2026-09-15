#!/usr/bin/env python3
"""Helper for hand-curating NP-FIDBench 2D peak lists (see CURATION_PROMPT.md).

  pixi run python3 spectra.py queue [--n 60]                  # stratified list of compounds to curate
  pixi run python3 spectra.py pick  NP0350544 HMBC [--snr 3]  # picker JSON (peaks + flags) to stdout
  pixi run python3 spectra.py plot  NP0350544 HMBC out.png [--f2 0.5 4.5] [--f1 10 80] [--snr 3]
  pixi run python3 spectra.py info  NP0350544                 # solvent, MHz, SMILES, experiments

Run from ~/Workspace (the master pixi env has nmrglue + rdkit + matplotlib).
"""
import argparse, glob, json, os, sys
ROOT = os.path.expanduser('~/Workspace/AgentBench')
sys.path.insert(0, os.path.join(ROOT, 'harness'))
COMP = os.path.join(ROOT, 'npmrd', 'compounds')

def cdir(npid): return os.path.join(COMP, npid)

def cmd_info(a):
    m = json.load(open(os.path.join(cdir(a.npid), 'benchmark_meta.json')))
    print(json.dumps({k: m.get(k) for k in ('npid', 'name', 'formula', 'smiles', 'nmr_solvent', 'nmr_freq_mhz', 'experiments', 'source_dois')}, indent=1))
    from fidproc import Compound
    print(json.dumps(Compound(cdir(a.npid)).list_spectra(), indent=1))

def cmd_pick(a):
    from fidproc import Compound, public
    r = public(Compound(cdir(a.npid)).process(a.experiment, snr=a.snr))
    print(json.dumps(r, indent=1))

def cmd_plot(a):
    from fidproc import Compound
    from plots import render_2d, render_1d
    r = Compound(cdir(a.npid)).process(a.experiment, snr=a.snr)
    png = render_2d(r, f2_range=a.f2, f1_range=a.f1) if r.get('f1_nucleus') else render_1d(r, ppm_range=a.f2)
    open(a.out, 'wb').write(png); print(a.out, len(r['peaks']), 'picked peaks in full spectrum')

def cmd_queue(a):
    rows = []
    for f in sorted(glob.glob(os.path.join(COMP, '*', 'benchmark_meta.json'))):
        m = json.load(open(f)); ex = m.get('experiments', {})
        if all(k in ex for k in ('HSQC', 'HMBC', 'COSY')):
            rows.append((m['npid'], m.get('nmr_solvent') or 'unknown', m.get('formula', ''), int(m.get('nmr_freq_mhz') or 0)))
    import random; random.seed(0); random.shuffle(rows)
    by = {}
    for r in rows: by.setdefault(r[1], []).append(r)
    order = sorted(by, key=lambda s: -len(by[s]))
    out, i = [], 0
    while len(out) < a.n and any(by.values()):
        s = order[i % len(order)]; i += 1
        if by[s]: out.append(by[s].pop())
    done = {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(os.path.dirname(__file__), 'curated', '*.json'))}
    print('npid\tsolvent\tformula\tMHz\tstatus')
    for r in out: print('\t'.join(map(str, r)) + '\t' + ('done' if r[0] in done else 'todo'))

ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
p = sp.add_parser('info'); p.add_argument('npid'); p.set_defaults(f=cmd_info)
p = sp.add_parser('pick'); p.add_argument('npid'); p.add_argument('experiment'); p.add_argument('--snr', type=float); p.set_defaults(f=cmd_pick)
p = sp.add_parser('plot'); p.add_argument('npid'); p.add_argument('experiment'); p.add_argument('out')
p.add_argument('--f2', type=float, nargs=2); p.add_argument('--f1', type=float, nargs=2); p.add_argument('--snr', type=float); p.set_defaults(f=cmd_plot)
p = sp.add_parser('queue'); p.add_argument('--n', type=int, default=60); p.set_defaults(f=cmd_queue)
a = ap.parse_args(); a.f(a)
