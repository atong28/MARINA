#!/usr/bin/env python3
"""Combine results/full (exhaustively curated compounds: central keep probabilities) with
results/tier1 (wider per-molecule spread) into the single parameter file MARINA2.0 consumes.

  pixi run python3 MARINA/analysis/hmbc-cosy-calibration/finalize.py
"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
full = json.load(open(os.path.join(HERE, 'results/full/dropout_params.json')))
t1 = json.load(open(os.path.join(HERE, 'results/tier1/dropout_params.json')))
out = {
    'source': 'finalize.py (results/full for rates, results/tier1 for the per-molecule spread)',
    'model': 'HMBC 2J/3J carbon-bound pair: keep = clip(m * P_hmbc[ptype][ctype], 0, 1); '
             'HMBC 4J: not in the ceiling (observed rate < 1%); exchangeable-proton pairs: keep = keep_exch[solvent_class] '
             '(per-molecule coin, not per peak); COSY: keep = clip(m * cosy[class]); COSY 4J: not in the ceiling. '
             'm ~ LogNormal(mu, sigma) drawn once per molecule per epoch, clipped to [m_min, m_max].',
    'ptype': 'proton carrier: CH3 / CH2 / CH (sp3), arom (aromatic C-H), olef (non-aromatic sp2 C-H), exch (O-H, N-H, S-H)',
    'ctype': 'HMBC target carbon: protonated / quaternary (no H, no C=O) / carbonyl (C=O)',
    'P_hmbc': {pt: {ct: v['keep'] for ct, v in d.items()} for pt, d in full['P_hmbc'].items()},
    'P_hmbc_n': {pt: {ct: v['n'] for ct, v in d.items()} for pt, d in full['P_hmbc'].items()},
    'P_hmbc_pessimistic_tier1': {pt: {ct: v['keep'] for ct, v in d.items()} for pt, d in t1['P_hmbc'].items()},
    'keep_exch': {k: v['keep'] for k, v in full['keep_exch'].items()},
    'cosy': {k: v['keep'] for k, v in full['cosy'].items() if k != '4J'},
    'molecule_multiplier': {'mu': full['molecule_multiplier']['lognormal_mu'],
                            'sigma': t1['molecule_multiplier']['lognormal_sigma'],
                            'm_min': 0.2, 'm_max': 1.6,
                            'note': 'mu from the exhaustively curated set (centres m at ~0.85); sigma from tier 1, which includes '
                                    'partially mapped and dilute deposits and so spans the deposit-quality range the model will meet'},
    'ceiling_membership': {'HMBC': '2J + 3J only (drop 4J: 5 of 456 possible observed, 2 at high/medium confidence)',
                           'COSY': 'vicinal 3J + geminal for CH2 with two distinct predicted shifts (drop 4J: 0 of 103 at high/medium)'},
}
json.dump(out, open(os.path.join(HERE, 'results/marina2_dropout_params.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
