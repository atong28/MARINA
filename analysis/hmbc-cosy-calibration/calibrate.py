#!/usr/bin/env python3
"""MARINA2.0 HMBC/COSY dropout calibration: ceiling (topologically possible) vs curated (observed).

  cd ~/Workspace && pixi run python3 MARINA/analysis/hmbc-cosy-calibration/calibrate.py \
      [--mnova /tmp/mnova/curated_mnova.json] [--tier full|tier1|tier12]

Reads curated/<NPID>.json (never modified), builds the ceiling from each file's `smiles` parsed
as-is, matches curated peaks to ceiling pairs (assign indices first, shift windows for the rest),
and writes results/<tier>/{per_compound.md,rates.md,rates.json,dropout_params.json}.

Denominator rule: a (compound, modality) enters the not-observed class only when the compound is
in the well-assigned tier AND that modality has >=1 curated peak. All other ok compounds contribute
observed peaks only (ceiling-membership / recall checks), never "dropped".
"""
import argparse, collections, glob, json, math, os, re
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.abspath(__file__))
CUR = os.path.join(HERE, 'curated')
OUT = os.path.join(HERE, 'results')

TIER1 = ['NP0351403', 'NP0351238', 'NP0354171', 'NP0351071', 'NP0351438', 'NP0351241', 'NP0351815',
         'NP0351101', 'NP0352202', 'NP0352091', 'NP0352115', 'NP0351319']
TIER2 = ['NP0350908', 'NP0350924', 'NP0351848', 'NP0350731']
# curator remarks say the HMBC network was mapped exhaustively ("fully", "complete", "clean")
FULL = ['NP0351238', 'NP0351403', 'NP0351438', 'NP0354171', 'NP0352202', 'NP0351071', 'NP0351319']
PROTIC = {'CD3OD', 'D2O'}
TOL_H, TOL_C = 0.03, 0.3          # shift-window fallback (curated axes are internally consistent)
PRIOR = {'HMBC': {'2J': 0.8, '3J': 1.0, '4J': 0.4}, 'COSY': {'3J': 1.0, 'gem': 1.0, '4J': 0.4}}  # design-doc D6
PRIOR_N = 10.0                    # pseudo-count strength for the smoothed parameters


# ----------------------------------------------------------------------------- structure helpers
def parse_assign(x, exch):
    """-> (atom_idx or None, tag) ; tag in {'', 'a', 'b', 'NH', 'OH', ...}"""
    if x is None:
        return None, 'none'
    if isinstance(x, int):
        return x, ''
    s = str(x)
    m = re.fullmatch(r'(\d+)([ab])?', s)
    if m:
        return int(m.group(1)), (m.group(2) or '')
    for e in exch:                     # 'NH' / 'OH' -> exchangeable_protons_seen entry
        if e.get('type') == s and isinstance(e.get('assign'), int):
            return e['assign'], s
    return None, s


def proton_type(atom):
    sym = atom.GetSymbol()
    if sym != 'C':
        return 'exch'
    nh = atom.GetTotalNumHs()
    if atom.GetIsAromatic():
        return 'arom'
    if any(b.GetBondType() == Chem.BondType.DOUBLE for b in atom.GetBonds()) and nh:
        return 'olef'
    return {3: 'CH3', 2: 'CH2', 1: 'CH'}.get(nh, 'CH')


def carbon_type(atom):
    if atom.GetTotalNumHs() > 0:
        return 'protonated'
    if any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(atom).GetSymbol() == 'O' for b in atom.GetBonds()):
        return 'carbonyl'
    return 'quaternary'


def build(smiles, diastereotopic=frozenset()):
    """Ceiling for one structure. Returns dict with mol, dm, ranks, pair tables keyed by
    symmetry-unique (rank_h, rank_c) with the representative atom indices."""
    mol = Chem.MolFromSmiles(smiles)
    dm = Chem.GetDistanceMatrix(mol)
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    hb = [a.GetIdx() for a in mol.GetAtoms() if a.GetTotalNumHs() > 0]
    hmbc, cosy = {}, {}
    for h in hb:
        ah = mol.GetAtomWithIdx(h)
        for c in range(mol.GetNumAtoms()):
            ac = mol.GetAtomWithIdx(c)
            if ac.GetSymbol() != 'C' or c == h:
                continue
            d = int(dm[h][c])
            if d > 3:
                continue
            key = (ranks[h], ranks[c])
            hmbc.setdefault(key, dict(h=h, c=c, d=d, nb=d + 1, jcls=f'{d + 1}J',
                                      ptype=proton_type(ah), ctype=carbon_type(ac)))
        for h2 in hb:
            if h2 <= h:
                continue
            d = int(dm[h][h2])
            if d > 2:
                continue
            key = tuple(sorted((ranks[h], ranks[h2])))
            cosy.setdefault(key, dict(a=h, b=h2, d=d, jcls=f'{d + 2}J',
                                      ptype=proton_type(ah), ptype2=proton_type(mol.GetAtomWithIdx(h2))))
        # geminal: a CH2 (or CH2 on N) with two protons; whether they are diastereotopic is unknown
        if ah.GetTotalNumHs() == 2 and ah.GetSymbol() == 'C' and h in diastereotopic:
            key = (ranks[h], ranks[h], 'gem')
            cosy.setdefault(key, dict(a=h, b=h, d=0, jcls='gem', ptype='CH2', ptype2='CH2'))
    return dict(mol=mol, dm=dm, ranks=ranks, hmbc=hmbc, cosy=cosy)


# ----------------------------------------------------------------------------- Mnova J lookup
def mnova_j(rec, mol):
    """|J| lookups keyed by (h_atom, c_atom) and unordered (h,h) in the curated mol's indexing.
    Mnova numbers atoms 1..N in RDKit order of ITS (non-isomeric canonical) smiles; map via
    substructure match of that mol onto the curated mol (stereo ignored)."""
    if rec is None:
        return {}, {}
    mm = Chem.MolFromSmiles(rec['smiles'])
    flat = Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=False))
    # match mm onto curated mol (curated mol may carry stereo; matching ignores chirality by default)
    match = mol.GetSubstructMatch(mm)
    if not match or len(match) != mm.GetNumAtoms():
        return {}, {}
    to_cur = {i + 1: match[i] for i in range(mm.GetNumAtoms())}     # mnova number -> curated idx
    jch, jhh = {}, {}
    p = rec['predictions']['hsqc']
    for c in p.get('C') or []:
        for a in c['atom']:
            ci = to_cur.get(a['index'])
            for j in c.get('js') or []:
                for b in j['atom']:
                    hi = to_cur.get(b['index'])
                    if ci is None or hi is None:
                        continue
                    v = abs(float(j['j']['value']))
                    jch[(hi, ci)] = max(jch.get((hi, ci), 0.0), v)
    for h in p.get('H') or []:
        for a in h['atom']:
            hi = to_cur.get(a['index'])
            for j in h.get('js') or []:
                for b in j['atom']:
                    h2 = to_cur.get(b['index'])
                    if hi is None or h2 is None:
                        continue
                    v = abs(float(j['j']['value']))
                    k = tuple(sorted((hi, h2)))
                    jhh[k] = max(jhh.get(k, 0.0), v)
    return jch, jhh


def jbin(v):
    if v is None:
        return 'noJ'
    return '<2' if v < 2 else ('2-4' if v < 4 else ('4-8' if v < 8 else '>=8'))


# ----------------------------------------------------------------------------- matching
def match_compound(d, S, jch, jhh, diag):
    """Returns per-modality lists of (ceiling_key or None, peak, how, status)."""
    mol, ranks = S['mol'], S['ranks']
    exch = d.get('exchangeable_protons_seen') or []
    sp = d.get('spectra') or {}
    # known shifts from assigned peaks (for the shift-window fallback)
    h_shift = collections.defaultdict(list)   # atom -> [ppm]
    c_shift = collections.defaultdict(list)
    for p in sp.get('HSQC', {}).get('peaks', []):
        a = p.get('assign')
        if a:
            hi, _ = parse_assign(a[0], exch)
            ci, _ = parse_assign(a[1], exch)
            if hi is not None:
                h_shift[hi].append(p['f2_ppm'])
            if ci is not None:
                c_shift[ci].append(p['f1_ppm'])
    for p in sp.get('HMBC', {}).get('peaks', []):
        a = p.get('assign')
        if a:
            hi, _ = parse_assign(a[0], exch)
            ci, _ = parse_assign(a[1], exch)
            if hi is not None:
                h_shift[hi].append(p['f2_ppm'])
            if ci is not None:
                c_shift[ci].append(p['f1_ppm'])
    for e in exch:
        if isinstance(e.get('assign'), int):
            h_shift[e['assign']].append(e['ppm'])

    def atoms_near(table, ppm, tol):
        return sorted({a for a, v in table.items() if any(abs(x - ppm) <= tol for x in v)})

    out = {'HSQC': [], 'HMBC': [], 'COSY': []}
    for p in sp.get('HSQC', {}).get('peaks', []):
        a = p.get('assign')
        hi = parse_assign(a[0], exch)[0] if a else None
        out['HSQC'].append((hi, p, 'assign' if hi is not None else 'none'))

    for p in sp.get('HMBC', {}).get('peaks', []):
        a = p.get('assign')
        hi = ci = None
        how = 'none'
        if a and len(a) == 2:
            hi, _ = parse_assign(a[0], exch)
            ci, _ = parse_assign(a[1], exch)
            how = 'assign' if (hi is not None and ci is not None) else 'assign-partial'
        if hi is None or ci is None:
            hs = [hi] if hi is not None else atoms_near(h_shift, p['f2_ppm'], TOL_H)
            cs = [ci] if ci is not None else atoms_near(c_shift, p['f1_ppm'], TOL_C)
            cands = [(h, c) for h in hs for c in cs if (ranks[h], ranks[c]) in S['hmbc']]
            keys = {(ranks[h], ranks[c]) for h, c in cands}
            if len(keys) == 1:
                hi, ci = cands[0]
                how = 'shift'
            else:
                diag['hmbc_unmatched' if not keys else 'hmbc_ambiguous'] += 1
                out['HMBC'].append((None, p, how + '/unresolved'))
                continue
        key = (ranks[hi], ranks[ci])
        if key not in S['hmbc']:
            d_ = int(S['dm'][hi][ci]) if hi != ci else 0
            diag['hmbc_outside_ceiling'] += 1
            diag['hmbc_outside_dist'][d_] += 1
            out['HMBC'].append((None, p, how + f'/outside(d={d_})'))
            continue
        out['HMBC'].append((key, p, how))

    for p in sp.get('COSY', {}).get('peaks', []):
        a = p.get('assign')
        ha = hb = None
        ta = tb = ''
        how = 'none'
        if a and len(a) == 2:
            ha, ta = parse_assign(a[0], exch)
            hb, tb = parse_assign(a[1], exch)
            how = 'assign' if (ha is not None and hb is not None) else 'assign-partial'
        if ha is None or hb is None:
            hs1 = [ha] if ha is not None else atoms_near(h_shift, p['f2_ppm'], TOL_H)
            hs2 = [hb] if hb is not None else atoms_near(h_shift, p['f1_ppm'], TOL_H)
            cands = []
            for x in hs1:
                for y in hs2:
                    k = (ranks[x], ranks[x], 'gem') if x == y else tuple(sorted((ranks[x], ranks[y])))
                    if k in S['cosy']:
                        cands.append((x, y, k))
            keys = {k for _, _, k in cands}
            if len(keys) == 1:
                ha, hb, _ = cands[0]
                how = 'shift'
            else:
                diag['cosy_unmatched' if not keys else 'cosy_ambiguous'] += 1
                out['COSY'].append((None, p, how + '/unresolved'))
                continue
        key = (ranks[ha], ranks[ha], 'gem') if ha == hb else tuple(sorted((ranks[ha], ranks[hb])))
        if key not in S['cosy']:
            d_ = int(S['dm'][ha][hb])
            diag['cosy_outside_ceiling'] += 1
            diag['cosy_outside_dist'][d_] += 1
            out['COSY'].append((None, p, how + f'/outside(d={d_})'))
            continue
        out['COSY'].append((key, p, how))
    return out


# ----------------------------------------------------------------------------- stats
def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


class Tally:
    def __init__(self):
        self.d = collections.defaultdict(lambda: [0, 0, 0])   # key -> [observed(high/med), observed_low, possible]

    def add(self, key, status):
        t = self.d[key]
        t[2] += 1
        if status == 'obs':
            t[0] += 1
        elif status == 'obs_low':
            t[1] += 1

    def rows(self):
        out = []
        for k, (o, ol, n) in sorted(self.d.items(), key=lambda x: str(x[0])):
            lo, hi = wilson(o, n)
            out.append(dict(key=k, observed=o, observed_low=ol, possible=n,
                            rate=o / n if n else float('nan'), rate_incl_low=(o + ol) / n if n else float('nan'),
                            ci=(lo, hi)))
        return out


def fmt_rows(rows, label):
    lines = [f'| {label} | possible | observed | +low | rate | rate incl. low | 95% CI |', '|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        k = r['key'] if isinstance(r['key'], str) else ' / '.join(map(str, r['key']))
        lines.append(f"| {k} | {r['possible']} | {r['observed']} | {r['observed_low']} | {r['rate']:.2f} | {r['rate_incl_low']:.2f} | {r['ci'][0]:.2f}–{r['ci'][1]:.2f} |")
    return '\n'.join(lines)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mnova', default='/tmp/mnova/curated_mnova.json')
    ap.add_argument('--tier', default='tier1', choices=['full', 'tier1', 'tier12'],
                    help='which compounds define the not-observed class (full ⊂ tier1 ⊂ tier12)')
    a = ap.parse_args()
    global OUT
    OUT = os.path.join(OUT, a.tier)
    os.makedirs(OUT, exist_ok=True)
    mn = json.load(open(a.mnova)) if os.path.exists(a.mnova) else {}
    tier = {'full': set(FULL), 'tier1': set(TIER1), 'tier12': set(TIER1) | set(TIER2)}[a.tier]

    per = []
    T = {'HMBC': collections.defaultdict(Tally), 'COSY': collections.defaultdict(Tally), 'HSQC': Tally()}
    diag = collections.Counter()
    diag['hmbc_outside_dist'] = collections.Counter()
    diag['cosy_outside_dist'] = collections.Counter()
    how_c = collections.Counter()
    assigned_dist = collections.Counter()
    frac_per_mol = []
    for f in sorted(glob.glob(os.path.join(CUR, 'NP*.json'))):
        d = json.load(open(f))
        if d['status'] != 'ok':
            continue
        # CH2 groups the curated HSQC shows as two distinct protons (a/b tags or two peaks on one atom)
        hs_by_atom = collections.defaultdict(set)
        for p in (d.get('spectra') or {}).get('HSQC', {}).get('peaks', []):
            if p.get('assign'):
                ai, _ = parse_assign(p['assign'][0], d.get('exchangeable_protons_seen') or [])
                if ai is not None:
                    hs_by_atom[ai].add(round(p['f2_ppm'], 2))
        S = build(d['smiles'], frozenset(k for k, v in hs_by_atom.items() if len(v) >= 2))
        jch, jhh = mnova_j(mn.get(d['npid']), S['mol'])
        m = match_compound(d, S, jch, jhh, diag)
        in_tier = d['npid'] in tier
        solvent = d.get('solvent') or 'unknown'
        row = dict(npid=d['npid'], tier=in_tier, solvent=solvent,
                   hsqc_n=len(m['HSQC']), hmbc_n=len(m['HMBC']), cosy_n=len(m['COSY']),
                   hmbc_possible=sum(1 for v in S['hmbc'].values() if v['d'] <= 2),
                   cosy_possible=sum(1 for v in S['cosy'].values() if v['jcls'] == '3J'))
        # observed sets
        obs = {'HMBC': {}, 'COSY': {}}
        for mod in ('HMBC', 'COSY'):
            for key, p, how in m[mod]:
                how_c[(mod, how.split('/')[0])] += 1
                if key is None:
                    continue
                st = 'obs_low' if p.get('confidence') == 'low' else 'obs'
                if obs[mod].get(key) != 'obs':
                    obs[mod][key] = st
                if mod == 'HMBC':
                    assigned_dist[S['hmbc'][key]['jcls']] += 1
        row['hmbc_matched'] = sum(1 for k in obs['HMBC'])
        row['cosy_matched'] = sum(1 for k in obs['COSY'])
        # HSQC sanity: protonated carbons observed
        if in_tier and m['HSQC']:
            pc = {S['ranks'][x] for x in range(S['mol'].GetNumAtoms())
                  if S['mol'].GetAtomWithIdx(x).GetSymbol() == 'C' and S['mol'].GetAtomWithIdx(x).GetTotalNumHs() > 0}
            seen = {S['ranks'][hi] for hi, p, how in m['HSQC'] if hi is not None}
            for r in pc:
                T['HSQC'].add('protonated C', 'obs' if r in seen else 'no')
            row['hsqc_rate'] = len(pc & seen) / len(pc) if pc else float('nan')
        # tallies
        for mod, table in (('HMBC', S['hmbc']), ('COSY', S['cosy'])):
            denom_ok = in_tier and len(m[mod]) > 0
            for key, v in table.items():
                st = obs[mod].get(key)
                if st is None:
                    if not denom_ok:
                        continue            # partial compound: never count as dropped
                    st = 'no'
                jv = None
                if mod == 'HMBC':
                    jv = jch.get((v['h'], v['c']))
                    sc = 'protic' if solvent in PROTIC else 'aprotic'
                    strata = [('bond', v['jcls']), ('bond×solvent', (v['jcls'], solvent)),
                              ('bond×solventclass', (v['jcls'], sc)),
                              ('bond×ptype', (v['jcls'], v['ptype'])), ('bond×ctype', (v['jcls'], v['ctype'])),
                              ('bond×J', (v['jcls'], jbin(jv)))]
                    if v['ptype'] == 'exch':
                        strata.append(('exch×solventclass', (v['jcls'], sc)))
                    elif v['jcls'] in ('2J', '3J'):
                        strata.append(('ptype×ctype (2J+3J)', (v['ptype'], v['ctype'])))
                else:
                    jv = jhh.get(tuple(sorted((v['a'], v['b']))))
                    strata = [('bond', v['jcls']), ('bond×solvent', (v['jcls'], solvent)),
                              ('bond×ptype', (v['jcls'], tuple(sorted((v['ptype'], v['ptype2']))))),
                              ('bond×J', (v['jcls'], jbin(jv)))]
                for name, key2 in strata:
                    T[mod][name].add(key2, st)
        if in_tier and m['HMBC']:
            poss = [k for k, v in S['hmbc'].items() if v['d'] <= 2]
            got = sum(1 for k in poss if obs['HMBC'].get(k) == 'obs')
            frac_per_mol.append((d['npid'], got, len(poss), got / len(poss)))
            row['hmbc_frac_2J3J'] = got / len(poss)
        per.append(row)

    # ------------------------------------------------------------------ write per-compound
    L = ['# Per-compound match table', '',
         '`n` = curated peaks in file (must equal LEDGER); `matched` = peaks resolved to a ceiling pair; '
         '`possible` = symmetry-unique ceiling pairs (HMBC 2J+3J, COSY vicinal 3J); `frac` = observed/possible (tier compounds only).', '',
         '| NPID | tier | solvent | HSQC n | HSQC rate | HMBC n | matched | possible | frac 2J+3J | COSY n | matched | possible |',
         '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in per:
        L.append(f"| {r['npid']} | {'T' if r['tier'] else ''} | {r['solvent']} | {r['hsqc_n']} | {r.get('hsqc_rate', float('nan')):.2f} | "
                 f"{r['hmbc_n']} | {r['hmbc_matched']} | {r['hmbc_possible']} | {r.get('hmbc_frac_2J3J', float('nan')):.2f} | "
                 f"{r['cosy_n']} | {r['cosy_matched']} | {r['cosy_possible']} |")
    L += ['', '## Matching diagnostics', '',
          f"- how peaks were resolved: {dict(how_c)}",
          f"- HMBC peaks outside the 2–4 bond ceiling: {diag['hmbc_outside_ceiling']} (heavy-atom distance histogram {dict(diag['hmbc_outside_dist'])})",
          f"- HMBC unresolved (no candidate / ambiguous): {diag['hmbc_unmatched']} / {diag['hmbc_ambiguous']}",
          f"- COSY peaks outside the ceiling: {diag['cosy_outside_ceiling']} ({dict(diag['cosy_outside_dist'])}); unresolved {diag['cosy_unmatched']} / {diag['cosy_ambiguous']}",
          f"- matched HMBC peaks by bond class (all ok compounds): {dict(assigned_dist)}",
          f"- Mnova J available for {sum(1 for r in per if mn.get(r['npid']))} / {len(per)} ok compounds"]
    open(os.path.join(OUT, 'per_compound.md'), 'w').write('\n'.join(L) + '\n')

    # ------------------------------------------------------------------ rates
    R = ['# Observation rates (ceiling pair observed in the curated spectrum)', '',
         f"Denominators: {a.tier} compounds ({len(tier)}), per modality only where the compound has >=1 curated peak. "
         'observed = confidence high/medium; +low = additional low-confidence observations. CI = Wilson 95% on `rate`.', '']
    rates = {}
    R += ['## HSQC sanity (protonated carbons seen in HSQC)', '', fmt_rows(T['HSQC'].rows(), 'class'), '']
    for mod in ('HMBC', 'COSY'):
        rates[mod] = {}
        for name in ('bond', 'bond×solventclass', 'bond×solvent', 'bond×ptype', 'bond×ctype', 'exch×solventclass', 'ptype×ctype (2J+3J)', 'bond×J'):
            if name not in T[mod]:
                continue
            rows = T[mod][name].rows()
            rates[mod][name] = rows
            R += [f'## {mod} by {name}', '', fmt_rows(rows, name), '']
    R += ['## Per-molecule observed fraction of the HMBC 2J+3J ceiling (tier compounds)', '',
          '| NPID | observed | possible | fraction |', '|---|---:|---:|---:|']
    for n, g, p, fr in frac_per_mol:
        R.append(f'| {n} | {g} | {p} | {fr:.2f} |')
    if frac_per_mol:
        fr = [x[3] for x in frac_per_mol]
        R += ['', f'mean {sum(fr)/len(fr):.2f}, min {min(fr):.2f}, max {max(fr):.2f}, n={len(fr)}']
    open(os.path.join(OUT, 'rates.md'), 'w').write('\n'.join(R) + '\n')
    json.dump({'tier': sorted(tier), 'rates': {m: {n: [dict(r, key=str(r['key'])) for r in rows] for n, rows in v.items()} for m, v in rates.items()},
               'per_compound': per, 'frac_per_mol': frac_per_mol, 'diag': {k: (dict(v) if isinstance(v, dict) else v) for k, v in diag.items()}},
              open(os.path.join(OUT, 'rates.json'), 'w'), indent=1)

    # ------------------------------------------------------------------ dropout parameters
    # Final parameterisation (what MARINA2.0's loader consumes):
    #   HMBC keep(pair) = m · P[ptype][ctype] for 2J/3J carbon-bound protons (2J and 3J pooled: their
    #   pooled rates differ by <0.05 while ptype×ctype spans 0.11–1.0), keep_4J for 4J pairs,
    #   keep_exch[solventclass] for exchangeable protons; m is a per-molecule multiplier drawn from
    #   `molecule_multiplier` (observed/expected ratio over the tier compounds), clipped to [0,1] after.
    #   COSY keep = m · P_cosy[class]; gem applies only to CH2 with two distinct predicted shifts.
    # Each cell is Beta-smoothed toward its pooled parent rate with PRIOR_N pseudo-counts.
    def smooth(o, n, prior):
        return (o + PRIOR_N * prior) / (n + PRIOR_N)
    bond = {r['key']: r for r in T['HMBC']['bond'].rows()}
    pooled23 = (bond['2J']['observed'] + bond['3J']['observed']) / max(1, bond['2J']['possible'] + bond['3J']['possible'])
    P = {}
    for r in T['HMBC']['ptype×ctype (2J+3J)'].rows():
        pt, ct = r['key']
        P.setdefault(pt, {})[ct] = dict(raw=r['rate'], n=r['possible'], keep=round(smooth(r['observed'], r['possible'], pooled23), 3))
    ex = {r['key'][1]: r for r in T['HMBC']['exch×solventclass'].rows() if r['key'][0] in ('2J', '3J')}
    exch_pool = collections.defaultdict(lambda: [0, 0])
    for r in T['HMBC']['exch×solventclass'].rows():
        if r['key'][0] in ('2J', '3J'):
            exch_pool[r['key'][1]][0] += r['observed']; exch_pool[r['key'][1]][1] += r['possible']
    keep_exch = {sc: dict(raw=o / n if n else float('nan'), n=n, keep=round(smooth(o, n, 0.05), 3)) for sc, (o, n) in exch_pool.items()}
    r4 = bond.get('4J')
    keep_4J = dict(raw=r4['rate'], raw_incl_low=r4['rate_incl_low'], n=r4['possible'], keep=round(smooth(r4['observed'], r4['possible'], 0.02), 3)) if r4 else None
    cosy = {}
    cb = {r['key']: r for r in T['COSY']['bond'].rows()}
    for cls, prior in (('3J', 0.3), ('gem', 0.5), ('4J', 0.02)):
        if cls in cb:
            r = cb[cls]
            cosy[cls] = dict(raw=r['rate'], raw_incl_low=r['rate_incl_low'], n=r['possible'], keep=round(smooth(r['observed'], r['possible'], prior), 3))
    # per-molecule multiplier: observed / expected under P over the compound's 2J+3J carbon-bound pairs
    mult = []
    for f in sorted(glob.glob(os.path.join(CUR, 'NP*.json'))):
        d = json.load(open(f))
        if d['status'] != 'ok' or d['npid'] not in tier or not (d.get('spectra') or {}).get('HMBC', {}).get('peaks'):
            continue
        S = build(d['smiles'])
        m = match_compound(d, S, {}, {}, collections.Counter({'hmbc_outside_dist': collections.Counter(), 'cosy_outside_dist': collections.Counter()}))
        obs_keys = {key for key, p, how in m['HMBC'] if key is not None and p.get('confidence') != 'low'}
        exp = got = 0.0
        for key, v in S['hmbc'].items():
            if v['d'] <= 2 and v['ptype'] != 'exch':
                exp += P.get(v['ptype'], {}).get(v['ctype'], {}).get('keep', pooled23)
                got += key in obs_keys
        if exp > 0:
            mult.append((d['npid'], got, round(exp, 1), round(got / exp, 2)))
    ratios = sorted(x[3] for x in mult)
    lr = [math.log(max(x, 1e-3)) for x in ratios]
    mu = sum(lr) / len(lr) if lr else 0.0
    sd = math.sqrt(sum((x - mu) ** 2 for x in lr) / max(1, len(lr) - 1)) if len(lr) > 1 else 0.0
    params = {'source': 'calibrate.py', 'tier': sorted(tier), 'prior_strength': PRIOR_N,
              'model': 'keep = clip(m * P[ptype][ctype], 0, 1) for 2J/3J carbon-bound HMBC pairs; keep_4J for 4J; '
                       'keep_exch[solventclass] for OH/NH/SH protons; COSY keep = clip(m * cosy[class]); m ~ molecule_multiplier',
              'pooled_2J3J_rate': round(pooled23, 3), 'bond_rates': {k: dict(raw=v['rate'], n=v['possible']) for k, v in bond.items()},
              'P_hmbc': P, 'keep_4J': keep_4J, 'keep_exch': keep_exch, 'cosy': cosy,
              'molecule_multiplier': dict(n=len(ratios), values=ratios, min=ratios[0] if ratios else None, max=ratios[-1] if ratios else None,
                                          lognormal_mu=round(mu, 3), lognormal_sigma=round(sd, 3),
                                          table=[dict(npid=a, observed=b, expected=c, ratio=r_) for a, b, c, r_ in mult]),
              'per_molecule_observed_fraction_2J3J': dict(n=len(frac_per_mol), values=sorted(x[3] for x in frac_per_mol))}
    json.dump(params, open(os.path.join(OUT, 'dropout_params.json'), 'w'), indent=1)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
