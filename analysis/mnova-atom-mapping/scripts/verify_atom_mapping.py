import os
import json, sys, glob, collections
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')

def canon(s):
    m=Chem.MolFromSmiles(s); return Chem.MolToSmiles(m,isomericSmiles=False,canonical=True)

stats=collections.Counter(); names=collections.Counter()
hh_dist=collections.Counter(); ch_dist=collections.Counter()
bad_examples=[]
n=0
for f in sorted(glob.glob(os.environ.get('MNOVA_SHARDS', '/tmp/mnova/p*.jsonl'))):
    for line in open(f):
        d=json.loads(line); n+=1
        stats['status_'+str(d['status'])]+=1
        if d['status']!='SUCCESS': continue
        p=d['predictions']['hsqc']
        stats['hsqc_'+str(p['status'])]+=1
        if p['status']!='SUCCESS': continue
        m=Chem.MolFromSmiles(d['smiles'])
        if m is None: stats['rdkit_fail']+=1; continue
        stats['smiles_is_canon']+= (canon(d['smiles'])==d['smiles'])
        atoms=d['atoms']
        if len(atoms)!=m.GetNumHeavyAtoms():
            stats['natoms_mismatch']+=1; bad_examples.append(('natoms',d['idx'],d['smiles'],len(atoms),m.GetNumHeavyAtoms())); continue
        # name check: element + H count
        ok=True
        for a in atoms:
            i=int(a['number'])-1; ra=m.GetAtomWithIdx(i)
            nh=ra.GetTotalNumHs(); exp=ra.GetSymbol()+('' if nh==0 else 'H' if nh==1 else f'H{nh}')
            names[a['name']]+=1
            if a['name']!=exp:
                ok=False; stats['name_mismatch_atoms']+=1
                if len(bad_examples)<20: bad_examples.append(('name',d['idx'],d['smiles'],i+1,a['name'],exp))
        stats['mol_all_names_ok' if ok else 'mol_name_mismatch']+=1
        if not ok: continue
        dm=Chem.GetDistanceMatrix(m)
        # J-coupling topological distances (heavy-atom distance; H-H across atoms a,b => bonds = d(a,b)+2)
        for h in p['H'] or []:
            stats['H_entries']+=1
            if h.get('js'): stats['H_with_js']+=1
            for a in h['atom']:
                ai=a['index']-1
                for j in h.get('js') or []:
                    for b in j['atom']:
                        bi=b['index']-1
                        hh_dist[int(dm[ai][bi])+2 if ai!=bi else 'gem']+=1
        for c in p['C'] or []:
            stats['C_entries']+=1
            if c.get('js'): stats['C_with_js']+=1
            for a in c['atom']:
                ai=a['index']-1
                for j in c.get('js') or []:
                    for b in j['atom']:
                        bi=b['index']-1
                        ch_dist[int(dm[ai][bi])+1]+=1
        # multi-atom entries?
        stats['H_multiatom']+=sum(len(h['atom'])>1 for h in p['H'] or [])
        stats['C_multiatom']+=sum(len(c['atom'])>1 for c in p['C'] or [])
print('records',n)
for k,v in sorted(stats.items()): print(f'{k:28s}{v}')
print('names',names.most_common(40))
print('HH J bond-distance hist',sorted(hh_dist.items(),key=lambda x:str(x[0])))
print('CH J bond-distance hist',sorted(ch_dist.items()))
for b in bad_examples[:20]: print(b)
