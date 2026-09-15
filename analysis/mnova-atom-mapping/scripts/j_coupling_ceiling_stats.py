import os
import json, glob, collections, numpy as np
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
hh=collections.defaultdict(list); ch=collections.defaultdict(list)
per_mol=[]  # (nheavy, n_hsqc, n_cosy3, n_cosy34, n_hmbc23, n_hmbc234, n_hmbc23_J>=2)
for f in sorted(glob.glob(os.environ.get('MNOVA_SHARDS', '/tmp/mnova/p*.jsonl'))):
    for line in open(f):
        d=json.loads(line)
        if d['status']!='SUCCESS': continue
        p=d['predictions']['hsqc']
        if p['status']!='SUCCESS': continue
        m=Chem.MolFromSmiles(d['smiles'])
        if m is None or len(d['atoms'])!=m.GetNumHeavyAtoms(): continue
        dm=Chem.GetDistanceMatrix(m)
        hs=set(); cosy3=set(); cosy34=set(); hmbc23=set(); hmbc234=set(); hmbc23j=set()
        pass
        for h in p['H'] or []:
            for a in h['atom']:
                ai=a['index']-1
                for j in h.get('js') or []:
                    for b in j['atom']:
                        bi=b['index']-1
                        if ai==bi: continue
                        nb=int(dm[ai][bi])+2; jv=abs(j['j']['value'])
                        hh[nb].append(jv)
                        key=tuple(sorted((ai,bi)))
                        if nb==3: cosy3.add(key)
                        if nb in(3,4): cosy34.add(key)
        for c in p['C'] or []:
            for a in c['atom']:
                ai=a['index']-1
                for j in c.get('js') or []:
                    for b in j['atom']:
                        bi=b['index']-1
                        nb=int(dm[ai][bi])+1; jv=abs(j['j']['value'])
                        ch[nb].append(jv)
                        if nb==1: hs.add((ai,bi))
                        if nb in(2,3): hmbc23.add((ai,bi)); 
                        if nb in(2,3) and jv>=2: hmbc23j.add((ai,bi))
                        if nb in(2,3,4): hmbc234.add((ai,bi))
        per_mol.append((m.GetNumHeavyAtoms(),len(hs),len(cosy3),len(cosy34),len(hmbc23),len(hmbc234),len(hmbc23j)))
def q(v): 
    v=np.array(v); return f'n={len(v):7d} med={np.median(v):5.2f} p10={np.percentile(v,10):5.2f} p90={np.percentile(v,90):5.2f} frac>=1Hz={np.mean(v>=1):.2f} frac>=2Hz={np.mean(v>=2):.2f} frac>=4Hz={np.mean(v>=4):.2f}'
print('H-H |J| by bond count'); [print(f'  {k}J_HH: {q(hh[k])}') for k in sorted(hh) if len(hh[k])>50]
print('C-H |J| by bond count'); [print(f'  {k}J_CH: {q(ch[k])}') for k in sorted(ch) if len(ch[k])>50]
pm=np.array(per_mol)
print('per-molecule means over',len(pm),'mols:')
for name,col in zip(['heavy','HSQC(1J entries)','COSY 3J pairs','COSY 3+4J pairs','HMBC 2+3J pairs','HMBC 2+3+4J pairs','HMBC 2+3J & |J|>=2Hz'],pm.T):
    print(f'  {name:24s} mean={col.mean():6.1f} med={np.median(col):5.0f} p90={np.percentile(col,90):5.0f} max={col.max()}')
