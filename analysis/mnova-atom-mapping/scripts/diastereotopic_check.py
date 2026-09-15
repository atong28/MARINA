import json, glob, collections
multi=collections.Counter(); names=collections.Counter(); gem_between_entries=0; gem_self=0; n=0; ex=None
for f in sorted(glob.glob('/tmp/mnova/p*.jsonl')):
    for line in open(f):
        d=json.loads(line)
        p=d['predictions']['hsqc']
        if p['status']!='SUCCESS' or not p['H']: continue
        n+=1
        name={int(a['number']):a['name'] for a in d['atoms']}
        cnt=collections.Counter(a['index'] for h in p['H'] for a in h['atom'])
        for idx,c in cnt.items():
            multi[c]+=1
            if c>1: names[name[idx]]+=1
        # do the two entries for a CH2 have different shifts?
        for idx,c in cnt.items():
            if c==2:
                sh=[h['shift']['value'] for h in p['H'] if any(a['index']==idx for a in h['atom'])]
                if sh[0]!=sh[1]: gem_between_entries+=1
                else: gem_self+=1
                if ex is None: ex=(d['smiles'],idx,name[idx],sh,[h.get('js') for h in p['H'] if any(a['index']==idx for a in h['atom'])])
print('mols',n,'entries-per-atom-index hist',dict(multi))
print('atom names with >1 H entry',names.most_common())
print('CH2 pairs with different shifts',gem_between_entries,'identical',gem_self)
print('example',ex)
