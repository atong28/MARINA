import pickle, collections
db=pickle.load(open('/tmp/mnova/db/index.pkl','rb'))
db_smiles={v['smiles'] for v in db.values()}
rec={}
for line in open('/tmp/mnova/all_records.tsv'):
    f=line.rstrip('\n').split('\t')
    if len(f)<8: continue
    smi,st,hst,na,nh,nhj,nc,ncj=f[0],f[1],f[2],*map(int,f[3:8])
    rec[smi]=(st,hst,na,nh,nhj,nc,ncj)
print('mnova records',len(rec),'unique smiles; MARINA-DB molecules',len(db_smiles))
c=collections.Counter()
for s in db_smiles:
    r=rec.get(s)
    if r is None: c['no_record']+=1; continue
    if r[1]!='SUCCESS': c['hsqc_not_success']+=1; continue
    if r[3]==0 or r[5]==0: c['empty_H_or_C']+=1; continue
    c['ok']+=1
    if r[4]==0 and r[6]==0: c['ok_but_no_js']+=1
    elif r[4]<r[3] or r[6]<r[5]: c['ok_partial_js']+=1
for k,v in sorted(c.items()): print(f'{k:20s}{v:8d}  {100*v/len(db_smiles):.2f}%')
