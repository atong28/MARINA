import json, sys, subprocess
# stream every shard: smiles, hsqc status, nH entries, nH with js, nC entries, nC with js, natoms
out=open('/tmp/mnova/all_records.tsv','w')
lst=subprocess.run(['unzip','-Z1','Snapshots/RawData/mnova_predictions.zip'],capture_output=True,text=True).stdout.split()
for name in sorted(lst):
    p=subprocess.Popen(['unzip','-p','Snapshots/RawData/mnova_predictions.zip',name],stdout=subprocess.PIPE)
    for line in p.stdout:
        d=json.loads(line); pr=d['predictions']['hsqc']
        H=pr.get('H') or []; C=pr.get('C') or []
        out.write('\t'.join(map(str,[d['smiles'],d['status'],pr['status'],len(d['atoms'] or []),len(H),sum(1 for h in H if h.get('js')),len(C),sum(1 for c in C if c.get('js'))]))+'\n')
    p.wait()
out.close(); print('done')
