import json,glob,os,collections
LR={"E20":0.003,"E21":0.006,"E22":0.006,"E24":0.003,"E27":0.006}
meta={}
for m in glob.glob('/root/fusiondance-data/cloud/fleet/*/fusiondance-data/runs/E2[0-7]-*/meta.json'):
    j=json.load(open(m)); cfg=j.get('config') or {}; tr=cfg.get('train',{}); co=j.get('cli_overrides') or {}
    lr=co.get('lr',tr.get('lr')) if isinstance(co,dict) else tr.get('lr'); mt=(co.get('max_tokens') if isinstance(co,dict) else None) or tr.get('max_tokens')
    meta[os.path.basename(os.path.dirname(m))]=(lr,mt,tr.get('dataset'))
disp=collections.Counter(); rows=[]
for f in sorted(glob.glob('/root/f2all/E2[0-7]-2026*.json')):
    d=json.load(open(f)); run=d['run']; arm=run.split('-')[0]; seed=int(d['seed'])
    if arm not in LR: why='not an F2 cell (E23/E25/E26)'
    elif run not in meta: why='no run meta pulled'
    else:
        lr,mt,ds=meta[run]
        if mt!=20000000: why=f'not 20M tokens ({mt})'
        elif ds!='code-py-v2': why=f'dataset {ds}'
        elif lr is None or abs(lr-LR[arm])>1e-12: why=f'not the registered LR ({lr})'
        elif seed==0: why='seed 0 (LR selection, excluded as registered)'
        else: why='INCLUDED'
    disp[why]+=1; rows.append({'run':run,'cell':arm,'seed':seed,'disposition':why})
inc=[r for r in rows if r['disposition']=='INCLUDED']; dup=collections.Counter((r['cell'],r['seed']) for r in inc)
json.dump({'n_files':len(rows),'counts':dict(disp),'duplicate_cell_seed_among_included':{f'{k[0]}-s{k[1]}':v for k,v in dup.items() if v>1},'rows':rows},open('/mnt/c/Users/mreca/Desktop/Gogeta-0/experiments/checks/results/f2_input_manifest.json','w'),indent=1)
print(len(rows)); [print(f'{v:4d}  {k}') for k,v in disp.most_common()]; print('duplicates among included:', {k:v for k,v in dup.items() if v>1})
