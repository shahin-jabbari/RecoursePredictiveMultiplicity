"""Independent numeric verification plus exact preservation of the pilot."""
from pathlib import Path
import json
import numpy as np
from common import *
from expansion import pilot_rows,record_key
import numeric_verify

def verify(root):
    root=Path(root);cfg=read(root/'protocol.json');result=numeric_verify.verify(root);bad=result['failures'];audit=read(root/'pilot_preservation.json');pilot=Path(cfg['pilot_run'])
    for path,h in audit['original_sha256'].items():
        if sha(path)!=h:bad.append('Original pilot changed: '+path)
    for path,item in audit['copied_files'].items():
        if sha(root/path)!=item['sha256'] or sha(item['source'])!=item['sha256']:bad.append('Copied pilot artifact changed: '+path)
    old={record_key(r):r for r in pilot_rows(cfg)};found=0
    for line in open(root/'records.jsonl'):
        row=json.loads(line);k=record_key(row)
        if k in old:
            a={p:v for p,v in old[k].items() if p not in ['point_file','budget_index']};b={p:v for p,v in row.items() if p not in ['point_file','budget_index']}
            if a!=b:bad.append('Pilot scientific record changed: '+str(k))
            expected_index=cfg['budgets'][row['dataset']].index(row['budget'])
            if row['budget_index']!=expected_index:bad.append('Expanded budget address: '+str(k))
            if sha(root/row['point_file'])!=sha(pilot/old[k]['point_file']):bad.append('Pilot point/trajectory bytes changed: '+str(k))
            found+=1
    if found!=cfg['preserved_pilot_records'] or found!=len(old):bad.append(f'Pilot preservation count: {found} expected {cfg["preserved_pilot_records"]}')
    reused_panels=0
    for key,item in read(root/'model_inventory.json').items():
        if not item['pilot_panel_reused']:continue
        ds,seed=key.split('/');rel=Path('blocks')/ds/f'seed_{seed}';a=dict(np.load(pilot/rel/'panel.npz'));b=dict(np.load(root/rel/'panel.npz'));take=min(len(a['selected_X']),cfg['instances'])
        for field in ['A','b','unseen_A','unseen_b','free']:
            if not np.array_equal(a[field],b[field]):bad.append('Pilot panel mismatch: '+key+' '+field)
        for field in ['selected_X','selected_indices','test_ids']:
            if not np.array_equal(a[field][:take],b[field][:take]):bad.append('Pilot cohort prefix mismatch: '+key+' '+field)
        if read(pilot/rel/'cohort.json')['random_order']!=read(root/rel/'cohort.json')['random_order']:bad.append('Pilot saved cohort order mismatch: '+key)
        reused_panels+=1
    first_new=[p.stat().st_mtime for p in (root/'blocks').glob('*/seed_*/cases/*/budget_*/points.npz') if str(p.relative_to(root)) not in audit['copied_files']]
    if first_new and min(first_new)<(root/'panels_frozen.json').stat().st_mtime:bad.append('New interventions precede panel freeze')
    result.update(passed=not bad,verified_utc=utc(),pilot_preservation_passed=not any('Pilot' in f or 'pilot' in f for f in bad),preserved_pilot_records=found,preserved_pilot_panels=reused_panels,new_records=result['counts']['records']-found)
    return result

if __name__=='__main__':
    import sys
    root=Path(sys.argv[1]);v=verify(root);save(root/'verification_recheck.json',v);print(json.dumps(v,indent=2))
