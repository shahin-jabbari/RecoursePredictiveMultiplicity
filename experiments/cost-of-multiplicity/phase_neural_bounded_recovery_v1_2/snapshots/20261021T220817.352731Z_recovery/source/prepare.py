import csv, html, shutil
from collections import Counter
from pathlib import Path
import numpy as np
from common import *
from extension import PILOT, selected_positions, check_preservation

def prepare(root,droot):
 root=Path(root);droot=Path(droot);root.mkdir(parents=True,exist_ok=False);droot.mkdir(parents=True,exist_ok=False)
 source_config=read(TRADE/'config.json');assert read(TRADE/'verification.json')['status']=='passed'
 cfg=dict(version='1.1',study='neural_cost_and_disparity_bounded_expansion',source_run=str(TRADE),disparity_result=str(droot),datasets=DATASETS,
  source_seeds=list(range(401,411)),pilot_seeds=list(range(401,411)),pilot_people_per_seed=4,model_counts=[6,12,24],budgets=source_config['budgets'],groups=GROUPS,
  maximum_optimization_seconds_per_solve=10.,threads=1,seed=0,expected_cases=240,expected_cost_comparisons=720,expected_tasks=6480,expected_new_tasks=5508,preserved_tasks=972,preserved_cases=36,source_pilot=str(PILOT),runtime_cap_seconds=23*3600,
  source_cases=1200,source_orpm_settings=26400,full_cost_tasks=32400,training=False,new_unseen_models=False,
  success_logit_threshold=0.,budget_tolerance=1e-8,cost_certificate_tolerance='1e-4*(1+maximum_budget)',
  inference='pointwise descriptive Student t intervals across seed means; expanded n=10 df=9; reused cohort n=10 df=9; numerical bounds separate',
  scope='saved neural-only heterogeneous architecture sets; twelve 20-ReLU and twelve 50/100-ReLU networks, balanced at each nested size',
  cohort_rule='Ten seeds, four saved people per dataset/seed. Preserve both pilot people first where present; append in saved order to reach two per group where available, then fill shortages in saved order. Synthetic/Polish first four. No new people outside the saved twenty.',
  structural_estimands='capped singleton, hardest singleton and common-action minimum standardized L2 at dataset maximum budget; cost and success partial-identification bounds',
  limits='10 seconds per actual optimization; supervisor stops the entire worker after 23 wall-clock hours including verification and reporting. No automatic restart or further expansion.',
  expected_maximum_optimization_hours=5508*10/3600)
 save(root/'protocol.json',cfg)
 shutil.copytree(Path(__file__).parent,root/'source',ignore=shutil.ignore_patterns('__pycache__'))
 hashes={str(TRADE/n):sha(TRADE/n) for n in ['protocol.json','config.json','verification.json','model_inventory.json','results.csv']}
 cases=[];full_cases=[];group_counts=[];inventory={};source_demand=[]
 for ds in DATASETS:
  for seed in range(401,411):
   old=TRADE/'blocks'/ds/f'seed_{seed}';selection=read(old/'selection.json');bank=Path(selection['source_bank'])
   pre=read(bank/'preprocessing.json');meta=read(bank/'models.json')
   for p in [old/'selection.json',old/'cohort.npz',bank/'models.json',bank/'preprocessing.json',bank/'split_and_predictions.npz']:
    hashes[str(p)]=sha(p)
   with np.load(bank/'split_and_predictions.npz',allow_pickle=False) as z:X=z['X_test'][selection['test_positions']].astype(np.float32).astype(float)
   labs=labels(X,pre,ds);counts=Counter(labs.tolist())
   for g in sorted(counts):group_counts.append(dict(dataset=ds,seed=seed,group=g,saved_people=counts[g]))
   for i,iid in enumerate(selection['instance_ids']):full_cases.append(dict(dataset=ds,seed=seed,instance_id=int(iid),group=str(labs[i]),group_weight=counts[str(labs[i])]/len(X)))
   nets={}
   for j in range(24,48):
    p=bank/f'model_{j:04d}.npz';hashes[str(p)]=sha(p);nets[j]=load_net(p)
    assert meta['models'][j]['training_settings']['passed']
    assert [list(w.shape)[::-1] for w,b in nets[j]]==meta['models'][j]['linear_dimensions']
   zz=logits([nets[j] for j in range(24,48)],X)
   for architecture,slots in [('shallow',range(12)),('deep',range(12,24))]:
    source_demand.append(dict(dataset=ds,seed=seed,kind='singleton_'+architecture,tasks=240,nontrivial=int(np.sum(zz[:,list(slots)]<0))))
   for n in [6,12,24]:source_demand.append(dict(dataset=ds,seed=seed,kind='joint_'+str(n),tasks=20,nontrivial=int(np.sum(np.min(zz[:,np.array(indices(n))-24],axis=1)<0))))
   inventory[f'{ds}/{seed}']=dict(source_bank=str(bank),models=24,source_indices=list(range(24,48)),architecture_counts={'20_relu':12,'50_100_relu':12},retrained=False)
   if seed not in cfg['pilot_seeds']:continue
   chosen=selected_positions(ds,seed,labs)
   bdest=root/'blocks'/ds/f'seed_{seed}'/'bank';bdest.mkdir(parents=True)
   for name in ['models.json','preprocessing.json']+[f'model_{j:04d}.npz' for j in range(24,48)]:shutil.copy2(bank/name,bdest/name)
   for position in chosen:
    iid=int(selection['instance_ids'][position]);key=f'{ds}_{seed}_{iid}';person=old/f'person_{iid}';dest=root/'cases'/key;dest.mkdir(parents=True)
    f=person/'factual.npz';hashes[str(f)]=sha(f);shutil.copy2(f,dest/'factual.npz')
    factual=arrays(f);np.testing.assert_array_equal(factual['factual'],X[position]);free=list(map(int,pre['actionable_indices']))
    assert set(free).isdisjoint(factual['immutable']) and len(free)+len(factual['immutable'])==len(X[position])
    points=[];orpm=[]
    for n in cfg['model_counts']:
     for B in cfg['budgets'][ds]:
      directory=person/'neural'/f'models_{n}'/f'budget_{B:g}'/'ORPM'
      result=read(directory/'result.json');e=arrays(directory/'evaluation.npz')
      for name in ['result.json','evaluation.npz']:hashes[str(directory/name)]=sha(directory/name)
      np.testing.assert_array_equal(e['factual'],factual['factual'])
      calc=logits([nets[j] for j in indices(n)],e['point']);np.testing.assert_allclose(calc,e['logits'],rtol=2e-8,atol=1e-8)
      assert bool((calc>=0).all())==result['joint_validity']
      assert feasible(e['point'],factual['factual'],B,free,factual['lower'],factual['upper'])
      points.append(e['point']);orpm.append(dict(models=n,budget=float(B),joint_validity=result['joint_validity'],average_validity=result['average_validity'],cost=result['cost'],source_artifact=result['artifact']))
    np.savez_compressed(dest/'saved_orpm_points.npz',points=np.stack(points));save(dest/'saved_orpm.json',orpm)
    c=dict(key=key,dataset=ds,seed=seed,instance_id=iid,source_position=position,group=str(labs[position]),group_weight=counts[str(labs[position])]/len(X),bank=str(bdest.relative_to(root)),free=free,maximum_budget=float(max(cfg['budgets'][ds])))
    cases.append(c)
   print(utc(),'PREPARED_SAVED_BLOCK',ds,seed,flush=True)
 assert len(cases)==240 and len(full_cases)==1200
 # Interleave datasets, so early timing covers every dataset.
 ordered=[]
 for seed in cfg['pilot_seeds']:
  for slot in range(4):
   for ds in DATASETS:ordered.append([c for c in cases if c['seed']==seed and c['dataset']==ds][slot])
 save(root/'cohort_plan.json',ordered);save(root/'model_inventory.json',inventory);save(root/'full_cohort.json',full_cases)
 save(root/'group_counts.json',group_counts);save(root/'source_compute_demand.json',source_demand)
 look={(r['dataset'],r['seed'],r['instance_id']):r for r in full_cases}
 with open(TRADE/'results.csv',newline='') as f,open(root/'saved_orpm_results.csv','w',newline='') as out:
  reader=csv.DictReader(f);writer=csv.DictWriter(out,fieldnames=reader.fieldnames+['group','group_weight']);writer.writeheader();count=0
  for row in reader:
   if row['family']!='neural' or row['method']!='ORPM':continue
   c=look[(row['dataset'],int(row['seed']),int(row['instance_id']))];row.update(group=c['group'],group_weight=c['group_weight']);writer.writerow(row);count+=1
 assert count==26400
 save(root/'input_sha256.json',hashes)
 # Preserve every prior task and corresponding selected input byte-for-byte.
 pilot_plan=read(PILOT/'cohort_plan.json');prior={c['key']:c for c in pilot_plan}
 for c in ordered:
  if c['key'] not in prior:continue
  assert c==prior[c['key']]
  dest=root/'cases'/c['key'];old=PILOT/'cases'/c['key']
  for name in ['factual.npz','saved_orpm_points.npz','saved_orpm.json']:
   if name!='saved_orpm_points.npz':assert sha(dest/name)==sha(old/name)
   else:np.testing.assert_array_equal(arrays(dest/name)['points'],arrays(old/name)['points'])
   shutil.copy2(old/name,dest/name)
  shutil.copytree(old/'tasks',dest/'tasks')
 shutil.copy2(PILOT/'timing.jsonl',root/'pilot_timing.jsonl')
 save(root/'preserved_case_keys.json',list(prior))
 preserved={str(p.relative_to(PILOT)):sha(p) for p in PILOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
 save(root/'pilot_source_sha256.json',preserved)
 # Task demand uses actual saved factual acceptance, before any new optimization.
 task_plan=[]
 for c in ordered:
  x0=arrays(root/'cases'/c['key']/'factual.npz')['factual']
  nets={j:load_net(root/c['bank']/f'model_{j:04d}.npz') for j in range(24,48)}
  tasks=[(f'single_{j}',[j],'singleton_shallow' if j<36 else 'singleton_deep') for j in range(24,48)]+[(f'joint_{n}',indices(n),'joint_'+str(n)) for n in [6,12,24]]
  for task,js,kind in tasks:
   task_plan.append(dict(case=c['key'],dataset=c['dataset'],seed=c['seed'],instance_id=c['instance_id'],task=task,kind=kind,reused=c['key'] in prior,nontrivial=bool(logits([nets[j] for j in js],x0).min()<0)))
 save(root/'task_plan.json',task_plan)
 save(root/'extension_preservation.json',check_preservation(root,check_original=True))
 text=(Path(__file__).parent/'PROTOCOL_TEMPLATE.txt').read_text()
 for p in [root,droot]:
  (p/'PROTOCOL.txt').write_text(text);(p/'PROTOCOL.html').write_text('<!doctype html><meta charset="utf-8"><title>Neural multiplicity bounded expansion protocol</title><style>body{max-width:1000px;margin:3em auto;font:17px/1.6 system-ui;padding:0 2em}pre{white-space:pre-wrap}</style><pre>'+html.escape(text)+'</pre>')
  save(p/'shared_run.json',dict(computation=str(root),cost_report=str(root/'cost_report'),disparity_report=str(droot)))
 frozen={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
 save(root/'freeze.json',dict(utc=utc(),before_new_optimization=True,sha256=frozen))
 save(root/'preparation_verification.json',dict(passed=True,cases=240,source_cases=1200,source_orpm_rows=26400,pilot_settings=sum(len(cfg['budgets'][c['dataset']])*3 for c in cases),all_pilot_orpm_forward_and_feasibility_checks_passed=True,no_model_training=True))
 save(root/'progress.json',dict(status='prepared',completed_tasks=972,preserved_tasks=972,new_completed_tasks=0,expected_tasks=6480,updated_utc=utc()))
 from eta import remaining
 save(root/'runtime_estimate.json',remaining(root))
 return cfg
