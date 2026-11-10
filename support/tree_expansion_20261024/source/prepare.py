"""Expansion preparation: immutable saved banks and outcomes, no fitting or solving."""
from common import *
import shutil

def link(src,dst):
 src=Path(src).resolve();dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists() or dst.is_symlink():raise RuntimeError(f'Refuse to overwrite {dst}')
 dst.symlink_to(src,target_is_directory=src.is_dir())

def preservation_inputs(source):
 source=Path(source);out={}
 for rel,digest in read(source/'freeze.json')['sha256'].items():
  path=source/rel
  if sha(path)!=digest:raise RuntimeError(f'Source frozen input changed: {path}')
  out[str(path)]=digest
 for name in ['protocol.json','freeze.json','verification.json','worker_exit.json','summaries.jsonl','records.jsonl','cohort_plan.json','cohorts.json','task_plan.json']:
  path=source/name
  if path.exists():out[str(path)]=sha(path)
 return out

def audit_prefix(root,cell,source_dir,source_id):
 from verify import independent_prob,verify_rejected
 models,x,c,meta=bank(root,cell['dataset'],cell['models'],cell['seed'],cell['person'])
 assert int(meta['selected_source_ids'][cell['person']])==int(source_id)
 rs=rows(source_dir/'rounds.jsonl');assert len(rs)==100
 losses=(1-independent_prob(models,[r['point'] for r in rs]))**2
 logw=np.full(len(models),-np.log(len(models)));eta=np.sqrt(np.log(len(models))/200);best=x.numpy();bl=(1-independent_prob(models,best)[0])**2
 factual=bl.copy()
 for j,r in enumerate(rs):
  w=np.exp(logw);assert r['round']==j+1 and np.allclose(w,r['weights'],atol=1e-12,rtol=0)
  assert point_check(r['point'],x,cell['budget'],c)
  assert np.allclose(losses[j],r['losses'],atol=1e-12,rtol=0)
  norm=lambda a:float(np.linalg.norm(np.asarray(a,dtype=float)-x.numpy().astype(float)))
  if (float(losses[j].max()),norm(r['point']))<(float(bl.max()),norm(best)):best=np.asarray(r['point'],np.float32);bl=losses[j]
  assert np.array_equal(best,np.asarray(r['best_point'],np.float32)) and np.allclose(bl,r['best_losses'],atol=1e-12,rtol=0)
  assert abs(norm(best)-r['best_cost'])<1e-10
  assert r['joint_valid']==bool((independent_prob(models,best)[0]>=.5).all())
  assert r['oracle']['lower_bound']<=min(float(np.min(losses[:j+1]@w)),float(factual@w))+1e-7
  verify_rejected(models,x,cell['budget'],c,r['oracle'],source_dir/r['oracle'].get('artifact_subdir','orpm_solver'),'weighted_loss')
  logw+=eta*losses[j];logw-=np.max(logw);logw-=np.log(np.exp(logw).sum())
 return True

def prepare(root):
 from verify import verify_bank
 root=Path(root);p=read(root/'protocol.json')
 if (root/'freeze.json').exists():raise RuntimeError('Already prepared; no implicit restart')
 begin=time.perf_counter();sources={k:Path(v) for k,v in p['source_runs'].items()};C,T,K=(sources[k] for k in ['convergence','tradeoff','multiplicity'])
 preserve={}
 for name,src in [('convergence',C),('tradeoff',T),('multiplicity',K)]:
  assert read(src/'worker_exit.json')['completed'] and read(src/'verification.json')['passed']
  preserve.update(preservation_inputs(src));stage(root,'source_preservation_audit',source=name)
 inventories=[]
 for ds in p['datasets']:
  for seed in p['seeds']:
   start=time.perf_counter();src=C/'banks'/ds/f'seed_{seed}';dest=root/'banks'/ds/f'seed_{seed}';dest.mkdir(parents=True)
   meta=read(src/'bank.json');positions=meta['eligible_random_order'][:20];assert len(positions)==20
   for child in src.iterdir():
    if child.name!='bank.json':link(child,dest/child.name)
   a=np.load(src/'split.npz');meta['selected_test_positions']=positions;meta['selected_source_ids']=[int(a['test_ids'][i]) for i in positions]
   write(dest/'bank.json',meta);assert verify_bank(root,p,ds,seed)==20
   item=dict(dataset=ds,seed=seed,models=24,selected_people=20,eligible_people=meta['eligible_count'],seconds=time.perf_counter()-start,candidates_trained=0,reused_bank=True)
   inventories.append(item);append(root/'timing.jsonl',dict(kind='preparation_bank',utc=now(),**item))
   write(root/'progress.json',dict(status='preparing',banks_completed=len(inventories),expected_banks=60,completed_trajectories=0,expected_trajectories=10800,utc=now()))
   if len(inventories)%6==0:stage(root,'BANKS_AUDITED',banks=len(inventories),total=60)
 write(root/'model_inventory.json',dict(banks=inventories,banks_completed=60,total_selected_models=1440,total_candidates_trained=0))
 cells=[]
 for person in range(20):
  for bi in range(3):
   for m in p['model_counts']:
    for seed in p['seeds']:
     for ds in p['datasets']:
      b=p['positive_budgets'][ds][bi];cells.append(dict(key=cell_key(ds,seed,person,m,b),dataset=ds,seed=seed,person=person,models=m,budget=b))
 plan=dict(cells=cells,people=1200,expected_trajectories=10800,expected_rounds=2160000,expected_diagnostics=162000,expected_references=10800,scope_shortfall_from_target=0)
 write(root/'task_plan.json',plan)
 # Map by saved test position, never by the expanded disparity cohort's ordinal.
 extra={(z['dataset'],z['seed'],z['position']):z for z in read(K/'cohort_plan.json') if not z['preserved']}
 reuse={'complete':{},'prefixes':{},'references':{}}
 old_summaries={z['key']:z for z in rows(C/'summaries.jsonl')};old_timings={z['key']:z for z in rows(C/'timing.jsonl') if z['kind']=='trajectory'}
 for i,cell in enumerate(cells):
  key,ds,seed,person,m,b=(cell[k] for k in ['key','dataset','seed','person','models','budget']);out=root/'trajectories'/key
  meta=read(root/'banks'/ds/f'seed_{seed}'/'bank.json');position=meta['selected_test_positions'][person];sid=meta['selected_source_ids'][person]
  if person<3:
   src=C/'trajectories'/key;assert read(src/'completed.json')==old_summaries[key]
   link(src,out);reuse['complete'][key]=str(src)
   append(root/'summaries.jsonl',old_summaries[key]);append(root/'timing.jsonl',old_timings[key])
   for n in ['setting.json','completed.json','rounds.jsonl','diagnostics.jsonl','reference.json']:
    preserve[str(src/n)]=sha(src/n)
   continue
  prefix=None
  if person<5:
   prefix=T/'cells'/key/'ORPM';refdir=T/'cells'/key/'DirectMinimax'
   assert int(read(prefix/'outcome.json')['source_person_id'])==sid
   assert int(read(refdir/'outcome.json')['source_person_id'])==sid
   assert (refdir/'solver_outcome.json').exists()
   reuse['references'][key]=dict(directory=str(refdir),source_id=sid)
   for n in ['outcome.json','solver_outcome.json']:preserve[str(refdir/n)]=sha(refdir/n)
  elif (ds,seed,position) in extra:
   old=extra[(ds,seed,position)];assert old['source_id']==sid
   prefix=K/'cells'/cell_key(ds,seed,old['person'],m,b)/'ORPM'
   assert int(read(prefix/'outcome.json')['source_person_id'])==sid
  if prefix is not None:
   audit_prefix(root,cell,prefix,sid)
   reuse['prefixes'][key]=dict(directory=str(prefix),rounds=100,source_id=sid,sha256=sha(prefix/'rounds.jsonl'))
   for n in ['rounds.jsonl','outcome.json']:preserve[str(prefix/n)]=sha(prefix/n)
  if i%180==0:stage(root,'PREFIX_IMPORT_AUDIT',visited=i,total=10800,prefixes=len(reuse['prefixes']))
 assert (len(reuse['complete']),len(reuse['prefixes']),len(reuse['references']))==(1620,1629,1080)
 plan.update(preserved_complete_trajectories=1620,preserved_prefix_trajectories=1629,preserved_rounds=486900,new_orpm_calls=1673100,new_diagnostics=137700,new_references=8100)
 write(root/'task_plan.json',plan);write(root/'reuse_manifest.json',reuse)
 controls=[]
 for ds in p['datasets']:
  for seed in p['seeds']:
   mm,x,c,meta=bank(root,ds,24,seed,0);a=np.load(root/'banks'/ds/f'seed_{seed}'/'split.npz')
   for person,pos in enumerate(meta['selected_test_positions']):
    pr=rt.probabilities(mm,a['X_test'][pos])
    for m in p['model_counts']:controls.append(dict(dataset=ds,seed=seed,person=person,models=m,budget=0,losses=(1-pr[:m])**2,joint_valid=bool((pr[:m]>=.5).all()),cost=0,mixture_gap=[0,0],point_gap=[0,0],analytic=True))
 write(root/'zero_budget_controls.json',controls)
 write(root/'source_preservation.json',dict(utc=now(),sha256=preserve,all_source_frozen_inputs_checked=True))
 write(root/'calibration_preservation.json',dict(files=[dict(path=k,sha256=v) for k,v in preserve.items()]))
 paths=[*root.glob('banks/*/seed_*/bank.json'),*(root/'source').rglob('*.py')]+[root/n for n in ['protocol.json','task_plan.json','reuse_manifest.json','zero_budget_controls.json','source_preservation.json','calibration_preservation.json']]
 write(root/'freeze.json',dict(utc=now(),all_inputs_frozen_before_recourse=True,sha256=manifest(root,paths),banks=60,people=1200,linked_source_inputs_covered_by='source_preservation.json'))
 write(root/'preparation_verification.json',dict(passed=True,utc=now(),native_predictions_match=True,all_banks_and_cohorts_frozen=True,candidates_trained=0,banks=60,people=1200,prefix_actions_weights_return_rule_bounds_independently_checked=1629,original_results_unchanged=True,seconds=time.perf_counter()-begin))
 write(root/'progress.json',dict(status='running',completed_trajectories=1620,expected_trajectories=10800,completed_rounds=324000,preserved_prefix_rounds=162900,new_orpm_calls=0,expected_new_orpm_calls=1673100,expected_rounds=2160000,utc=now()))
 stage(root,'ALL_INPUTS_FROZEN',banks=60,people=1200,complete_reused=1620,prefixes_reused=1629)
 return plan
