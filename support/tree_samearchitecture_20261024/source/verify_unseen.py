from experiment import *
from legacy_verify import verify_method
from fractions import Fraction
import gzip,tempfile

def verify_banks(root):
 p=read(root/'protocol.json');checks=0;all_seeds=set()
 for ds in p['datasets']:
  for seed in p['seeds']:
   b=root/'banks'/ds/f'seed_{seed}';a=np.load(b/'split.npz');meta=read(b/'bank.json');ad=read(b/'admission.json')
   ids=[set(a[k].tolist()) for k in ['train_ids','validation_ids','test_ids']];assert not ids[0]&ids[1] and not ids[0]&ids[2] and not ids[1]&ids[2]
   models=[rt.TreeModel.load(b/'candidates'/f'{i:04d}') for i in meta['selected_candidate_indices']]
   assert len(models)==24 and len({fingerprint(m) for m in models})==24
   cutoff=ad['minimum_admissible_accuracy']
   if p['arm']=='random_forest_pilot':
    verified=[];fps=set()
    for e in ad['candidates']:
     m=rt.TreeModel.load(b/'candidates'/f"{e['index']:04d}");fp=fingerprint(m);assert fp==e['fingerprint'];assert e['first_unique']==(fp not in fps);fps.add(fp)
     score=float(((native_saved_prediction(m,a['X_validation'])>=.5)==a['y_validation']).mean());assert abs(score-e['validation_accuracy'])<1e-12
     assert e['fit_seed']==seed_for(p['arm'],ds,seed,0,e['index']);assert e['fit_seed'] not in all_seeds;all_seeds.add(e['fit_seed']);verified.append(score)
    assert cutoff==max(verified[:96])-.05
    selected=[e['index'] for e in ad['candidates'] if e['first_unique'] and e['validation_accuracy']>=cutoff-1e-12][:24]
    assert selected==meta['selected_candidate_indices']==ad['selected']
   source=Path(p['source_root']);source_meta=read(source/'banks'/ds/f'seed_{seed}'/'bank.json');co=read(root/'cohorts.json')[ds][str(seed)]
   assert co==read(source/'cohorts.json')[ds][str(seed)] and co['positions']==source_meta['eligible_random_order'][:5]
   factual=a['X_test'][co['positions']];ca=read(b/'cohort_audit.json')
   assert np.array_equal(common_eligibility(models,factual),ca['common_recourse_needed'])
   assert np.allclose(independent_prob(models,factual),ca['seen_factual_probabilities'],rtol=0,atol=1e-12)
   excluded={fingerprint(m) for m in models}
   for stream,panel in [(1,'refit'),(2,'bootstrap')]:
    pd=b/'panels'/panel;adp=read(pd/'admission.json');assert adp['minimum_admissible_accuracy']==cutoff
    fps=set(excluded);selected=[]
    for e in adp['candidates']:
     folder=pd/'candidates'/f"{e['index']:04d}";m=rt.TreeModel.load(folder);fp=fingerprint(m)
     assert fp==e['fingerprint'];unique=fp not in fps;fps.add(fp)
     score=float(((native_saved_prediction(m,a['X_validation'])>=.5)==a['y_validation']).mean())
     assert abs(score-e['validation_accuracy'])<1e-12 and e['distinct_from_seen_and_prior_candidates']==unique
     assert e['accepted']==(unique and score>=cutoff-1e-12)
     if e['accepted']:selected.append(e['index'])
     fitseed=seed_for(p['arm'],ds,seed,stream,e['index']);assert fitseed==e['fit_seed'] and fitseed not in all_seeds;all_seeds.add(fitseed)
     assert m.aggregation=='probability' and len(m.trees)==(10 if p['arm']=='random_forest_pilot' else 1)
     for t in m.trees:
      stack=[(0,0)]
      while stack:
       node,depth=stack.pop();assert depth<=4
       if t.left[node]!=-1:stack.extend([(int(t.left[node]),depth+1),(int(t.right[node]),depth+1)])
     if stream==2:
      idx=np.load(folder/'bootstrap_indices.npy');expected=np.random.default_rng(fitseed+500_000_000).integers(0,len(a['y_train']),len(a['y_train']));assert np.array_equal(idx,expected)
    assert selected==adp['selected'] and len(selected)==50 and adp['candidates'][-1]['accepted']
    chosen=[rt.TreeModel.load(pd/'candidates'/f'{i:04d}') for i in selected];saved=np.load(pd/'native_prediction_audit.npz')
    for name,X in [('test',a['X_test']),('validation',a['X_validation'])]:
     assert np.allclose(independent_prob(chosen,X),saved[name],rtol=0,atol=1e-12)
    checks+=1
   stage(root,'independent_bank_check',dataset=ds,seed=seed,panels_checked=checks)
 return checks

def verify_reference(root,cell,row):
 mm,x,c,meta=bank(root,cell['dataset'],1,cell['seed'],0);saved=read(root/row['shared_reference']);info=saved['info'];assert np.array_equal(row['point'],saved['point'])
 assert row['reference_success']==bool(info['success']) and point_check(row['point'],x,cell['budget'],c)
 if info['analytic']:
  assert np.array_equal(row['point'],x.numpy()) and independent_prob(mm,row['point'])[0,0]>=.5 and row['solver_calls']==0;return
 assert row['solver_calls']==1 and info['returned_ORPM_candidate'] is False and info['ORPM_warm_start'] is False
 import gurobipy as gp
 o=rt.TreeOracle(mm,x,cell['budget'],c,time_limit=30,seed=cell['seed']);original=o.problem
 path=(root/row['shared_reference']).parent/'solver/formulation.mps.gz'
 try:
  with tempfile.NamedTemporaryFile(suffix='.mps') as f:
   f.write(gzip.decompress(path.read_bytes()));f.flush();o.problem=gp.read(f.name,env=o.env)
  model=o.problem;expr=model.getObjective();assert isinstance(expr,gp.QuadExpr) and model.ModelSense==1
  q=defaultdict(float)
  for j in range(expr.size()):q[(expr.getVar1(j).VarName,expr.getVar2(j).VarName)]+=expr.getCoeff(j)
  assert q=={(v.VarName,v.VarName):1. for v in o.x}
  le=expr.getLinExpr();linear=defaultdict(float)
  for j in range(le.size()):linear[le.getVar(j).VarName]+=le.getCoeff(j)
  for v,z in zip(o.x,o.origin):assert abs(linear.get(v.VarName,0)+2*z)<1e-8
  assert abs(le.getConstant()-sum(o.origin**2))<1e-7
  found=False
  for con in model.getConstrs():
   r=model.getRow(con)
   if con.Sense=='>' and con.RHS==.5 and r.size()==1 and r.getVar(0).VarName=='probability_0' and r.getCoeff(0)==1:found=True
  assert found
  if info['success']:
   assert independent_prob(mm,saved['point'])[0,0]>=.5 and audit_encoding(o,saved['point'],'min_cost')['passed']
   assert abs(cost(saved['point'],x)**2-info['objective'])<1e-7 and info['lower_bound']<=info['objective']+1e-7
  if info.get('witness_audit'):assert audit_encoding(o,info['witness_audit']['point'],'min_cost')['passed']
  if info.get('certificate_rejected'):assert info['lower_bound']==0 and not info['global_certificate']
  if info.get('incumbent_rejected'):
   from verify_boundary import verify_budget_proof
   verify_budget_proof(mm,x,cell['budget'],c,info)
 finally:o.problem.dispose();original.dispose();o.env.dispose()

def verify(root):
 start=time.perf_counter();p=read(root/'protocol.json');plan=read(root/'task_plan.json');frozen_check(root);panels=verify_banks(root)
 records=rows(root/'records.jsonl');assert len(records)==plan['expected_method_rows'] and len({r['record_id'] for r in records})==len(records)
 rounds=0;rejections=0
 if p['arm']=='decision_tree':
  assert sha(root/'records.jsonl')==sha(Path(p['source_root'])/'records.jsonl')
 else:
  expected={record_id(c,m) for c in plan['cells'] for m in p['methods']};assert {r['record_id'] for r in records}==expected
  references=set()
  for c in plan['cells']:
   for method in p['methods']:
    row=read(method_dir(root,c,method)/'outcome.json');assert row==next(z for z in records if z['record_id']==row['record_id'])
    if method=='SingleForestReference' and c['budget']>0:
     if row['shared_reference'] not in references:verify_reference(root,c,row);references.add(row['shared_reference'])
    else:
     v=verify_method(root,c,method);rounds+=v['rounds'];rejections+=v['rejections']
  assert rounds==4800 and len(references)==24
  timing=rows(root/'timing.jsonl');counts=Counter(t['method'] for t in timing if t['kind']=='method')
  assert counts==dict(ORPM=48,UniformForest=48,DirectMinimax=48)
  assert len([t for t in timing if t['kind']=='reference_shared'])==24
  assert len({(t['key'],t['method']) for t in timing if t['kind']=='method'})==144
 evalrows=rows(root/'evaluation.jsonl');byid={(r['record_id'],r['panel']):r for r in evalrows}
 assert len(evalrows)==len(byid)==plan['expected_evaluation_rows']
 assert set(byid)=={(r['record_id'],panel) for r in records for panel in PANELS}
 for ds in p['datasets']:
  for seed in p['seeds']:
   b=root/'banks'/ds/f'seed_{seed}';a=np.load(b/'split.npz');ca=read(b/'cohort_audit.json');group=[r for r in records if r['dataset']==ds and r['seed']==seed]
   seen_models,x,c,meta=bank(root,ds,24,seed,0);pts=np.asarray([r['point'] for r in group],dtype=np.float32);seen=independent_prob(seen_models,pts)
   assert np.allclose(seen,np.asarray([r['probabilities'] for r in group]),atol=1e-12,rtol=0)
   for j,r in enumerate(group):
    _,x,c,_=bank(root,ds,24,seed,r['person']);assert point_check(r['point'],x,r['budget'],c)
    assert r['joint_validity']==bool((seen[j,:r['models']]>=.5).all())
    assert abs(cost(r['point'],x)-r['cost'])<1e-10 and r['source_person_id']==ca['source_ids'][r['person']]
   for panel in PANELS:
    pd=b/'panels'/panel;ad=read(pd/'admission.json');models=[rt.TreeModel.load(pd/'candidates'/f'{i:04d}') for i in ad['selected']]
    native=np.stack([native_saved_prediction(m,pts) for m in models],axis=1);fact=np.stack([native_saved_prediction(m,a['X_test'][ca['positions']]) for m in models],axis=1)
    saved=np.load(root/'predictions'/ds/f'seed_{seed}'/f'{panel}.npz');assert saved['record_ids'].tolist()==[r['record_id'] for r in group]
    assert np.allclose(native,saved['probabilities'],atol=1e-12,rtol=0) and np.allclose(fact,saved['factual'],atol=1e-12,rtol=0)
    for j,r in enumerate(group):
     expected=evaluation_metrics(native[j],r['joint_validity'],fact[r['person']],ca['common_recourse_needed'][r['person']]);got=byid[(r['record_id'],panel)]
     for k,v in expected.items():assert abs(got[k]-v)<1e-12,(r['record_id'],panel,k)
   stage(root,'verified_evaluation_bank',dataset=ds,seed=seed)
 frozen_check(root)
 result=dict(passed=True,utc=now(),method_rows=len(records),evaluation_rows=len(evalrows),panels=panels,rounds=rounds,rejected_certificates=rejections,source_and_input_preservation=True,independent_native_predictions=True,no_optimization_in_verification=True,full_forest_launch_authorized=False,seconds=time.perf_counter()-start)
 write(root/'verification.json',result);return result
