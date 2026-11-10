from common import *
import gzip,tempfile

def independent_prob(models,X):
 X=np.atleast_2d(np.asarray(X,dtype=np.float32));columns=[]
 for model in models:
  trees=[]
  for t in model.trees:
   result=np.empty(len(X));stack=[(0,np.arange(len(X)))]
   while stack:
    node,ids=stack.pop()
    if not len(ids):continue
    if t.left[node]==-1:result[ids]=t.probability[node]
    else:
     mask=X[ids,int(t.feature[node])].astype(float)<=float(t.threshold[node]);stack.extend([(int(t.left[node]),ids[mask]),(int(t.right[node]),ids[~mask])])
   trees.append(result)
  columns.append(np.mean(trees,axis=0))
 return np.stack(columns,axis=1)

def verify_rejected(models,x0,b,c,info,artifact,objective):
 if info.get('incumbent_rejected'):
  from verify_boundary import verify_budget_proof
  verify_budget_proof(models,x0,b,c,info)
 if not info.get('certificate_rejected'):return
 import gurobipy as gp
 oracle=rt.TreeOracle(models,x0,b,c,time_limit=1,seed=390)
 # Read original complete archived encoding; independently fill all variables at the saved witness.
 old=oracle.problem
 with tempfile.NamedTemporaryFile(suffix='.mps') as tmp:
  with gzip.open(artifact/'formulation.mps.gz','rb') as f:tmp.write(f.read());tmp.flush()
  oracle.problem=gp.read(tmp.name,env=oracle.env)
 try:
  audit=audit_encoding(oracle,info['witness_audit']['point'],objective)
  assert audit['passed'],audit
  assert info['lower_bound']==0 and not info['global_certificate']
 finally:oracle.problem.dispose();old.dispose();oracle.env.dispose()

def verify_cell(root,p,cell):
 ds,seed,person,m,budget=(cell[k] for k in ['dataset','seed','person','models','budget'])
 mm,x,c,meta=bank(root,ds,m,seed,person);out=root/'trajectories'/cell['key']
 factual=(1-independent_prob(mm,x.numpy())[0])**2
 nround=ndiag=nref=rejected=incumbents_rejected=ntrajectory=0
 rs=rows(out/'rounds.jsonl');dsrows=rows(out/'diagnostics.jsonl');diagnostics={d['round']:d for d in dsrows}
 assert len(diagnostics)==len(dsrows)
 assert 1<=len(rs)<=200 and sorted(diagnostics)==p['checkpoints']
 assert len(rs)==200 and read(out/'completed.json')['rounds']==200
 ntrajectory+=1
 pts=np.asarray([r['point'] for r in rs],dtype=np.float32);bestpts=np.asarray([r['best_point'] for r in rs],dtype=np.float32)
 losses=(1-independent_prob(mm,pts))**2;bestloss=(1-independent_prob(mm,bestpts))**2;eta=np.sqrt(np.log(m)/200);logw=np.full(m,-np.log(m));weights=[];lower=[];bestpoint=x.numpy();bl=factual.copy();first=None
 for j,r in enumerate(rs):
  assert r['round']==j+1;w=np.exp(logw);assert np.allclose(w,r['weights'],atol=1e-12,rtol=0)
  assert point_check(pts[j],x,budget,c) and point_check(bestpts[j],x,budget,c)
  assert np.allclose(losses[j],r['losses'],atol=1e-12,rtol=0) and np.allclose(bestloss[j],r['best_losses'],atol=1e-12,rtol=0)
  cost=lambda z:float(np.linalg.norm(np.asarray(z,dtype=float)-x.numpy().astype(float)))
  if (float(losses[j].max()),cost(pts[j]))<(float(bl.max()),cost(bestpoint)):bestpoint=pts[j];bl=losses[j]
  assert np.array_equal(bestpoint,bestpts[j]);assert abs(cost(bestpoint)-r['best_cost'])<1e-10
  assert r['joint_valid']==bool((independent_prob(mm,bestpoint)[0]>=.5).all())
  info=r['oracle'];assert info['lower_bound']<=min(float(np.min(losses[:j+1]@w)),float(factual@w))+1e-7
  verify_rejected(mm,x,budget,c,info,out/info.get('artifact_subdir','orpm_solver'),'weighted_loss');rejected+=int(info['certificate_rejected']);incumbents_rejected+=int(info.get('incumbent_rejected',False))
  weights.append(w);lower.append(info['lower_bound']);wb=np.mean(weights,axis=0);L=float(np.mean(lower));U=min(float(factual@wb),float(np.min(losses[:j+1]@wb)))
  if j+1 in diagnostics:
   d=diagnostics[j+1];assert np.allclose(d['weights'],wb,atol=1e-12,rtol=0);assert point_check(d['point'],x,budget,c)
   dl=(1-independent_prob(mm,d['point'])[0])**2;assert np.allclose(dl,d['losses'],atol=1e-12,rtol=0)
   info=d['oracle'];verify_rejected(mm,x,budget,c,info,out/info.get('artifact_subdir','diagnostic_solver'),'weighted_loss');rejected+=int(info['certificate_rejected']);incumbents_rejected+=int(info.get('incumbent_rejected',False));L=max(L,info['lower_bound']);U=min(U,float(dl@wb))
  assert L<=U+1e-7 and abs(L-r['dual_lower'])<1e-10 and abs(U-r['dual_upper'])<1e-10
  P=float(losses[:j+1].mean(0).max());Pb=float(bl.max());assert P>=L-1e-7 and Pb>=L-1e-7
  for name,val in [('mixture_gap',P),('point_gap',Pb)]:assert np.allclose(r[name],[max(0,val-U),max(0,val-L)],atol=1e-10,rtol=0)
  if r['mixture_gap'][1]<=.005 and first is None:first=j+1
  assert r['first_certified_round']==first
  logw+=eta*losses[j];logw-=np.max(logw);logw-=np.log(np.exp(logw).sum())
 nround+=len(rs);ndiag+=len(dsrows)
 ref=read(out/'reference.json');assert point_check(ref['point'],x,budget,c);rl=(1-independent_prob(mm,ref['point'])[0])**2;assert np.allclose(rl,ref['losses'],atol=1e-12,rtol=0);assert ref['oracle']['lower_bound']<=min(rl.max(),bl.max())+1e-7
 verify_rejected(mm,x,budget,c,ref['oracle'],out/'reference_solver','minimax');rejected+=int(ref['oracle']['certificate_rejected']);nref+=1
 return dict(rounds=nround,diagnostics=ndiag,references=nref,rejected_certificates=rejected,rejected_incumbents=incumbents_rejected)

def verify_bank(root,p,ds,seed):
 b=root/'banks'/ds/f'seed_{seed}';models,x,c,meta=bank(root,ds,24,seed,0);a=np.load(b/'split.npz');native=np.load(b/'native_prediction_audit.npz')
 assert np.allclose(independent_prob(models,a['X_test']).T,native['test'],rtol=0,atol=1e-12)
 assert np.allclose(independent_prob(models,a['X_validation']).T,native['validation'],rtol=0,atol=1e-12)
 ids=[set(a[k].tolist()) for k in ['train_ids','validation_ids','test_ids']]
 assert not ids[0]&ids[1] and not ids[0]&ids[2] and not ids[1]&ids[2]
 admission=read(b/'admission.json');fingerprints=set();accuracies=[]
 for e in admission['candidates']:
  model=rt.TreeModel.load(b/'candidates'/f"{e['index']:04d}");t=model.trees[0];h=hashlib.sha256()
  for z in [t.left,t.right,t.feature,t.threshold,t.probability]:h.update(np.ascontiguousarray(z).tobytes())
  fp=h.hexdigest();assert fp==e['fingerprint'];assert (fp not in fingerprints)==e['first_unique'];fingerprints.add(fp)
  accuracy=float(((independent_prob([model],a['X_validation'])[:,0]>=.5)==a['y_validation']).mean())
  assert abs(accuracy-e['validation_accuracy'])<1e-12;accuracies.append(accuracy)
  assert e['fit_seed']==seed+1000+e['index']
 assert 96<=len(accuracies)<=p['primary_family']['candidate_reserve_limit']
 reference=max(accuracies[:96]);assert abs(reference-admission['best_validation_accuracy'])<1e-12
 selected=[e['index'] for e in admission['candidates'] if e['first_unique'] and accuracies[e['index']]>=reference-.05-1e-12][:24]
 assert len(selected)==24 and selected==meta['selected_candidate_indices']==admission['selected']
 if len(accuracies)>96:assert selected[-1]==len(accuracies)-1
 pred=native['test'].T;eligible=np.ones(len(pred),dtype=bool)
 for m in [6,12,24]:eligible &=2*(pred[:,:m]<.5).sum(1)>=m
 order=np.random.default_rng(seed).permutation(len(pred));ordered=[int(i) for i in order if eligible[i]]
 assert ordered==meta['eligible_random_order']
 assert meta['selected_test_positions']==ordered[:p['people_per_dataset_seed']]
 assert meta['selected_source_ids']==[int(a['test_ids'][i]) for i in meta['selected_test_positions']]
 return len(meta['selected_test_positions'])

def verify(root):
 root=Path(root);p=read(root/'protocol.json');plan=read(root/'task_plan.json');begin=time.perf_counter();deadline_check(root)
 assert read(root/'preparation_verification.json')['passed']
 for rel,digest in read(root/'freeze.json')['sha256'].items():assert sha(root/rel)==digest,('frozen_input_changed',rel)
 summaries=rows(root/'summaries.jsonl');timing=rows(root/'timing.jsonl');trajectories=[z for z in timing if z.get('kind')=='trajectory'];preparation=[z for z in timing if z.get('kind')=='preparation_bank']
 assert len(timing)==len(trajectories)+len(preparation)
 assert len(preparation)==len(p['datasets'])*len(p['seeds'])
 assert len({(z['dataset'],z['seed']) for z in preparation})==len(preparation)
 expected={z['key']:z for z in plan['cells']};assert len(expected)==plan['expected_trajectories']
 ss={z['key']:z for z in summaries};tt={z['key']:z for z in trajectories}
 assert len(summaries)==len(ss)==len(trajectories)==len(tt)==len(expected) and set(ss)==set(tt)==set(expected)
 people=0;recomputed=set()
 for ds in p['datasets']:
  for seed in p['seeds']:
   deadline_check(root);count=verify_bank(root,p,ds,seed);people+=count
   for person in range(count):
    for m in p['model_counts']:
     for b in p['positive_budgets'][ds]:recomputed.add(cell_key(ds,seed,person,m,b))
 assert recomputed==set(expected) and people==plan['people']
 totals=dict(rounds=0,diagnostics=0,references=0,rejected_certificates=0,rejected_incumbents=0)
 for ordinal,(key,cell) in enumerate(expected.items(),1):
  deadline_check(root)
  assert ss[key]==read(root/'trajectories'/key/'completed.json')
  assert ss[key]=={k:v for k,v in tt[key].items() if k not in ['kind','utc']}
  for k in ['dataset','seed','person','models','budget']:assert ss[key][k]==cell[k]
  result=verify_cell(root,p,cell)
  for k,v in result.items():totals[k]+=v
  if ordinal%30==0:stage(root,'independent_verification',verified_trajectories=ordinal,total=len(expected))
 assert (totals['rounds'],totals['diagnostics'],totals['references'])==(plan['expected_rounds'],plan['expected_diagnostics'],plan['expected_references'])
 controls=read(root/'zero_budget_controls.json');assert len(controls)==people*len(p['model_counts'])
 assert len({(z['dataset'],z['seed'],z['person'],z['models']) for z in controls})==len(controls)
 for row in controls:
  mm,x,c,meta=bank(root,row['dataset'],row['models'],row['seed'],row['person']);pr=independent_prob(mm,x.numpy())[0]
  assert np.allclose(row['losses'],(1-pr)**2,rtol=0,atol=1e-12) and row['joint_valid']==bool((pr>=.5).all())
  assert row['cost']==0 and row['mixture_gap']==[0,0] and row['point_gap']==[0,0]
 for item in read(root/'calibration_preservation.json')['files']:assert sha(item['path'])==item['sha256']
 result=dict(passed=True,utc=now(),source_model_cohort_preservation=True,independent_predictions=True,admission_and_cohorts_rechecked=True,action_feasibility=True,weights_and_gap_arithmetic=True,unique_cells_and_timing_kinds_checked=True,calibration_unchanged=True,trajectories=len(expected),banks=len(preparation),people=people,seconds=time.perf_counter()-begin,**totals)
 write(root/'verification.json',result);return result

if __name__=='__main__':verify(Path(sys.argv[1]))
