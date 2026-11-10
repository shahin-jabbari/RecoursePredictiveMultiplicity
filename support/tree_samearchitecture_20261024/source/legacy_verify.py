from common import *
from source_verify import independent_prob,verify_rejected
from fractions import Fraction
import collections

def verify_reference_independent(model,x,b,c,saved):
 origin=x.numpy().astype(float);factual=float(independent_prob([model],origin)[0,0])
 if factual>=.5:
  assert saved['info']['status']=='reference_already_accepts' and np.array_equal(saved['point'],origin);return
 tree=model.trees[0];stack=[(0,np.asarray(c.lower).copy(),np.asarray(c.upper).copy())];candidates=[]
 while stack:
  node,lo,hi=stack.pop()
  if tree.left[node]!=-1:
   f=int(tree.feature[node]);threshold=float(tree.threshold[node]);left=np.float32(threshold)
   if float(left)>threshold:left=np.nextafter(left,np.float32(-np.inf))
   right=np.nextafter(left,np.float32(np.inf));lhi=hi.copy();lhi[f]=min(lhi[f],float(left));rlo=lo.copy();rlo[f]=max(rlo[f],float(right))
   stack.extend([(int(tree.left[node]),lo.copy(),lhi),(int(tree.right[node]),rlo,hi.copy())]);continue
  if np.any(lo>hi):continue
  qpoint=np.clip(origin,lo,hi).astype(np.float32);q=sum((Fraction(float(a))-Fraction(float(o)))**2 for a,o in zip(qpoint,origin))
  if q>Fraction(float(b))**2:continue
  prob=independent_prob([model],qpoint)[0,0];candidates.append(((1-prob)**2,cost(qpoint,x),node,qpoint))
 assert candidates
 expected=min(candidates,key=lambda z:z[:3]);assert np.array_equal(expected[3],saved['point'])
 assert abs(expected[0]-saved['info']['objective'])<1e-12 and abs(expected[0]-saved['info']['lower_bound'])<1e-12

def verify_method(root,cell,method):
 d=method_dir(root,cell,method);row=read(d/'outcome.json');models,x,c,meta=bank(root,cell['dataset'],24,cell['seed'],cell['person']);m=cell['models'];mm=models[:m];b=cell['budget']
 assert row['record_id']==record_id(cell,method)
 for k in ['key','dataset','seed','person','models','budget']:assert row[k]==cell[k]
 assert point_check(row['point'],x,b,c)
 pr=independent_prob(models,row['point'])[0];assert np.allclose(pr,row['probabilities'],atol=1e-12,rtol=0)
 assert row['source_person_id']==meta['selected_source_ids'][cell['person']]
 assert row['joint_validity']==bool((pr[:m]>=.5).all()) and row['full24_validity']==bool((pr>=.5).all())
 assert row['mean_acceptance']==float((pr[:m]>=.5).mean()) and row['full24_mean_acceptance']==float((pr>=.5).mean())
 assert abs(cost(row['point'],x)-row['cost'])<1e-12 and abs(row['worst_probability']-pr[:m].min())<1e-12
 assert row['reference_already_accepts']==bool(independent_prob(models[:1],x.numpy())[0,0]>=.5)
 if b==0:
  assert np.array_equal(row['point'],x.numpy()) and row['analytic'] and row['solver_calls']==0;return dict(rounds=0,rejections=0)
 if method=='SingleTreeReference':
  saved=read(Path(root)/row['shared_reference']);verify_reference_independent(models[0],x,b,c,saved)
  assert np.array_equal(row['point'],saved['point']);assert row['solver_calls']==0;return dict(rounds=0,rejections=0)
 if method=='ORPM':
  rs=rows(d/'rounds.jsonl');assert len(rs)==100
  points=np.asarray([z['point'] for z in rs],dtype=np.float32);ll=(1-independent_prob(mm,points))**2
  best=x.numpy();bl=(1-independent_prob(mm,best)[0])**2;factual=bl.copy();logw=np.full(m,-np.log(m));eta=np.sqrt(np.log(m)/200);infos=[]
  for j,r in enumerate(rs):
   assert r['round']==j+1 and point_check(points[j],x,b,c);w=np.exp(logw)
   assert np.allclose(w,r['weights'],rtol=0,atol=1e-12) and np.allclose(ll[j],r['losses'],rtol=0,atol=1e-12)
   if (float(ll[j].max()),cost(points[j],x))<(float(bl.max()),cost(best,x)):best=points[j];bl=ll[j]
   assert np.array_equal(best,r['best_point']) and np.allclose(bl,r['best_losses'],rtol=0,atol=1e-12)
   assert abs(cost(best,x)-r['best_cost'])<1e-10 and r['joint_valid']==bool((independent_prob(mm,best)[0]>=.5).all())
   info=r['oracle'];assert info['lower_bound']>=0 and info['lower_bound']<=min(float(np.min(ll[:j+1]@w)),float(factual@w))+1e-7
   assert abs(float(ll[j]@w)-info['objective'])<1e-7
   verify_rejected(mm,x,b,c,info,d/'orpm_solver','weighted_loss');infos.append(info)
   logw+=eta*ll[j];logw-=np.max(logw);logw-=np.log(np.exp(logw).sum())
  assert np.array_equal(best,row['point']) and abs(max(z['lower_bound'] for z in infos)-row['objective_lower'])<1e-12
  objective=float(bl.max());nround=100
 else:
  saved=read(d/'solver_outcome.json');info=saved['oracle'];assert np.array_equal(row['point'],saved['point'])
  if row['reused']:assert not info.get('incumbent_rejected') and not info.get('certificate_rejected') and info['incumbent_verified']
  else:assert saved['orpm_warm_start'] is False and saved['returned_ORPM_candidate'] is False
  objective=float(((1-pr[:m])**2).max() if method=='DirectMinimax' else ((1-pr[:m])**2).mean());kind='minimax' if method=='DirectMinimax' else 'weighted_loss'
  assert abs(info['objective']-objective)<1e-7 and info['lower_bound']<=objective+1e-7
  verify_rejected(mm,x,b,c,info,d/'solver',kind);infos=[info];nround=0
 assert abs(row['objective']-objective)<1e-7 and row['objective_lower']<=objective+1e-7
 assert row['solver_calls']==len(infos) and abs(row['solver_seconds']-sum(i['solver_seconds'] for i in infos))<1e-7
 for name,test in [('timeout_fraction',lambda z:z['solver_status']==9),('native_certified_fraction',lambda z:z['global_certificate']),('rejected_certificate_fraction',lambda z:z.get('certificate_rejected',False)),('rejected_incumbent_fraction',lambda z:z.get('incumbent_rejected',False))]:assert abs(row[name]-np.mean([test(z) for z in infos]))<1e-12
 return dict(rounds=nround,rejections=sum(z.get('certificate_rejected',False) for z in infos))

def verify(root,pilot=False):
 root=Path(root);begin=time.perf_counter();plan=read(root/'task_plan.json');p=read(root/'protocol.json')
 assert read(root/'preparation_verification.json')['passed'] and read(root/'reuse_verification.json')['passed']
 for rel,digest in read(root/'freeze.json')['sha256'].items():assert sha(root/rel)==digest,('frozen_input_changed',rel)
 for path,digest in read(root/'source_preservation.json')['sha256'].items():assert sha(path)==digest,('source_changed',path)
 # Independently compare imported bytes to the same current source prefix/reference.
 for item in read(root/'reuse_manifest.json')['imports']:
  with Path(item['source_rounds']).open('rb') as f:prefix=b''.join(next(f) for _ in range(100))
  assert hashlib.sha256(prefix).hexdigest()==item['orpm_prefix_sha256']
  assert sha(item['source_direct'])==item['direct_sha256']
 cells=[c for c in plan['cells'] if not pilot or c['pilot']];records=rows(root/'records.jsonl');byid={r['record_id']:r for r in records}
 assert len(byid)==len(records)
 expected={record_id(c,method) for c in cells for method in p['methods']}
 if pilot:assert expected.issubset(byid) and len(expected)==192
 else:assert set(byid)==expected and len(records)==25200
 totals=dict(rounds=0,rejections=0);reference={}
 for n,cell in enumerate(cells,1):
  assert (root/'cells'/cell['key']/'completed.json').exists()
  for method in p['methods']:
   result=verify_method(root,cell,method)
   for k,v in result.items():totals[k]+=v
   row=read(method_dir(root,cell,method)/'outcome.json');assert byid[record_id(cell,method)]==row
   if method=='SingleTreeReference':reference.setdefault((cell['dataset'],cell['seed'],cell['person'],cell['budget']),[]).append(row)
  if n%30==0:stage(root,'pilot_verification' if pilot else 'independent_verification',verified_cells=n,expected_cells=len(cells))
 for group in reference.values():
  group.sort(key=lambda z:z['models']);point=group[0]['point'];assert all(np.array_equal(point,z['point']) for z in group)
  assert all(int(a['joint_validity'])>=int(b['joint_validity']) for a,b in zip(group,group[1:]))
 timing=rows(root/'timing.jsonl');method_times=[r for r in timing if r['kind']=='method'];assert len({(r['key'],r['method']) for r in method_times})==len(method_times)
 if not pilot:
  counts=collections.Counter(r['method'] for r in method_times);assert counts==dict(ORPM=3780,UniformTree=5400,DirectMinimax=3780),counts
  assert len([r for r in timing if r['kind']=='reference_shared'])==1800
  assert totals['rounds']==540000
 result=dict(passed=True,utc=now(),pilot=pilot,cells=len(cells),records=len(expected),source_preservation=True,cohort_prefix_preservation=True,predictions_feasibility_updates_and_return_rules=True,rejected_certificates_independently_audited=True,unique_records_and_timings=True,no_new_optimization=True,seconds=time.perf_counter()-begin,**totals)
 write(root/('pilot_verification.json' if pilot else 'verification.json'),result);return result
