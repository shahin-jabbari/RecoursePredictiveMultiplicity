from common import *
from fractions import Fraction
import gzip,shutil

def pack_models(folder):
 for path in Path(folder).rglob('*.mps'):
  with path.open('rb') as src,gzip.open(str(path)+'.gz','wb') as dst:shutil.copyfileobj(src,dst)
  with gzip.open(str(path)+'.gz','rb') as f:assert hashlib.sha256(f.read()).hexdigest()==sha(path)
  path.unlink()

def stats(infos):
 return dict(solver_calls=len(infos),solver_seconds=sum(z.get('solver_seconds',0.) for z in infos),
  timeout_fraction=float(np.mean([z['solver_status']==9 for z in infos])) if infos else None,
  native_certified_fraction=float(np.mean([z['global_certificate'] for z in infos])) if infos else None,
  rejected_certificate_fraction=float(np.mean([z.get('certificate_rejected',False) for z in infos])) if infos else None,
  rejected_incumbent_fraction=float(np.mean([z.get('incumbent_rejected',False) for z in infos])) if infos else None)

def outcome(root,cell,method,point,**extra):
 mm,x,c,meta=bank(root,cell['dataset'],24,cell['seed'],cell['person']);pr=rt.probabilities(mm,point);m=cell['models']
 assert point_check(point,x,cell['budget'],c)
 return dict(**cell,method=method,record_id=record_id(cell,method),point=np.asarray(point,dtype=np.float32),
  source_person_id=meta['selected_source_ids'][cell['person']],probabilities=pr,
  joint_validity=bool((pr[:m]>=.5).all()),mean_acceptance=float((pr[:m]>=.5).mean()),
  full24_validity=bool((pr>=.5).all()),full24_mean_acceptance=float((pr>=.5).mean()),
  worst_probability=float(pr[:m].min()),cost=cost(point,x),reference_already_accepts=bool(rt.probabilities(mm[:1],x)[0]>=.5),**extra)

def single_reference(model,x,b,c):
 factual=float(rt.probabilities([model],x)[0]);candidates=[]
 if factual>=.5:return x.clone(),dict(status='reference_already_accepts',analytic=True,candidates=[],objective=(1-factual)**2,lower_bound=None,global_certificate=False)
 origin=x.numpy().astype(float)
 for node,lo,hi,path in model.trees[0].leaves(len(x)):
  lower=np.maximum(lo,c.lower);upper=np.minimum(hi,c.upper)
  if np.any(lower>upper):continue
  point=np.clip(origin,lower,upper).astype(np.float32)
  q=sum((Fraction(float(a))-Fraction(float(o)))**2 for a,o in zip(point,origin))
  if q>Fraction(float(b))**2:continue
  if not point_check(point,x,b,c):raise RuntimeError('Analytic reference point failed feasibility')
  prob=float(rt.probabilities([model],point)[0]);assert int(model.trees[0].apply(point)[0])==node
  candidates.append(dict(leaf=node,point=point,probability=prob,loss=(1-prob)**2,cost=cost(point,x),exact_squared_distance=str(q)))
 assert candidates
 chosen=min(candidates,key=lambda z:(z['loss'],z['cost'],z['leaf']))
 return torch.as_tensor(chosen['point']),dict(status='enumerated_single_tree_loss',analytic=True,candidates=candidates,selected_leaf=chosen['leaf'],objective=chosen['loss'],lower_bound=chosen['loss'],global_certificate=True)

def run_orpm(root,cell,d):
 mm,x,c,meta=bank(root,cell['dataset'],cell['models'],cell['seed'],cell['person']);m=len(mm);b=cell['budget'];start=time.perf_counter()
 oracle=rt.TreeOracle(mm,x,b,c,time_limit=1,abs_gap=1e-5,seed=cell['seed'],artifact_dir=d/'orpm_solver')
 logw=np.full(m,-np.log(m));eta=np.sqrt(np.log(m)/200);best=x.clone();bl=loss(mm,x);allpoints=[];allloss=[];infos=[]
 try:
  for t in range(1,101):
   deadline_check(root);write(root/'stage.json',dict(stage='orpm',utc=now(),key=cell['key'],round=t))
   w=np.exp(logw);point,info=checked_solve(oracle,w,known_points=(best,));l=loss(mm,point);info['artifact_subdir']='orpm_solver'
   if allloss:
    cached=np.asarray(allloss)@w;k=int(np.argmin(cached))
    if info['lower_bound']>float(cached[k])+1e-7:
     audit=audit_encoding(oracle,allpoints[k],'weighted_loss')
     if not audit['passed']:raise RuntimeError('Earlier action failed full encoded witness audit')
     info.update(lower_bound=0.,certificate_rejected=True,witness_audit=audit,global_certificate=False,certificate_rejection='bound exceeds earlier feasible action at current weights')
   if (float(l.max()),cost(point,x))<(float(bl.max()),cost(best,x)):best=point.clone();bl=l.copy()
   allpoints.append(point.numpy().copy());allloss.append(l);infos.append(info)
   append(d/'rounds.jsonl',dict(round=t,weights=w,point=point.numpy(),losses=l,best_point=best.numpy(),best_losses=bl,best_cost=cost(best,x),joint_valid=bool((rt.probabilities(mm,best)>=.5).all()),oracle=info))
   logw+=eta*l;logw-=np.max(logw);logw-=np.log(np.exp(logw).sum())
  if getattr(oracle,'excluded_budget_leaf_combinations',None):oracle.problem.write(str(d/'orpm_solver/final_formulation.mps'))
 finally:oracle.close()
 pack_models(d)
 return outcome(root,cell,'ORPM',best,reused=False,analytic=False,rounds=100,seconds=time.perf_counter()-start,timing_scope='new_complete_method_wall',objective=float(bl.max()),objective_lower=max(z['lower_bound'] for z in infos),**stats(infos))

def direct(root,cell,method,d):
 mm,x,c,meta=bank(root,cell['dataset'],cell['models'],cell['seed'],cell['person']);start=time.perf_counter();objective='minimax' if method=='DirectMinimax' else 'weighted_loss';w=None if method=='DirectMinimax' else np.ones(len(mm))/len(mm)
 oracle=rt.TreeOracle(mm,x,cell['budget'],c,time_limit=30,abs_gap=1e-5,seed=cell['seed'],artifact_dir=d/'solver')
 try:
  point,info=checked_solve(oracle,w,objective,known_points=())
  # Separate post-solve certificate audit; never change the native returned point using ORPM.
  prior=read(method_dir(root,cell,'ORPM')/'outcome.json')['point'];pl=loss(mm,prior);value=float(pl.max()) if method=='DirectMinimax' else float(pl@w)
  if info['lower_bound']>value+1e-7:
   audit=audit_encoding(oracle,prior,objective)
   if not audit['passed']:raise RuntimeError('Post-solve ORPM bound witness failed encoded audit')
   info.update(lower_bound=0.,certificate_rejected=True,certificate_rejection='postsolve_ORPM_witness_bound_audit_only',witness_audit=audit,global_certificate=False)
  if getattr(oracle,'excluded_budget_leaf_combinations',None):oracle.problem.write(str(d/'solver/final_formulation.mps'))
 finally:oracle.close()
 pack_models(d);write(d/'solver_outcome.json',dict(point=point.numpy(),oracle=info,orpm_warm_start=False,returned_ORPM_candidate=False))
 return outcome(root,cell,method,point,reused=False,analytic=False,seconds=time.perf_counter()-start,timing_scope='new_complete_method_wall',objective=info['objective'],objective_lower=info['lower_bound'],**stats([info]))

def run_method(root,cell,method):
 d=method_dir(root,cell,method);d.mkdir(parents=True,exist_ok=True)
 if (d/'outcome.json').exists():return read(d/'outcome.json')
 start=time.perf_counter();mm,x,c,meta=bank(root,cell['dataset'],cell['models'],cell['seed'],cell['person'])
 stage(root,'method',key=cell['key'],method=method)
 if cell['budget']==0:
  result=outcome(root,cell,method,x,reused=False,analytic=True,status='zero_budget_factual',seconds=time.perf_counter()-start,timing_scope='analytic_zero',**stats([]))
 elif method=='ORPM':result=run_orpm(root,cell,d)
 elif method in ['UniformTree','DirectMinimax']:result=direct(root,cell,method,d)
 else:
  cache=Path(root)/'reference_cache'/f"{cell['dataset']}_s{cell['seed']}_p{cell['person']}_b{cell['budget']:g}.json"
  if not cache.exists():
   point,info=single_reference(mm[0],x,cell['budget'],c);write(cache,dict(point=point.numpy(),info=info,seconds=time.perf_counter()-start))
   append(root/'timing.jsonl',dict(kind='reference_shared',utc=now(),dataset=cell['dataset'],seed=cell['seed'],person=cell['person'],budget=cell['budget'],seconds=time.perf_counter()-start))
  saved=read(cache)
  result=outcome(root,cell,method,saved['point'],reused=False,analytic=True,status=saved['info']['status'],seconds=saved['seconds'],timing_scope='shared_reference_once_per_person_budget',shared_reference=str(cache.relative_to(root)),objective=saved['info']['objective'],objective_lower=saved['info']['lower_bound'],**stats([]))
 write(d/'outcome.json',result)
 if method!='SingleTreeReference' and cell['budget']>0:
  append(root/'timing.jsonl',dict(kind='method',utc=now(),key=cell['key'],dataset=cell['dataset'],seed=cell['seed'],person=cell['person'],models=cell['models'],budget=cell['budget'],method=method,seconds=result['seconds']))
 return result
