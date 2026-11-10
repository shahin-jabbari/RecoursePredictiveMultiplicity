from experiment import *
from methods import run_orpm,direct,outcome,pack_models,stats
from fractions import Fraction
import gurobipy as gp

def reference_solve(mm,x,b,c,seed,d,known):
 """One native minimum-cost solve, with post-solve audits only; no ORPM replacement."""
 if rt.probabilities(mm,x)[0]>=.5:
  return x.clone(),dict(status='factual_already_accepted',analytic=True,success=True,solver_called=False,objective=0.,lower_bound=0.,global_certificate=True)
 oracle=rt.TreeOracle(mm,x,b,c,time_limit=30,abs_gap=1e-5,seed=seed,artifact_dir=d/'solver')
 try:
  for var in oracle.problem.getVars():var.Start=gp.GRB.UNDEFINED
  point=oracle.solve(objective='min_cost');info=dict(point.recourse_info);model=oracle.problem
  try:raw=float(model.ObjBound);raw=raw if np.isfinite(raw) else None
  except Exception:raw=None
  info.update(raw_solver_bound=raw,solution_count=int(model.SolCount),certificate_rejected=False,incumbent_rejected=False,analytic=False,solver_called=True,success=False,ORPM_warm_start=False,returned_ORPM_candidate=False)
  if info['status']=='incumbent_failed_validation':
   v=info['verification']
   if not(v.get('leaf_box_consistent') and v.get('predictions_match') and v.get('threshold_satisfied') and v.get('candidate_feasible') is False):raise RuntimeError('Invalid forest reference encoding/prediction/threshold')
   from boundary_recovery import proof_for_leaves
   proof,_=proof_for_leaves(oracle,info['selected_leaves'])
   info.update(incumbent_rejected=True,budget_rejection_proof=proof,certificate_rejected=True,lower_bound=0.,global_certificate=False,objective=None,returned_action_verified=True,additional_optimization_calls=0)
   point=x.clone()
  if info['incumbent_verified']:
   if not point_check(point,x,b,c) or rt.probabilities(mm,point)[0]<.5:raise RuntimeError('Minimum-cost reference invalid returned point')
   if not audit_encoding(oracle,point,'min_cost')['passed']:raise RuntimeError('Forest reference failed complete encoded witness audit')
   info['success']=True
  witnesses=[]
  if info['success']:witnesses.append(point)
  for z in known:
   if point_check(z,x,b,c) and rt.probabilities(mm,z)[0]>=.5:witnesses.append(torch.as_tensor(z,dtype=torch.float32))
  if witnesses:
   witness=min(witnesses,key=lambda z:cost(z,x));upper=cost(witness,x)**2
   if info['solver_status'] in [3,4,5] or raw is not None and raw>upper+1e-7 or info['bound_inconsistent_with_verified_candidate']:
    audit=audit_encoding(oracle,witness,'min_cost')
    if not audit['passed']:raise RuntimeError('Contradictory forest reference certificate and invalid encoded witness')
    info.update(certificate_rejected=True,witness_audit=audit,lower_bound=0.,global_certificate=False,certificate_rejection='postsolve feasible witness; native returned point unchanged')
  info['lower_bound']=max(0.,info.get('lower_bound') or 0.)
  if info['solver_status'] in [4,5]:info['lower_bound']=0.;info['global_certificate']=False
  if info['incumbent_rejected']:info['incumbent_verified']=False;info['global_certificate']=False
  info['not_global_infeasibility']=info['solver_status']==3
  info['status_label']='accepted_native_point' if info['success'] else ('numerical_exclusion_within_budget' if info['solver_status']==3 and not info['certificate_rejected'] else 'unresolved_no_accepted_native_point')
  # If no accepted native action, factual is solely an explicit failed-reference output.
  if not info['success']:point=x.clone();info['objective']=None
  return point,info
 finally:oracle.close();pack_models(d)

def run_method(root,cell,method):
 d=method_dir(root,cell,method);d.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
 if (d/'outcome.json').exists():raise RuntimeError('Existing scientific output: no implicit rerun')
 mm,x,c,meta=bank(root,cell['dataset'],cell['models'],cell['seed'],0)
 stage(root,'forest_method',key=cell['key'],method=method)
 if cell['budget']==0:
  result=outcome(root,cell,method,x,reused=False,analytic=True,status='zero_budget_factual',seconds=0.,timing_scope='analytic_zero',**stats([]))
 elif method=='ORPM':result=run_orpm(root,cell,d)
 elif method in ['UniformForest','DirectMinimax']:result=direct(root,cell,method,d)
 else:
  key=f"{cell['dataset']}_s{cell['seed']}_p0_b{cell['budget']:g}";cache=root/'reference_cache'/key/'result.json'
  if not cache.exists():
   prior=read(method_dir(root,cell,'ORPM')/'outcome.json')['point']
   pt,info=reference_solve(mm[:1],x,cell['budget'],c,cell['seed'],cache.parent,[prior]);elapsed=time.perf_counter()-start
   write(cache,dict(point=pt.numpy(),info=info,seconds=elapsed))
   append(root/'timing.jsonl',dict(kind='reference_shared',dataset=cell['dataset'],seed=cell['seed'],person=0,budget=cell['budget'],seconds=elapsed,utc=now()))
  saved=read(cache);info=saved['info'];infos=[info] if info['solver_called'] else []
  result=outcome(root,cell,method,saved['point'],reused=False,analytic=info['analytic'],reference_success=info['success'],seconds=saved['seconds'],timing_scope='shared_reference_once',shared_reference=str(cache.relative_to(root)),objective=info['objective'],objective_lower=info['lower_bound'],**stats(infos))
 write(d/'outcome.json',result)
 if cell['budget']>0 and method!='SingleForestReference':append(root/'timing.jsonl',dict(kind='method',dataset=cell['dataset'],seed=cell['seed'],person=0,models=cell['models'],budget=cell['budget'],key=cell['key'],method=method,seconds=result['seconds'],utc=now()))
 return result
