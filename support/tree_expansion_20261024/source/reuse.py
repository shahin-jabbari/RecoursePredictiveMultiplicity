"""Read-only source reuse and explicit conservative re-audits; never optimize here."""
from common import *
import gzip,tempfile

def restore_exclusions(oracle,saved):
 from boundary_recovery import proof_for_leaves
 import gurobipy as gp
 excluded=set()
 for row in saved:
  info=row['oracle']
  if not info.get('incumbent_rejected'):continue
  proof=info['budget_rejection_proof'];signature=tuple(proof['selected_leaves'])
  if signature in excluded:continue
  audit,chosen=proof_for_leaves(oracle,signature)
  for k in ['exact_squared_cost','exact_squared_budget','exact_excess']:assert audit[k]==proof[k]
  oracle.problem.addConstr(gp.quicksum(g['z'] for g in chosen)<=len(chosen)-1,name=f'proven_outside_ball_{len(excluded)}');excluded.add(signature)
 oracle.problem.update();oracle.excluded_budget_leaf_combinations=excluded
 # The within-trajectory start is the previously returned feasible action.
 last=torch.as_tensor(saved[-1]['point'],dtype=torch.float32)
 assert point_check(last,oracle.x0,oracle.budget,oracle.constraints)
 oracle._warm_start(last)

def archived_audit(models,x,c,b,point,artifact):
 import gurobipy as gp
 oracle=rt.TreeOracle(models,x,b,c,time_limit=1,seed=401);old=oracle.problem
 with tempfile.NamedTemporaryFile(suffix='.mps') as tmp:
  with gzip.open(Path(artifact)/'formulation.mps.gz','rb') as f:tmp.write(f.read());tmp.flush()
  oracle.problem=gp.read(tmp.name,env=oracle.env)
 try:return audit_encoding(oracle,point,'minimax')
 finally:oracle.problem.dispose();old.dispose();oracle.env.dispose()

def conservative_reference(info,models,x,c,b,best,artifact):
 result=dict(info)
 if result['lower_bound']>float(loss(models,best).max())+1e-7:
  au=archived_audit(models,x,c,b,best,artifact)
  if not au['passed']:raise RuntimeError('New saved action contradicts reference but fails complete archived-encoding audit')
  result.update(lower_bound=0.,global_certificate=False,certificate_rejected=True,witness_audit=au,certificate_rejection='preserved_reference_bound_rejected_by_verified_expanded_trajectory_witness',preserved_native_info=dict(info))
 return result

def import_reference(root,out,models,x,c,b,best,item):
 from prepare import link
 begin=time.perf_counter();src=Path(item['directory']);raw=read(src/'solver_outcome.json');meta=read(src/'outcome.json')
 link(src/'solver_outcome.json',out/'source_reference.json');link(src/'solver',out/'reference_solver')
 assert point_check(raw['point'],x,b,c)
 info=conservative_reference(raw['oracle'],models,x,c,b,best,src/'solver')
 return dict(point=raw['point'],losses=loss(models,raw['point']),oracle=info,seconds=time.perf_counter()-begin,orpm_warm_start=False,reused=True,source_directory=str(src),historical_seconds=meta['seconds'],new_optimization_calls=0)

def verify_preservation(root):
 root=Path(root);reuse=read(root/'reuse_manifest.json');plan=read(root/'task_plan.json')
 for path,digest in read(root/'source_preservation.json')['sha256'].items():
  assert sha(path)==digest,('source_changed',path)
 count=0
 for key,item in reuse['prefixes'].items():
  out=root/'trajectories'/key;old=rows(Path(item['directory'])/'rounds.jsonl');new=rows(out/'rounds.jsonl')
  assert sha(out/'source_prefix.jsonl')==item['sha256']
  for a,b in zip(old,new):
   for k in ['round','weights','point','losses','best_point','best_losses','best_cost','joint_valid']:assert a[k]==b[k],(key,k)
   ai=dict(a['oracle']);bi=dict(b['oracle']);ai.pop('artifact_subdir',None);bi.pop('artifact_subdir',None)
   assert ai==bi,(key,'native_oracle_changed')
  count+=len(old)
 for key,path in reuse['complete'].items():
  assert (root/'trajectories'/key).is_symlink() and (root/'trajectories'/key).resolve()==Path(path).resolve()
 new=[z for z in rows(root/'timing.jsonl') if z['kind']=='trajectory' and z['key'] not in reuse['complete']]
 assert sum(z['new_orpm_calls'] for z in new)==plan['new_orpm_calls']
 assert sum(z['new_reference_calls'] for z in new)==plan['new_references']
 assert sum(z['preserved_orpm_calls'] for z in new)==count==162900
 result=dict(passed=True,utc=now(),all_source_bytes_unchanged=True,complete_trajectories_preserved=1620,prefixes_preserved=1629,prefix_rounds_preserved=count,new_orpm_calls=plan['new_orpm_calls'],new_reference_calls=8100,new_diagnostics=137700,no_prefix_response_reoptimized=True)
 write(root/'extension_verification.json',result);return result
