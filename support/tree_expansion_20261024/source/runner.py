from common import *
import faulthandler,traceback,gzip,shutil
from prepare import prepare,link
from functools import lru_cache
from reuse import restore_exclusions,import_reference,verify_preservation

def gap(P,L,U):
 if L>U+1e-7 or P<L-1e-7:raise RuntimeError(f'Contradictory gap bounds: P={P} L={L} U={U}')
 return [max(0.,float(P-U)),max(0.,float(P-L))]
def trajectory(root,ds,m,b,ordinal,seed,person):
 deadline_check(root);protocol=read(root/'protocol.json');expected=read(root/'task_plan.json')['expected_trajectories'];models,x0,c,meta=bank(root,ds,m,seed,person);key=cell_key(ds,seed,person,m,b);out=root/'trajectories'/key
 if out.exists():raise RuntimeError('Trajectory already exists; no implicit rerun or resume')
 out.mkdir(parents=True);historical_seconds=0.;oracle_subdir='orpm_solver';diagnostic_subdir='diagnostic_solver'
 reuse=cached_reuse(root);prefix=reuse['prefixes'].get(key);saved=[]
 if prefix:
  src=Path(prefix['directory']);saved=rows(src/'rounds.jsonl');assert len(saved)==100 and sha(src/'rounds.jsonl')==prefix['sha256']
  link(src/'rounds.jsonl',out/'source_prefix.jsonl');link(src/'orpm_solver',out/'saved_orpm_solver')
  write(out/'reuse.json',dict(prefix=prefix,source_bytes_retained=True,native_optimizer_rebuilt=True))
 begin=time.perf_counter();stage(root,'trajectory',dataset=ds,models=m,budget=b,trajectory=ordinal,total=expected)
 write(out/'setting.json',dict(dataset=ds,models=m,budget=b,seed=seed,person_source_id=meta['selected_source_ids'][person],factual=x0.numpy(),immutable_indices=c.immutable,rounds=200,eta=float(np.sqrt(np.log(m)/200))))
 oracle=rt.TreeOracle(models,x0,b,c,time_limit=1,abs_gap=1e-5,seed=seed,artifact_dir=out/oracle_subdir)
 if saved:
  restore_exclusions(oracle,saved)
 buildtime=time.perf_counter()-begin
 evaluator=rt.TreeOracle(models,x0,b,c,time_limit=2,abs_gap=1e-5,seed=seed,artifact_dir=out/diagnostic_subdir)
 diagbuild=time.perf_counter()-begin-buildtime
 eta=np.sqrt(np.log(m)/200);logw=np.full(m,-np.log(m));best=x0.clone();factual_loss=loss(models,x0);bestloss=factual_loss.copy()
 allpoints=[];alllosses=[];allweights=[];lbs=[];firsthit=None;diag_seconds=0.;solve_seconds=0.;eval_solve_seconds=0.;records=[]
 stage(root,'TRAJECTORY_STARTED',key=key)
 try:
  for t in range(1,201):
   deadline_check(root)
   stage_data=dict(dataset=ds,seed=seed,person=person,models=m,budget=b,trajectory=ordinal,round=t)
   write(root/'stage.json',dict(stage='orpm_oracle',utc=now(),**stage_data))
   w=np.exp(logw)
   if t<=len(saved):
    prior=saved[t-1];assert np.allclose(w,prior['weights'],atol=1e-12,rtol=0)
    x=torch.as_tensor(prior['point'],dtype=torch.float32);info=dict(prior['oracle']);info['artifact_subdir']='saved_orpm_solver';l=loss(models,x)
    assert np.allclose(l,prior['losses'],atol=1e-12,rtol=0)
   else:
    x,info=checked_solve(oracle,w,known_points=(best,));info['artifact_subdir']=oracle_subdir;l=loss(models,x);solve_seconds+=info['solver_seconds']
   if alllosses:
    cached=np.asarray(alllosses)@w;k=int(np.argmin(cached))
    if info['lower_bound']>float(cached[k])+1e-7:
     if t<=len(saved):raise RuntimeError('Preserved prefix lower bound is inconsistent; no silent mutation')
     au=audit_encoding(oracle,allpoints[k],'weighted_loss')
     if not au['passed']:raise RuntimeError('Earlier action failed full audit after contradictory lower bound')
     info.update(lower_bound=0.,certificate_rejected=True,witness_audit=au,global_certificate=False,certificate_rejection='bound exceeds earlier feasible action at current weights')
   if (float(l.max()),float(np.linalg.norm(x.numpy().astype(float)-x0.numpy().astype(float))))<(float(bestloss.max()),float(np.linalg.norm(best.numpy().astype(float)-x0.numpy().astype(float)))):best=x.clone();bestloss=l.copy()
   allpoints.append(x.numpy().copy());alllosses.append(l);allweights.append(w);lbs.append(info['lower_bound'])
   wb=np.mean(allweights,axis=0);L=float(np.mean(lbs));mat=np.asarray(alllosses);U=min(float(factual_loss@wb),float(np.min(mat@wb)));diag=None
   if t in protocol['checkpoints']:
    write(root/'stage.json',dict(stage='independent_checkpoint',utc=now(),**stage_data));st=time.perf_counter()
    dx,di=checked_solve(evaluator,wb,known_points=(best,));di['artifact_subdir']=diagnostic_subdir;dl=loss(models,dx);upper=min(U,float(dl@wb))
    if di['lower_bound']>upper+1e-7:
     witnesses=[x0.numpy(),dx.numpy(),*allpoints];k=int(np.argmin([float(loss(models,z)@wb) for z in witnesses]));au=audit_encoding(evaluator,witnesses[k],'weighted_loss')
     if not au['passed']:raise RuntimeError('Checkpoint contradictory bound witness failed full audit')
     di.update(lower_bound=0.,certificate_rejected=True,witness_audit=au,global_certificate=False,certificate_rejection='checkpoint bound exceeds saved action')
    L=max(L,di['lower_bound']);U=upper;elapsed=time.perf_counter()-st;diag_seconds+=elapsed;eval_solve_seconds+=di['solver_seconds']
    diag=dict(round=t,weights=wb,point=dx.numpy(),losses=dl,oracle=di,seconds=elapsed);append(out/'diagnostics.jsonl',diag)
   P=float(mat.mean(axis=0).max());Pb=float(bestloss.max());gm=gap(P,L,U);gs=gap(Pb,L,U)
   if gm[1]<=.005 and firsthit is None:firsthit=t
   row=dict(round=t,weights=w,point=x.numpy(),losses=l,best_point=best.numpy(),best_losses=bestloss,best_cost=float(np.linalg.norm(best.numpy().astype(float)-x0.numpy().astype(float))),joint_valid=bool((rt.probabilities(models,best)>=.5).all()),mixture_primal=P,point_primal=Pb,dual_lower=L,dual_upper=U,mixture_gap=gm,point_gap=gs,first_certified_round=firsthit,oracle=info,checkpoint=diag is not None)
   append(out/'rounds.jsonl',row);records.append(row)
   logw=logw+eta*l;logw-=np.max(logw);logw-=np.log(np.exp(logw).sum())
   if t<=len(saved):
    for field in ['round','weights','point','losses','best_point','best_losses','best_cost','joint_valid']:
     assert clean(row[field])==prior[field],('prefix_field_changed',field,t)
   if t%10==0 or t==1 or t==148:
    write(root/'progress.json',dict(status='running',completed_trajectories=ordinal-1,expected_trajectories=expected,completed_rounds=(ordinal-1)*200+t,preserved_complete_trajectories=1620,preserved_prefix_rounds=162900,new_orpm_calls=new_calls_before(root,ordinal)+max(0,t-len(saved)),expected_new_orpm_calls=1673100,expected_rounds=expected*200,current=stage_data,utc=now()))
  algorithm_wall=time.perf_counter()-begin+historical_seconds-diag_seconds-diagbuild
 finally:
  for obj,sub in [(oracle,oracle_subdir),(evaluator,diagnostic_subdir)]:
   if getattr(obj,'excluded_budget_leaf_combinations',None):obj.problem.write(str(out/sub/'final_formulation.mps'))
  oracle.close();evaluator.close()
 stage(root,'direct_minimax_reference',dataset=ds,models=m,budget=b)
 refstart=time.perf_counter()
 if key in reuse['references']:
  ref=import_reference(root,out,models,x0,c,b,best,reuse['references'][key]);rx=torch.as_tensor(ref['point'],dtype=torch.float32)
 else:
  reference=rt.TreeOracle(models,x0,b,c,time_limit=30,abs_gap=1e-5,seed=seed,artifact_dir=out/'reference_solver')
  try:
   rx,ri=checked_solve(reference,objective='minimax',known_points=(best,))
   ref=dict(point=rx.numpy(),losses=loss(models,rx),oracle=ri,seconds=time.perf_counter()-refstart,orpm_warm_start=False)
  finally:
   if getattr(reference,'excluded_budget_leaf_combinations',None):reference.problem.write(str(out/'reference_solver/final_formulation.mps'))
   reference.close()
 write(out/'reference.json',ref)
 # Portable safe export: only Python gzip handles compression, never the solver.
 for p in out.glob('*solver*/*.mps'):
  if p.parent.is_symlink():continue
  with p.open('rb') as src,gzip.open(str(p)+'.gz','wb') as dst:shutil.copyfileobj(src,dst)
  if sha(p)!=hashlib.sha256(gzip.open(str(p)+'.gz','rb').read()).hexdigest():raise RuntimeError('MPS compression checksum mismatch')
  p.unlink()
 ri=ref['oracle'];point_excess=gap(float(bestloss.max()),ri['lower_bound'],min(float(loss(models,rx).max()),float(bestloss.max())))
 summary=dict(key=key,dataset=ds,seed=seed,person=person,models=m,budget=b,rounds=200,first_certified_round=firsthit,final_mixture_gap=records[-1]['mixture_gap'],final_point_gap=records[-1]['point_gap'],point_optimality_excess=point_excess,validity_50=records[49]['joint_valid'],validity_100=records[99]['joint_valid'],validity_200=records[-1]['joint_valid'],mixture_gap_100=records[99]['mixture_gap'],cost_200=records[-1]['best_cost'],reference_status=ri['solver_status'],reference_certified=ri['global_certificate'],orpm_solver_seconds=solve_seconds,diagnostic_solver_seconds=eval_solve_seconds,reference_solver_seconds=0. if key in reuse['references'] else ri['solver_seconds'],algorithm_wall_seconds=algorithm_wall,diagnostic_wall_seconds=diag_seconds+diagbuild,reference_wall_seconds=ref['seconds'],total_seconds=time.perf_counter()-begin+historical_seconds,new_orpm_calls=200-len(saved),preserved_orpm_calls=len(saved),reused_reference=key in reuse['references'],new_reference_calls=int(key not in reuse['references']))
 write(out/'completed.json',summary);append(root/'timing.jsonl',dict(kind='trajectory',utc=now(),**summary));append(root/'summaries.jsonl',summary)
 write(root/'progress.json',dict(status='running',completed_trajectories=ordinal,expected_trajectories=expected,completed_rounds=ordinal*200,preserved_complete_trajectories=1620,preserved_prefix_rounds=162900,new_orpm_calls=new_calls_before(root,ordinal+1),expected_new_orpm_calls=1673100,expected_rounds=expected*200,utc=now()))
 from eta import remaining
 
 if ordinal%30==0:write(root/'runtime_estimate.json',remaining(root))
 stage(root,'TRAJECTORY_COMPLETED',key=key,completed=ordinal,seconds=summary['total_seconds'])


@lru_cache(maxsize=8)
def cached_reuse(root):return read(root/'reuse_manifest.json')
@lru_cache(maxsize=8)
def call_prefix_counts(root):
 reuse=cached_reuse(root);plan=read(root/'task_plan.json');counts=[0]
 for z in plan['cells']:
  counts.append(counts[-1]+(0 if z['key'] in reuse['complete'] else 100 if z['key'] in reuse['prefixes'] else 200))
 return counts

def new_calls_before(root,ordinal):return call_prefix_counts(root)[ordinal-1]

def run(root):
 root=Path(root);prepare(root);plan=read(root/'task_plan.json')
 for ordinal,cell in enumerate(plan['cells'],1):
  if cell['key'] in cached_reuse(root)['complete']:continue
  trajectory(root,cell['dataset'],cell['models'],cell['budget'],ordinal,cell['seed'],cell['person'])
 stage(root,'independent_verification')
 from verify import verify
 verify(root)
 verify_preservation(root)
 stage(root,'analysis_and_reporting')
 from analyze import analyze
 analyze(root)
 deadline_check(root)
 write(root/'progress.json',dict(status='completed',completed_trajectories=plan['expected_trajectories'],expected_trajectories=plan['expected_trajectories'],completed_rounds=plan['expected_rounds'],expected_rounds=plan['expected_rounds'],new_orpm_calls=plan['new_orpm_calls'],preserved_rounds=plan['preserved_rounds'],utc=now()))
 stage(root,'TREE_CONVERGENCE_EXPANSION_COMPLETED',trajectories=plan['expected_trajectories'])

if __name__=='__main__':
 root=Path(sys.argv[1]);native=(root/'native_traceback.log').open('a');faulthandler.enable(native,all_threads=True)
 try:run(root)
 except BaseException as e:
  write(root/'failure.json',dict(utc=now(),type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),stage=read(root/'stage.json') if (root/'stage.json').exists() else None));traceback.print_exc();raise
