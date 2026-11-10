import faulthandler,os,sys,time,traceback
from pathlib import Path
import numpy as np
from common import *
from solver import environment,solve
from extension import check_preservation

def run(root):
 root=Path(root);cfg=read(root/'protocol.json');started=time.perf_counter();completed=cfg['recovery_completed_tasks'];preserved={tuple(x) for x in read(root/'recovery_completed_keys.json')}
 def mark(**kw):save(root/'stage.json',dict(updated_utc=utc(),**kw))
 def progress(status,**kw):save(root/'progress.json',dict(status=status,completed_tasks=completed,preserved_tasks=cfg['preserved_tasks'],new_completed_tasks=completed-cfg['preserved_tasks'],completed_since_recovery=completed-cfg['recovery_completed_tasks'],expected_tasks=cfg['expected_tasks'],expected_new_tasks=cfg['expected_new_tasks'],updated_utc=utc(),elapsed_seconds=time.perf_counter()-started,**kw))
 for p,h in read(root/'input_sha256.json').items():assert sha(p)==h
 save(root/'runtime_preservation.json',check_preservation(root,check_original=True))
 with environment() as env:
  for c in read(root/'cohort_plan.json'):
   directory=root/'cases'/c['key'];f=arrays(directory/'factual.npz');x0=f['factual'];points=arrays(directory/'saved_orpm_points.npz')['points']
   nets={j:load_net(root/c['bank']/f'model_{j:04d}.npz') for j in range(24,48)}
   tasks=[(f'single_{j}',[j],'singleton_shallow' if j<36 else 'singleton_deep') for j in range(24,48)]+[(f'joint_{n}',indices(n),'joint_'+str(n)) for n in [6,12,24]]
   for task,js,kind in tasks:
    if (c['key'],task) in preserved:continue
    dest=directory/'tasks'/task;dest.mkdir(parents=True)
    context=dict(case=c['key'],dataset=c['dataset'],seed=c['seed'],instance_id=c['instance_id'],task=task,kind=kind)
    progress('optimizing',**context)
    point,info=solve([nets[j] for j in js],x0,c['maximum_budget'],c['free'],f['lower'],f['upper'],env,cfg['maximum_optimization_seconds_per_solve'],dest,points,mark=lambda **kw:mark(**context,**kw))
    np.savez_compressed(dest/'point.npz',point=point)
    save(dest/'result.json',dict(**context,indices=js,info=info))
    append(root/'timing.jsonl',dict(**context,updated_utc=utc(),seconds=info['seconds'],optimization_seconds=info['optimization_seconds'],solver_status=info['solver_status'],certified=info['certified']))
    completed+=1
   from analysis import timing_estimate
   save(root/'runtime_estimate.json',timing_estimate(root));progress('case_completed',case=c['key']);print(utc(),'CASE_COMPLETED',completed,'/',cfg['expected_tasks'],c['key'],flush=True)
 progress('verifying');mark(stage='independent_verification')
 from verify import verify
 save(root/'verification.json',verify(root))
 progress('reporting');mark(stage='analysis_and_reports')
 from analysis import analyze
 analyze(root)
 save(root/'verification.json',verify(root,True))
 save(root/'extension_verification.json',check_preservation(root,check_original=True))
 progress('completed');mark(stage='completed')
 print(utc(),'NEURAL_MULTIPLICITY_EXPANSION_COMPLETED',completed,flush=True)
if __name__=='__main__':
 root=Path(sys.argv[1]);native=open(root/'native_traceback.log','a',buffering=1);faulthandler.enable(file=native,all_threads=True)
 try:run(root)
 except BaseException as e:
  save(root/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),utc=utc(),stage=read(root/'stage.json') if (root/'stage.json').exists() else None))
  traceback.print_exc();sys.exit(1)
