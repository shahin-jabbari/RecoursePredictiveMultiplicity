from pathlib import Path
from datetime import datetime,timezone
import sys,shutil,html,gzip,hashlib,re,time
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'source'))
from common import *
from recovery import FAILED,check_recovery_preservation
from numerical_guard import encoded_witness_audit,reject_contradicted_certificate
from solver import environment,network_bounds
import gurobipy as gp
assert read(HERE/'validation_evidence/validation.json')['passed']
assert read(HERE/'diagnostics/saved_results_audit.json')['passed']
assert read(FAILED/'worker_exit.json')['completed'] is False
original_deadline=read(FAILED/'deadline.json');assert time.time()<original_deadline['deadline_epoch']
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'_recovery'
root=BASE/'cost-of-multiplicity/phase_neural_bounded_recovery_v1_2/results'/stamp
droot=BASE/'disparity-of-multiplicity/phase_neural_bounded_recovery_v1_2/results'/stamp
root.mkdir(parents=True);droot.mkdir(parents=True)
save(HERE/'active_run.json',dict(root=str(root),disparity=str(droot)))
# Check and preserve the failed run rather than editing its source or outcomes.
for p,h in read(FAILED/'freeze.json')['sha256'].items():assert sha(FAILED/p)==h,p
source_manifest={str(p.relative_to(FAILED)):sha(p) for p in FAILED.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
save(root/'failed_source_sha256.json',source_manifest)
for name in ['blocks','cases']:shutil.copytree(FAILED/name,root/name)
for name in ['cohort_plan.json','model_inventory.json','full_cohort.json','group_counts.json','source_compute_demand.json','saved_orpm_results.csv','input_sha256.json','pilot_source_sha256.json','preserved_case_keys.json','pilot_timing.jsonl','task_plan.json','timing.jsonl']:
 shutil.copy2(FAILED/name,root/name)
# Before deriving the interrupted result, freeze every prior copied input/result.
prior={str(p.relative_to(root)):sha(p) for name in ['blocks','cases'] for p in (root/name).rglob('*') if p.is_file()}
save(root/'copied_prior_sha256.json',prior)
shutil.copytree(HERE/'source',root/'source',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(HERE/'validation_evidence',root/'validation_evidence')
shutil.copytree(HERE/'diagnostics',root/'diagnostics',ignore=shutil.ignore_patterns('original_problem.mps','__pycache__'))
shutil.copy2(__file__,root/'preparation_script.py')
shutil.copy2(FAILED/'PROTOCOL.txt',root/'original_protocol.txt')
save(root/'original_deadline.json',original_deadline);save(root/'deadline.json',original_deadline)
cfg=read(FAILED/'protocol.json');cfg.update(version='1.2',study='neural_cost_and_disparity_bounded_recovery',disparity_result=str(droot),source_failed_run=str(FAILED),absolute_deadline_epoch=original_deadline['deadline_epoch'],absolute_deadline_utc=original_deadline['deadline_utc'],recovery_completed_tasks=1323,previous_completed_tasks=1322,recovered_interrupted_tasks=1,recovery_remaining_tasks=5157,numerical_failure_rule='Reject certificates contradicted by independently checked feasible actions; require exported-model witness consistency, use zero lower bound and verified feasible upper bound, uncertified outcome. Encoding errors still stop. No retry or solver parameter change.',limits='Original absolute cutoff 2026-10-22T19:11:17Z retained. No fresh wall-clock allowance. Ten seconds per task, one thread, unchanged model and parameters. No study optimization retry for the interrupted task.')
save(root/'protocol.json',cfg)
# Recover interrupted output from the ORIGINAL logged optimizer run, never from replays.
c=next(c for c in read(root/'cohort_plan.json') if c['key']=='Synthetic_402_1728');folder=root/'cases'/c['key'];dest=folder/'tasks/joint_24';f=arrays(folder/'factual.npz');pts=arrays(folder/'saved_orpm_points.npz')['points'];js=indices(24);nets=[load_net(root/c['bank']/f'model_{j:04d}.npz') for j in js]
valid=[x for x in pts if feasible(x,f['factual'],c['maximum_budget'],c['free'],f['lower'],f['upper']) and logits(nets,x).min()>=0]
point=min(valid,key=lambda x:np.linalg.norm(x-f['factual']));cost=float(np.linalg.norm(point-f['factual']))
log=(dest/'solver.log').read_text();assert 'Model is infeasible' in log and 'Solution count 0' in log
match=re.search(r'Explored (\d+) nodes .* in ([0-9.]+) seconds',log);assert match
optimization_seconds=float(match.group(2));node_count=int(match.group(1))
with environment() as env:
 import tempfile
 with tempfile.TemporaryDirectory(prefix='recovery-mps-audit-') as tmp:
  p=Path(tmp)/'problem.mps'
  with gzip.open(dest/'problem.mps.gz','rb') as src,open(p,'wb') as dst:shutil.copyfileobj(src,dst)
  digest=sha(p);m=gp.read(str(p),env=env)
  audit=encoded_witness_audit(m,nets,f['factual'],c['maximum_budget'],c['free'],point);binaries=sum(v.VType=='B' for v in m.getVars());m.dispose()
start=datetime.fromisoformat(read(FAILED/'progress.json')['updated_utc']);end=datetime.fromisoformat(read(FAILED/'failure.json')['utc']);wall=(end-start).total_seconds()
info=dict(binary_count=binaries,neuron_bounds=[[(l.tolist(),u.tolist()) for l,u in network_bounds(net,f['factual'],c['maximum_budget'],c['free'])] for net in nets],solver_status=3,solver_solutions=0,normalized_objective=None,normalized_best_bound=None,optimization_seconds=optimization_seconds,seconds=wall,node_count=node_count,export_uncompressed_sha256=digest,available=True,native_incumbent=False,cost_upper=cost,capped_upper=min(cost,c['maximum_budget']),upper_source='saved_ORPM',forward_error=None,raw_point=None,recovered_from_original_run=True,recovery_optimization_seconds=0.,timing_reconstructed_from_logs=True,source_solver_log_sha256=sha(FAILED/'cases'/c['key']/'tasks/joint_24/solver.log'))
info=reject_contradicted_certificate(info,'infeasibility_contradicted_by_verified_witness',audit)
context=dict(case=c['key'],dataset=c['dataset'],seed=c['seed'],instance_id=c['instance_id'],task='joint_24',kind='joint_24')
np.savez_compressed(dest/'point.npz',point=point);save(dest/'result.json',dict(**context,indices=js,info=info))
append(root/'timing.jsonl',dict(**context,updated_utc=read(FAILED/'failure.json')['utc'],seconds=wall,optimization_seconds=optimization_seconds,solver_status=3,certified=False,recovered_without_resolve=True,timing_reconstructed_from_logs=True))
completed=[]
for c in read(root/'cohort_plan.json'):
 for p in (root/'cases'/c['key']/'tasks').glob('*/result.json'):completed.append([c['key'],p.parent.name])
assert len(completed)==1323 and len({tuple(x) for x in completed})==1323
save(root/'recovery_completed_keys.json',completed)
save(root/'recovery.json',dict(passed=True,prior_completed_tasks=1322,recovered_interrupted_tasks=1,pending_tasks=5157,previous_results_byte_identical=True,interrupted_task_not_resolved=True,original_deadline_retained=True,scientific_solver_parameters_unchanged=True,diagnostic_runs_excluded_from_study=True,utc=utc()))
check_recovery_preservation(root)
text=(root/'original_protocol.txt').read_text()+'''\n\nRECOVERY AMENDMENT 1.2 (October 21, 2026)\n\nThe first expansion stopped after 350 new tasks when Gurobi 12.0.2 declared Synthetic seed 402 person 1728, joint 24 models, infeasible despite a feasible saved ORPM action. That action has L2 cost 0.99999948 within radius 5, minimum logit 2.96094, and maximum exported-model row violation about 1.05e-14. Fixing all encoded variables to the action is feasible. The original settings reproduce the false infeasibility result; a diagnostic with cuts disabled times out instead. The precise internal numerical mechanism is not established. These diagnostic solves are excluded from scientific outcomes.\n\nAll 1,322 previously stored task results and the original failed run are preserved byte-for-byte. The interrupted task is reconstructed from its original exported model, status and log, with no additional study solve; its solver-reported infeasibility is explicitly rejected. There are 1,323 saved outcomes at recovery start and 5,157 tasks remaining. Original 350 timing rows remain an exact prefix; the reconstructed timing is labeled and excluded from rate fitting.\n\nFor all subsequent tasks, a solver infeasibility claim or cost lower bound contradicted by an independently verified feasible action is treated as a rejected numerical certificate. The action must also satisfy the saved optimization encoding; otherwise the run still stops on an integrity error. Rejected certificates supply NO infeasibility or positive cost lower-bound claim: cost lower bound is zero, upper bound comes from the verified feasible action, and the outcome stays uncertified. Raw status and certificate rejection are reported separately. No action constraint, model, cohort, optimization parameter, warm start, retry allowance or 10-second limit changes. This is an amendment to result validation and failure handling, not tuning solver performance.\n\nThe original absolute deadline remains 2026-10-22 19:11:17 UTC, or 3:11 p.m. EDT October 22. Recovery does not restart the 23-hour allowance; diagnosis, interruption and recovery consume the original calendar window. The independent supervisor includes verification/reporting and stops the whole process group. A cutoff remains incomplete, with partial results retained.\n'''
for d in [root,droot]:
 (d/'PROTOCOL.txt').write_text(text);(d/'PROTOCOL.html').write_text('<!doctype html><meta charset="utf-8"><title>Neural multiplicity recovery protocol</title><style>body{max-width:1000px;margin:3em auto;font:17px/1.6 system-ui;padding:0 2em}pre{white-space:pre-wrap}</style><pre>'+html.escape(text)+'</pre>')
 save(d/'shared_run.json',dict(computation=str(root),cost_report=str(root/'cost_report'),disparity_report=str(droot)))
# Mutable timing and deadline files are audited by prefix/absolute-value rules.
exclude={'timing.jsonl','deadline.json'}
frozen={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and str(p.relative_to(root)) not in exclude}
save(root/'freeze.json',dict(utc=utc(),before_remaining_optimization=True,sha256=frozen))
save(root/'preparation_verification.json',dict(passed=True,preserved_task_results=1322,recovered_from_existing_log=1,pending_tasks=5157,cohort_unchanged=True,original_deadline_retained=True))
save(root/'progress.json',dict(status='prepared_recovery',completed_tasks=1323,preserved_tasks=972,new_completed_tasks=351,completed_since_recovery=0,expected_tasks=6480,expected_new_tasks=5508,updated_utc=utc()))
from eta import remaining
eta=remaining(root);save(root/'runtime_estimate.json',eta)
print('RECOVERY_PREPARED',root,flush=True);print({k:v for k,v in eta.items() if k!='details'},flush=True)
