"""Use only new task timings for progress; prior timings calibrate per-dataset rates."""
from collections import defaultdict
import time
from common import *
def remaining(root):
 root=Path(root);cfg=read(root/'protocol.json');plan=[x for x in read(root/'task_plan.json') if not x['reused']]
 rows=[json.loads(x) for x in (root/'timing.jsonl').read_text().splitlines()] if (root/'timing.jsonl').exists() else []
 prior=[json.loads(x) for x in (root/'pilot_timing.jsonl').read_text().splitlines()]
 done={(x['case'],x['task']) for x in rows};assert len(done)==len(rows)
 pending=[x for x in plan if (x['case'],x['task']) not in done];details=[];total=0.
 for ds in DATASETS:
  for kind in ['singleton_shallow','singleton_deep','joint_6','joint_12','joint_24']:
   current=[r['seconds'] for r in rows if r['dataset']==ds and r['kind']==kind and r['optimization_seconds']>0 and not r.get('timing_reconstructed_from_logs',False)]
   baseline=[r['seconds'] for r in prior if r['dataset']==ds and r['kind']==kind and r['optimization_seconds']>0 and not r.get('timing_reconstructed_from_logs',False)]
   # Small new samples cannot immediately discard the calibrated prior.
   sample=current if len(current)>=6 else current+baseline
   rate=float(np.mean(sample)) if sample else 11.
   future=[p for p in pending if p['dataset']==ds and p['kind']==kind];n=sum(p['nontrivial'] for p in future)
   seconds=n*rate+(len(future)-n)*.002;total+=seconds
   details.append(dict(dataset=ds,kind=kind,new_observations=len(current),remaining_tasks=len(future),remaining_nontrivial=n,mean_wall_seconds=rate,projected_seconds=seconds))
 estimate=(1.15*total+900)/3600;low=(.7*total+300)/3600;high=(1.7*total+1800)/3600
 deadline=read(root/'deadline.json') if (root/'deadline.json').exists() else {}
 return dict(updated_utc=utc(),new_completed_tasks=len(rows),preserved_tasks=cfg['preserved_tasks'],completed_tasks=cfg['preserved_tasks']+len(rows),completed_since_recovery=max(0,cfg['preserved_tasks']+len(rows)-cfg['recovery_completed_tasks']),expected_tasks=cfg['expected_tasks'],remaining_hours=estimate,remaining_planning_hours=[low,high],remaining_optimization_cap_hours=sum(p['nontrivial'] for p in pending)*10/3600,maximum_new_optimization_cap_hours=len(plan)*10/3600,deadline_utc=deadline.get('deadline_utc'),hours_until_cutoff=max(0,(deadline['deadline_epoch']-time.time())/3600) if deadline else None,details=details,note='Measured pilot rates refined by new task wall times. Optimization time is already included. Planning range, not statistical CI; reporting allowance included. Original absolute cutoff is retained; recovery does not reset it.')
