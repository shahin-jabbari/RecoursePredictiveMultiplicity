"""Read-only projection. New wall time includes solves, checkpoints, construction and I/O."""
import json,time,datetime,statistics
from pathlib import Path
from collections import defaultdict
from functools import lru_cache

def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()] if Path(p).exists() else []
@lru_cache(maxsize=8)
def inputs(root):
 root=Path(root);p=read(root/'protocol.json');reuse=read(root/'reuse_manifest.json');plan=read(root/'task_plan.json');cal=defaultdict(list)
 for z in rows(Path(p['source_runs']['convergence'])/'summaries.jsonl'):cal[(z['dataset'],z['models'],z['budget'])].append(z)
 return p,reuse,plan,cal

def remaining(root):
 root=Path(root);p,reuse,plan,cal=inputs(str(root));timing=[z for z in rows(root/'timing.jsonl') if z.get('kind')=='trajectory'];done={z['key'] for z in timing};measured=defaultdict(list)
 for z in timing:
  if z['key'] in reuse['complete']:continue
  k=(z['dataset'],z['models'],z['budget'],z['key'] in reuse['prefixes'],z['key'] in reuse['references']);measured[k].append(z['total_seconds'])
 totals=defaultdict(float);ncal=nnew=0
 for z in plan['cells']:
  if z['key'] in done:continue
  pref=z['key'] in reuse['prefixes'];ref=z['key'] in reuse['references'];k=(z['dataset'],z['models'],z['budget'],pref,ref)
  if len(measured[k])>=3:seconds=statistics.mean(measured[k]);nnew+=1
  else:
   old=cal[k[:3]]
   seconds=statistics.mean(v['algorithm_wall_seconds']*(.5 if pref else 1)+v['diagnostic_wall_seconds']+(0 if ref else v['reference_wall_seconds']) for v in old)
   seconds+=.1 if pref else 0;ncal+=1
  totals[z['dataset']]+=seconds
 reserve=5400.;central=sum(totals.values())+reserve
 if len(done)==plan['expected_trajectories']:
  pr=read(root/'progress.json') if (root/'progress.json').exists() else {}
  central=0 if pr.get('status')=='completed' else reserve
 low=sum(totals.values())*.65+1800;high=sum(totals.values())*1.8+10800
 now=time.time()
 return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed_trajectories=len(done),expected_trajectories=10800,preserved_complete=1620,new_completed=len(done)-1620,remaining_trajectories=10800-len(done),central_remaining_hours=central/3600,planning_remaining_hours=[low/3600,high/3600],estimated_finish_utc=datetime.datetime.fromtimestamp(now+central,datetime.timezone.utc).isoformat(),remaining_hours_by_dataset={k:v/3600 for k,v in totals.items()},matched_new_rates_used_for_cells=nnew,historical_rates_used_for_cells=ncal,verification_reporting_reserve_hours=reserve/3600,deadline_active=False,planning_not_confidence_interval=True,assumes_awake=True,partial_current_trajectory_not_subtracted=True)
if __name__=='__main__':
 import sys
 print(json.dumps(remaining(Path(sys.argv[1])),indent=2))
