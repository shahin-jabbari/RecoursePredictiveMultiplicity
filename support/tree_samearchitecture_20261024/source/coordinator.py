"""Run only the two explicitly authorized arms, sequentially to protect pilot timing."""
import os,sys,json,time,fcntl,subprocess,datetime,traceback
from pathlib import Path

def write(p,d):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())

def run(root):
 lock=(root/'coordinator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (root/'launch.json').exists():raise RuntimeError('Coordinator already launched; no restart')
 import hashlib
 pre=read(root/'prelaunch_verification.json');assert pre['passed']
 for rel,digest in pre['deployment_sha256'].items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest,('deployment_changed',rel)
 start=time.time();launch=dict(coordinator_pid=os.getpid(),started_utc=now(),started_epoch=start,arms=['decision_tree','random_forest_pilot'],full_forest_authorized=False,deadline_active=False)
 write(root/'launch.json',launch);lock.seek(0);lock.truncate();lock.write(json.dumps(launch));lock.flush();os.fsync(lock.fileno())
 results={}
 for arm in launch['arms']:
  r=root/arm
  with (r/'supervisor.log').open('a') as log:
   child=subprocess.Popen([sys.executable,'-B','-u',str(r/'source/supervisor_unseen.py'),str(r)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
   write(root/'progress.json',dict(active_arm=arm,supervisor_pid=child.pid,utc=now(),finished=results,full_forest_authorized=False))
   code=child.wait();exitfile=r/'worker_exit.json';results[arm]=read(exitfile) if exitfile.exists() else dict(completed=False,supervisor_returncode=code,error='missing worker_exit.json')
  # Failure of one independent arm does not authorize any restart, tuning, or full study.
 write(root/'worker_exit.json',dict(completed=all(z.get('completed') for z in results.values()),utc=now(),arms=results,elapsed_hours=(time.time()-start)/3600,full_forest_launched=False))
 write(root/'progress.json',dict(active_arm=None,status='completed' if all(z.get('completed') for z in results.values()) else 'finished_with_failure',utc=now(),finished=results,full_forest_authorized=False))
 print(json.dumps(results),flush=True)

if __name__=='__main__':
 root=Path(sys.argv[1])
 try:run(root)
 except BaseException as e:write(root/'coordinator_failure.json',dict(utc=now(),type=type(e).__name__,message=str(e),traceback=traceback.format_exc()));raise
