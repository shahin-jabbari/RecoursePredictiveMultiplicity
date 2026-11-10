"""Independent supervisor: deadline includes the entire worker and its process group."""
import json,os,sys,signal,socket,subprocess,traceback,fcntl,time
from datetime import datetime,timezone
from common import *

def terminate_group(child,grace=5.):
 try:os.killpg(child.pid,signal.SIGTERM)
 except ProcessLookupError:pass
 end=time.monotonic()+grace
 while time.monotonic()<end:
  child.poll()
  try:os.killpg(child.pid,0)
  except ProcessLookupError:break
  time.sleep(.05)
 try:os.killpg(child.pid,signal.SIGKILL)
 except ProcessLookupError:pass
 return child.wait()

def wait_with_deadline(child,deadline):
 while True:
  code=child.poll()
  if code is not None:return code,False
  left=deadline-time.time()
  if left<=0:return terminate_group(child),True
  try:return child.wait(timeout=min(1.,left)),False
  except subprocess.TimeoutExpired:pass

def complete(root):
 cfg=read(root/'protocol.json');p=read(root/'progress.json');v=read(root/'verification.json') if (root/'verification.json').exists() else {}
 ext=read(root/'extension_verification.json') if (root/'extension_verification.json').exists() else {}
 return bool(p.get('status')=='completed' and p.get('completed_tasks')==cfg['expected_tasks'] and p.get('new_completed_tasks')==cfg['expected_new_tasks'] and v.get('passed') and v.get('report_checks') and ext.get('passed') and (root/'analysis.json').exists() and (root/'cost_report/figures_tables.zip').exists() and (Path(cfg['disparity_result'])/'figures_tables.zip').exists() and 'NEURAL_MULTIPLICITY_EXPANSION_COMPLETED' in (root/'run.log').read_text() and not (root/'failure.json').exists())

def supervise(root,command=None,validation=False):
 root=Path(root);cfg=read(root/'protocol.json');child=None
 with open(root/'worker.lock','a+') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (root/'launch.json').exists(),'This run was already launched; no automatic resume'
  started=time.time();deadline=cfg['absolute_deadline_epoch']
  assert deadline>started,'Original study deadline already passed'
  command=command or [sys.executable,'-B','-u',str(root/'source/worker.py'),str(root)]
  with open(root/'run.log','xb') as log:
   child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
  try:
   awake=None
   if not validation and Path('/usr/bin/caffeinate').exists():awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(child.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).pid
   deadline_iso=datetime.fromtimestamp(deadline,timezone.utc).isoformat()
   identity=dict(pid=child.pid,supervisor_pid=os.getpid(),started_utc=utc(),hostname=socket.gethostname(),root=str(root),command=command,caffeinate_pid=awake,deadline_utc=deadline_iso,process_group=child.pid)
   save(root/'deadline.json',dict(deadline_epoch=deadline,deadline_utc=deadline_iso,wall_seconds=cfg['runtime_cap_seconds'],includes_solving_verification_reporting=True,cutoff_is_incomplete=True))
   save(root/'launch.json',identity);lock.seek(0);lock.truncate();lock.write(json.dumps(identity)+'\n');lock.flush();os.fsync(lock.fileno())
   print(utc(),'WORKER_STARTED',identity,flush=True)
   code,cutoff=wait_with_deadline(child,deadline)
   completed=bool(code==0 and not cutoff and complete(root));sig=signal.Signals(-code).name if code<0 else None
   record=dict(utc=utc(),returncode=code,signal=sig,completed=completed,deadline_reached=cutoff,stage=read(root/'stage.json') if (root/'stage.json').exists() else None,worker_pid=child.pid,supervisor_pid=os.getpid())
   save(root/'worker_exit.json',record)
   if cutoff:
    save(root/'runtime_limit.json',dict(reason='Frozen 23-hour wall-clock allowance exhausted',**record))
    progress=read(root/'progress.json') if (root/'progress.json').exists() else {};progress.update(status='runtime_limit_reached',updated_utc=utc());save(root/'progress.json',progress)
    (root/'runtime_limit_summary.txt').write_text('INCOMPLETE AT PREDECLARED WALL-CLOCK LIMIT\nThe worker process group was stopped. Saved completed tasks and source artifacts remain.\nThis is not study completion or solver infeasibility. Missing tasks are not failures and must not be included in complete-case comparisons without an explicit partial-design label. No automatic restart.\n'+json.dumps(progress,indent=2)+'\n')
   elif not completed and not (root/'failure.json').exists():save(root/'failure.json',dict(error='Native '+str(sig) if sig else 'Worker exited before verified completion',**record))
   print(utc(),'WORKER_EXIT',record,flush=True)
   return record
  except BaseException:
   if child and child.poll() is None:terminate_group(child)
   raise
if __name__=='__main__':
 try:supervise(Path(sys.argv[1]))
 except BaseException as e:save(Path(sys.argv[1])/'supervisor_failure.json',dict(utc=utc(),error=repr(e),traceback=traceback.format_exc()));raise
