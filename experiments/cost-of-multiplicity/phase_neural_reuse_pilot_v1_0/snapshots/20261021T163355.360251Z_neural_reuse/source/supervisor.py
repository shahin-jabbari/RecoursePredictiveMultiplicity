import json,os,sys,signal,socket,subprocess,traceback,fcntl
from pathlib import Path
from common import *

def supervise(root):
 root=Path(root)
 with open(root/'worker.lock','a+') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  command=[sys.executable,'-B','-u',str(root/'source/worker.py'),str(root)]
  with open(root/'run.log','xb') as log:child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
  awake=None
  if Path('/usr/bin/caffeinate').exists():awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(child.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).pid
  identity=dict(pid=child.pid,supervisor_pid=os.getpid(),started_utc=utc(),hostname=socket.gethostname(),root=str(root),command=command,caffeinate_pid=awake)
  save(root/'launch.json',identity);lock.seek(0);lock.truncate();lock.write(json.dumps(identity)+'\n');lock.flush()
  print(utc(),'WORKER_STARTED',identity,flush=True);code=child.wait();progress=read(root/'progress.json');cfg=read(root/'protocol.json')
  complete=bool(code==0 and progress.get('status')=='completed' and progress.get('completed_tasks')==972 and (root/'verification.json').exists() and read(root/'verification.json').get('report_checks') and (root/'cost_report/figures_tables.zip').exists() and (Path(cfg['disparity_result'])/'figures_tables.zip').exists() and 'NEURAL_MULTIPLICITY_PILOT_COMPLETED' in (root/'run.log').read_text())
  sig=signal.Signals(-code).name if code<0 else None
  record=dict(utc=utc(),returncode=code,signal=sig,completed=complete,stage=read(root/'stage.json') if (root/'stage.json').exists() else None,worker_pid=child.pid,supervisor_pid=os.getpid())
  save(root/'worker_exit.json',record)
  if not complete and not (root/'failure.json').exists():save(root/'failure.json',dict(error='Native '+str(sig) if sig else 'Worker exited before verified completion',**record))
  print(utc(),'WORKER_EXIT',record,flush=True)
if __name__=='__main__':
 try:supervise(Path(sys.argv[1]))
 except BaseException as e:save(Path(sys.argv[1])/'supervisor_failure.json',dict(utc=utc(),error=repr(e),traceback=traceback.format_exc()));raise
