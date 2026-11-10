import os,sys,json,time,datetime,subprocess,fcntl,traceback,socket,zipfile
from pathlib import Path

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,p)
def complete(root):
 req=['verification.json','extension_verification.json','analysis.json','report.txt','report.html','figures_tables.zip','progress.json']
 if not all((root/n).exists() for n in req) or (root/'failure.json').exists():return False
 v=read(root/'verification.json');e=read(root/'extension_verification.json');a=read(root/'analysis.json');pr=read(root/'progress.json')
 if not(v.get('passed') and e.get('passed') and a.get('completed') and v.get('trajectories')==10800 and v.get('rounds')==2160000 and v.get('diagnostics')==162000 and v.get('references')==10800 and pr.get('status')=='completed' and e.get('new_orpm_calls')==1673100):return False
 if len(list((root/'figures').glob('*.png')))!=54 or len(list((root/'figures').glob('*.pdf')))!=54:return False
 with zipfile.ZipFile(root/'figures_tables.zip') as z:
  if z.testzip() is not None:return False
 return 'TREE_CONVERGENCE_EXPANSION_COMPLETED' in (root/'run.log').read_text()
def supervise(root):
 root=Path(root);lock=(root/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (root/'launch.json').exists():raise RuntimeError('Already launched; no implicit restart')
 assert read(root/'prelaunch_verification.json')['passed']
 assert read(root/'protocol.json')['absolute_deadline_epoch'] is None
 start=time.time();cmd=[sys.executable,'-B','-u',str(root/'source/runner.py'),str(root)]
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLCONFIGDIR='/tmp/tree_expansion_mpl')
 with (root/'run.log').open('a') as log:
  worker=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,pass_fds=(lock.fileno(),))
  guard=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(worker.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  launch=dict(worker_pid=worker.pid,supervisor_pid=os.getpid(),started_epoch=start,started_utc=now(),command=cmd,root=str(root),host=socket.gethostname(),process_group=worker.pid,caffeinate_pid=guard.pid,deadline_active=False,study='tree_convergence_expansion_10x20')
  write(root/'launch.json',launch);lock.seek(0);lock.truncate();lock.write(json.dumps(launch));lock.flush();os.fsync(lock.fileno())
  rc=worker.wait();success=rc==0 and complete(root);result=dict(utc=now(),returncode=rc,signal=-rc if rc<0 else None,completed=success,elapsed_hours=(time.time()-start)/3600,last_stage=read(root/'stage.json') if (root/'stage.json').exists() else None)
  write(root/'worker_exit.json',result)
  if not success and not (root/'failure.json').exists():write(root/'failure.json',dict(type='PREMATURE_WORKER_EXIT',**result))
  print(json.dumps(result),flush=True)
  return success
if __name__=='__main__':
 root=Path(sys.argv[1])
 try:supervise(root)
 except BaseException as e:write(root/'supervisor_failure.json',dict(utc=now(),type=type(e).__name__,message=str(e),traceback=traceback.format_exc()));raise
