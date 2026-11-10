"""Launch/resume a detached, immutable-source convergence study without overwriting results."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,fcntl,hashlib,json,os,shutil,subprocess,sys,uuid
HERE=Path(__file__).resolve().parent
DEFAULT_PROJECT=Path('/Users/sj987/Documents/ChatGPT/Code/Predictive_Multiplicity_Recourse/V7')

def atomic(path,data):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,path)

def launch(project=DEFAULT_PROJECT,config=None,root=None,python=None,new=False,parent=None):
    project=Path(project).expanduser().resolve();config=Path(config or HERE/'config.json').resolve()
    c=json.loads(config.read_text());signature=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:12]
    parent=Path(parent) if parent else project/'results/V7/convergence_paper'/c['profile']
    parent.mkdir(parents=True,exist_ok=True);pointer=parent/f'latest_{signature}.json'
    # Serialize creation as well as execution to prevent concurrent double launches.
    with (parent/'launch.lock').open('a+') as create_lock:
        fcntl.flock(create_lock,fcntl.LOCK_EX)
        if root is None and pointer.exists() and not new:root=Path(json.loads(pointer.read_text())['root'])
        if root is None:
            root=parent/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'_'+uuid.uuid4().hex[:10]);root.mkdir()
            source=root/'source';source.mkdir();shutil.copytree(project/'utils',source/'utils',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            for path in HERE.glob('*.py'):
                shutil.copy2(path,source/path.name)
            shutil.copy2(config,root/'config.json');atomic(pointer,dict(root=str(root)))
        root=Path(root).resolve()
        if json.loads((root/'config.json').read_text())!=c:raise ValueError('Resume configuration mismatch')
        lock=(root/'worker.lock').open('a+')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close();return dict(status='already_running',root=str(root))
        try:
            progress=root/'progress.json'
            if progress.exists() and json.loads(progress.read_text()).get('status')=='completed':return dict(status='already_completed',root=str(root))
            env=os.environ.copy();env.update(RECOURSE_V7_PROJECT=str(project),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',
                MPLCONFIGDIR=str(root/'matplotlib'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
            with (root/'run.log').open('ab',buffering=0) as log:
                proc=subprocess.Popen([python or sys.executable,'-u',str(root/'source/convergence_study.py'),
                    '--root',str(root),'--config',str(root/'config.json'),'--lock-fd',str(lock.fileno())],
                    cwd=str(project),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,
                    start_new_session=True,close_fds=True,pass_fds=(lock.fileno(),))
            if sys.platform=='darwin':subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(proc.pid)],stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
            info=dict(status='started',pid=proc.pid,root=str(root),log=str(root/'run.log'),created_utc=datetime.now(timezone.utc).isoformat())
            atomic(root/'launch.json',info);return info
        finally:lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',default=str(DEFAULT_PROJECT));p.add_argument('--config');p.add_argument('--root');p.add_argument('--parent');p.add_argument('--new',action='store_true')
    a=p.parse_args();print(json.dumps(launch(project=a.project,config=a.config,root=a.root,new=a.new,parent=a.parent),indent=2))
