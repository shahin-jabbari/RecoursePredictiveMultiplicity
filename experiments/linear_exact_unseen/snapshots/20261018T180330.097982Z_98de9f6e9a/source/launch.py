"""Launch only the frozen pilot in a fresh directory; no resume or full-study mode."""
import os,sys,fcntl,subprocess,shutil,uuid
from pathlib import Path
from datetime import datetime,timezone
from common import read,save,sha,utc
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    root=HERE/'results'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'_'+uuid.uuid4().hex[:10]);root.mkdir(parents=True,exist_ok=False)
    for name in ['protocol.json','PROTOCOL.txt','full_study_PROPOSED.json']:shutil.copy2(HERE/name,root/name)
    src=root/'source';src.mkdir()
    for path in HERE.glob('*.py'):shutil.copy2(path,src/path.name)
    shutil.copytree(HERE/'frozen_utils',src/'frozen_utils',ignore=shutil.ignore_patterns('__pycache__'))
    manifest={str(p.relative_to(root)):sha(p) for p in sorted(src.rglob('*')) if p.is_file()};manifest.update({n:sha(root/n) for n in ['protocol.json','PROTOCOL.txt','full_study_PROPOSED.json']})
    save(root/'freeze.json',dict(frozen_utc=utc(),source_sha256=manifest,protocol_version='1.0',purpose='pilot only; analysis included before outcomes'))
    lock=os.open(root/'worker.lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1')
    with open(root/'run.log','xb') as log:
        proc=subprocess.Popen([sys.executable,'-u',str(src/'runner.py'),'--root',str(root),'--lock-fd',str(lock)],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,pass_fds=(lock,))
    os.ftruncate(lock,0);os.write(lock,(str(proc.pid)+'\n').encode());os.close(lock)
    awake=None
    if Path('/usr/bin/caffeinate').exists():awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(proc.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True).pid
    save(root/'launch.json',dict(pid=proc.pid,caffeinate_pid=awake,started_utc=utc(),python=sys.executable,root=str(root),command=[str(src/'runner.py'),'--root',str(root)],scope='pilot only',expected_records=read(root/'protocol.json')['expected_records']))
    (HERE/'CURRENT_RUN.txt').write_text(str(root)+'\n')
    print(str(root));print('PID',proc.pid)
