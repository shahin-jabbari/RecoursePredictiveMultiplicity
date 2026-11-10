"""Launch one authorized full expansion; no automatic resume or concurrent duplicate."""
import os,sys,fcntl,subprocess,shutil,uuid,argparse
from pathlib import Path
from datetime import datetime,timezone
from common import read,save,sha,utc
HERE=Path(__file__).resolve().parent

def launch(config,output):
    cfg=read(config);root=Path(output)/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'_'+uuid.uuid4().hex[:10])
    lock_parent=Path(output);lock_parent.mkdir(parents=True,exist_ok=True)
    phase_fd=os.open(lock_parent/'active_worker.lock',os.O_CREAT|os.O_RDWR,0o600)
    try:fcntl.flock(phase_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('An experiment worker already holds this phase lock; no duplicate launch allowed')
    root.mkdir(exist_ok=False)
    for path in [config,HERE/'PROTOCOL.txt']:shutil.copy2(path,root/('protocol.json' if path==config else path.name))
    src=root/'source';src.mkdir()
    for path in HERE.glob('*.py'):shutil.copy2(path,src/path.name)
    shutil.copytree(HERE/'frozen_utils',src/'frozen_utils',ignore=shutil.ignore_patterns('__pycache__'))
    manifest={str(p.relative_to(root)):sha(p) for p in sorted(src.rglob('*')) if p.is_file()};manifest.update({n:sha(root/n) for n in ['protocol.json','PROTOCOL.txt']})
    save(root/'freeze.json',dict(frozen_utc=utc(),source_sha256=manifest,protocol_version=cfg['protocol_version'],purpose='authorized expansion' if not cfg.get('validation') else 'labeled validation only'))
    shutil.copy2(HERE/'runtime_estimate.json',root/'runtime_estimate.json')
    worker_fd=os.open(root/'worker.lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(worker_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLBACKEND='Agg',MPLCONFIGDIR=str(root/'plot_cache'),PYTHONDONTWRITEBYTECODE='1')
    cmd=[sys.executable,'-u',str(src/'runner.py'),'--root',str(root),'--lock-fd',str(worker_fd)]
    with open(root/'run.log','xb') as log:proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,pass_fds=(worker_fd,phase_fd))
    for fd in [worker_fd,phase_fd]:os.ftruncate(fd,0);os.write(fd,(str(proc.pid)+'\n').encode());os.close(fd)
    awake=None
    if Path('/usr/bin/caffeinate').exists():awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(proc.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True).pid
    save(root/'launch.json',dict(pid=proc.pid,caffeinate_pid=awake,started_utc=utc(),python=sys.executable,root=str(root),command=cmd,scope='full expansion' if not cfg.get('validation') else 'validation',expected_records=cfg['expected_records'],preserved_records=cfg['preserved_pilot_records'],new_records=cfg['new_records']))
    (lock_parent/'CURRENT_RUN.txt').write_text(str(root)+'\n');print(root);print('PID',proc.pid)
    return root
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,default=HERE/'protocol.json');ap.add_argument('--output',type=Path,default=HERE/'results');args=ap.parse_args();launch(args.config,args.output)
