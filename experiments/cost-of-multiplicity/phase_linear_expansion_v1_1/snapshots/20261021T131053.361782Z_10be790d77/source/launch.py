import os,sys,fcntl,subprocess
from pathlib import Path
from design import read,save,sha,utc

def launch(root):
    root=Path(root);assert read(root/'preparation_verification.json')['passed']
    assert not (root/'launch.json').exists() and not (root/'run.log').exists(), 'This frozen run was already launched; automatic resume prohibited'
    for n,h in read(root/'freeze.json')['sha256'].items():assert sha(root/n)==h,'Source drift '+n
    fds=[]
    for path in [root.parents[2]/'expansion_active.lock',root/'worker.lock']:
        fd=os.open(path,os.O_CREAT|os.O_RDWR,0o600)
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Active worker already owns lock '+str(path))
        fds.append(fd)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',MPLCONFIGDIR='/tmp/multiplicity-expansion-matplotlib')
    cmd=[sys.executable,'-u',str(root/'source/supervisor.py'),str(root)]+list(map(str,fds))
    with open(root/'supervisor.log','xb') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,pass_fds=tuple(fds))
    for fd in fds:os.close(fd)
    save(root/'supervisor_launch.json',dict(pid=p.pid,utc=utc(),command=cmd))
    for folder in [root.parents[2],Path(read(root/'analysis_protocol.json')['disparity_result']).parents[2]]:
        save(folder/'active_expansion.json',dict(result=str(root),supervisor_pid=p.pid,status='running',started_utc=utc()))
    print(str(root),flush=True);print('SUPERVISOR',p.pid,flush=True)
if __name__=='__main__':launch(Path(sys.argv[1]))
