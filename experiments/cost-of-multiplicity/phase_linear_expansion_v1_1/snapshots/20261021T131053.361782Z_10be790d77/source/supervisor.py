import os,sys,subprocess,traceback,signal,socket
from pathlib import Path
from design import read,save,utc

def supervise(root,fds):
    root=Path(root);command=[sys.executable,'-u',str(root/'source/worker.py'),str(root)]
    with open(root/'run.log','xb') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,pass_fds=tuple(fds))
    awake=None
    if Path('/usr/bin/caffeinate').exists():
        awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(child.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).pid
    identity=dict(pid=child.pid,supervisor_pid=os.getpid(),caffeinate_pid=awake,started_utc=utc(),root=str(root),command=command,hostname=socket.gethostname(),expected_new_settings=14190,preserved_settings=26400)
    save(root/'launch.json',identity)
    import json
    for fd in fds:os.lseek(fd,0,os.SEEK_SET);os.ftruncate(fd,0);os.write(fd,(json.dumps(identity)+'\n').encode());os.fsync(fd)
    print(utc(),'WORKER_STARTED',identity,flush=True)
    code=child.wait();p=read(root/'progress.json');acfg=read(root/'analysis_protocol.json');droot=Path(acfg['disparity_result'])
    v=read(root/'verification.json') if (root/'verification.json').exists() else {}
    complete=bool(code==0 and p.get('status')=='completed' and p.get('completed_settings')==40590 and v.get('passed') and v.get('report_checks') and (root/'cost_report/figures_tables.zip').exists() and (droot/'figures_tables.zip').exists() and 'LINEAR_MULTIPLICITY_EXPANSION_COMPLETED' in (root/'run.log').read_text())
    stage=read(root/'stage.json') if (root/'stage.json').exists() else None
    signame=signal.Signals(-code).name if code<0 else None
    exit_record=dict(utc=utc(),returncode=code,signal=signame,completed=complete,stage=stage,worker_pid=child.pid,supervisor_pid=os.getpid())
    save(root/'worker_exit.json',exit_record)
    if not complete:
        if not (root/'failure.json').exists():save(root/'failure.json',dict(error='Native signal '+str(signame) if signame else 'Worker exited before verified completion',**exit_record,native_traceback=str(root/'native_traceback.log')))
        p.update(status='failed',updated_utc=utc());save(root/'progress.json',p)
    print(utc(),'WORKER_EXIT',exit_record,flush=True)
    return 0 if complete else 1
if __name__=='__main__':
    root=Path(sys.argv[1]);fds=[int(x) for x in sys.argv[2:]]
    try:code=supervise(root,fds)
    except BaseException as e:
        save(root/'supervisor_failure.json',dict(utc=utc(),error=repr(e),traceback=traceback.format_exc()));code=1
    finally:
        for fd in fds:os.close(fd)
    sys.exit(code)
