"""Read-only process/progress inspection; never launches or stops an experiment."""
from pathlib import Path
import argparse,fcntl,json

def status(root):
    root=Path(root);out={'root':str(root)}
    for name in ['launch','progress','runtime_estimate','verification','failure']:
        p=root/(name+'.json')
        if p.exists():out[name]=json.loads(p.read_text())
    out['running']=False
    if (root/'worker.lock').exists():
        with (root/'worker.lock').open('rb') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
            except BlockingIOError:out['running']=True
    if 'failure' in out and out.get('progress',{}).get('status') not in ['failed','interrupted']:
        out['earlier_failure']=out.pop('failure')
    if 'runtime_estimate' in out:out['runtime_estimate'].pop('conditions',None)
    if (root/'preflight.json').exists():
        p=json.loads((root/'preflight.json').read_text())
        out['preflight']=dict(status=p['status'],blocks_passed=len(p['blocks']),blocks_total=p['expected'],
                             failures=p['failures'])
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();print(json.dumps(status(a.root),indent=2))
