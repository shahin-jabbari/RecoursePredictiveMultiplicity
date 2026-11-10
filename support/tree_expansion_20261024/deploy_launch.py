from pathlib import Path
import json,shutil,subprocess,os,hashlib,datetime
W=Path(__file__).resolve().parent;plan=json.loads((W/'campaign_plan.json').read_text());campaign=Path(plan['campaign_root']);root=Path(plan['stages'][0]['run_root']);v=json.loads((W/'prelaunch_verification.json').read_text())
assert v['passed']
for rel,digest in v['validated_source_sha256'].items():assert hashlib.sha256((W/'source'/rel).read_bytes()).hexdigest()==digest,rel
campaign.mkdir(parents=True,exist_ok=False);root.mkdir(parents=True,exist_ok=False)
for name in ['campaign_plan.json','campaign_state.json','CONTINUATION.md','PLAN.html']:shutil.copy2(W/name,campaign/name)
shutil.copytree(W/'source',root/'source',ignore=shutil.ignore_patterns('__pycache__'))
for name in ['protocol.json','prelaunch_verification.json']:shutil.copy2(W/name,root/name)
shutil.copytree(W/'validation_evidence',root/'validation_evidence',symlinks=True)
shutil.copy2(W/'test_expansion.py',root/'validation_evidence/test_expansion.py')
# Launch exactly once, detached; the supervisor owns the lock and worker lifetime.
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/tree_expansion_mpl')
with (root/'supervisor.log').open('ab',buffering=0) as f:
 proc=subprocess.Popen(['/Users/sj987/miniconda3/bin/python','-B','-u',str(root/'source/supervisor.py'),str(root)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,env=env)
state=json.loads((campaign/'campaign_state.json').read_text());state.update(status='running',supervisor_pid=proc.pid,last_update_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());state['stages'][0]['status']='running';tmp=campaign/'campaign_state.json.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(campaign/'campaign_state.json')
print(json.dumps(dict(campaign_root=str(campaign),run_root=str(root),supervisor_pid=proc.pid)))
