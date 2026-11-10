"""Labeled synthetic integration fixtures; none of these are scientific study data."""
import os,sys,json,shutil,copy,hashlib
from pathlib import Path
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W/'source'))
import runner,verify,reuse,common
from common import np,torch,rc,rt,write,read,rows,sha
from sklearn.tree import DecisionTreeClassifier
X=np.array([[-2],[-1],[-.5],[.5],[1],[2]],np.float32);y=np.array([0,0,0,1,1,1])
clf=DecisionTreeClassifier(max_depth=1,random_state=1).fit(X,y);models=[rt.TreeModel.from_estimator(clf) for _ in range(6)];x=torch.tensor([-1.],dtype=torch.float32);con=rc.Constraints(np.array([-np.inf]),np.array([np.inf]),[],'immutable')
meta={'selected_source_ids':[987654]}
runner.bank=lambda *a,**k:(models,x,con,meta)
verify.bank=runner.bank
fixture=W/'validation_evidence'
if fixture.exists():raise RuntimeError('Fixtures already exist; no accidental overwrite')
fixture.mkdir();protocol=read(W/'protocol.json');key='Fixture_s401_p0_m6_b2';cell=dict(key=key,dataset='Fixture',seed=401,person=0,models=6,budget=2)
original=runner.checked_solve;calls=[]
def counted(*args,**kwargs):
 calls.append((args[0].time_limit if hasattr(args[0],'time_limit') else None,kwargs.get('objective','weighted_loss')))
 return original(*args,**kwargs)
runner.checked_solve=counted

def init(name,pref=None,ref=None):
 root=fixture/name;root.mkdir();write(root/'protocol.json',protocol);write(root/'task_plan.json',dict(cells=[cell],expected_trajectories=1));write(root/'reuse_manifest.json',dict(complete={},prefixes={key:pref} if pref else {},references={key:ref} if ref else {}));return root
fresh=init('synthetic_fresh');runner.trajectory(fresh,'Fixture',6,2,1,401,0);assert len(calls)==216,len(calls)
result=verify.verify_cell(fresh,protocol,cell);assert result['rounds']==200
# A frozen source fixture mimics saved trade-off format without modifying its generating run.
source=fixture/'synthetic_saved_method';source.mkdir();rr=(fresh/'trajectories'/key/'rounds.jsonl').read_bytes().splitlines(keepends=True);(source/'rounds.jsonl').write_bytes(b''.join(rr[:100]));(source/'orpm_solver').symlink_to((fresh/'trajectories'/key/'orpm_solver').resolve(),target_is_directory=True)
refdir=fixture/'synthetic_saved_reference';refdir.mkdir();ref=read(fresh/'trajectories'/key/'reference.json');write(refdir/'solver_outcome.json',dict(point=ref['point'],oracle=ref['oracle']));write(refdir/'outcome.json',dict(seconds=ref['seconds'],source_person_id=987654));(refdir/'solver').symlink_to((fresh/'trajectories'/key/'reference_solver').resolve(),target_is_directory=True)
prefix=dict(directory=str(source),rounds=100,source_id=987654,sha256=sha(source/'rounds.jsonl'));refitem=dict(directory=str(refdir),source_id=987654)
calls.clear();continued=init('synthetic_prefix_continuation',prefix,refitem);runner.trajectory(continued,'Fixture',6,2,1,401,0)
assert len(calls)==115,('repeated_old_optimization',len(calls));result=verify.verify_cell(continued,protocol,cell);assert result['rounds']==200
assert sha(continued/'trajectories'/key/'source_prefix.jsonl')==prefix['sha256']
old=rows(source/'rounds.jsonl');new=rows(continued/'trajectories'/key/'rounds.jsonl')
for a,b in zip(old,new):
 for k in ['round','weights','point','losses','best_point','best_losses','best_cost','joint_valid']:assert a[k]==b[k]
 ai=dict(a['oracle']);bi=dict(b['oracle']);ai.pop('artifact_subdir',None);bi.pop('artifact_subdir',None);assert ai==bi
# A deliberately contradictory certificate must be downgraded only after a full encoding audit.
raw=copy.deepcopy(ref['oracle']);raw['lower_bound']=1.;raw['global_certificate']=True
checked=reuse.conservative_reference(raw,models,x,con,2,torch.tensor(ref['point']),refdir/'solver')
assert checked['certificate_rejected'] and checked['lower_bound']==0 and not checked['global_certificate'];assert checked['preserved_native_info']==raw
# Tampered saved weights must be detected before any response optimization or checkpoint.
tamper=fixture/'tampered_prefix';tamper.mkdir();bad=copy.deepcopy(old);bad[0]['weights'][0]+=.1
(tamper/'rounds.jsonl').write_text(''.join(json.dumps(z)+'\n' for z in bad));(tamper/'orpm_solver').symlink_to((source/'orpm_solver').resolve(),target_is_directory=True)
badprefix=dict(prefix,directory=str(tamper),sha256=sha(tamper/'rounds.jsonl'));badroot=init('deliberate_fault',badprefix);calls.clear()
try:runner.trajectory(badroot,'Fixture',6,2,1,401,0)
except AssertionError:pass
else:raise AssertionError('Tampered prefix accepted')
assert not calls,'Optimization occurred after invalid saved first-round weights'
# No stale deadline, and duplicate launches/trajectories remain fail-closed.
common.deadline_check(continued)
try:runner.trajectory(continued,'Fixture',6,2,1,401,0)
except RuntimeError as e:assert 'already exists' in str(e)
else:raise AssertionError('Repeated trajectory accepted')
write(fixture/'checks.json',dict(passed=True,synthetic_only=True,fresh_calls=216,continued_calls=115,old_response_repeats=0,full_cell_independent_verification=True,prefix_bytes_preserved=True,tampering_rejected_before_optimization=True,contradictory_reference_rejected_after_full_encoding_audit=True,no_historical_deadline=True,duplicate_refused=True))
print('ALL EXPANSION INTEGRATION CHECKS PASSED')
