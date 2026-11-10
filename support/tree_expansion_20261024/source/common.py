import os,sys,json,hashlib,time,datetime,platform
from pathlib import Path
import numpy as np
import torch
from utils import recourse_core as rc,recourse_trees as rt
BASE=Path('/Users/sj987/Documents/ChatGPT/Code/Predictive_Multiplicity_Recourse/V7')
rc.PROJECT_ROOT=BASE
torch.set_num_threads(1)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,Path):return str(x)
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(clean(x),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def append(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(clean(x),allow_nan=False)+'\n');f.flush()
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()] if Path(p).exists() else []
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def manifest(root,paths):return {str(p.relative_to(root)):sha(p) for p in paths if p.is_file()}
def stage(root,name,**kwargs):write(root/'stage.json',dict(stage=name,utc=now(),**kwargs));print(now(),name,clean(kwargs),flush=True)
def bank(root,ds,m=24,seed=401,person=0):
 b=Path(root)/'banks'/ds/f'seed_{seed}';a=np.load(b/'split.npz',allow_pickle=False);meta=read(b/'bank.json')
 models=[rt.TreeModel.load(b/'candidates'/f'{i:04d}') for i in meta['selected_candidate_indices'][:m]]
 pos=meta['selected_test_positions'][person];x=torch.from_numpy(a['X_test'][pos].copy());imm=meta['immutable_indices']
 lo=np.full(len(x),-np.inf);hi=np.full(len(x),np.inf);lo[imm]=x.numpy()[imm];hi[imm]=x.numpy()[imm]
 return models,x,rc.Constraints(lo,hi,imm,'immutable'),meta

def cell_key(ds,seed,person,m,b):return f'{ds}_s{seed}_p{person}_m{m}_b{b:g}'
def expected_cells(root):return read(Path(root)/'task_plan.json')['cells']
def deadline_check(root):
 limit=read(Path(root)/'protocol.json').get('absolute_deadline_epoch')
 if limit is not None and time.time()>=limit:raise RuntimeError('Study deadline reached')

def point_check(point,x0,budget,c):
 a=np.asarray(point,dtype=np.float32).astype(float);x=x0.numpy().astype(float)
 return bool(np.isfinite(a).all() and np.linalg.norm(a-x)<=budget+1e-7 and np.array_equal(a[c.immutable],x[c.immutable]) and np.all(a>=c.lower-1e-7) and np.all(a<=c.upper+1e-7))
def loss(models,x):return (1-rt.probabilities(models,x))**2

def audit_encoding(oracle,point,objective):
 """Evaluate every bound, linear/quadratic row and indicator at a constructed witness; no optimize call."""
 import gurobipy as gp
 m=oracle.problem;m.update();a=np.asarray(point,dtype=np.float32).astype(float);v={}
 for var,value in zip(oracle.x,a):v[var.VarName]=float(value)
 for f,var in oracle.cell_indices.items():v[var.VarName]=int(np.searchsorted(oracle.split_positions[f],a[f],side='left'))
 for group in oracle.groups:
  for g in group:v[g['z'].VarName]=int(np.all(a>=g['lower']) and np.all(a<=g['upper']))
 probabilities=rt.probabilities(oracle.models,a)
 for var,value in zip(oracle.p,probabilities):v[var.VarName]=float(value)
 v[oracle.epigraph.VarName]=float(((1-probabilities)**2).max())
 if set(v)!=set(z.VarName for z in m.getVars()):raise RuntimeError('Unmapped variables in independent witness audit')
 def lin(e):return e.getConstant()+sum(e.getCoeff(i)*v[e.getVar(i).VarName] for i in range(e.size()))
 def quad(e):return lin(e.getLinExpr())+sum(e.getCoeff(i)*v[e.getVar1(i).VarName]*v[e.getVar2(i).VarName] for i in range(e.size()))
 def violation(value,sense,rhs):return max(0,value-rhs) if sense=='<' else max(0,rhs-value) if sense=='>' else abs(value-rhs)
 worst=0.;counts=dict(variables=0,linear=0,quadratic=0,indicators=0)
 for z in m.getVars():
  val=v[z.VarName];worst=max(worst,z.LB-val,val-z.UB,abs(val-round(val)) if z.VType!='C' else 0);counts['variables']+=1
 for c in m.getConstrs():worst=max(worst,violation(lin(m.getRow(c)),c.Sense,c.RHS));counts['linear']+=1
 for c in m.getQConstrs():worst=max(worst,violation(quad(m.getQCRow(c)),c.QCSense,c.QCRHS));counts['quadratic']+=1
 for c in m.getGenConstrs():
  if c.GenConstrType!=gp.GRB.GENCONSTR_INDICATOR:raise RuntimeError('Unsupported general constraint audit')
  z,active,expr,sense,rhs=m.getGenConstrIndicator(c)
  if v[z.VarName]==active:worst=max(worst,violation(lin(expr),sense,rhs))
  counts['indicators']+=1
 passed=worst<=1e-7 and point_check(a,oracle.x0,oracle.budget,oracle.constraints)
 return dict(passed=passed,max_constraint_violation=worst,counts=counts,point=a,objective_kind=objective,probabilities=probabilities,method='all_variables_rows_and_indicators_no_optimization')

def checked_solve(oracle,weights=None,objective='weighted_loss',known_points=(),recovered_info=None):
 if recovered_info is None:
  p=oracle.solve(weights,objective);info=dict(p.recourse_info);m=oracle.problem
  try:raw=float(m.ObjBound)
  except Exception:raw=None
  info.update(raw_solver_bound=raw,solution_count=int(m.SolCount),certificate_rejected=False)
 else:
  info=dict(recovered_info);p=oracle.x0.clone();oracle.calls+=1
  info.update(reconstructed_from_saved_solve=True,additional_optimization_calls=0,certificate_rejected=False)
 if info['status']=='incumbent_failed_validation':
  from boundary_recovery import apply_rejection
  p,info=apply_rejection(oracle,info,weights,objective,known_points)

 if not point_check(p,oracle.x0,oracle.budget,oracle.constraints):raise RuntimeError('Returned action violates original action region')
 options=[p,oracle.x0,*known_points]
 def objective_value(x):
  l=loss(oracle.models,x);return float(np.dot(weights,l)) if objective=='weighted_loss' else float(l.max())
 witness=min(options,key=objective_value);upper=objective_value(witness)
 contradictory=info['solver_status'] in [3,4,5] or info['bound_inconsistent_with_verified_candidate'] or (info['lower_bound'] is not None and info['lower_bound']>upper+1e-7)
 if contradictory:
  audit=audit_encoding(oracle,witness,objective)
  if not audit['passed']:raise RuntimeError('Contradictory solver certificate AND failed full witness audit: '+json.dumps(clean(audit)))
  info.update(certificate_rejected=True,certificate_rejection='raw certificate contradicts fully audited feasible witness',witness_audit=audit,lower_bound=0.,global_certificate=False)
 elif info['lower_bound'] is None:info['lower_bound']=0.
 info['verified_objective_upper']=upper
 return p,info
