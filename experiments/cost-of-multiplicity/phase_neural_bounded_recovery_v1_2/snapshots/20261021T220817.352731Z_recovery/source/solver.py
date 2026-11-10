"""Finite-radius minimum L2, exact multilayer ReLU encoding and numerical MIP bounds.

The requested object is min(c,tau). A proof of no accepting point in the tau
ball implies capped cost tau, not global infeasibility. Saved ORPM witnesses
tighten upper bounds AFTER optimization and never initialize the solver.
"""
import gzip, shutil, time
from pathlib import Path
import numpy as np
import gurobipy as gp
from common import logits, feasible, sha
from numerical_guard import encoded_witness_audit, reject_contradicted_certificate

def environment():
 env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start();return env
def network_bounds(net,x0,tau,free):
 bounds=[]
 for i,(W,b) in enumerate(net):
  if i==0:
   c=W@x0+b;r=tau*np.linalg.norm(W[:,free],axis=1);lo=c-r;hi=c+r
  else:
   pos=np.maximum(W,0);neg=np.minimum(W,0)
   lo=pos@al+neg@ah+b;hi=pos@ah+neg@al+b
  pad=1e-8*(1+np.maximum(abs(lo),abs(hi)));lo-=pad;hi+=pad
  bounds.append((lo,hi));al=np.maximum(lo,0);ah=np.maximum(hi,0)
 return bounds
def build(nets,x0,tau,free,lower,upper,env,seconds):
 m=gp.Model('neural_capped_minimum_L2',env=env)
 m.Params.OutputFlag=0;m.Params.Threads=1;m.Params.Seed=0;m.Params.TimeLimit=seconds
 m.Params.FeasibilityTol=1e-9;m.Params.IntFeasTol=1e-9;m.Params.OptimalityTol=1e-9
 m.Params.MIPGap=1e-6;m.Params.MIPGapAbs=1e-9;m.Params.NumericFocus=2;m.Params.DualReductions=0
 # Normalize only the displacement variables and objective, not network outputs.
 y=[m.addVar(lb=max(-1.,float((lower[j]-x0[j])/tau)),ub=min(1.,float((upper[j]-x0[j])/tau)),name=f'u_{j}') for j in free]
 q=gp.quicksum(v*v for v in y);m.addQConstr(q<=1.,name='normalized_L2_ball')
 zs=[];nb=0;bound_records=[]
 for ni,net in enumerate(nets):
  bounds=network_bounds(net,x0,tau,free);bound_records.append([(a.tolist(),b.tolist()) for a,b in bounds]);previous=None
  for layer,(W,b) in enumerate(net):
   following=[]
   for h in range(len(b)):
    if layer==0:expr=gp.LinExpr(float(W[h]@x0+b[h]))+gp.quicksum(float(tau*W[h,j])*v for j,v in zip(free,y))
    else:expr=gp.LinExpr(float(b[h]))+gp.quicksum(float(w)*v for w,v in zip(W[h],previous))
    lo,hi=map(float,(bounds[layer][0][h],bounds[layer][1][h]))
    if layer==len(net)-1:
     z=m.addVar(lb=lo,ub=hi,name=f'logit_{ni}');m.addConstr(z==expr);m.addConstr(z>=0.,name=f'accept_{ni}');zs.append(z)
    elif hi<=0:following.append(0.)
    elif lo>=0:following.append(expr)
    else:
     z=m.addVar(lb=0.,ub=hi,name=f'h_{ni}_{layer}_{h}');a=m.addVar(vtype=gp.GRB.BINARY,name=f'a_{ni}_{layer}_{h}');nb+=1
     m.addConstr(z>=expr);m.addConstr(z<=expr-lo*(1-a));m.addConstr(z<=hi*a);following.append(z)
   previous=following
 m.setObjective(q,gp.GRB.MINIMIZE);m.update()
 return m,y,zs,dict(binary_count=nb,neuron_bounds=bound_records)
def safe_export(m,folder):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 plain=folder/'problem.mps';compressed=folder/'problem.mps.gz'
 m.write(str(plain))
 original=sha(plain)
 with open(plain,'rb') as src,gzip.open(compressed,'wb') as dst:shutil.copyfileobj(src,dst)
 import hashlib
 h=hashlib.sha256()
 with gzip.open(compressed,'rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 assert h.hexdigest()==original
 plain.unlink();return original
def solve(nets,x0,tau,free,lower,upper,env,seconds,folder,witnesses,mark=lambda **kw:None):
 started=time.perf_counter();z0=logits(nets,x0)
 if np.min(z0)>=0:
  return x0.copy(),dict(status='factual_already_accepted',solver_status=None,available=True,cost_lower=0.,cost_upper=0.,capped_lower=0.,capped_upper=0.,certified=True,infeasible_within_horizon=False,seconds=time.perf_counter()-started,optimization_seconds=0.,binary_count=0,upper_source='factual',native_incumbent=False)
 if not len(free):
  return x0.copy(),dict(status='immutable_rejection',solver_status=None,available=False,cost_lower=None,cost_upper=None,capped_lower=tau,capped_upper=tau,certified=True,infeasible_within_horizon=True,seconds=time.perf_counter()-started,optimization_seconds=0.,binary_count=0,upper_source=None,native_incumbent=False)
 mark(stage='encoding',models=len(nets));m,y,zs,audit=build(nets,x0,tau,free,lower,upper,env,seconds)
 try:
  folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
  m.Params.OutputFlag=1;m.Params.LogToConsole=0;m.Params.LogFile=str(folder/'solver.log')
  mark(stage='exporting',models=len(nets));export_hash=safe_export(m,folder)
  mark(stage='optimizing',models=len(nets));m.optimize();status=int(m.Status);count=int(m.SolCount)
  if status in [gp.GRB.NUMERIC,gp.GRB.INF_OR_UNBD,gp.GRB.UNBOUNDED]:raise RuntimeError(f'Invalid solver termination status {status}')
  try:bound=float(m.ObjBound)
  except (gp.GurobiError,AttributeError):bound=None
  if bound is not None and (not np.isfinite(bound) or abs(bound)>1e50):bound=None
  candidates=[(np.asarray(x,float),'saved_ORPM') for x in witnesses];raw=None;forward_error=None;native_valid=False
  if count:
   raw=x0.copy();raw[free]+=tau*np.array([v.X for v in y]);encoded=np.array([z.X for z in zs]);actual=logits(nets,raw)
   forward_error=float(np.max(abs(encoded-actual)))
   if forward_error>1e-5*(1+float(np.max(abs(encoded)))):raise RuntimeError('Independent network/encoded-logit disagreement')
   native_valid=bool(feasible(raw,x0,tau,free,lower,upper) and actual.min()>=0)
   candidates.append((raw,'MIQCP'))
   # Deterministic radial numerical repair, accepted only after strict forward checks.
   for factor in [1-1e-9,1+1e-9,1-1e-7,1+1e-7,1-1e-5,1+1e-5]:candidates.append((x0+factor*(raw-x0),'MIQCP_radial_repair'))
  valid=[(p,src) for p,src in candidates if feasible(p,x0,tau,free,lower,upper) and logits(nets,p).min()>=0]
  point,origin=min(valid,key=lambda v:np.linalg.norm(v[0]-x0)) if valid else (x0.copy(),None)
  upper_cost=float(np.linalg.norm(point-x0)) if valid else None
  infeasible=status==gp.GRB.INFEASIBLE
  rejection=None;rejection_audit=None
  if infeasible and valid:
   rejection='infeasibility_contradicted_by_verified_witness'
   rejection_audit=encoded_witness_audit(m,nets,x0,tau,free,point)
   infeasible=False;bound=None
  raw_lower=tau*np.sqrt(max(0.,bound or 0.));lo=tau if infeasible else min(tau,raw_lower)
  hi=min(tau,upper_cost) if valid else tau
  if lo>hi+1e-5*(1+tau):
   if not valid:raise RuntimeError(f'Contradictory cost bounds without witness {lo} > {hi}')
   rejection='lower_bound_contradicted_by_verified_witness'
   rejection_audit=encoded_witness_audit(m,nets,x0,tau,free,point)
   lo=raw_lower=0.
  numerical_clip=max(0.,lo-hi);lo=min(lo,hi)
  certified=bool(infeasible or (valid and hi-lo<=1e-4*(1+tau)))
  info=dict(audit,status='certified_no_acceptance_within_horizon' if infeasible else ('certified_cost' if certified else ('cost_bounds_with_witness' if valid else 'unresolved_no_valid_witness')),
   solver_status=status,solver_solutions=count,normalized_objective=float(m.ObjVal) if count else None,normalized_best_bound=bound,
   optimization_seconds=float(m.Runtime),seconds=time.perf_counter()-started,node_count=float(m.NodeCount),export_uncompressed_sha256=export_hash,
   available=bool(valid),native_incumbent=native_valid,cost_lower=None if infeasible else float(min(raw_lower,hi) if valid else raw_lower),cost_upper=upper_cost,
   capped_lower=float(lo),capped_upper=float(hi),certified=certified,infeasible_within_horizon=infeasible,upper_source=origin,
   numerical_lower_clip=float(numerical_clip),forward_error=forward_error,raw_point=None if raw is None else raw.tolist())
  if rejection:info=reject_contradicted_certificate(info,rejection,rejection_audit)
  return point,info
 finally:m.dispose()
