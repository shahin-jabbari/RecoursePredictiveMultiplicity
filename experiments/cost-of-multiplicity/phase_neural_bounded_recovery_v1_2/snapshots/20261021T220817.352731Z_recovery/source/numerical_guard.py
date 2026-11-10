"""Reject contradicted numerical certificates; never relax the action constraints.
No optimization, warm start, or retries are performed by this guard.
"""
import numpy as np

def encoded_witness_audit(model,nets,x0,tau,free,point):
 values={f'u_{j}':float((point[j]-x0[j])/tau) for j in free}
 for ni,net in enumerate(nets):
  a=np.asarray(point,float)
  for layer,(W,b) in enumerate(net):
   pre=W@a+b
   if layer==len(net)-1:values[f'logit_{ni}']=float(pre[0])
   else:
    for h,z in enumerate(pre):values[f'h_{ni}_{layer}_{h}']=float(max(0,z));values[f'a_{ni}_{layer}_{h}']=float(z>=0)
    a=np.maximum(0,pre)
 worst=0.;worst_name=None
 def check(value,name):
  nonlocal worst,worst_name
  if value>worst:worst=float(value);worst_name=name
 for v in model.getVars():
  x=values[v.VarName];check(max(0,v.LB-x,x-v.UB),v.VarName)
  if v.VType in ['B','I']:check(abs(x-round(x)),v.VarName+'_integrality')
 for c in model.getConstrs():
  row=model.getRow(c);lhs=sum(row.getCoeff(i)*values[row.getVar(i).VarName] for i in range(row.size()))
  check(abs(lhs-c.RHS) if c.Sense=='=' else max(0,lhs-c.RHS) if c.Sense=='<' else max(0,c.RHS-lhs),c.ConstrName)
 for c in model.getQConstrs():
  row=model.getQCRow(c);lin=row.getLinExpr();lhs=sum(lin.getCoeff(i)*values[lin.getVar(i).VarName] for i in range(lin.size()))+sum(row.getCoeff(i)*values[row.getVar1(i).VarName]*values[row.getVar2(i).VarName] for i in range(row.size()))
  check(abs(lhs-c.QCRHS) if c.QCSense=='=' else max(0,lhs-c.QCRHS) if c.QCSense=='<' else max(0,c.QCRHS-lhs),c.QCName)
 audit=dict(passed=bool(worst<=1e-7),max_constraint_violation=worst,worst_constraint=worst_name,variables=model.NumVars,linear_constraints=model.NumConstrs,quadratic_constraints=model.NumQConstrs,tolerance=1e-7)
 if not audit['passed']:raise RuntimeError('Feasible network point fails encoded model audit: '+str(audit))
 return audit

def reject_contradicted_certificate(info,reason,audit):
 assert audit['passed'] and info['available']
 info.update(status='numerical_certificate_rejected_with_witness',numerical_issue=reason,solver_certificate_rejected=True,certificate_rejection_audit=audit,raw_normalized_best_bound=info.get('normalized_best_bound'),normalized_best_bound=None,cost_lower=0.,capped_lower=0.,infeasible_within_horizon=False,certified=False,numerical_lower_clip=0.)
 return info
