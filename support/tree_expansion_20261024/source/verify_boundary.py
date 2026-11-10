"""Independent path-based validation of rejected leaf combinations."""
from fractions import Fraction
import numpy as np

def verify_budget_proof(models,x0,budget,constraints,info):
 if not info.get('incumbent_rejected'):return
 proof=info['budget_rejection_proof'];selected=proof['selected_leaves'];d=len(x0)
 lo=np.maximum(np.asarray(constraints.lower,float),x0.numpy().astype(float)-budget)
 hi=np.minimum(np.asarray(constraints.upper,float),x0.numpy().astype(float)+budget)
 origin=x0.numpy().astype(float);k=0
 for model in models:
  for tree in model.trees:
   target=int(selected[k]);k+=1
   stack=[(0,[])]
   path=None
   while stack:
    node,conditions=stack.pop()
    if tree.left[node]==-1:
     if node==target:path=conditions;break
    else:
     f=int(tree.feature[node]);t=float(tree.threshold[node]);rounded=np.float32(t)
     if float(rounded)>t:rounded=np.nextafter(rounded,np.float32(-np.inf))
     right=np.nextafter(rounded,np.float32(np.inf))
     stack.append((int(tree.left[node]),conditions+[(f,'upper',float(rounded))]))
     stack.append((int(tree.right[node]),conditions+[(f,'lower',float(right))]))
   assert path is not None
   for f,side,v in path:
    if side=='upper':hi[f]=min(hi[f],v)
    else:lo[f]=max(lo[f],v)
 assert np.all(lo<=hi)
 near=np.clip(origin,lo,hi).astype(np.float32)
 assert np.array_equal(near,np.asarray(proof['nearest_point'],np.float32))
 assert np.array_equal(near[constraints.immutable].astype(float),origin[constraints.immutable])
 q=sum((Fraction(float(a))-Fraction(float(b)))**2 for a,b in zip(near,origin));cap=Fraction(float(budget))**2
 assert q>cap and str(q)==proof['exact_squared_cost'] and str(q-cap)==proof['exact_excess']
 assert info['lower_bound']==0 and not info['global_certificate'] and info['additional_optimization_calls']==0
 assert info['incumbent_verified'] is False and info['returned_action_verified'] is True
 return True
