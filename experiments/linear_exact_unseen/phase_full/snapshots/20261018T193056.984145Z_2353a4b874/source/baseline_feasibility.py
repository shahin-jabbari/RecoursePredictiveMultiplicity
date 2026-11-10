"""Repair only numerical feasibility drift; reject material solver violations."""
import numpy as np

POLICY = 'float64_numerical_boundary_repair_v1'

def repair_baseline_point(point, origin, budget, constraints):
    raw=np.asarray(point,dtype=np.float64).copy();origin=np.asarray(origin,dtype=np.float64)
    if raw.shape!=origin.shape or not np.isfinite(raw).all():
        raise ValueError('Baseline returned a nonfinite point or an incompatible shape')
    if budget<0 or not np.isfinite(budget):raise ValueError('Invalid budget')
    lo=np.asarray(constraints.lower,float);hi=np.asarray(constraints.upper,float)
    if (origin<lo).any() or (origin>hi).any():raise ValueError('The factual must lie in the feasible box')
    cost=float(np.linalg.norm(raw-origin))
    budget_excess=max(0.,cost-budget)
    box_excess=max(0.,float(np.max(lo-raw)),float(np.max(raw-hi)))
    immutable_excess=float(np.max(np.abs(raw[constraints.immutable]-origin[constraints.immutable]))) if len(constraints.immutable) else 0.
    # The old solver explicitly allowed 1e-5; account for float32 norm/conversion
    # roundoff around that boundary, without allowing an arbitrary infeasible result.
    allowance=1e-5+8*np.finfo(np.float32).eps*max(1.,budget,float(np.linalg.norm(origin)),float(np.linalg.norm(raw)))
    largest=max(budget_excess,box_excess,immutable_excess)
    if largest>allowance:
        raise ValueError(f'Material baseline feasibility violation {largest:.12g}; numerical allowance {allowance:.12g}')
    point=np.clip(raw,lo,hi);point[constraints.immutable]=origin[constraints.immutable]
    d=point-origin;norm=float(np.linalg.norm(d))
    if norm>budget:
        point=origin+d*((budget/norm)*(1-1e-12)) if budget else origin.copy()
    point[constraints.immutable]=origin[constraints.immutable]
    final_cost=float(np.linalg.norm(point-origin))
    if final_cost>budget+1e-12*max(1.,budget) or (point<lo).any() or (point>hi).any():
        raise ValueError('Numerical repair did not restore strict encoded feasibility')
    audit=dict(policy=POLICY,applied=bool(np.any(point!=raw)),distance=float(np.linalg.norm(point-raw)),
        raw_cost=cost,returned_cost=final_cost,budget=float(budget),raw_budget_excess=budget_excess,
        raw_box_excess=box_excess,raw_immutable_excess=immutable_excess,numerical_allowance=allowance)
    return point,audit
