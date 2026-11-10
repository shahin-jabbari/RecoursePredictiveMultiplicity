"""Reject numerically infeasible tree incumbents without relaxing the action ball.

Every excluded leaf combination is proved unreachable using exact rational
arithmetic on the saved float32 endpoints. No optimization or retry occurs here.
"""
from fractions import Fraction
import numpy as np
import torch


def proof_for_leaves(oracle, selected_leaves):
    if len(selected_leaves) != len(oracle.groups):
        raise RuntimeError('Incomplete selected-leaf record')
    chosen=[]
    for group,node in zip(oracle.groups,selected_leaves):
        match=[g for g in group if g['node']==int(node)]
        if len(match)!=1: raise RuntimeError('Selected leaf absent or ambiguous in encoding')
        chosen.append(match[0])
    lower=np.maximum.reduce([oracle.lower]+[g['lower'] for g in chosen])
    upper=np.minimum.reduce([oracle.upper]+[g['upper'] for g in chosen])
    if not np.all(lower<=upper): raise RuntimeError('Inconsistent leaf boxes; not a budget-boundary rejection')
    nearest=np.maximum(lower,np.minimum(upper,oracle.origin)).astype(np.float32)
    # All finite endpoints and the factual are representable float32 numbers.
    if not np.array_equal(nearest.astype(float),np.maximum(lower,np.minimum(upper,oracle.origin))):
        raise RuntimeError('Nearest leaf point is not exactly representable')
    if not np.isfinite(nearest).all(): raise RuntimeError('Nonfinite candidate')
    from common import rt
    predictions=rt.probabilities(oracle.models,nearest)
    expected=[];cursor=0
    for model in oracle.models:
        count=len(model.trees)
        expected.append(np.mean([g['value'] for g in chosen[cursor:cursor+count]]));cursor+=count
    if not np.allclose(predictions,expected,rtol=0,atol=1e-12):
        raise RuntimeError('Saved-tree traversal disagrees with selected leaves')
    if not np.array_equal(nearest[oracle.constraints.immutable].astype(float),oracle.origin[oracle.constraints.immutable]):
        raise RuntimeError('Immutable-coordinate mismatch')
    q=sum((Fraction(float(a))-Fraction(float(o)))**2 for a,o in zip(nearest,oracle.origin))
    cap=Fraction(float(oracle.budget))**2
    cost=float(np.linalg.norm(nearest.astype(float)-oracle.origin))
    if not(q>cap and cost>oracle.budget+1e-7):
        raise RuntimeError('Not a proven out-of-budget leaf combination')
    return dict(passed=True,method='exact_rational_squared_distance_to_intersection_of_representable_leaf_boxes',
                selected_leaves=[int(n) for n in selected_leaves],nearest_point=nearest,
                candidate_probabilities=predictions,candidate_cost=cost,budget=float(oracle.budget),
                exact_squared_cost=str(q),exact_squared_budget=str(cap),exact_excess=str(q-cap),
                cost_excess=cost-oracle.budget),chosen


def apply_rejection(oracle,info,weights,objective,known_points=()):
    from common import loss,point_check,audit_encoding,append,clean
    if objective not in ['weighted_loss','minimax']:
        raise RuntimeError('Recovery only supports the approved loss objectives')
    v=info.get('verification',{})
    if not(v.get('leaf_box_consistent') and v.get('predictions_match') and
           v.get('threshold_satisfied') and v.get('candidate_feasible') is False):
        raise RuntimeError('Invalid encoding/prediction or unsupported validation failure')
    proof,chosen=proof_for_leaves(oracle,info['selected_leaves'])
    if abs(proof['candidate_cost']-v['candidate_cost'])>1e-12 or not np.allclose(
            proof['candidate_probabilities'],v['candidate_probabilities'],rtol=0,atol=1e-12):
        raise RuntimeError('Recorded candidate differs from independently reconstructed leaf point')
    options=[oracle.x0,*known_points]
    if not all(point_check(x,oracle.x0,oracle.budget,oracle.constraints) for x in options):
        raise RuntimeError('Recovery fallback includes an invalid saved action')
    score=lambda x: float(loss(oracle.models,x)@weights) if objective=='weighted_loss' else float(loss(oracle.models,x).max())
    point=min(options,key=score).clone()
    # The exclusion is a valid inequality, not a smaller budget or a changed threshold.
    import gurobipy as gp
    signature=tuple(info['selected_leaves'])
    excluded=getattr(oracle,'excluded_budget_leaf_combinations',set())
    added=signature not in excluded
    if added:
        name=f'proven_outside_ball_{len(excluded)}'
        oracle.problem.addConstr(gp.quicksum(g['z'] for g in chosen)<=len(chosen)-1,name=name)
        oracle.problem.update();excluded.add(signature);oracle.excluded_budget_leaf_combinations=excluded
    else:name='already_excluded_combination'
    audit=audit_encoding(oracle,point,objective)
    if not audit['passed']: raise RuntimeError('Feasible fallback failed complete encoded-witness audit')
    oracle._warm_start(point)
    raw=dict(info)
    info=dict(info,status='budget_infeasible_incumbent_rejected',native_status=raw['status'],
              raw_native_info=raw,incumbent_rejected=True,incumbent_verified=False,
              returned_action_verified=True,objective=score(point),lower_bound=0.,
              absolute_gap=score(point),global_certificate=False,certificate_rejected=True,
              certificate_rejection='Numerically infeasible incumbent; use conservative zero lower bound',
              budget_rejection_proof=proof,exclusion_cut=name,witness_audit=audit,
              model_probabilities=__import__('common').rt.probabilities(oracle.models,point),
              fallback_rule='best weighted-loss (or minimax-loss) of factual and saved best action; no new optimization',
              additional_optimization_calls=0)
    if oracle.artifact_dir:
        append(oracle.artifact_dir/'budget_rejections.jsonl',dict(call=oracle.calls,proof=proof,cut_name=name,cut_added=added,witness_audit=audit))
    return point,info
