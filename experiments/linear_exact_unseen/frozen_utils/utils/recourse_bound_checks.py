"""Track independent lower-bound sources and revoke them when witnesses disagree.

Nonnegative component losses permit transporting a lower bound between weight
vectors. Store original sources, never transported aggregates, so revoking a
source also revokes every use of it. Witness checks are necessary consistency
checks, not proofs of floating-point solver correctness.
"""
import numpy as np

DIAGNOSTIC_VERSION = 'witness_checked_v2'
DIAGNOSTIC_POLICY = dict(version=DIAGNOSTIC_VERSION,
    solver_tolerances=dict(FeasibilityTol=1e-9,IntFeasTol=1e-9,OptimalityTol=1e-9,
                          NumericFocus=3,MIPGap=0,MIPGapAbs=1e-10),
    minimum_tangent_slope=1e-8,solver_bound_guard=1e-7,
    bound_guard_is_rigorous_error_enclosure=False,
    discard_contradicted_sources_and_transports=True,
    diagnostic_witnesses='all saved feasible responses, plus local/global evaluation responses',
    evaluation_is_post_hoc=True,changes_orpm_trajectory=False)


class DualBoundLedger:
    def __init__(self, losses):
        self.losses=np.asarray(losses,float).copy()
        if self.losses.ndim!=2 or not np.isfinite(self.losses).all() or (self.losses<0).any():
            raise ValueError('Witness losses must be finite and nonnegative')
        self.sources=[];self.invalidations=[]

    def add_witnesses(self,losses):
        values=np.atleast_2d(np.asarray(losses,float))
        if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid witness losses')
        self.losses=np.vstack([self.losses,values])
        self.check()

    def add_source(self,round,weights,value,numerical=False,label=''):
        weights=np.asarray(weights,float)
        if not np.isfinite(value) or value<0 or (weights<0).any() or not np.isclose(weights.sum(),1.):
            raise ValueError('Invalid lower-bound source')
        source=dict(round=int(round),weights=weights.copy(),value=float(value),
                    numerical=bool(numerical),label=label,invalidated=False)
        self.sources.append(source);self.check()
        return source

    def check(self):
        for s in self.sources:
            if s['invalidated']:continue
            upper=float(np.min(self.losses@s['weights']))
            if s['value']>upper:
                if not s['numerical']:
                    raise RuntimeError(f"Independent analytic lower bound contradicts witness: {s['value']} > {upper}")
                s['invalidated']=True
                self.invalidations.append(dict(round=s['round'],label=s['label'],discarded_lower=s['value'],
                    witness_upper=upper,reason='feasible_witness_contradiction'))

    def interval(self,round,weights):
        w=np.asarray(weights,float);lower=0.;used=None
        for s in self.sources:
            if s['invalidated'] or s['round']>round:continue
            positive=s['weights']>0
            scale=float(np.min(w[positive]/s['weights'][positive]))
            candidate=scale*s['value']
            if candidate>lower:lower=candidate;used=s['label']
        upper=float(np.min(self.losses@w))
        if lower>upper:raise RuntimeError('Bound ledger failed its consistency check')
        return lower,upper,used
