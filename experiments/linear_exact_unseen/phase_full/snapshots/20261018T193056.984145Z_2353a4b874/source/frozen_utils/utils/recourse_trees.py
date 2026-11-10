"""Tree/forest recourse on original predictions, with auditable solver bounds.

Leaf encodings follow the mathematical-programming approach of Parmentier &
Vidal (ICML 2021), not a vendored OCEAN implementation. Recourse coordinates
are float32, matching sklearn's prediction input conversion. Leaf endpoints
are adjacent representable float32 values; no arbitrary split margin is used.
All global bounds are solver bounds subject to its numerical tolerances.
Shared integer split cells enforce consistent branch choices even when two
float32 leaf endpoints are closer than the solver's numerical resolution.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.stats import binom, norm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from . import recourse_core as rc
from . import recourse_baselines as rb
from . import recourse_results as rr


def f32_ceil(value):
    value = np.asarray(value, dtype=float)
    rounded = value.astype(np.float32)
    return np.where(rounded.astype(float) < value,
                    np.nextafter(rounded, np.float32(np.inf)), rounded).astype(float)


def f32_floor(value):
    value = np.asarray(value, dtype=float)
    rounded = value.astype(np.float32)
    return np.where(rounded.astype(float) > value,
                    np.nextafter(rounded, np.float32(-np.inf)), rounded).astype(float)


@dataclass
class TreeSpec:
    left: np.ndarray
    right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    probability: np.ndarray

    @classmethod
    def from_estimator(cls, estimator, class_labels=None):
        tree = estimator.tree_
        values = tree.value[:, 0, :]
        index = np.flatnonzero(np.asarray(estimator.classes_ if class_labels is None else class_labels) == 1)
        p = values[:, index[0]] / values.sum(axis=1) if len(index) else np.zeros(len(values))
        return cls(tree.children_left.copy(), tree.children_right.copy(),
                   tree.feature.copy(), tree.threshold.copy(), p.copy())

    def apply(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        nodes = np.zeros(len(X), dtype=int)
        while True:
            active = np.flatnonzero(self.left[nodes] != -1)
            if not len(active):
                break
            n = nodes[active]
            go_left = X[active, self.feature[n]].astype(float) <= self.threshold[n]
            nodes[active] = np.where(go_left, self.left[n], self.right[n])
        return nodes

    def leaves(self, dimension):
        out = []
        def visit(node, lower, upper, path):
            if self.left[node] == -1:
                out.append((node, lower, upper, path))
                return
            feature, threshold = int(self.feature[node]), float(self.threshold[node])
            left_max = float(f32_floor(threshold))
            right_min = float(np.nextafter(np.float32(left_max), np.float32(np.inf)))
            hi = upper.copy(); hi[feature] = min(hi[feature], left_max)
            visit(self.left[node], lower.copy(), hi, path + [(feature, threshold, True)])
            lo = lower.copy(); lo[feature] = max(lo[feature], right_min)
            visit(self.right[node], lo, upper.copy(), path + [(feature, threshold, False)])
        visit(0, np.full(dimension, -np.inf), np.full(dimension, np.inf), [])
        return out


class TreeModel:
    """Safe array-based sklearn-compatible predictor; no autograd is implied."""
    def __init__(self, trees, dimension, family='decision_tree', aggregation='probability', metadata=None):
        if aggregation not in {'probability', 'hard_vote'}:
            raise ValueError('aggregation must be probability or hard_vote')
        self.trees, self.dimension, self.family = list(trees), int(dimension), family
        self.aggregation = aggregation
        self.metadata = metadata or {}
        self.recourse_context = None

    @classmethod
    def from_estimator(cls, estimator, aggregation='probability', metadata=None):
        is_forest = isinstance(estimator, RandomForestClassifier)
        estimators = estimator.estimators_ if is_forest else [estimator]
        return cls([TreeSpec.from_estimator(t, estimator.classes_) for t in estimators], estimator.n_features_in_,
                   'random_forest' if is_forest else 'decision_tree', aggregation, metadata)

    def leaf_value(self, tree, node):
        p = tree.probability[node]
        # sklearn breaks individual-tree ties in favor of class 0.
        return (p > .5).astype(float) if self.aggregation == 'hard_vote' else p

    def predict_probability(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        return np.mean([self.leaf_value(t, t.apply(X)) for t in self.trees], axis=0)

    def __call__(self, X):
        tensor = isinstance(X, torch.Tensor)
        values = X.detach().cpu().numpy() if tensor else X
        result = self.predict_probability(values)
        return torch.as_tensor(result, dtype=X.dtype if tensor else torch.float32,
                               device=X.device if tensor else 'cpu')

    def logits(self, X):
        return torch.logit(self(X).clamp(1e-6, 1-1e-6))

    def save(self, directory):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        arrays = {f'tree_{i}/{key}': getattr(t, key) for i, t in enumerate(self.trees)
                  for key in ['left', 'right', 'feature', 'threshold', 'probability']}
        rr._npz(directory/'trees.npz', arrays)
        rr.atomic_json(directory/'model.json', {'family': self.family, 'dimension': self.dimension,
            'aggregation': self.aggregation, 'tree_count': len(self.trees), 'metadata': self.metadata})

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        info = json.loads((directory/'model.json').read_text())
        with np.load(directory/'trees.npz', allow_pickle=False) as arrays:
            trees = [TreeSpec(**{key: arrays[f'tree_{i}/{key}'].copy() for key in
                      ['left', 'right', 'feature', 'threshold', 'probability']})
                     for i in range(info['tree_count'])]
        return cls(trees, info['dimension'], info['family'], info['aggregation'], info['metadata'])


def probabilities(models, point):
    point = point.detach().cpu().numpy() if isinstance(point, torch.Tensor) else point
    return np.asarray([m.predict_probability(point)[0] for m in models], dtype=float)


class TreeOracle:
    """Reusable mixed-integer quadratic oracle for a fixed model set and factual.

    Float32 leaf boxes are disjoint. Continuous x variables relax the float32
    domain, so the solver objective bound remains a lower bound. For every
    selected leaf combination, projecting the float32 factual onto the leaf
    box yields the closest representable point with those predictions.
    Integer cell indices additionally enforce that leaves from different trees
    agree on each feature interval; tiny coordinate gaps alone cannot do this
    reliably under finite-precision mixed-integer feasibility tolerances.
    """
    def __init__(self, models, x0, budget, constraints=None, time_limit=10.0,
                 abs_gap=1e-5, seed=42, artifact_dir=None):
        import gurobipy as gp
        if not models or budget < 0 or not np.isfinite(budget) or time_limit <= 0 or abs_gap < 0:
            raise ValueError('Require models, finite nonnegative budget/gap and positive time limit.')
        self.models = models
        self.x0 = torch.as_tensor(x0, dtype=torch.float32).detach().cpu().clone()
        self.origin = self.x0.numpy().astype(float)
        self.budget = float(budget)
        self.constraints = constraints or rc.constraints_for(models, self.x0)
        self.lower = f32_ceil(np.maximum(self.constraints.lower, self.origin-budget))
        self.upper = f32_floor(np.minimum(self.constraints.upper, self.origin+budget))
        if np.any(self.lower > self.upper):
            raise ValueError('The action domain is empty.')
        self.abs_gap = abs_gap
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.calls = 0
        self.env = gp.Env(empty=True); self.env.setParam('OutputFlag', 0); self.env.start()
        self.problem = gp.Model('tree_recourse', env=self.env)
        m = self.problem
        m.Params.TimeLimit = float(time_limit)
        m.Params.MIPGap = 0.0
        m.Params.MIPGapAbs = float(abs_gap)
        m.Params.FeasibilityTol = 1e-9
        m.Params.IntFeasTol = 1e-9
        m.Params.Threads = 1
        m.Params.Seed = int(seed) % 2000000000
        self.x = [m.addVar(lb=lo, ub=hi, name=f'x_{i}')
                  for i, (lo, hi) in enumerate(zip(self.lower, self.upper))]
        # Each distinct representable split boundary partitions a feature into
        # ordered cells. The integer distance between adjacent cells is one,
        # regardless of how small the corresponding coordinate gap is.
        cuts=[set() for _ in self.x]
        for model in models:
            for tree in model.trees:
                for node in np.flatnonzero(tree.left!=-1):
                    feature=int(tree.feature[node])
                    cut=float(f32_floor(tree.threshold[node]))
                    if self.lower[feature]<=cut<self.upper[feature]:
                        cuts[feature].add(cut)
        self.split_positions=[np.asarray(sorted(values),dtype=float) for values in cuts]
        self.cell_indices={f:m.addVar(lb=0,ub=len(values),vtype=gp.GRB.INTEGER,
                                     name=f'split_cell_{f}')
                           for f,values in enumerate(self.split_positions) if len(values)}
        self.distance = gp.quicksum((v-float(a))*(v-float(a)) for v,a in zip(self.x,self.origin))
        m.addQConstr(self.distance <= self.budget**2, name='cost_budget')
        self.groups, self.p = [], []
        self.model_constraints = []
        for j, model in enumerate(models):
            tree_values = []
            for k, tree in enumerate(model.trees):
                group = []
                for node, lo, hi, path in tree.leaves(len(self.x)):
                    lower, upper = np.maximum(lo,self.lower), np.minimum(hi,self.upper)
                    nearest = np.maximum(lower,np.minimum(upper,self.origin))
                    if np.any(lower > upper) or np.sum((nearest-self.origin)**2) > budget**2 + 1e-10:
                        continue
                    z = m.addVar(vtype=gp.GRB.BINARY, name=f'leaf_{j}_{k}_{node}')
                    for f in np.flatnonzero(lower > self.lower):
                        m.addGenConstrIndicator(z, True, self.x[f] >= float(lower[f]))
                        if f in self.cell_indices:
                            cell=int(np.searchsorted(self.split_positions[f],lower[f],side='left'))
                            m.addGenConstrIndicator(z,True,self.cell_indices[f]>=cell)
                    for f in np.flatnonzero(upper < self.upper):
                        m.addGenConstrIndicator(z, True, self.x[f] <= float(upper[f]))
                        if f in self.cell_indices:
                            cell=int(np.searchsorted(self.split_positions[f],upper[f],side='left'))
                            m.addGenConstrIndicator(z,True,self.cell_indices[f]<=cell)
                    group.append({'z':z, 'node':int(node), 'lower':lower, 'upper':upper,
                                  'value':float(model.leaf_value(tree,np.asarray(node)))})
                m.addConstr(gp.quicksum(g['z'] for g in group) == 1, name=f'one_leaf_{j}_{k}')
                tree_values.append(gp.quicksum(g['value']*g['z'] for g in group))
                self.groups.append(group)
            p = m.addVar(lb=0, ub=1, name=f'probability_{j}')
            m.addConstr(p == gp.quicksum(tree_values)/len(model.trees))
            self.p.append(p)
        self.epigraph = m.addVar(lb=0,ub=1,name='worst_loss')
        m.update()
        self._warm_start(self.x0)
        if self.artifact_dir:
            self.artifact_dir.mkdir(parents=True,exist_ok=True)
            m.Params.LogFile = str(self.artifact_dir/'solver.log')
            rr.atomic_json(self.artifact_dir/'formulation.json', {
                'domain':'float32 coordinates; continuous relaxation with representable leaf bounds',
                'factual':self.origin, 'budget':budget, 'lower':self.lower, 'upper':self.upper,
                'models':len(models), 'trees':len(self.groups), 'leaf_variables':sum(map(len,self.groups)),
                'encoding':'shared_integer_split_cells_v1',
                'integer_cell_variables':len(self.cell_indices),
                'split_positions':self.split_positions,
                'time_limit_per_solve':time_limit, 'absolute_gap_target':abs_gap,
                'feasibility_tolerance':1e-9, 'split_margin':None})

    def close(self):
        self.problem.dispose(); self.env.dispose()

    def _warm_start(self, point):
        values = point.detach().cpu().numpy().astype(float)
        for v, a in zip(self.x,values): v.Start=float(a)
        for feature,variable in self.cell_indices.items():
            variable.Start=int(np.searchsorted(self.split_positions[feature],values[feature],side='left'))
        for group in self.groups:
            for g in group:
                g['z'].Start = float(np.all(values>=g['lower']) and np.all(values<=g['upper']))
        for p, value in zip(self.p, probabilities(self.models,point)): p.Start=float(value)

    def solve(self, weights=None, objective='weighted_loss', threshold=.5):
        import gurobipy as gp
        m = self.problem
        if self.model_constraints:
            m.remove(self.model_constraints); self.model_constraints=[]; m.update()
        if objective == 'weighted_loss':
            w = np.asarray(weights,dtype=float)
            if w.shape != (len(self.p),) or np.any(w<0) or not np.isfinite(w).all() or not np.isclose(w.sum(),1):
                raise ValueError('Oracle weights must be a probability distribution over whole models.')
            expr = gp.quicksum(float(v)*(1-p)*(1-p) for v,p in zip(w,self.p))
        elif objective == 'minimax':
            self.model_constraints = [m.addQConstr((1-p)*(1-p)<=self.epigraph) for p in self.p]
            expr = self.epigraph
        elif objective == 'min_cost':
            if len(self.models) != 1 or not np.isfinite(threshold):
                raise ValueError('Minimum-cost baseline expects one deployed model and a finite threshold.')
            self.model_constraints = [m.addConstr(self.p[0]>=float(threshold))]
            expr = self.distance
        else:
            raise ValueError(f'Unknown objective {objective}')
        m.setObjective(expr,gp.GRB.MINIMIZE)
        m.update()
        if self.artifact_dir and self.calls == 0:
            m.write(str(self.artifact_dir/'formulation.mps'))
        start=time.perf_counter(); m.optimize(); elapsed=time.perf_counter()-start
        lower_bound = None
        if m.Status not in {gp.GRB.INFEASIBLE,gp.GRB.INF_OR_UNBD,gp.GRB.UNBOUNDED}:
            try:
                if np.isfinite(m.ObjBound): lower_bound=max(0.0,float(m.ObjBound))
            except (AttributeError,gp.GurobiError): pass
        point=self.x0.clone(); chosen=[]; verified=False
        verification={}
        status='infeasible' if m.Status==gp.GRB.INFEASIBLE else 'no_incumbent'
        if m.SolCount:
            chosen=[max(group,key=lambda g:g['z'].X) for group in self.groups]
            lo=np.maximum.reduce([self.lower]+[g['lower'] for g in chosen])
            hi=np.minimum.reduce([self.upper]+[g['upper'] for g in chosen])
            proposed=np.maximum(lo,np.minimum(hi,self.origin)).astype(np.float32)
            candidate=torch.from_numpy(proposed)
            actual=probabilities(self.models,candidate)
            encoded=np.asarray([p.X for p in self.p])
            verification=dict(leaf_box_consistent=bool(np.all(lo<=hi)),
                maximum_leaf_box_conflict=float(max(0.0,np.max(lo-hi))),
                candidate_feasible=rc.feasible(candidate,self.x0,self.budget,self.constraints,atol=1e-7),
                candidate_probabilities=actual,encoded_probabilities=encoded,
                predictions_match=bool(np.max(np.abs(actual-encoded))<=1e-6),
                threshold_satisfied=bool(objective!='min_cost' or actual[0]>=threshold-1e-10),
                candidate_cost=float(np.linalg.norm(proposed.astype(float)-self.origin)))
            verified=all(verification[key] for key in ['leaf_box_consistent','candidate_feasible',
                                                       'predictions_match','threshold_satisfied'])
            if verified:
                point=candidate
                status='optimal' if m.Status==gp.GRB.OPTIMAL else 'feasible_incumbent'
                self._warm_start(point)
            else: status='incumbent_failed_validation'
        losses=(1-probabilities(self.models,point))**2
        value=(float(np.dot(w,losses)) if objective=='weighted_loss' else
               float(losses.max()) if objective=='minimax' else float(np.sum((point.numpy()-self.origin)**2)))
        if objective=='min_cost' and not verified: value=None
        # A fallback factual is feasible for loss objectives, but not necessarily for a threshold baseline.
        bound_inconsistent=bool(value is not None and lower_bound is not None and lower_bound>value+1e-7)
        if bound_inconsistent:lower_bound=None
        gap=None if value is None or lower_bound is None else max(0.0,value-lower_bound)
        info={'status':status,'solver_status':int(m.Status),'objective_kind':objective,
              'objective':value,'solver_incumbent_objective':float(m.ObjVal) if m.SolCount else None,
              'lower_bound':lower_bound,'absolute_gap':gap,'incumbent_verified':verified,
              'bound_inconsistent_with_verified_candidate':bound_inconsistent,
              'global_certificate':bool(verified and gap is not None and gap<=self.abs_gap+1e-7),
              'bound_domain':'float32 recourse coordinates; solver numerical tolerances apply',
              'solver_seconds':elapsed,'node_count':float(m.NodeCount),'selected_leaves':[g['node'] for g in chosen],
              'model_probabilities':probabilities(self.models,point),'weights':weights,
              'encoding':'shared_integer_split_cells_v1','verification':verification,
              'threshold':threshold if objective=='min_cost' else None,'gradient_evaluations':0}
        self.calls+=1
        if self.artifact_dir: rr._append(self.artifact_dir/'solves.jsonl',rr.jsonable(info))
        return rc._tag(point,**info)


def tree_orpm(models,x0,budget,T=50,time_limit=10.0,abs_gap=1e-5,constraints=None,artifact_dir=None,seed=42):
    if T<1: raise ValueError('T must be positive')
    start=time.perf_counter()
    oracle=TreeOracle(models,x0,budget,constraints,time_limit,abs_gap,seed,artifact_dir)
    build_seconds=time.perf_counter()-start
    n=len(models); logw=np.full(n,-np.log(n)); eta=np.sqrt(np.log(n)/T)
    best=x0.clone(); upper=float(((1-probabilities(models,best))**2).max()); lower=0.0
    points=[]; weights=[]; rounds=[]; best_points=[]
    try:
        for t in range(T):
            w=np.exp(logw)
            xt=oracle.solve(w)
            losses=(1-probabilities(models,xt))**2
            current=float(losses.max())
            if (current,float(torch.linalg.vector_norm(xt-x0))) < (upper,float(torch.linalg.vector_norm(best-x0))):
                best,upper=xt.clone(),current
            info=xt.recourse_info
            if info['lower_bound'] is not None: lower=max(lower,info['lower_bound'])
            row={'round':t+1,'current_worst_loss':current,'best_worst_loss':upper,
                 'minimax_lower_bound':lower,'minimax_gap_upper_bound':max(0.0,upper-lower),
                 'weighted_oracle_gap':info['absolute_gap'],'oracle_status':info['status'],
                 'elapsed_seconds':time.perf_counter()-start, 'weights':w,'losses':losses,
                 'point':xt.numpy(),'best_point':best.numpy(),'oracle':info}
            rounds.append(row); points.append(xt.numpy()); weights.append(w); best_points.append(best.numpy())
            if rr.current_run(): rr.event('tree_orpm_round',**row)
            if artifact_dir: rr._append(Path(artifact_dir)/'rounds.jsonl',rr.jsonable(row))
            logw+=eta*losses; shift=logw.max();logw-=shift+np.log(np.exp(logw-shift).sum())
    finally:
        oracle.close()
    best=rc._tag(best,status='best_feasible_candidate',gradient_evaluations=0,
        build_seconds=build_seconds,rounds=T,return_rule='best worst-model loss, then cost',
        minimax_lower_bound=lower,minimax_gap_upper_bound=max(0.0,upper-lower),
        theorem='no convex averaged-point guarantee',solver_seconds=sum(r['oracle']['solver_seconds'] for r in rounds),
        certified_oracle_rounds=sum(r['oracle']['global_certificate'] for r in rounds),
        failed_oracle_rounds=sum(not r['oracle']['incumbent_verified'] for r in rounds),
        timeout_oracle_rounds=sum(r['oracle']['solver_status']==9 for r in rounds))
    best.recourse_trace=dict(points=np.asarray(points),best_points=np.asarray(best_points),weights=np.asarray(weights),
        current_worst_loss=np.asarray([r['current_worst_loss'] for r in rounds]),
        best_worst_loss=np.asarray([r['best_worst_loss'] for r in rounds]),
        lower_bounds=np.asarray([r['minimax_lower_bound'] for r in rounds]),
        gap_upper_bounds=np.asarray([r['minimax_gap_upper_bound'] for r in rounds]),
        elapsed_seconds=np.asarray([r['elapsed_seconds'] for r in rounds]))
    return best,rounds


class SmoothTreeModel(torch.nn.Module):
    """FOCUS-style soft splits; our surrogate adaptation, not the FOCUS algorithm."""
    def __init__(self,model,temperature=.1):
        super().__init__()
        if temperature<=0: raise ValueError('temperature must be positive')
        self.model=model; self.temperature=temperature; self.recourse_context=model.recourse_context
        self.paths=[[(path,float(model.leaf_value(tree,np.asarray(node))))
                     for node,lo,hi,path in tree.leaves(model.dimension)] for tree in model.trees]

    def forward(self,X):
        X=X.reshape(-1,self.model.dimension)
        predictions=[]
        for leaves in self.paths:
            p=X.sum(dim=1)*0
            for path,value in leaves:
                mass=torch.ones(len(X),dtype=X.dtype,device=X.device)
                for f,threshold,left in path:
                    score=(threshold-X[:,f])/self.temperature
                    mass=mass*torch.sigmoid(score if left else -score)
                p=p+mass*value
            predictions.append(p)
        return torch.stack(predictions).mean(dim=0)

    def logits(self,X): return torch.logit(self(X).clamp(1e-6,1-1e-6))


def adv_surrogate(models,x0,budget,T=50,temperature=.1,constraints=None):
    smooth=rc.ModelSet([SmoothTreeModel(m,temperature) for m in models],context=getattr(models,'context',None))
    history=rb.ADV(x0,smooth,budget,rc.recourse_loss,T=T,constraints=constraints)
    terminal=history[-1]
    points=getattr(terminal,'recourse_trace',{}).get('points',np.asarray([x0.numpy()]+[p.detach().numpy() for p in history]))
    c=constraints or rc.constraints_for(models,x0)
    feasible=[torch.tensor(p,dtype=x0.dtype) for p in points if rc.feasible(torch.tensor(p,dtype=x0.dtype),x0,budget,c)]
    best=min(feasible,key=lambda p:(float(((1-probabilities(models,p))**2).max()),float(torch.linalg.vector_norm(p-x0))))
    best=rc._tag(best,**terminal.recourse_info)
    best.recourse_info.update(temperature=temperature,adaptation='soft-split ADV; select best original-model candidate')
    best.recourse_trace={'points':points}
    return best


def roar_surrogate(models,x0,budget,T=50,seed=42,constraints=None):
    surrogates=rb.linearize_models_lime(models,x0.numpy(),models.context.X_train,seed=seed)
    a0,b0=rc.linear_parameters(surrogates[0])
    delta=max(max(float((a-a0).abs().max()),float((b-b0).abs())) for a,b in map(rc.linear_parameters,surrogates))
    point=rb.ROAR(x0,surrogates[0],delta,budget,rc.recourse_loss,T=T,constraints=constraints)[-1]
    point.recourse_info.update(adaptation='ROAR on anchored LIME logit surrogates',
        surrogate_diagnostics=[m.surrogate_diagnostics for m in surrogates])
    trace=dict(getattr(point,'recourse_trace',{}))
    trace.update(surrogate_coefficients=np.stack([rc.linear_parameters(m)[0].numpy() for m in surrogates]),
                 surrogate_intercepts=np.asarray([float(rc.linear_parameters(m)[1]) for m in surrogates]))
    point.recourse_trace=trace
    return point


def forest_robust_threshold(n_trees,risk=.1,confidence=None):
    """Forel et al. Direct-SAA and equation-(8) Agresti-Coull inflation.

    confidence explicitly denotes central normal coverage (e.g. .95 -> z=1.96),
    avoiding ambiguity in the paper's beta/quantile convention. Thresholds above
    one remain infeasible; they must not be clipped or silently weakened.
    """
    if n_trees<1 or not 0<risk<1 or (confidence is not None and not 0<=confidence<1):
        raise ValueError('Invalid forest size, risk, or confidence.')
    required=int(math.ceil(n_trees/2))
    threshold=brentq(lambda p:binom.sf(required-1,n_trees,p)-(1-risk),0.,1.,xtol=1e-13)
    if confidence is not None:
        ac=(n_trees*threshold+2)/(n_trees+4)
        threshold+=norm.ppf((1+confidence)/2)*np.sqrt(ac*(1-ac)/n_trees)
    return float(threshold)


def robx_constrained(deployed,x0,budget,X_train,T=50,constraints=None,time_limit=10.,
                     abs_gap=1e-5,seed=42,K=100,sigma=.1,tau=.7,alpha=.1,anchors=3,
                     max_anchor_candidates=200,artifact_dir=None):
    """RobX with an exact min-cost base and projected feasible training anchors.

    Projection, a candidate cap, and a step cap are explicit adaptations for the
    common actionability/budget protocol. They do not inherit RobX's theorem.
    """
    c=constraints or rc.constraints_for([deployed],x0)
    oracle=TreeOracle([deployed],x0,budget,c,time_limit,abs_gap,seed,artifact_dir)
    try: base=oracle.solve(objective='min_cost')
    finally: oracle.close()
    rng=np.random.default_rng(seed)
    probes=rng.normal(size=(K,len(x0)))*sigma
    scores=[];trace=[base.numpy()]
    def stability(p):
        outputs=deployed.predict_probability(np.asarray(p)+probes)
        return float(outputs.mean()-outputs.std())
    def stable(p):
        value=stability(p);scores.append(value)
        return deployed.predict_probability(p)[0]>=.5 and value>=tau
    chosen=None
    if base.recourse_info['incumbent_verified'] and stable(base.numpy()): chosen=base
    else:
        order=np.argsort(np.linalg.norm(np.asarray(X_train)-base.numpy(),axis=1),kind='stable')
        conservative=[]
        for index in order[:max_anchor_candidates]:
            p=rc.project_l2_ball(torch.tensor(X_train[index],dtype=x0.dtype),x0,budget,constraints=c)
            if stable(p.numpy()):
                conservative.append(p)
                if len(conservative)==anchors: break
        candidates=[]
        for anchor in conservative:
            current=base.clone()
            for step in range(T):
                current=rc.project_l2_ball(alpha*anchor+(1-alpha)*current,x0,budget,constraints=c)
                trace.append(current.numpy())
                if stable(current.numpy()):
                    candidates.append(current);break
            else:
                candidates.append(anchor)
        if candidates: chosen=min(candidates,key=lambda p:float(torch.linalg.vector_norm(p-x0)))
    success=chosen is not None
    point=chosen if success else base
    point=rc._tag(point,status='stable_candidate' if success else 'no_stable_candidate_found',
        gradient_evaluations=0,base_solver=base.recourse_info,K=K,sigma=sigma,tau=tau,alpha=alpha,
        anchors=anchors,max_anchor_candidates=max_anchor_candidates,stable_success=success,
        adaptation='RobX with projected anchors, common random probes, and bounded search; no transferred theorem')
    point.recourse_trace=dict(points=np.asarray(trace),stability_scores=np.asarray(scores),probe_offsets=probes)
    return point


def train_tree_pool(X,y,dataset_name,seed=42,family='decision_tree',num_models=5,num_unseen=5,
                    max_instances=20,aggregation='probability',max_depth=4,n_estimators=25,
                    min_samples_leaf=5,epsilon=.05,pool_multiplier=4,actionable_indices=None,
                    max_pool_multiplier=None):
    if family not in {'decision_tree','random_forest'} or num_models<1 or num_unseen<0:
        raise ValueError('Invalid family or requested model counts.')
    if max_pool_multiplier is None:
        max_pool_multiplier=max(32,pool_multiplier)
    for name,value in [('pool_multiplier',pool_multiplier),('max_pool_multiplier',max_pool_multiplier)]:
        if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<1:
            raise ValueError(f'{name} must be a positive integer.')
    if max_pool_multiplier<pool_multiplier:
        raise ValueError('max_pool_multiplier must be at least pool_multiplier.')
    ctx=rc.prepare_split(X,y,seed,dataset_name,actionable_indices)
    ctx.epsilon=epsilon
    train_start=time.perf_counter()
    admission_history=[]
    def extend_group(pool,target,offset,label):
        # Continue the existing deterministic stream; never retrain old candidates.
        for i in range(len(pool),target):
            model_seed=seed+offset+i
            if family=='decision_tree':
                estimator=DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,
                    max_features='sqrt',random_state=model_seed)
            else:
                estimator=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,max_features='sqrt',bootstrap=True,
                    random_state=model_seed,n_jobs=1)
            before=time.perf_counter();estimator.fit(ctx.X_train,ctx.y_train)
            model=TreeModel.from_estimator(estimator,aggregation,{
                'seed':model_seed,'group':label,'parameters':estimator.get_params(),
                'training_seconds':time.perf_counter()-before})
            model.recourse_context=ctx
            accuracy=float(((model.predict_probability(ctx.X_validation)>=.5)==ctx.y_validation).mean())
            model.metadata['validation_accuracy']=accuracy
            pool.append(model)
            if rr.current_run():
                base=rr.current_run().path/f'seed_{seed}_fold_{rr.current_run().scope.get("fold",0)}'
                model.save(base/(label+'_pool')/f'model_{i:04d}')
                rr.event('tree_model_trained',index=i,**model.metadata)
    def admit_group(count,offset,label,reference=None):
        if count==0:return [],[],[],reference
        pool=[]
        target=pool_multiplier*count
        limit=max_pool_multiplier*count
        while True:
            extend_group(pool,target,offset,label)
            scores=[m.metadata['validation_accuracy'] for m in pool]
            # Recompute against the best of ALL trained optimization candidates.
            # The unseen pool always uses the final optimization reference.
            group_reference=max(scores) if label=='optimization' else reference
            eligible=[m for m in pool if (
                m.metadata['validation_accuracy']>=group_reference-epsilon-1e-12
                if label=='optimization' else
                abs(m.metadata['validation_accuracy']-group_reference)<=epsilon+1e-12)]
            attempt=dict(group=label,candidates_trained=len(pool),eligible_models=len(eligible),
                requested_models=count,reference_validation_accuracy=group_reference,
                epsilon=epsilon,candidate_limit=limit)
            admission_history.append(attempt)
            rr.event('tree_pool_admission',**attempt)
            if len(eligible)>=count:
                return pool,scores,eligible,group_reference
            if len(pool)>=limit:
                message=(f'Only {len(eligible)}/{count} {label} models meet the unchanged '
                    f'validation tolerance after {len(pool)} candidates (cap {limit}). '
                    'Increase max_pool_multiplier explicitly to search further.')
                if rr.current_run():
                    path=rr.current_run().path/f'seed_{seed}_fold_{rr.current_run().scope.get("fold",0)}'
                    rr.atomic_json(path/'tree_selection_failure.json',dict(
                        status='model_generation_failed',error=message,
                        admission_history=admission_history,validation_scores=scores))
                raise RuntimeError(message)
            next_target=min(2*target,limit)
            print(f'  {dataset_name} seed {seed}: {label} pool has {len(eligible)}/{count} '
                  f'eligible models; expanding {target} -> {next_target} candidates '
                  f'(epsilon={epsilon:g} unchanged)',flush=True)
            target=next_target
    pool,scores,eligible,reference=admit_group(num_models,1000,'optimization')
    models=rc.ModelSet(eligible[:num_models],context=ctx)
    eval_pool,eval_scores,eligible_eval,_=admit_group(num_unseen,100000,'unseen',reference)
    unseen=rc.ModelSet(eligible_eval[:num_unseen],context=ctx)
    candidates=rc.select_candidates(models,max_instances,order='test')
    meta={'reference_validation_accuracy':reference,'epsilon':epsilon,
          'selection_protocol':'bounded_adaptive_pool_v1',
          'pool_multiplier':pool_multiplier,'max_pool_multiplier':max_pool_multiplier,
          'admission_history':admission_history,
          'optimization_candidates_trained':len(pool),'unseen_candidates_trained':len(eval_pool),
          'optimization_validation_scores':scores,'unseen_validation_scores':eval_scores,
          'selected_optimization_seeds':[m.metadata['seed'] for m in models],
          'selected_unseen_seeds':[m.metadata['seed'] for m in unseen],
          'candidate_rule':ctx.candidate_selection['mode'],
          'candidate_selection':dict(ctx.candidate_selection),
          'requested_instances':max_instances,'actual_instances':len(candidates),
          'training_seconds':time.perf_counter()-train_start,'aggregation':aggregation,
          'model_scope':'one entire forest is one competing model'}
    models.metadata=meta;unseen.metadata=meta
    if rr.current_run():
        path=rr.current_run().path/f'seed_{seed}_fold_{rr.current_run().scope.get("fold",0)}'
        rr.atomic_json(path/'tree_selection.json',meta)
        for label,group in [('optimization_pool',pool),('unseen_pool',eval_pool)]:
            for i,model in enumerate(group):model.save(path/label/f'model_{i:04d}')
        for label,group in [('optimization',models),('unseen',unseen)]:
            for i,model in enumerate(group):model.save(path/label/f'model_{i:04d}')
        rr.current_run().save_arrays(str(path.relative_to(rr.current_run().path)/'tree_candidates.npz'),
            candidate_ids=np.asarray(ctx.candidate_ids),
            candidates=np.stack([p.numpy() for p in candidates]) if candidates else np.empty((0,len(ctx.columns))))
        for split in ['train','validation','test']:
            data=getattr(ctx,'X_'+split)
            rr.current_run().save_arrays(str(path.relative_to(rr.current_run().path)/f'tree_predictions_{split}.npz'),
                optimization=np.stack([m.predict_probability(data) for m in models],axis=1),
                unseen=np.stack([m.predict_probability(data) for m in unseen],axis=1) if unseen else np.empty((len(data),0)))
    return models,unseen,candidates
