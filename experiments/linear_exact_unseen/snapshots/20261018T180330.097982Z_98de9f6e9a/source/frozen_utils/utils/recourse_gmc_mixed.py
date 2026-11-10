"""Nonconvex logistic-loss ORPM and global outer bounds for ReLU model sets.

Logistic loss is convex in a logit, but its composition with a neural network
need not be convex in the recourse vector. Global bounds here use an exact
mixed-integer ReLU encoding and tangent underestimators of logistic loss.
Local search supplies feasible upper bounds; it never supplies a global lower
bound. Optimization intervals are subject to the solver's numerical tolerances.
"""
import copy
import time
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit,logsumexp
from . import recourse_core as rc
from . import recourse_results as rr
from . import recourse_convergence as gap


def network(model):
    if isinstance(model,rc.FeatureMaskedModel):
        return model.base_model,model.feature_mask.detach().numpy().astype(float)
    base=model
    layer=base.linear if isinstance(base,rc.LogisticRegressionModel) else base.net[0]
    return base,np.ones(layer.in_features)


def layers(model):
    base,_=network(model)
    return [base.linear] if isinstance(base,rc.LogisticRegressionModel) else list(base.net[:-1])


def layer_bounds(model,origin,budget,constraints):
    """Conservative finite interval bounds; first affine map also uses the ball."""
    _,mask=network(model)
    lo=np.maximum(origin-budget,constraints.lower)*mask
    hi=np.minimum(origin+budget,constraints.upper)*mask
    result=[];first=True
    for layer in layers(model):
        if isinstance(layer,torch.nn.Linear):
            W=layer.weight.detach().numpy().astype(float);b=layer.bias.detach().numpy().astype(float)
            wp,wn=np.maximum(W,0),np.minimum(W,0)
            lower=wp@lo+wn@hi+b;upper=wp@hi+wn@lo+b
            if first:
                w=W*mask
                movable=w.copy();movable[:,constraints.immutable]=0
                center=w@origin+b;radius=budget*np.linalg.norm(movable,axis=1)
                lower=np.maximum(lower,center-radius);upper=np.minimum(upper,center+radius)
                first=False
            padding=1e-9*(1+np.maximum(np.abs(lower),np.abs(upper)))
            lo,hi=lower-padding,upper+padding
        elif isinstance(layer,torch.nn.ReLU):lo,hi=np.maximum(lo,0),np.maximum(hi,0)
        else:raise TypeError(type(layer).__name__)
        result.append((lo.copy(),hi.copy()))
    return result


def common_normalizer(sets,origin,budget,constraints):
    lower=[layer_bounds(m,origin,budget,constraints)[-1][0][0] for models in sets for m in models]
    return max(1e-15,float(np.logaddexp(0,-np.asarray(lower)).max()))


class LocalLogisticOracle:
    def __init__(self,models,origin,budget,constraints,normalizer,maxiter=50):
        self.origin=np.asarray(origin,float);self.budget=budget;self.c=constraints
        self.normalizer=normalizer;self.maxiter=maxiter
        self.models=[copy.deepcopy(m,{id(models.context):models.context}).double() for m in models]
        for m in self.models:
            m.eval()
            for p in m.parameters():p.requires_grad_(False)
        self.lo=np.maximum(self.origin-budget,constraints.lower)
        self.hi=np.minimum(self.origin+budget,constraints.upper)

    def project(self,x):
        x=np.clip(np.asarray(x,float),self.lo,self.hi)
        d=x-self.origin;n=np.linalg.norm(d)
        if n>self.budget:x=self.origin+d*self.budget/n*(1-1e-12)
        return x

    def losses(self,x):
        with torch.no_grad():
            z=torch.tensor(x,dtype=torch.float64)
            return np.array([torch.nn.functional.softplus(-m.logits(z)).item() for m in self.models])/self.normalizer

    def solve(self,w,initial):
        def value_grad(x):
            z=torch.tensor(x,dtype=torch.float64,requires_grad=True)
            f=sum(float(a)*torch.nn.functional.softplus(-m.logits(z)).sum() for a,m in zip(w,self.models))
            g,=torch.autograd.grad(f,z)
            return f.item(),g.numpy()
        start=self.project(initial)
        result=minimize(value_grad,start,jac=True,method='SLSQP',bounds=list(zip(self.lo,self.hi)),
            constraints=[dict(type='ineq',fun=lambda x:self.budget**2-np.sum((x-self.origin)**2),
                              jac=lambda x:-2*(x-self.origin))],options=dict(maxiter=self.maxiter,ftol=1e-8))
        choices=[start,self.origin,self.project(result.x)]
        point=min(choices,key=lambda x:float(w@self.losses(x))).copy()
        return point,dict(objective=float(w@self.losses(point)),optimizer_success=bool(result.success),
            optimizer_iterations=int(result.nit),optimizer_message=str(result.message),
            bound_type='feasible_upper_only_nonconvex')


class NeuralLogisticBound:
    """Global lower bounds from an exact ReLU MIQCP with softplus minorants."""
    def __init__(self,models,origin,budget,constraints,normalizer,time_limit=1.,seed=42):
        import gurobipy as gp
        self.gp=gp;self.origin=np.asarray(origin,float);self.models=models;self.normalizer=normalizer
        self.time_limit=time_limit
        self.env=gp.Env(empty=True);self.env.setParam('OutputFlag',0);self.env.start()
        self.problem=gp.Model('mixed_logistic_global_outer_bound',env=self.env)
        p=self.problem;p.Params.Threads=1;p.Params.Seed=seed;p.Params.TimeLimit=time_limit
        p.Params.FeasibilityTol=1e-9;p.Params.IntFeasTol=1e-9;p.Params.OptimalityTol=1e-9
        p.Params.NumericFocus=3;p.Params.MIPGap=0;p.Params.MIPGapAbs=1e-10
        lo=np.maximum(origin-budget,constraints.lower);hi=np.minimum(origin+budget,constraints.upper)
        self.x=[p.addVar(lb=float(a),ub=float(b),name=f'x{i}') for i,(a,b) in enumerate(zip(lo,hi))]
        p.addQConstr(gp.quicksum((x-float(a))*(x-float(a)) for x,a in zip(self.x,origin))<=budget**2)
        self.z=[];self.v=[];self.activations=[];self.zbounds=[];self.component_lower=[]
        self.skipped_tiny_cuts=0
        for mi,model in enumerate(models):
            _,mask=network(model);variables=[x*float(a) for x,a in zip(self.x,mask)]
            bounds=layer_bounds(model,origin,budget,constraints);activation_vars=[]
            for li,layer in enumerate(layers(model)):
                if isinstance(layer,torch.nn.Linear):
                    W=layer.weight.detach().numpy().astype(float);b=layer.bias.detach().numpy().astype(float)
                    variables=[gp.quicksum(float(w)*x for w,x in zip(row,variables))+float(bi) for row,bi in zip(W,b)]
                else:
                    lower,upper=bounds[li-1];activated=[]
                    for j,(expression,l,u) in enumerate(zip(variables,lower,upper)):
                        if u<=0:activated.append(0.)
                        elif l>=0:activated.append(expression)
                        else:
                            h=p.addVar(lb=0,ub=float(u),name=f'h{mi}_{li}_{j}')
                            a=p.addVar(vtype=gp.GRB.BINARY,name=f'a{mi}_{li}_{j}')
                            p.addConstr(h>=expression)
                            p.addConstr(h<=expression-float(l)*(1-a));p.addConstr(h<=float(u)*a)
                            activated.append(h);activation_vars.append((li,j,h,a))
                    variables=activated
            zl,zu=float(bounds[-1][0][0]),float(bounds[-1][1][0])
            z=p.addVar(lb=zl,ub=zu,name=f'logit{mi}');p.addConstr(z==variables[0])
            self.component_lower.append(float(np.logaddexp(0,-zu))/normalizer)
            # Keep very small analytic bounds outside the numerical optimization.
            v=p.addVar(lb=0.,
                       ub=float(np.logaddexp(0,-zl))/normalizer,name=f'loss{mi}')
            self.z.append(z);self.v.append(v);self.activations.append(activation_vars);self.zbounds.append((zl,zu))
            knots={zl,zu,0.,*np.linspace(-12,12,25),-48.,-24.,24.,48.}
            for k in sorted(knots):
                if zl<=k<=zu:self.cut(mi,k)
        p.update()

    def cut(self,i,z):
        intercept=float(np.logaddexp(0,-z))/self.normalizer
        slope=-float(expit(-z))/self.normalizer
        # Omitting a tangent weakens the relaxation but cannot overstate its bound.
        # Tiny coefficients in the softplus tail made the old formulation fragile.
        if abs(slope)<1e-8:
            self.skipped_tiny_cuts+=1
            return
        self.problem.addConstr(self.v[i]>=intercept+slope*(self.z[i]-float(z)))

    def warm(self,point):
        for variable,value in zip(self.x,point):variable.Start=float(value)
        for i,model in enumerate(self.models):
            _,mask=network(model);value=np.asarray(point)*mask;preacts={}
            for li,layer in enumerate(layers(model)):
                if isinstance(layer,torch.nn.Linear):
                    value=layer.weight.detach().numpy().astype(float)@value+layer.bias.detach().numpy().astype(float)
                else:preacts[li]=value.copy();value=np.maximum(value,0)
            for li,j,h,a in self.activations[i]:
                h.Start=float(max(0,preacts[li][j]));a.Start=float(preacts[li][j]>=0)
            z=float(value[0]);self.z[i].Start=z;self.v[i].Start=float(np.logaddexp(0,-z))/self.normalizer

    def solve(self,w,incumbent,local):
        p=self.problem;gp=self.gp;started=time.perf_counter()
        p.setObjective(gp.quicksum(float(a)*v for a,v in zip(w,self.v)),gp.GRB.MINIMIZE)
        self.warm(incumbent);upper=float(w@local.losses(incumbent));best=incumbent.copy()
        analytic=max(0.,float(w@np.asarray(self.component_lower))-1e-12)
        lower=analytic;status=None;attempts=[];discarded=False
        for iteration in range(2):
            remaining=self.time_limit-(time.perf_counter()-started)
            if remaining<=.01:break
            p.Params.TimeLimit=remaining/(2-iteration);p.optimize();status=int(p.Status)
            try:bound=float(p.ObjBound)
            except (AttributeError,gp.GurobiError):bound=float('-inf')
            if p.SolCount:
                candidate=local.project([x.X for x in self.x]);value=float(w@local.losses(candidate))
                if value<upper:best,upper=candidate,value
                logits=[float(z.X) for z in self.z]
                for i,z in enumerate(logits):self.cut(i,z)
            usable=status not in {gp.GRB.NUMERIC,gp.GRB.INFEASIBLE,gp.GRB.INF_OR_UNBD,gp.GRB.UNBOUNDED}
            candidate=bound-1e-7 if np.isfinite(bound) and usable else None
            conflict=candidate is not None and candidate>upper
            conflict=conflict or lower>upper
            attempts.append(dict(status=status,raw_bound=bound if np.isfinite(bound) else None,
                guarded_bound=candidate,feasible_upper=upper,contradicted=bool(conflict),usable_status=usable))
            if conflict:
                discarded=True;lower=analytic
                # Retry within the existing wall-clock budget, without the old search state.
                p.reset();self.warm(best)
            elif candidate is not None:lower=max(lower,candidate)
        if analytic>upper:raise RuntimeError('Analytic component bound contradicts a feasible point')
        # A contradicted numerical bound is never clipped to an incumbent objective.
        if discarded:lower=analytic
        return best,dict(lower_bound=lower,upper_bound=upper,
            solver_status=status,solver_seconds=time.perf_counter()-started,bound_consistent=not discarded,
            bound_source='analytic_component_fallback' if discarded else 'analytic_and_numerical_outer_bound',
            numerical_policy='witness_checked_v2',attempts=attempts,analytic_component_lower=analytic,
            solver_bound_guard=1e-7,skipped_tiny_tangents=self.skipped_tiny_cuts,
            formulation='exact_ReLU_L2_MIQCP_logistic_tangent_minorants',
            bound_note='Numerical outer bound, subject to solver tolerances; the guard is not a rigorous error enclosure. Contradicted numerical bounds are discarded.')

    def close(self):self.problem.dispose();self.env.dispose()


def trajectory(models,x0,budget,normalizer,T=100,every=20,maxiter=50,dual_seconds=1.,seed=42,on_row=None):
    origin=np.asarray(x0,float);c=rc.constraints_for(models,torch.tensor(origin,dtype=torch.float32))
    local=LocalLogisticOracle(models,origin,budget,c,normalizer,maxiter)
    n=len(models);eta=float(np.sqrt(np.log(n)/T));logw=np.full(n,-np.log(n))
    point=origin.copy();best=point.copy();best_value=local.losses(best).max()
    points=[];weights=[];losses=[];bests=[];timings=[];infos=[]
    started=time.perf_counter()
    for t in range(1,T+1):
        w=np.exp(logw);point,info=local.solve(w,point);v=local.losses(point)
        assert np.all(v>=0) and np.all(v<=1+1e-8)
        if (v.max(),np.linalg.norm(point-origin))<(best_value,np.linalg.norm(best-origin)):
            best,best_value=point.copy(),float(v.max())
        points.append(point.copy());weights.append(w.copy());losses.append(v.copy());bests.append(best.copy())
        infos.append(info);timings.append(time.perf_counter()-started)
        rr.event('mixed_logistic_orpm_round',round=t,point=point,best_point=best,weights=w,losses=v,oracle=info)
        logw+=eta*v;logw-=logsumexp(logw)
    weights=np.asarray(weights);losses=np.asarray(losses)
    evaluator=NeuralLogisticBound(models,origin,budget,c,normalizer,dual_seconds,seed)
    rows=[]
    try:
        for t in gap.checkpoints(T,every):
            wb=weights[:t].mean(axis=0)
            index=int(np.argmin(losses[:t]@wb))
            choices=[origin,bests[t-1],points[index]]
            initial=min(choices,key=lambda x:float(wb@local.losses(x)))
            response,local_info=local.solve(wb,initial)
            response,info=evaluator.solve(wb,response,local)
            upper=min(info['upper_bound'],float(wb@local.losses(initial)))
            lower=info['lower_bound'];point_loss=float(local.losses(bests[t-1]).max())
            pg=gap.gap_interval(point_loss,lower,upper)
            mg=gap.gap_interval(float(losses[:t].mean(axis=0).max()),lower,upper)
            row=dict(round=t,**pg,mixed_primal=mg['primal'],mixed_gap_lower_bound=mg['gap_lower_bound'],
                mixed_gap_upper_bound=mg['gap_upper_bound'],normalizer=normalizer,eta=eta,
                average_weights=wb,best_point=bests[t-1],dual_response=response,evaluation_oracle=info,
                local_evaluation=local_info,algorithm_elapsed_seconds=timings[t-1],
                gap_kind='nonconvex_mixed_logistic_best_point',point_rule='best_feasible_candidate',
                certificate_status='global_outer_bound_interval' if info['bound_consistent'] else 'bound_inconsistency_fallback')
            rows.append(row)
            if on_row:on_row(row)
    finally:evaluator.close()
    return best,rows,dict(points=np.asarray(points),weights=weights,losses=losses,best_points=np.asarray(bests),
                         oracle_objectives=np.array([r['objective'] for r in infos]),normalizer=normalizer,eta=eta)
