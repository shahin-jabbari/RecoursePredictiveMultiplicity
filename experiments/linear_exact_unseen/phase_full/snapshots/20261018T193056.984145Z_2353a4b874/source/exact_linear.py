"""Exact convex linear-model benchmarks with independently evaluated bounds."""
import time
import numpy as np
import gurobipy as gp
from scipy.optimize import linprog


def environment():
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    return env


def problem(env, name, settings):
    m = gp.Model(name, env=env)
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    m.Params.Seed = 0
    m.Params.TimeLimit = settings['time_limit_seconds']
    m.Params.FeasibilityTol = settings['feasibility_tolerance']
    m.Params.OptimalityTol = settings['feasibility_tolerance']
    m.Params.BarQCPConvTol = settings['barrier_tolerance']
    m.Params.BarConvTol = settings['barrier_tolerance']
    m.Params.NumericFocus = 2
    m.Params.DualReductions = 0
    return m


def max_margin(A, b, x0, budget, free, settings, env):
    """max_y min_i(A_i(x0+y)+b_i), ||y||_2 <= B, fixed other entries.

    For any simplex weight w, w'z+B||A_free'w|| is a global upper bound.
    Return both a feasible point and a directly evaluated dual bound.
    """
    start = time.perf_counter()
    z = A@x0+b
    Q = A[:, free]
    n, d = Q.shape
    if budget == 0 or d == 0:
        w = np.eye(n)[int(np.argmin(z))]
        return x0.copy(), dict(status='optimal', solver_status=2,
            margin_lower=float(z.min()), margin_upper=float(z.min()), certificate_gap=0.,
            certificate_passed=True, dual_weights=w.tolist(), available=True,
            seconds=time.perf_counter()-start, singleton=True)
    with problem(env, 'linear_max_min_margin', settings) as m:
        y = m.addVars(d, lb=-budget, ub=budget, name='change')
        t = m.addVar(lb=-gp.GRB.INFINITY, name='minimum_logit')
        cs = [m.addConstr(gp.quicksum(Q[i,j]*y[j] for j in range(d))+float(z[i]) >= t)
              for i in range(n)]
        m.addQConstr(gp.quicksum(y[j]*y[j] for j in range(d)) <= budget*budget)
        m.setObjective(t, gp.GRB.MAXIMIZE)
        m.Params.QCPDual = 1
        m.optimize()
        status = int(m.Status)
        delta = np.array([y[j].X for j in range(d)]) if m.SolCount else np.zeros(d)
        raw_cost = float(np.linalg.norm(delta))
        if raw_cost > budget:
            # Radial repair makes the evaluated lower bound strictly feasible.
            delta *= (budget/raw_cost)*(1-1e-12)
        point = x0.copy(); point[free] += delta
        lower = float(np.min(A@point+b))
        try:
            w = np.abs(np.array([c.Pi for c in cs], float))
            if not np.isfinite(w).all() or w.sum() == 0: raise ValueError('missing dual weights')
            w /= w.sum()
        except (AttributeError, gp.GurobiError, ValueError):
            w = np.full(n, 1/n)
        upper = float(w@z+budget*np.linalg.norm(Q.T@w))
    tolerance = settings['certificate_tolerance']*(1+abs(lower))
    if upper-lower > tolerance:
        # Independent dual optimization; evaluation of any returned w remains valid.
        with problem(env, 'linear_margin_dual', settings) as m:
            ww = m.addVars(n, lb=0., ub=1., name='weight')
            vv = m.addVars(d, lb=-gp.GRB.INFINITY, name='weighted_normal')
            s = m.addVar(lb=0., name='dual_norm')
            m.addConstr(gp.quicksum(ww.values()) == 1.)
            for j in range(d):m.addConstr(vv[j] == gp.quicksum(Q[i,j]*ww[i] for i in range(n)))
            m.addQConstr(gp.quicksum(vv[j]*vv[j] for j in range(d)) <= s*s)
            m.setObjective(gp.quicksum(float(z[i])*ww[i] for i in range(n))+budget*s)
            m.optimize()
            if m.SolCount:
                candidate = np.maximum(0., np.array([ww[i].X for i in range(n)]))
                candidate /= candidate.sum()
                value = float(candidate@z+budget*np.linalg.norm(Q.T@candidate))
                if value < upper:w, upper = candidate, value
    gap = upper-lower
    assert gap >= -settings['certificate_tolerance']*(1+abs(upper)), (lower,upper)
    return point, dict(status='optimal' if status==2 else 'uncertified_solver_status',
        solver_status=status, margin_lower=lower, margin_upper=upper,
        certificate_gap=max(0.,gap), certificate_passed=bool(gap<=tolerance),
        dual_weights=w.tolist(), available=True, raw_cost=raw_cost,
        seconds=time.perf_counter()-start, singleton=False)


def min_l2(A, b, x0, free, settings, env):
    """Minimum L2 distance to the nominal all-logit>=0 intersection.

    Solve .5||y||^2. Any lambda>=0 gives dual lower bound
    -lambda'z-.5||A_free'lambda||^2. No positive logit margin is imposed.
    """
    start=time.perf_counter();z=A@x0+b;Q=A[:,free];n,d=Q.shape
    if np.min(z)>=0:
        return x0.copy(),dict(status='optimal',solver_status=2,available=True,
            cost_lower=0.,cost_upper=0.,primal_objective=0.,dual_objective=0.,
            certificate_gap=0.,certificate_passed=True,dual_multipliers=[0.]*n,
            seconds=time.perf_counter()-start)
    if not d:
        return x0.copy(),dict(status='certified_infeasible',solver_status=3,available=False,
            certificate_passed=True,seconds=time.perf_counter()-start,independent_infeasibility=True)
    with problem(env,'linear_minimum_L2',settings) as m:
        y=m.addVars(d,lb=-gp.GRB.INFINITY,name='change')
        cs=[m.addConstr(gp.quicksum(Q[i,j]*y[j] for j in range(d))+float(z[i]) >= 0.) for i in range(n)]
        m.setObjective(.5*gp.quicksum(y[j]*y[j] for j in range(d)))
        m.optimize();status=int(m.Status)
        if not m.SolCount:
            lp=linprog(np.zeros(d),A_ub=-Q,b_ub=z,bounds=[(None,None)]*d,method='highs')
            certified=bool(status==gp.GRB.INFEASIBLE and lp.status==2)
            return x0.copy(),dict(status='certified_infeasible' if certified else 'unresolved',
                solver_status=status,available=False,certificate_passed=certified,
                independent_infeasibility=certified,linprog_status=int(lp.status),seconds=time.perf_counter()-start)
        delta=np.array([y[j].X for j in range(d)]);point=x0.copy();point[free]+=delta
        primal=float(.5*delta@delta)
        multipliers=np.maximum(0.,np.array([c.Pi for c in cs]))
        dual=float(-multipliers@z-.5*np.sum((Q.T@multipliers)**2))
        residual=float(max(0.,-np.min(A@point+b)))
        gap=primal-dual
        tol=settings['certificate_tolerance']*(1+abs(primal))
        assert gap>=-tol,(primal,dual)
        passed=bool(gap<=tol and residual<=settings['evaluation_logit_tolerance'])
        return point,dict(status='optimal' if status==2 else 'feasible_incumbent',solver_status=status,
            available=True,cost_lower=float(np.sqrt(max(0.,2*dual))),cost_upper=float(np.sqrt(2*primal)),
            primal_objective=primal,dual_objective=dual,certificate_gap=max(0.,gap),
            logit_constraint_residual=residual,certificate_passed=passed,
            dual_multipliers=multipliers.tolist(),seconds=time.perf_counter()-start)


def self_check(settings):
    """Closed-form feasible, conflicting and immutable cases, not study outcomes."""
    results=[]
    with environment() as env:
        for name,A,b,x0,B,free,margin,cost in [
            ('one_halfspace',np.array([[1.,0.]]),np.array([-1.]),np.zeros(2),2.,[0,1],1.,1.),
            ('orthogonal_halfspaces',np.eye(2),np.array([-1.,-1.]),np.zeros(2),2.,[0,1],np.sqrt(2)-1,np.sqrt(2)),
            ('conflicting_halfspaces',np.array([[1.],[-1.]]),np.array([-1.,-1.]),np.zeros(1),2.,[0],-1.,None),
            ('immutable_obstruction',np.array([[1.,0.]]),np.array([-1.]),np.zeros(2),2.,[1],-1.,None),
            ('zero_budget',np.array([[1.]]),np.array([-1.]),np.zeros(1),0.,[0],-1.,1.)]:
            point,info=max_margin(A,b,x0,B,free,settings,env)
            assert abs(info['margin_lower']-margin)<2e-5 and info['certificate_passed'],(name,info)
            q,qi=min_l2(A,b,x0,free,settings,env)
            if cost is None:assert qi['status']=='certified_infeasible',(name,qi)
            else:assert abs(qi['cost_upper']-cost)<2e-5 and qi['certificate_passed'],(name,qi)
            results.append(dict(case=name,passed=True,max_margin=info,min_l2=qi))
    return dict(status='passed',gurobi_version=list(gp.gurobi.version()),analytic_cases=results)
