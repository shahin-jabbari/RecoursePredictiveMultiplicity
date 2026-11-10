"""Baselines with explicit uncertainty sets, budgets, constraints, and solver status."""
import copy
import time
import warnings
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import minimize
from . import recourse_core as rc
from . import recourse_results as rr


def linearize_models_lime(models, x0, X_train, seed=42):
    from lime.lime_tabular import LimeTabularExplainer
    X_train = np.asarray(X_train)
    explainer = LimeTabularExplainer(X_train, mode='regression',
        discretize_continuous=False, random_state=seed, verbose=False)
    surrogates = rc.ModelSet(context=getattr(models, 'context', None))
    for index, model in enumerate(models):
        if rc.is_linear(model):
            a, b = rc.linear_parameters(model)
            coeff, intercept = a.cpu().numpy(), float(b)
            diagnostics = {'method': 'exact_affine_logits', 'anchor_error': 0.0}
        else:
            def predict_logits(X):
                with torch.no_grad():
                    return model.logits(torch.tensor(X, dtype=torch.float32)).numpy().reshape(-1)
            explanation = explainer.explain_instance(np.asarray(x0), predict_logits,
                num_features=len(x0),
                num_samples=int((rr._ACTIVE_SETTINGS.get() or {}).get('lime_samples', 5000)))
            scaled_coeff = np.zeros(len(x0))
            for feature, value in explanation.local_exp[1]:
                scaled_coeff[feature] = value
            coeff = scaled_coeff/explainer.scaler.scale_
            fitted_intercept = float(explanation.intercept[1])-coeff@explainer.scaler.mean_
            actual_logit = float(predict_logits(np.asarray(x0).reshape(1, -1))[0])
            anchor_error = float(coeff@x0+fitted_intercept-actual_logit)
            # Explicit anchored local-logit surrogate, exact at the factual input.
            intercept = actual_logit-float(coeff@x0)
            diagnostics = {'method': 'anchored_LIME_logit_regression',
                           'unanchored_logit_error': anchor_error,
                           'lime_local_r2': float(explanation.score)}
        surrogate = rc.LogisticRegressionModel(len(x0), seed+index)
        rc.set_model_weights(surrogate, torch.tensor(np.r_[coeff, intercept], dtype=torch.float32))
        surrogate.surrogate_diagnostics = diagnostics
        surrogate.recourse_context = getattr(models, 'context', None)
        with torch.no_grad():
            factual = torch.tensor(x0, dtype=torch.float32)
            error = abs(float(surrogate(factual))-float(model(factual)))
        if error > 1e-5:
            raise ValueError('Surrogate must preserve the factual probability')
        diagnostics['factual_probability_error'] = error
        surrogates.append(surrogate)
    return surrogates


def ADV(x0, models, B, loss_func, T=100, lr=0.01, tol=1e-4,
        immutable_indices=None, constraints=None):
    c = constraints or rc.constraints_for(models, x0, immutable_indices)
    x = x0.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)
    history, best, best_value = [], x0.clone(), float('inf')
    evaluations = 0
    for _ in range(T):
        value = torch.stack([loss_func(m, x) for m in models]).max()
        if not torch.isfinite(value):
            raise FloatingPointError('Nonfinite ADV loss')
        if value.item() < best_value:
            best, best_value = x.detach().clone(), value.item()
        optimizer.zero_grad()
        x.grad = torch.autograd.grad(value, x)[0]
        evaluations += len(models)
        optimizer.step()
        with torch.no_grad():
            x.copy_(rc.project_l2_ball(x, x0, B, constraints=c))
        history.append(x.detach().clone())
    value = max(loss_func(m, x).item() for m in models)
    if value < best_value:
        best = x.detach().clone()
    if not history:
        raise ValueError('T must be positive')
    raw_points = np.asarray([x0.detach().cpu().numpy()]+[p.detach().cpu().numpy() for p in history])
    history[-1] = rc._tag(best, status='nonconvex_step_limit', gradient_evaluations=evaluations,
                          requested_steps=T, tolerance_role='diagnostic_only')
    rr.attach_trace(history[-1], points=raw_points)
    return history


def _constrained_score_solver(score, x0, budget, c, T, tol=1e-4, minimize_cost=False):
    start = x0.detach().cpu().numpy().astype(float)
    counts = {'gradients': 0}
    trace = []
    phases = []
    phase = 1
    def score_grad(arr):
        x = torch.tensor(arr, dtype=x0.dtype, device=x0.device, requires_grad=True)
        value = score(x)
        grad = torch.autograd.grad(value, x)[0]
        counts['gradients'] += 1
        if not torch.isfinite(value) or not torch.isfinite(grad).all():
            raise FloatingPointError('Nonfinite robust score')
        return float(value), grad.detach().cpu().numpy().astype(float)
    constraints = [{'type': 'ineq', 'fun': lambda x: budget**2-np.sum((x-start)**2),
                    'jac': lambda x: -2*(x-start)}]
    bounds = list(zip(np.maximum(c.lower, start-budget), np.minimum(c.upper, start+budget)))
    def callback(x):
        trace.append(torch.tensor(x, dtype=x0.dtype))
        phases.append(phase)
    phase1 = minimize(lambda x: tuple(-v for v in score_grad(x)), start, jac=True,
        method='SLSQP', bounds=bounds, constraints=constraints, callback=callback,
        options={'maxiter': T, 'ftol': tol})
    point = torch.tensor(phase1.x, dtype=x0.dtype)
    best_score = float(score(point))
    status = 'valid_incumbent' if best_score >= 0 and rc.feasible(point, x0, budget, c) else 'no_feasible_recourse_found'
    iterations = int(phase1.nit)
    if minimize_cost and status == 'valid_incumbent' and iterations < T:
        phase = 2
        score_constraint = {'type': 'ineq', 'fun': lambda x: score_grad(x)[0],
                            'jac': lambda x: score_grad(x)[1]}
        phase2 = minimize(lambda x: np.sum((x-start)**2), phase1.x,
            jac=lambda x: 2*(x-start), method='SLSQP', bounds=bounds,
            constraints=constraints+[score_constraint], callback=callback,
            options={'maxiter': T-iterations, 'ftol': tol})
        proposed = torch.tensor(phase2.x, dtype=x0.dtype)
        if rc.feasible(proposed, x0, budget, c) and float(score(proposed)) >= 0:
            point = proposed
            status = 'solver_converged' if phase2.success else 'valid_incumbent'
        iterations += int(phase2.nit)
    if not rc.feasible(point, x0, budget, c):
        point, status = x0.clone(), 'no_feasible_recourse_found'
    if not trace:
        trace.append(point)
        phases.append(phase)
    trace_points = np.asarray([x0.detach().cpu().numpy()]+[p.detach().cpu().numpy() for p in trace])
    trace[-1] = rc._tag(point, status=status, gradient_evaluations=counts['gradients'],
        solver_iterations=iterations, requested_steps=T, robust_score=float(score(point)),
        optimality_proven=False)
    rr.attach_trace(trace[-1], points=trace_points, phases=np.asarray([0]+phases),
                    robust_scores=np.asarray([float(score(torch.tensor(p, dtype=x0.dtype))) for p in trace_points]))
    return trace


def ROAR(x0, initial_model, delta_model, B, loss_func, T=100, lr=0.01,
         tol=1e-4, immutable_indices=None, constraints=None):
    """Fixed L-infinity parameter-box adversary; monotone score formulation.

    Maximizing this worst logit is equivalent to minimizing its target-1 MSE
    within the fixed L2 budget, but avoids sigmoid saturation. SLSQP uses T
    as its iteration cap; lr is retained for call compatibility, not used by it.
    """
    a, b = rc.linear_parameters(initial_model)
    c = constraints or rc.constraints_for([initial_model], x0, immutable_indices)
    score = lambda x: a@x+b-delta_model*(x.abs().sum()+1)
    history = _constrained_score_solver(score, x0, B, c, T, tol)
    history[-1].recourse_info.update(uncertainty='parameter_Linf_box', delta_model=delta_model,
                                    optimizer='SLSQP', legacy_lr=lr)
    return history


def _last_layer(model):
    mask = None
    if isinstance(model, rc.FeatureMaskedModel):
        mask, model = model.feature_mask, model.base_model
    if isinstance(model, rc.LogisticRegressionModel):
        layer = model.linear
        feature = lambda x: x if mask is None else x*mask
    else:
        layer = model.net[-2]
        feature = lambda x: model.net[:-2](x if mask is None else x*mask)
    theta = torch.cat([layer.weight.detach().reshape(-1), layer.bias.detach().reshape(-1)])
    return feature, theta


def prepare_ellice(model, X_train, epsilon=0.05, lambda_reg=1e-5):
    """Hessian ellipsoid from training loss, including the last-layer intercept."""
    feature, theta = _last_layer(model)
    with torch.no_grad():
        features = feature(torch.tensor(np.asarray(X_train), dtype=torch.float32)).double()
        aug = torch.cat([features, torch.ones(len(features), 1, dtype=torch.float64)], dim=1)
        p = torch.sigmoid(aug@theta.double())
        H = (aug.T*(p*(1-p)))@aug/len(aug)+lambda_reg*torch.eye(aug.shape[1], dtype=aug.dtype)
        inverse = torch.linalg.inv(H).to(theta.dtype)
    return {'feature': feature, 'theta': theta, 'H': H, 'H_inverse': inverse,
            'epsilon': epsilon, 'lambda_reg': lambda_reg}


def ElliCE(x0, models, B, loss_func, T=100, lr=0.01, tol=1e-4,
           immutable_indices=None, constraints=None, preparation=None):
    """Budgeted published Hessian-ellipsoid score constraint, with last-layer extension.

    The ellipsoid is centered on the designated deployed model models[0], not
    the covariance of a finite sample. Neural penultimate features make the
    input optimization heuristic. This baseline does not certify a heterogeneous
    finite set, which is checked independently by the common evaluator.
    """
    ctx = getattr(models, 'context', None) or getattr(models[0], 'recourse_context', None)
    if preparation is None:
        if ctx is None:
            raise ValueError('ElliCE requires training data/Hessian; provide preparation or model context.')
        preparation = prepare_ellice(models[0], ctx.X_train, ctx.epsilon)
    theta, inverse = preparation['theta'], preparation['H_inverse']
    def score(x):
        features = preparation['feature'](x).reshape(-1)
        aug = torch.cat([features, x.new_ones(1)])
        variance = torch.clamp(aug@inverse@aug, min=torch.finfo(x.dtype).eps)
        return theta@aug-torch.sqrt(2*preparation['epsilon']*variance)
    c = constraints or rc.constraints_for(models, x0, immutable_indices)
    history = _constrained_score_solver(score, x0, B, c, T, tol, minimize_cost=True)
    history[-1].recourse_info.update(uncertainty='training_Hessian_ellipsoid',
        epsilon=preparation['epsilon'], center='deployed_model_0', optimizer='SLSQP',
        legacy_lr=lr, convex_input_constraint=rc.is_linear(models[0]),
        variant='minimum_L2_with_budget_and_robust_score_constraint')
    return history


def _solver_outcome(status, solution_count, grb):
    if solution_count > 0:
        return 'optimal' if status == grb.OPTIMAL else 'feasible_incumbent'
    if status == grb.INFEASIBLE:
        return 'infeasible'
    if status == grb.TIME_LIMIT:
        return 'timeout_without_incumbent'
    return 'solver_no_solution'


def exact_milp_recourse_gurobi(x0, models, budget, lower_bounds=None,
        upper_bounds=None, M=1e4, eps=1e-3, time_limit=10.0,
        immutable_indices=None, constraints=None, return_info=False):
    """Minimum L1 with an L2 budget (MIQCP), valid neuron-specific ReLU bounds.

    M remains in the signature for compatibility; the unsafe fixed bound is
    superseded by interval bounds derived from the actual input region.
    Feasible time-limit incumbents are retained and independently validated.
    """
    import gurobipy as gp
    from gurobipy import GRB
    time_limit = (rr._ACTIVE_SETTINGS.get() or {}).get('miqcp_time_limit', time_limit)
    if budget < 0 or eps < 0 or M <= 0:
        raise ValueError('Invalid budget, margin or legacy M')
    c = constraints or rc.constraints_for(models, x0, immutable_indices)
    start = x0.detach().cpu().numpy().astype(float)
    lo = np.maximum(start-budget, c.lower)
    hi = np.minimum(start+budget, c.upper)
    if lower_bounds is not None:
        lo = np.maximum(lo, np.asarray(lower_bounds))
    if upper_bounds is not None:
        hi = np.minimum(hi, np.asarray(upper_bounds))
    if np.any(lo > hi):
        raise ValueError('Incompatible feature bounds')
    actual_constraints = rc.Constraints(lo, hi, c.immutable, c.policy)
    with gp.Env(empty=True) as environment:
        environment.setParam('OutputFlag', 0)
        environment.start()
        with gp.Model('Recourse_MIQCP', env=environment) as problem:
            run = rr.current_run()
            solver_directory = None
            if run:
                scope = run.scope
                solver_directory = run.path/f"seed_{scope['seed']}_fold_{scope['fold']}"/f"instance_{scope['instance_id']}"/str(scope['method'])
                solver_directory.mkdir(parents=True, exist_ok=True)
                problem.Params.LogToConsole = 0
                problem.Params.OutputFlag = 1
                problem.Params.LogFile = str(solver_directory/'solver.log')
            if time_limit is not None:
                problem.Params.TimeLimit = time_limit
            x = problem.addVars(len(start), lb=lo.tolist(), ub=hi.tolist(), name='x')
            diff = problem.addVars(len(start), lb=-GRB.INFINITY, name='change')
            absolute = problem.addVars(len(start), lb=0, name='abs_change')
            for i in range(len(start)):
                problem.addConstr(diff[i] == x[i]-start[i])
                problem.addConstr(absolute[i] >= diff[i])
                problem.addConstr(absolute[i] >= -diff[i])
            problem.addQConstr(gp.quicksum(diff[i]*diff[i] for i in range(len(start))) <= budget**2)
            problem.setObjective(gp.quicksum(absolute.values()), GRB.MINIMIZE)
            neuron_bounds = []
            for mi, model in enumerate(models):
                if isinstance(model, rc.FeatureMaskedModel):
                    mask = model.feature_mask.detach().numpy().astype(float)
                    base = model.base_model
                else:
                    mask, base = np.ones(len(start)), model
                variables = [x[i]*mask[i] for i in range(len(start))]
                lower = np.minimum(lo*mask, hi*mask)
                upper = np.maximum(lo*mask, hi*mask)
                layers = [base.linear] if isinstance(base, rc.LogisticRegressionModel) else list(base.net[:-1])
                for li, layer in enumerate(layers):
                    if isinstance(layer, torch.nn.Linear):
                        w = layer.weight.detach().numpy().astype(float)
                        b = layer.bias.detach().numpy().astype(float)
                        wp, wn = np.maximum(w, 0), np.minimum(w, 0)
                        next_lower = wp@lower+wn@upper+b
                        next_upper = wp@upper+wn@lower+b
                        variables = [gp.quicksum(w[j, k]*variables[k] for k in range(len(variables)))+b[j]
                                     for j in range(len(b))]
                        lower, upper = next_lower, next_upper
                    elif isinstance(layer, torch.nn.ReLU):
                        activated = []
                        for j, (expression, low, high) in enumerate(zip(variables, lower, upper)):
                            neuron_bounds.append((float(low), float(high)))
                            if high <= 0:
                                activated.append(0.0)
                            elif low >= 0:
                                activated.append(expression)
                            else:
                                h = problem.addVar(lb=0, ub=float(high), name=f'h{mi}_{li}_{j}')
                                active = problem.addVar(vtype=GRB.BINARY, name=f'a{mi}_{li}_{j}')
                                problem.addConstr(h >= expression)
                                problem.addConstr(h <= expression-low*(1-active))
                                problem.addConstr(h <= high*active)
                                activated.append(h)
                        variables = activated
                        lower, upper = np.maximum(lower, 0), np.maximum(upper, 0)
                    else:
                        raise TypeError(f'Unsupported layer: {type(layer).__name__}')
                if len(variables) != 1:
                    raise ValueError('Expected scalar logit')
                problem.addConstr(variables[0] >= eps)
            if solver_directory:
                problem.write(str(solver_directory/'problem.mps'))
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    rr.event('solver_incumbent', objective=model.cbGet(GRB.Callback.MIPSOL_OBJ),
                        bound=model.cbGet(GRB.Callback.MIPSOL_OBJBND),
                        runtime=model.cbGet(GRB.Callback.RUNTIME))
            problem.optimize(callback if run else None)
            status, count = problem.Status, problem.SolCount
            outcome = _solver_outcome(status, count, GRB)
            point = torch.tensor([x[i].X for i in range(len(start))], dtype=x0.dtype) if count else x0.clone()
            info = {'status': outcome, 'gurobi_status': int(status), 'solution_count': int(count),
                    'optimality_proven': status == GRB.OPTIMAL, 'margin': eps,
                    'objective': 'minimum_L1_with_L2_budget', 'time_limit': time_limit,
                    'runtime': float(problem.Runtime), 'neuron_bound_count': len(neuron_bounds),
                    'gradient_evaluations': 0, 'legacy_M_ignored': M}
            if count:
                # Validate the mathematical network in float64. Float32 forward
                # passes can lose the small margin when large terms cancel.
                # Separately require the ORIGINAL model to predict favorable.
                logits = [float(copy.deepcopy(m).double().logits(point.double())) for m in models]
                checked = rc.feasible(point, x0, budget, actual_constraints)
                checked = checked and all(np.isfinite(z) and z >= eps-1e-5 for z in logits)
                # Never turn an invalid numerical incumbent into a success.
                if not checked or not rc.valid_recourse(point, models):
                    info['status'] = 'incumbent_failed_validation'
                    point = x0.clone()
                info['incumbent_validated'] = checked
                info['validation_logits_float64'] = logits
                info['objective_value'] = float(problem.ObjVal)
                if problem.IsMIP:
                    info['mip_gap'] = float(problem.MIPGap)
            point = rc._tag(point, **info)
            rr.attach_trace(point, points=np.asarray([x0.detach().cpu().numpy(), point.detach().cpu().numpy()]),
                            neuron_bounds=np.asarray(neuron_bounds).reshape(-1, 2))
    return (point, info) if return_info else point


def milp_recourse(x0, models, budget, constraints=None, return_info=False):
    """Legacy LP baseline, now a QCP with the L2 budget inside the optimization.

    The existing margin=1.0 is preserved. Infeasibility at that stronger margin
    is reported explicitly; it is not proof that no class-1 point exists.
    """
    margin = 1.0
    if not all(rc.is_linear(m) for m in models):
        raise TypeError('This baseline requires affine-logit models/surrogates')
    return exact_milp_recourse_gurobi(x0, models, budget, eps=margin,
        constraints=constraints, return_info=return_info)
