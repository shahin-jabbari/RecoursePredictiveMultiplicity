"""V5 saddle-gap diagnostics. Solver intervals and sampling CIs are distinct.

For a feasible point x and simplex weights w, P=max_i loss_i(x) and
q(w)=min_z sum_i w_i loss_i(z). If L <= q(w) <= U, the gap is in
[max(0,P-U), P-L]. A local feasible response provides U, never L.
Tree diagnostics use a separate solver after the ORPM trajectory is fixed.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from . import recourse_core as rc
from . import recourse_results as rr
from . import recourse_statistics as rs
from . import recourse_reporting as rp

GAP_SCHEMA = 'v5-duality-gap-v1'


def checkpoints(T, every=5):
    if int(T) != T or T < 1 or int(every) != every or every < 1:
        raise ValueError('T and checkpoint spacing must be positive integers')
    return sorted({1, int(T), *range(int(every), int(T)+1, int(every))})


def gap_interval(primal, dual_lower, dual_upper, tolerance=1e-7):
    """Reject contradictory bounds rather than reporting a false zero gap."""
    p, lo, hi = map(float, (primal, dual_lower, dual_upper))
    if not np.isfinite([p, lo, hi]).all():
        raise ValueError('Gap inputs must be finite')
    if lo > hi+tolerance or lo > p+tolerance or hi < -tolerance:
        raise ValueError('Inconsistent duality-gap bounds')
    # Numerical discrepancies below tolerance are not mathematical certificates.
    lo = min(lo, hi, p)
    return dict(primal= p, dual_lower_bound=lo, dual_upper_bound=hi,
                gap_lower_bound=max(0., p-hi), gap_upper_bound=max(0., p-lo),
                dual_solve_uncertainty=max(0., hi-lo))


def regret_bound(n, eta, t, mean_oracle_error, B=1.):
    """Fixed-horizon exponential weights: log(n)/(eta*t)+eta*B^2/8.

The rate 2B sqrt(log(n)/t) would incorrectly retune eta at each plotted t.
For n=1, adversary regret is zero. The same bound applies to the uniform
distribution over responses without requiring convexity in feature space.
"""
    regret = 0. if n == 1 else np.log(n)/(eta*t)+eta*B*B/8.
    return float(regret+mean_oracle_error)


def linear_ball_box_lower(g, origin, radius, lo, hi):
    """Lower bound on min g.(x-origin) over a box intersected with an L2 ball.

For any lambda>0, minimize g.d+lambda*(||d||^2-radius^2) over the box.
The resulting dual value is a lower bound even if bisection is inexact.
"""
    g = np.asarray(g, float).copy()
    lower, upper = np.asarray(lo)-origin, np.asarray(hi)-origin
    g[(lower == 0) & (upper == 0)] = 0.
    if radius == 0 or not np.any(g):
        return 0.
    ball = -radius*np.linalg.norm(g)
    left, right = 0., max(1., np.linalg.norm(g)/(2*radius))
    for _ in range(80):
        multiplier = (left+right)/2
        d = np.clip(-g/(2*multiplier), lower, upper)
        if np.linalg.norm(d) > radius:
            left = multiplier
        else:
            right = multiplier
    d = np.clip(-g/(2*right), lower, upper)
    dual = g@d+right*(d@d-radius*radius)
    return float(max(ball, dual)-1e-12*(1+abs(ball)+abs(dual)))


class ConvexLogitOracle:
    """Float64 convex logistic loss; SLSQP search plus an independent bound.

The lower bound follows from convexity, not the optimizer's success flag.
The common normalization gives losses in [0,1] on the whole feasible ball.
The feasible domain is the continuous box/L2/immutable-feature relaxation.
"""
    def __init__(self, models, x0, budget, constraints=None, maxiter=200, ftol=1e-10):
        if not models or not all(rc.is_linear(m) for m in models):
            raise ValueError('Convex benchmark requires affine-logit models')
        if not np.isfinite(budget) or budget < 0 or maxiter < 1:
            raise ValueError('Require a finite nonnegative budget and maxiter>=1')
        self.origin = np.asarray(x0, dtype=float).copy()
        self.radius, self.maxiter, self.ftol = float(budget), int(maxiter), float(ftol)
        self.c = constraints or rc.constraints_for(models, torch.as_tensor(x0))
        self.lo = np.maximum(self.c.lower, self.origin-budget)
        self.hi = np.minimum(self.c.upper, self.origin+budget)
        self.lo[self.c.immutable] = self.origin[self.c.immutable]
        self.hi[self.c.immutable] = self.origin[self.c.immutable]
        if not np.isfinite(self.origin).all() or np.any(self.lo > self.origin) or np.any(self.hi < self.origin):
            raise ValueError('Feasible box must contain the factual')
        params = [rc.linear_parameters(m) for m in models]
        self.A = np.stack([a.detach().cpu().numpy().astype(float) for a, _ in params])
        self.b = np.array([float(b.detach().cpu()) for _, b in params])
        self.normalizer = max(float(np.logaddexp(0., -self.A@self.origin-self.b+
                                     budget*np.linalg.norm(self.A, axis=1)).max()), 1e-15)

    def losses(self, x):
        return np.logaddexp(0., -self.A@np.asarray(x)-self.b)/self.normalizer

    def _feasible_candidate(self, x):
        if not np.isfinite(x).all():
            return self.origin.copy()
        x = np.clip(x, self.lo, self.hi)
        d = x-self.origin
        norm = np.linalg.norm(d)
        if norm > self.radius:
            x = self.origin+d*(self.radius/norm)*(1.-1e-12)
        return x

    def solve(self, weights, initial=None):
        w = np.asarray(weights, float)
        if w.shape != self.b.shape or (w < 0).any() or not np.isfinite(w).all() or not np.isclose(w.sum(), 1.):
            raise ValueError('Weights must be a probability vector')
        start = self._feasible_candidate(self.origin if initial is None else np.asarray(initial, float))
        def fun(x): return float(w@self.losses(x))
        def jac(x): return -(w*expit(-self.A@x-self.b))@self.A/self.normalizer
        started = time.perf_counter()
        if self.radius == 0 or np.all(self.lo == self.hi):
            x, success, message, nit = self.origin.copy(), True, 'singleton feasible set', 0
        else:
            result = minimize(fun, start, jac=jac, method='SLSQP',
                bounds=list(zip(self.lo, self.hi)),
                constraints=[dict(type='ineq', fun=lambda x:self.radius**2-np.sum((x-self.origin)**2),
                                  jac=lambda x:-2*(x-self.origin))],
                options=dict(maxiter=self.maxiter, ftol=self.ftol))
            candidates = [start, self.origin, self._feasible_candidate(result.x)]
            x = min(candidates, key=fun).copy()
            success, message, nit = bool(result.success), str(result.message), int(result.nit)
        value, gradient = fun(x), jac(x)
        support = linear_ball_box_lower(gradient, self.origin, self.radius, self.lo, self.hi)
        lower = max(0., value+gradient@(self.origin-x)+support-1e-10*(1+abs(value)))
        if lower > value+1e-8:
            raise ValueError('Convex support bound exceeds feasible objective')
        lower = min(lower, value)
        return x, dict(objective=value, lower_bound=lower, absolute_gap=value-lower,
            optimizer_success=success, optimizer_message=message, optimizer_iterations=nit,
            status='gap_certified' if value-lower <= 1e-5 else 'feasible_with_bound',
            certificate='convex_supporting_plane_ball_box_dual_float64',
            numerical_tolerance=1e-8, solver_seconds=time.perf_counter()-started)


def convex_trajectory(models, x0, budget, T=25, every=5, constraints=None,
                      maxiter=200, on_row=None):
    """The paper's averaged-point gap, with a distinct evaluation solve at wbar."""
    oracle = ConvexLogitOracle(models, x0, budget, constraints, maxiter)
    selected = set(checkpoints(T, every))
    n = len(models); eta = np.sqrt(np.log(n)/T)
    logw = np.full(n, -np.log(n)); xsum = np.zeros_like(oracle.origin)
    wsum = np.zeros(n); lower_sum = error_sum = 0.
    rows, points, weight_history, loss_history = [], [], [], []
    warm = oracle.origin.copy(); algorithm_seconds = diagnostic_seconds = 0.
    for t in range(1, T+1):
        tick = time.perf_counter()
        w = np.exp(logw); x, info = oracle.solve(w, warm); warm = x
        losses = oracle.losses(x)
        if (losses < -1e-10).any() or (losses > 1+1e-8).any():
            raise ValueError('Observed losses violate B=1')
        points.append(x.copy()); weight_history.append(w.copy()); loss_history.append(losses.copy())
        xsum += x; wsum += w; lower_sum += info['lower_bound']; error_sum += info['absolute_gap']
        xb, wb = xsum/t, wsum/t
        logw += eta*losses; logw -= logsumexp(logw)
        algorithm_seconds += time.perf_counter()-tick
        rr.event('convex_orpm_round', round=t, point=x, average_point=xb, weights=w,
                 average_weights=wb, losses=losses, oracle=info, algorithm_elapsed_seconds=algorithm_seconds)
        if t not in selected:
            continue
        tick = time.perf_counter()
        response, evaluation = oracle.solve(wb, xb)
        # q(mean w) >= mean q(w); bounds on individual rounds can be averaged.
        lower = max(lower_sum/t, evaluation['lower_bound'])
        upper = min(evaluation['objective'], float(wb@oracle.losses(xb)),
                    min(float(wb@v) for v in loss_history))
        primal = float(oracle.losses(xb).max())
        row = dict(round=t, **gap_interval(primal, lower, upper),
            weighted_oracle_gap=info['absolute_gap'], mean_oracle_error_upper_bound=error_sum/t,
            theory_upper_bound=regret_bound(n, eta, t, error_sum/t),
            eta=eta, normalizer=oracle.normalizer, actual_models=n,
            point_rule='average_point', gap_kind='convex_averaged_point',
            certificate_status=evaluation['status'], certificate_domain='continuous_box_l2_float64',
            average_weights=wb.copy(), average_point=xb.copy(), dual_response=response,
            evaluation_oracle=evaluation, algorithm_elapsed_seconds=algorithm_seconds)
        diagnostic_seconds += time.perf_counter()-tick
        row.update(diagnostic_elapsed_seconds=diagnostic_seconds,
                   total_elapsed_seconds=algorithm_seconds+diagnostic_seconds)
        rows.append(row)
        if on_row is not None: on_row(row)
    trace = dict(points=np.asarray(points), weights=np.asarray(weight_history), losses=np.asarray(loss_history),
                 average_point=xb, average_weights=wb, normalizer=oracle.normalizer, eta=eta)
    return xb, rows, trace


def tree_gap_trajectory(models, x0, budget, rounds, constraints=None, every=5,
                        time_limit=1., abs_gap=1e-5, seed=42, artifact_dir=None):
    """Post-hoc evaluation does not alter ORPM points, weights, or warm starts.

Best-point gap uses max_i loss_i(x_best). Mixed gap uses
max_i mean_s loss_i(x_s), which is NOT loss at the averaged feature vector.
"""
    from . import recourse_trees as rt
    if not rounds:
        raise ValueError('A nonempty ORPM trajectory is required')
    selected = checkpoints(len(rounds), every)
    w = np.asarray([r['weights'] for r in rounds], float)
    losses = np.asarray([r['losses'] for r in rounds], float)
    if w.shape != losses.shape or not np.isfinite(losses).all():
        raise ValueError('Malformed saved trajectory')
    origin_loss = (1-rt.probabilities(models, x0))**2
    # The factual is a feasible incumbent even if a solver reports no incumbent.
    round_lb = np.array([max(0., r['oracle']['lower_bound'])
                         if r['oracle'].get('lower_bound') is not None else 0. for r in rounds])
    errors = np.maximum(0., (w*losses).sum(axis=1)-round_lb)
    eta = np.sqrt(np.log(len(models))/len(rounds))
    started = time.perf_counter()
    evaluator = rt.TreeOracle(models, x0, budget, constraints, time_limit, abs_gap, seed, artifact_dir)
    records = []
    try:
        for t in selected:
            wb = w[:t].mean(axis=0)
            response = evaluator.solve(wb)
            info = response.recourse_info
            best = torch.as_tensor(rounds[t-1]['best_point'], dtype=x0.dtype)
            best_losses = (1-rt.probabilities(models, best))**2
            upper = min(float(wb@origin_loss), float(wb@best_losses),
                        float((losses[:t]@wb).min()), float(wb@((1-rt.probabilities(models, response))**2)))
            lower = max(float(round_lb[:t].mean()), float(info.get('lower_bound') or 0.))
            consistent = lower <= upper+1e-7
            if not consistent:
                lower = 0.  # Preserve an honest interval and explicitly flag the inconsistent solver bound.
            point_gap = gap_interval(float(best_losses.max()), lower, upper)
            mixed_gap = gap_interval(float(losses[:t].mean(axis=0).max()), lower, upper)
            row = dict(round=t, **point_gap, mixed_primal=mixed_gap['primal'],
                mixed_gap_lower_bound=mixed_gap['gap_lower_bound'],
                mixed_gap_upper_bound=mixed_gap['gap_upper_bound'],
                mean_oracle_error_upper_bound=float(errors[:t].mean()),
                mixed_theory_upper_bound=regret_bound(len(models), eta, t, errors[:t].mean()),
                eta=eta, actual_models=len(models), average_weights=wb, best_point=best.numpy(),
                dual_response=response.numpy(), evaluation_oracle=info,
                gap_kind='tree_best_point', point_rule='best_feasible_candidate',
                certificate_domain='float32_tree_encoding_solver_tolerances',
                certificate_status='bound_inconsistency_fallback' if not consistent else
                    'trivial_nonnegative_bound' if lower == 0 and info.get('lower_bound') is None else 'solver_bound_interval',
                diagnostic_elapsed_seconds=time.perf_counter()-started,
                algorithm_elapsed_seconds=float(rounds[t-1]['elapsed_seconds']))
            row['total_elapsed_seconds'] = row['algorithm_elapsed_seconds']+row['diagnostic_elapsed_seconds']
            records.append(row)
            rr.event('tree_duality_checkpoint', **row)
            if artifact_dir:
                rr._append(Path(artifact_dir)/'duality_checkpoints.jsonl', rr.jsonable(row))
    finally:
        evaluator.close()
    return records


def scalar_row(row):
    return {k:v for k,v in row.items() if v is None or isinstance(v, (str, bool, int, float, np.number))}


@rr.experiment(settings_factory=lambda mode, seed: __import__('utils.recourse_campaign', fromlist=['profile_settings']).profile_settings(mode, seed, 'convergence_duality'))
def run_convex_duality_experiment(dataset_name, dataset, num_models=3, recourse_budget=2.,
        actionable_indices=None, folds=3, max_instances=5, seed=42, T=25, gap_every=5, maxiter=200):
    run = rr.current_run(); rows = []; outcomes = []
    rr.record_iteration_limits(T, T)
    rr.atomic_json(run.path/'duality_protocol.json', dict(schema=GAP_SCHEMA,
        loss='normalized affine-logit BCE', return_rule='average point, average pre-update weights',
        oracle='SLSQP feasible search with convex supporting-plane lower bound',
        domain='continuous box and L2 ball with immutable features', B=1,
        gap_checkpoints=checkpoints(T, gap_every), solver_maxiter=maxiter,
        gap_intervals='Optimization bounds, not confidence intervals',
        uncertainty='Pointwise seed-t intervals; each seed contributes its mean over factuals',
        theory='log(n)/(eta*t)+eta/8+mean per-round certified oracle error; eta fixed for T'))
    for fold in range(folds):
        current_seed = seed+fold
        rr.set_scope(seed=current_seed, fold=fold)
        print(f'{dataset_name}: convex gap, seed {current_seed} ({fold+1}/{folds})', flush=True)
        try:
            models, factuals = rc.train_and_select_candidates_diverse(dataset[0], dataset[1], num_models, 0, 0,
                max_instances=max_instances, seed=current_seed, actionable_indices=actionable_indices, dataset_name=dataset_name)
            run.save_seed(models, factuals, fold, current_seed)
        except Exception as exc:
            run.commit(dict(dataset=dataset_name, seed=current_seed, status='model_generation_failed', error=str(exc)))
            continue
        for position, factual in enumerate(factuals):
            iid = models.context.candidate_ids[position]
            identity = dict(dataset=dataset_name, seed=current_seed, instance_id=iid,
                budget=recourse_budget, config=f'{len(models)} logistic models', family='logistic', aggregation='probability')
            rr.set_scope(seed=current_seed, fold=fold, instance_id=iid, method='ORPM-convex')
            def on_row(row):
                record = dict(identity, **scalar_row(row)); rows.append(record)
                rr._append(run.path/'duality_checkpoints.jsonl', rr.jsonable(dict(identity, **row)))
                pd.DataFrame(rows).to_csv(run.path/'duality_convergence.csv', index=False)
            try:
                point, trajectory, trace = convex_trajectory(models, factual.numpy(), recourse_budget,
                    T=T, every=gap_every, constraints=rc.constraints_for(models, factual), maxiter=maxiter, on_row=on_row)
                run.save_arrays(f'seed_{current_seed}_fold_{fold}/instance_{iid}/convex_trace.npz', **trace)
                last = trajectory[-1]
                outcome = dict(identity, method='ORPM-convex', status='completed', feasible=True,
                    valid=bool((np.array([float(a@torch.as_tensor(point, dtype=a.dtype)+b)
                        for a,b in [rc.linear_parameters(m) for m in models]]) >= 0).all()),
                    cost=float(np.linalg.norm(point-factual.numpy())), point=point, factual=factual.numpy(),
                    max_loss=last['primal'], loss_kind='normalized_affine_logit_bce',
                    elapsed_seconds=last['total_elapsed_seconds'], **scalar_row(last))
                outcomes.append(outcome); run.commit(outcome)
                print(f'  factual {position+1}/{len(factuals)}: final gap '
                      f'[{last["gap_lower_bound"]:.5g}, {last["gap_upper_bound"]:.5g}]', flush=True)
            except Exception as exc:
                run.commit(dict(identity, status='solver_error', error=f'{type(exc).__name__}: {exc}'))
                print(f'  factual {iid} failed: {exc}', flush=True)
    result = dict(Dataset=dataset_name, Budget=recourse_budget, Config=f'{num_models} logistic models',
        Method='ORPM-convex', **{'Requested Seeds':folds, 'Completed Seeds':len({r['seed'] for r in outcomes}),
        'Evaluated Instances':len(outcomes), 'ORPM T':T, 'Gap Checkpoint Spacing':gap_every,
        'Gap Definition':'averaged point and averaged weights', 'Run Directory':str(run.path)})
    for label, metric in [('Gap Lower Bound','gap_lower_bound'), ('Gap Upper Bound','gap_upper_bound'),
                          ('Primal Loss','primal'), ('Diagnostic Time','diagnostic_elapsed_seconds')]:
        result.update(rs.columns(label, rs.metric_summary(outcomes, metric, bounds=(0.,np.inf))))
    if rows: plot_gap_convergence(pd.DataFrame(rows), run.path, show=False)
    return pd.DataFrame([result])


def plot_gap_convergence(raw, output_dir, show=True):
    """Export one dataset/config/budget at a time; no instance pseudoreplication."""
    import matplotlib.pyplot as plt
    if raw.empty: return []
    outputs = []
    groups = ['dataset','family','aggregation','budget','config','gap_kind']
    raw = raw.copy()
    for key in groups:
        if key not in raw: raw[key] = 'unspecified'
    for key, data in raw.groupby(groups, dropna=False):
        ds, family, aggregation, budget, config, kind = key
        metrics = [k for k in ['primal','dual_lower_bound','dual_upper_bound','gap_lower_bound','gap_upper_bound',
            'dual_solve_uncertainty','mixed_gap_lower_bound','mixed_gap_upper_bound','theory_upper_bound',
            'mixed_theory_upper_bound','mean_oracle_error_upper_bound','algorithm_elapsed_seconds',
            'diagnostic_elapsed_seconds','total_elapsed_seconds'] if k in data]
        seed = data.groupby(['seed','round'])[metrics].mean().reset_index()
        counts = data.groupby(['seed','round']).instance_id.nunique().rename('n_instances').reset_index()
        seed = seed.merge(counts, on=['seed','round'], validate='one_to_one')
        stats = rs.summarize_seed_frame(seed, ['round'], metrics, {m:(0.,np.inf) for m in metrics})
        name = f'duality_gap_{family}_{aggregation}_{config}_budget_{budget}'
        rp.save_table(data, output_dir, ds, name+'_instances')
        rp.save_table(seed, output_dir, ds, name+'_seed_means')
        rp.save_table(stats, output_dir, ds, name+'_summary')
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
        def line(ax, metric, label, style='-', ci=True):
            sub = stats[stats.metric.eq(metric)].sort_values('round')
            if sub.empty: return
            artist, = ax.plot(sub['round'], sub['mean'], style, label=label)
            if ci:
                ax.fill_between(sub['round'].to_numpy(), sub.ci_low.to_numpy(), sub.ci_high.to_numpy(),
                                color=artist.get_color(), alpha=.12)
        line(axes[0], 'gap_upper_bound', 'Gap upper bound')
        line(axes[0], 'gap_lower_bound', 'Gap lower bound', '--')
        line(axes[1], 'primal', 'Worst-model loss at returned point')
        line(axes[1], 'dual_lower_bound', 'Dual objective lower bound', '--')
        line(axes[1], 'dual_upper_bound', 'Dual objective upper bound', ':')
        if kind == 'tree_best_point':
            line(axes[2], 'mixed_gap_upper_bound', 'Mixed-strategy gap upper bound')
            line(axes[2], 'mixed_gap_lower_bound', 'Mixed-strategy gap lower bound', '--')
            line(axes[2], 'mixed_theory_upper_bound', 'Regret bound + oracle error', ':', False)
            axes[2].set_title('Distribution over iterates; not mean features')
        else:
            line(axes[2], 'gap_upper_bound', 'Measured gap upper bound')
            line(axes[2], 'theory_upper_bound', 'Regret bound + oracle error', '--', False)
            line(axes[2], 'mean_oracle_error_upper_bound', 'Mean certified oracle-error bound', ':')
            axes[2].set_title('Fixed-horizon learning rate and oracle accuracy')
        axes[0].set_title('Best-point gap' if kind == 'tree_best_point' else 'Averaged-point duality gap')
        axes[1].set_title('Primal and dual objectives at checkpoint')
        for ax, ylabel in zip(axes, ['Duality gap (loss units)', 'Loss (normalized BCE)' if kind != 'tree_best_point' else 'Squared probability loss', 'Gap / error (loss units)']):
            ax.set(xlabel='ORPM round', ylabel=ylabel); ax.grid(alpha=.2); ax.legend(fontsize=7)
            ax.set_ylim(bottom=0.)
        model_label=str(config).replace('_',' ')
        if family == 'random_forest':
            model_label += ' (hard voting)' if aggregation == 'hard_vote' else ' (probability averaging)'
        fig.suptitle(f'{rp.dataset_label(ds)} | {model_label} | budget {budget:g}\n'
            'Lines: optimization bounds. Shading: pointwise 95% seed-t CI for each bound; unavailable CIs omitted.')
        fig.tight_layout(); outputs.extend(rp.save_figure(fig, output_dir, ds, name, show))
    return outputs
