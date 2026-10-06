"""Leakage-free repeated-holdout experiments with preserved notebook settings."""
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from . import recourse_core as rc
from . import recourse_baselines as rb
from . import recourse_results as rr
from . import recourse_statistics as rs
from . import recourse_campaign as campaign
from .recourse_results import experiment_settings, read_records, load_models


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def save_records(records, name):
    if rr.current_run():
        rc.RUN_RECORDS[:] = records
        return str(rr.current_run().path/'instances.jsonl')
    rc.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in str(name))
    # Include configuration/seed in caller's name; rerunning the same run replaces it.
    path = rc.RESULTS_DIR/f'{name}_{rr._stamp()}_instances.jsonl'
    with path.open('w') as stream:
        for row in records:
            stream.write(json.dumps(_jsonable(row), allow_nan=False)+'\n')
    rc.RUN_RECORDS[:] = records
    return str(path)


def _commit(record, point=None, models=None, x0=None, c=None):
    if rr.current_run():
        return rr.current_run().commit(record, point, models, x0, c)
    return record


def _save_seed(models, candidates, fold, seed, **metadata):
    if rr.current_run():
        rr.current_run().save_seed(models, candidates, fold, seed, **metadata)


def _save_ellice(prepared, fold, seed):
    if rr.current_run() and prepared is not None:
        rr.current_run().save_arrays(f'seed_{seed}_fold_{fold}/ellice_preparation.npz',
            **{k: v for k, v in prepared.items() if k != 'feature'})


def _record(method, point, models, x0, recourse_budget, c, elapsed, **metadata):
    info = getattr(point, 'recourse_info', {'status': 'unreported'})
    is_feasible = rc.feasible(point, x0, recourse_budget, c)
    with torch.no_grad():
        probabilities = [float(m(point)) for m in models]
    finite = np.isfinite(probabilities).all()
    loss = max((p-1)**2 for p in probabilities) if finite else None
    return dict(metadata, method=method, valid=rc.valid_recourse(point, models, x0, recourse_budget, c),
        feasible=is_feasible, max_loss=loss, elapsed_seconds=elapsed,
        round_log_write_seconds=info.get('round_log_write_seconds', 0.0),
        elapsed_without_round_log_writes=max(0.0, elapsed-info.get('round_log_write_seconds', 0.0)),
        cost=float(torch.linalg.vector_norm(point-x0)), probabilities=probabilities if finite else None,
        point=point.tolist(), factual=x0.tolist(), solver=info,
        gradient_evaluations=info.get('gradient_evaluations', 0),
        status=info.get('status', 'unreported'), protocol='separate_iteration_caps_not_compute_matched',
        loss='MSE_probability_nonconvex_heuristic', feature_domain='continuous_relaxation')


def _summary(records, methods, folds, index):
    result = {'Requested Repetitions': folds, 'Protocol': 'repeated_stratified_holdout',
              'Comparison': 'separate iteration caps; compute effort reported separately',
              'Uncertainty': 'pointwise 95% seed-t CI; Std is sample SD of seed means'}
    for method in methods:
        rows = [r for r in records if r.get('method') == method and 'valid' in r]
        for prefix, metric, scale, bounds, unit in [
            ('Validity','valid',100.,(0.,100.),''),
            ('Time','elapsed_seconds',1.,(0.,np.inf),'s'),
            ('Loss','max_loss',1.,(0.,1.),''),
            ('Cost','cost',1.,(0.,np.inf),''),
        ]:
            result.update(rs.columns(f'{method} {prefix}', rs.metric_summary(rows,metric,scale,bounds),unit))
        result[f'{method} Evaluated Instances'] = len(rows)
        result[f'{method} Completed Repetitions'] = len({r['seed'] for r in rows})
        result[f'{method} Mean Gradient Evaluations'] = rs.metric_summary(rows,'gradient_evaluations')['mean']
        result[f'{method} Time Instance SD (s)'] = np.std([r['elapsed_seconds'] for r in rows],ddof=1) if len(rows)>1 else np.nan
        result[f'{method} Failure Count'] = sum(r['status'] in {
            'solver_error','timeout_without_incumbent','incumbent_failed_validation',
            'no_feasible_recourse_found','infeasible'} for r in rows)
        result[f'{method} Incomplete Oracle Count'] = sum(any(
            d.get('status') in {'step_limit','local_stationarity_only'}
            for d in r.get('solver',{}).get('oracle_diagnostics',[])) for r in rows)
    return pd.DataFrame(result,index=[index])


def _compare(dataset_name, dataset, counts, recourse_budget, folds, max_instances, seed, T,
        lr=0.01, actionable_indices=None, diverse=False, epsilon=0.05,
        integer_method=None, tolerance=None):
    orpm_T = rr.orpm_rounds(T, dataset_name, recourse_budget, counts)
    rr.record_iteration_limits(orpm_T, T)
    X, y = dataset
    X = rc._frame(X, dataset_name)
    methods = ['ORPM', 'ADV', 'ElliCE', 'ROAR']+(['MILP'] if integer_method else [])
    records = []
    for fold in range(folds):
        print(f'{dataset_name}: seed {seed+fold} ({fold+1}/{folds}); preparing models', flush=True)
        current_seed = seed+fold
        rr.set_scope(fold=fold, seed=current_seed)
        try:
            if tolerance is None:
                models, candidates = rc.train_and_select_candidates(X, y, *counts,
                    max_instances=max_instances, seed=current_seed, epsilon=epsilon,
                    actionable_indices=actionable_indices, diverse=diverse, dataset_name=dataset_name)
            else:
                models, candidates = generate_models_with_tolerance(X, y, tolerance, *counts,
                    num_selected=sum(counts), max_instances=max_instances, seed=current_seed)
        except Exception as exc:
            for method in methods:
                records.append(_commit(dict(method=method, fold=fold, seed=current_seed, status='model_generation_failed',
                    error=f'{type(exc).__name__}: {exc}', requested_counts=counts)))
            continue
        _save_seed(models, candidates, fold, current_seed)
        ctx = models.context
        prepared = None
        prep_error = None
        before = time.perf_counter()
        try:
            prepared = rb.prepare_ellice(models[0], ctx.X_train, ctx.epsilon)
        except Exception as exc:
            prep_error = exc
        preparation_time = time.perf_counter()-before
        _save_ellice(prepared, fold, current_seed)
        for position, x0 in enumerate(candidates):
            print(f'{dataset_name}: seed {seed+fold}, factual {position+1}/{len(candidates)}, budget {recourse_budget:g}', flush=True)
            c = rc.constraints_for(models, x0)
            before = time.perf_counter()
            surrogate_error = None
            try:
                surrogates = rb.linearize_models_lime(models, x0.numpy(), ctx.X_train,
                                                      seed=current_seed+position)
                first_a, first_b = rc.linear_parameters(surrogates[0])
                delta_model = max(max(float((a-first_a).abs().max()), float((b-first_b).abs()))
                                  for a, b in map(rc.linear_parameters, surrogates))
            except Exception as exc:
                surrogate_error, surrogates, delta_model = exc, None, None
            lime_time = time.perf_counter()-before
            if rr.current_run() and surrogates is not None:
                rr.current_run().save_arrays(f'seed_{current_seed}_fold_{fold}/instance_{ctx.candidate_ids[position]}/surrogates.npz',
                    weights=np.stack([rc.linear_parameters(m)[0].numpy() for m in surrogates]),
                    biases=np.asarray([float(rc.linear_parameters(m)[1]) for m in surrogates]),
                    delta_model=delta_model)
            metadata = {'dataset': rc.canonical_name(dataset_name), 'fold': fold, 'seed': current_seed,
                'instance_id': ctx.candidate_ids[position], 'actual_models': len(models),
                'requested_counts': counts, 'model_metadata': models.metadata,
                'budget': recourse_budget, 'T': T, 'actionability_policy': c.policy,
                'feature_names': ctx.columns, 'favorable_raw_label': rc.FAVORABLE_LABEL.get(ctx.name, 1)}
            for method in methods:
                rr.set_scope(fold=fold, seed=current_seed, instance_id=ctx.candidate_ids[position], method=method)
                before = time.perf_counter()
                overhead = 0.0
                try:
                    if method == 'ORPM':
                        point = rc.solve_robust_recourse_with_oracle(x0, models, recourse_budget,
                            rc.recourse_loss, rc.oracle_gradient_descent, T=orpm_T, B=1)[0]
                    elif method == 'ADV':
                        point = rb.ADV(x0, models, recourse_budget, rc.recourse_loss,
                                       T=T, lr=lr, constraints=c)[-1]
                    elif method == 'ElliCE':
                        if prep_error is not None:
                            raise prep_error
                        point = rb.ElliCE(x0, models, recourse_budget, rc.recourse_loss,
                            T=T, lr=lr, constraints=c, preparation=prepared)[-1]
                        overhead = preparation_time/max(1, len(candidates))
                    elif method == 'ROAR':
                        if surrogate_error is not None:
                            raise surrogate_error
                        point = rb.ROAR(x0, surrogates[0], delta_model, recourse_budget,
                            rc.recourse_loss, T=T, lr=lr, constraints=c)[-1]
                        overhead = lime_time
                    elif integer_method == 'original_models':
                        point = rb.exact_milp_recourse_gurobi(x0, models, recourse_budget, constraints=c)
                    else:
                        if surrogate_error is not None:
                            raise surrogate_error
                        point = rb.milp_recourse(x0, surrogates, recourse_budget, constraints=c)
                        overhead = lime_time
                    elapsed = time.perf_counter()-before+overhead
                    record = _record(method, point, models, x0, recourse_budget, c, elapsed, **metadata)
                    if method == 'ROAR' or (method == 'MILP' and integer_method != 'original_models'):
                        record['surrogate_diagnostics'] = [s.surrogate_diagnostics for s in surrogates]
                except Exception as exc:
                    point = rc._tag(x0.clone(), status='solver_error',
                                    error=f'{type(exc).__name__}: {exc}', gradient_evaluations=0)
                    record = _record(method, point, models, x0, recourse_budget, c,
                                     time.perf_counter()-before+overhead, **metadata)
                    record['valid'] = False
                record.update(T=orpm_T if method == 'ORPM' else T, ORPM_T=orpm_T, baseline_T=T)
                record['preparation_seconds_amortized'] = preparation_time/max(1, len(candidates)) if method == 'ElliCE' else 0.0
                record['surrogate_seconds'] = lime_time if method == 'ROAR' or (method == 'MILP' and integer_method != 'original_models') else 0.0
                records.append(_commit(record, point, models, x0, c))
    config = f'{sum(counts)} Models ({counts[0]}LR+{counts[1]}NN1+{counts[2]}NN2)'
    key = (f'{dataset_name}_{counts}_b{recourse_budget}_T{T}_seed{seed}_tol{tolerance}'
           f'_folds{folds}_instances{max_instances}_diverse{diverse}_integer{integer_method}')
    path = save_records(records, key)
    result = _summary(records, methods, folds, config)
    result['Instance Records'] = path
    result['ORPM T'] = orpm_T
    result['Baseline T'] = T
    result['Actual Models'] = sum(counts) if any('actual_models' in r for r in records) else 0
    result['Model Generation Failures'] = sum(r['status'] == 'model_generation_failed' for r in records)//len(methods)
    if tolerance is not None:
        spreads = [r['model_metadata']['achieved_accuracy_spread'] for r in records if 'model_metadata' in r]
        result['Achieved Accuracy Spread'] = np.mean(spreads) if spreads else np.nan
    return result


@rr.experiment
def comparison_with_baselines_linear(dataset_name, dataset, recourse_budget=5.0,
        folds=5, max_instances=5, seed=42, T=200, lr=0.01):
    result = _compare(dataset_name, dataset, (3, 0, 0), recourse_budget, folds, max_instances, seed, T, lr)
    for method in ['ORPM', 'ADV', 'ElliCE', 'ROAR']:
        mean = result[f'{method} Validity Mean'].iloc[0]
        row = result.iloc[0]
        result[f'{method} Validity'] = rs.format_ci(mean, row[f'{method} Validity CI95 Low'],
            row[f'{method} Validity CI95 High'],row[f'{method} Validity CI Status'])
        result[f'{method} Loss'] = result.get(f'{method} Loss Mean', np.nan)
    result.index = [dataset_name]
    return result


@rr.experiment
def comparison_with_baselines_nonlinear(dataset_name, dataset, num_models=2,
        recourse_budget=2.0, folds=1, max_instances=5, seed=42, T=100, include_miqcp=True):
    return _compare(dataset_name, dataset, (0, 0, num_models), recourse_budget,
                    folds, max_instances, seed, T,
                    integer_method='original_models' if include_miqcp else None)


@rr.experiment
def comparison_with_baselines_nonlinear_v2(dataset_name, dataset, num_linear=1,
        num_nn_single=1, num_nn_double=1, recourse_budget=2.0, folds=1,
        max_instances=5, seed=42, T=100, actionable_indices=None):
    return _compare(dataset_name, dataset, (num_linear, num_nn_single, num_nn_double),
        recourse_budget, folds, max_instances, seed, T, actionable_indices=actionable_indices, diverse=True)


@rr.experiment
def comparison_with_baselines_nonlinear_diverse(dataset_name, dataset,
        recourse_budget=2.0, folds=1, max_instances=5, seed=42, T=100,
        actionable_indices=None, epsilon=0.05):
    return _compare(dataset_name, dataset, (1, 1, 1), recourse_budget, folds,
        max_instances, seed, T, actionable_indices=actionable_indices, diverse=True,
        epsilon=epsilon, integer_method='surrogate_models_margin1')


@rr.experiment
def run_generalization_experiment(dataset_name, dataset, num_opt_models=5, num_eval_models=10,
        recourse_budget=5.0, max_instances=10, seed=42, T=20, folds=2):
    orpm_T = rr.orpm_rounds(T, dataset_name, recourse_budget, (num_opt_models,0,0))
    rr.record_iteration_limits(orpm_T, T)
    X, y = dataset
    X = rc._frame(X, dataset_name)
    records = []
    generation_failures = []
    for fold in range(folds):
        print(f'{dataset_name}: seed {seed+fold} ({fold+1}/{folds}); preparing models', flush=True)
        rr.set_scope(fold=fold, seed=seed+fold)
        try:
            models, _ = rc.train_and_select_candidates(X, y, num_opt_models+num_eval_models, 0, 0,
                max_instances=max_instances, seed=seed+fold, epsilon=0.05, dataset_name=dataset_name)
        except Exception as exc:
            generation_failures.append(_commit({'dataset': dataset_name, 'fold': fold, 'seed': seed+fold,
                'status': 'model_generation_failed', 'error': f'{type(exc).__name__}: {exc}'}))
            continue
        opt = rc.ModelSet(models[:num_opt_models], context=copy.copy(models.context))
        unseen = rc.ModelSet(models[num_opt_models:], context=copy.copy(models.context))
        # Eligibility depends only on the deployed optimization model, never unseen models.
        candidates = rc.select_candidates(opt, max_instances)
        models.context.candidate_ids = opt.context.candidate_ids
        _save_seed(models, candidates, fold, seed+fold,
            optimization_model_indices=list(range(num_opt_models)),
            evaluation_model_indices=list(range(num_opt_models, len(models))))
        started = time.perf_counter()
        preparation, preparation_error = None, None
        try:
            preparation = rb.prepare_ellice(opt[0], opt.context.X_train, opt.context.epsilon)
        except Exception as exc:
            preparation_error = exc
        preparation_time = time.perf_counter()-started
        _save_ellice(preparation, fold, seed+fold)
        for j, x0 in enumerate(candidates):
            print(f'{dataset_name}: seed {seed+fold}, factual {j+1}/{len(candidates)}, budget {recourse_budget:g}', flush=True)
            c = rc.constraints_for(opt, x0)
            points, times = {}, {}
            for method in ['ORPM', 'ElliCE']:
                rr.set_scope(fold=fold, seed=seed+fold, instance_id=opt.context.candidate_ids[j], method=method)
                started = time.perf_counter()
                try:
                    if method == 'ORPM':
                        point = rc.solve_robust_recourse_with_oracle(x0, opt, recourse_budget,
                            rc.recourse_loss, rc.oracle_gradient_descent, T=orpm_T, B=1)[0]
                    else:
                        if preparation_error is not None: raise preparation_error
                        point = rb.ElliCE(x0, opt, recourse_budget, rc.recourse_loss,
                            T=T, lr=0.01, preparation=preparation, constraints=c)[-1]
                    points[method] = point
                except Exception as exc:
                    points[method] = rc._tag(x0.clone(), status='solver_error',
                        error=f'{type(exc).__name__}: {exc}', gradient_evaluations=0)
                times[method] = time.perf_counter()-started
                if method == 'ElliCE':
                    times[method] += preparation_time/max(1, len(candidates))
            for method, point in points.items():
                for split_name, evaluation in [('opt', opt), ('eval', unseen)]:
                    row = _record(method+'_'+split_name, point, evaluation, x0, recourse_budget, c, times[method],
                        dataset=dataset_name, fold=fold, seed=seed+fold,
                        instance_id=opt.context.candidate_ids[j],
                        budget=recourse_budget, T=orpm_T if method == 'ORPM' else T, ORPM_T=orpm_T, baseline_T=T,
                        optimization_model_count=len(opt), evaluation_model_count=len(unseen))
                    if point.recourse_info['status'] == 'solver_error':
                        row['valid'] = False
                    row['model_fraction_valid'] = float(np.mean(np.asarray(row['probabilities']) >= .5)) if row['probabilities'] is not None and row['feasible'] and row['status'] != 'solver_error' else 0.0
                    row['model_hash'] = hashlib.sha256(b''.join(p.detach().numpy().tobytes()
                        for m in models for p in m.parameters())).hexdigest()
                    records.append(_commit(row, point, evaluation, x0, c))
    key = f'{dataset_name}_generalization_opt{num_opt_models}_eval{num_eval_models}_b{recourse_budget}_T{T}_seed{seed}'
    path = save_records(records+generation_failures, key)
    summary = _summary(records,['ORPM_opt','ORPM_eval','ElliCE_opt','ElliCE_eval'],folds,dataset_name).iloc[0].to_dict()
    names = {'ORPM_opt': 'ORPM Opt Validity (%)', 'ORPM_eval': 'ORPM Generalization (%)',
             'ElliCE_opt': 'ElliCE Opt Validity (%)', 'ElliCE_eval': 'ElliCE Generalization (%)'}
    for method, label in names.items():
        summary[label] = rs.format_ci(summary[f'{method} Validity Mean'],
            summary[f'{method} Validity CI95 Low'],summary[f'{method} Validity CI95 High'],
            summary[f'{method} Validity CI Status'])
        rows = [r for r in records if r['method'] == method]
        summary.update(rs.columns(f'{method} Model Fraction',rs.metric_summary(rows,'model_fraction_valid',100.,(0.,100.))))
    summary.update({'ORPM T':orpm_T,'Baseline T':T,'Optimization Models':num_opt_models,'Unseen Models':num_eval_models})
    summary.update({'Protocol': 'independent repeated stratified holdouts', 'Instance Records': path,
                    'Model Generation Failures': len(generation_failures),
                    'Requested Repetitions': folds, 'Completed Repetitions': len({r['fold'] for r in records})})
    return pd.DataFrame(summary, index=[dataset_name])


def _select_accuracy_window(pool, requested, tolerance):
    """Choose the widest attainable validation-accuracy spread <= tolerance.

    Keep exact architecture counts. The finite pool may not attain the requested
    spread; report achieved spread and subset IDs rather than relabeling it.
    """
    if tolerance < 0:
        raise ValueError('tolerance must be nonnegative')
    ordered = sorted(pool, key=lambda m: (m['acc'], m['id']))
    best = None
    for low in range(len(ordered)):
        for high in range(low, len(ordered)):
            if ordered[high]['acc']-ordered[low]['acc'] > tolerance+1e-12:
                break
            selected, needed = [], list(requested)
            endpoints = [ordered[low]] if low == high else [ordered[low], ordered[high]]
            valid = True
            for candidate in endpoints:
                if needed[candidate['kind']] <= 0:
                    valid = False
                    break
                selected.append(candidate)
                needed[candidate['kind']] -= 1
            if not valid:
                continue
            for candidate in reversed(ordered[low:high+1]):
                if candidate['id'] not in {m['id'] for m in selected} and needed[candidate['kind']] > 0:
                    selected.append(candidate)
                    needed[candidate['kind']] -= 1
            if any(needed):
                continue
            spread = max(m['acc'] for m in selected)-min(m['acc'] for m in selected)
            key = (spread, np.mean([m['acc'] for m in selected]))
            if best is None or key > best[0]:
                best = (key, sorted(selected, key=lambda m: m['id']))
    if best is None:
        raise ValueError(f'No subset satisfies accuracy tolerance {tolerance} and architecture counts {requested}')
    return best[1]


@rr.experiment(settings_factory=lambda mode, seed: campaign.profile_settings(mode,seed,'convergence'))
def run_convergence_experiment(dataset_name, dataset, num_linear=1, num_nn_single_layer=1,
        num_nn_double_layer=1, recourse_budget=2.0, actionable_indices=None,
        folds=2, max_instances=20, seed=42, T=50):
    """MSE objective trajectories, with the same mode and persistence as comparisons."""
    baseline_T = T
    T = rr.orpm_rounds(T,kind='convergence')
    rr.record_iteration_limits(T,baseline_T)
    X, y = dataset
    X = rc._frame(X, dataset_name)
    records, trajectories, trajectory_seeds = [], [], []
    counts = (num_linear, num_nn_single_layer, num_nn_double_layer)
    for fold in range(folds):
        print(f'{dataset_name}: seed {seed+fold} ({fold+1}/{folds}); preparing models', flush=True)
        rr.set_scope(fold=fold, seed=seed+fold)
        try:
            models, candidates = rc.train_and_select_candidates_diverse(X, y, *counts,
                max_instances=max_instances, seed=seed+fold, actionable_indices=actionable_indices,
                dataset_name=dataset_name)
        except Exception as exc:
            records.append(_commit({'method': 'ORPM', 'fold': fold, 'seed': seed+fold,
                'status': 'model_generation_failed', 'error': f'{type(exc).__name__}: {exc}'}))
            continue
        _save_seed(models, candidates, fold, seed+fold)
        for position, x0 in enumerate(candidates):
            print(f'{dataset_name}: seed {seed+fold}, factual {position+1}/{len(candidates)}, budget {recourse_budget:g}', flush=True)
            rr.set_scope(fold=fold, seed=seed+fold, instance_id=models.context.candidate_ids[position], method='ORPM')
            c = rc.constraints_for(models, x0)
            started = time.perf_counter()
            try:
                point, wbar, history, _, _ = rc.solve_robust_recourse_with_oracle(x0, models,
                    recourse_budget, rc.recourse_loss, rc.oracle_gradient_descent, T=T, B=1)
                elapsed = time.perf_counter()-started
                curve = [max(rc.recourse_loss(m, x).item() for m in models) for x in history]
                trajectories.append(curve)
                trajectory_seeds.append(seed+fold)
            except Exception as exc:
                elapsed = time.perf_counter()-started
                point = rc._tag(x0.clone(), status='solver_error', error=f'{type(exc).__name__}: {exc}')
            row = _record('ORPM', point, models, x0, recourse_budget, c, elapsed,
                dataset=dataset_name, fold=fold, seed=seed+fold, instance_id=models.context.candidate_ids[position],
                actual_models=len(models), requested_counts=counts, budget=recourse_budget, T=T,
                measurement='maximum_probability_MSE_nonconvex_heuristic')
            if point.recourse_info['status'] == 'solver_error': row['valid'] = False
            records.append(_commit(row, point, models, x0, c))
    config = f'{sum(counts)} Models ({counts[0]}LR+{counts[1]}NN1+{counts[2]}NN2)'
    result = _summary(records, ['ORPM'], folds, config)
    result['Instance Records'] = save_records(records, 'objective_trajectories')
    result['ORPM T'] = T
    if trajectories:
        unique_seeds = sorted(set(trajectory_seeds))
        matrix = np.asarray(trajectories)
        seed_means = np.stack([matrix[np.asarray(trajectory_seeds)==seed].mean(axis=0) for seed in unique_seeds])
        stats = [rs.mean_ci(seed_means[:,j],(0.,1.)) for j in range(T)]
        curves = pd.DataFrame({'round':np.arange(1,T+1),
            'mean_max_loss':[v['mean'] for v in stats],'std_max_loss':[v['sd'] for v in stats],
            'ci95_low':[v['ci_low'] for v in stats],'ci95_high':[v['ci_high'] for v in stats],
            'ci95_unbounded_low':[v['ci_low_unbounded'] for v in stats],
            'ci95_unbounded_high':[v['ci_high_unbounded'] for v in stats],
            'se_max_loss':[v['se'] for v in stats],
            'n_seeds':[v['n_seeds'] for v in stats],'ci_status':[v['ci_status'] for v in stats]})
        if rr.current_run():
            curves.to_csv(rr.current_run().path/'convergence.csv',index=False)
            pd.DataFrame([{'seed':seed,'round':j+1,'mean_max_loss':float(seed_means[i,j])}
                          for i,seed in enumerate(unique_seeds) for j in range(T)]).to_csv(
                              rr.current_run().path/'convergence_seed_means.csv',index=False)
        result.attrs['trajectory'] = {'dataset':dataset_name,'config':config,'budget':recourse_budget,
            'mean':curves.mean_max_loss.to_numpy(),'std':curves.std_max_loss.to_numpy(),
            'ci_low':curves.ci95_low.to_numpy(),'ci_high':curves.ci95_high.to_numpy(),
            'n_seeds':len(unique_seeds),'ci_status':curves.ci_status.tolist(),
            'uncertainty':'pointwise 95% seed-t CI; not simultaneous',
            'individual_trajectories':trajectories,'trajectory_seeds':trajectory_seeds,
            'seed_means':seed_means}
    return result


def generate_models_with_tolerance(X, y, tolerance, num_linear=1, num_nn_single_layer=1,
        num_nn_double_layer=1, num_selected=3, max_instances=10, seed=42):
    requested = [num_linear, num_nn_single_layer, num_nn_double_layer]
    if num_selected != sum(requested):
        raise ValueError('num_selected must equal the requested architecture counts')
    ctx = rc.prepare_split(X, y, seed)
    d = len(ctx.columns)
    factories = [lambda s: rc.LogisticRegressionModel(d, s),
                 lambda s: rc.NeuralNetworkModelSingleLayer(d, 20, s),
                 lambda s: rc.NeuralNetworkModelDoubleLayers(d, 50, 100, s)]
    pool = []
    # Expanded candidate pool is a selection fix, not an increase in competing-set size.
    for kind, count in enumerate(requested):
        for i in range(3*count):
            model_seed = seed+kind*100+i
            model = factories[kind](model_seed)
            rc.train_model(model, ctx.X_train, ctx.y_train, epochs=50, lr=0.01, seed=model_seed)
            pool.append({'model': model, 'kind': kind, 'id': kind*100+i,
                         'acc': rc._accuracy(model, ctx.X_validation, ctx.y_validation)})
    selected = _select_accuracy_window(pool, requested, tolerance)
    models = rc.ModelSet([item['model'] for item in selected], context=ctx)
    for model in models:
        model.recourse_context = ctx
    achieved = max(i['acc'] for i in selected)-min(i['acc'] for i in selected)
    models.metadata = {'requested_accuracy_spread': tolerance, 'achieved_accuracy_spread': achieved,
        'selected_pool_ids': [i['id'] for i in selected], 'pool_size': len(pool),
        'actual_count': len(models), 'requested_counts': requested,
        'cohort_reference': 'fixed_candidate_pool_model_0',
        'parameter_closeness_claim': False, 'selection_set': 'validation'}
    models.metadata['candidate_pool'] = [{k: v for k, v in item.items() if k != 'model'} | {
        'training_artifact': getattr(item['model'], 'training_artifact', None)} for item in pool]
    # Same deployed reference across tolerances; selected competitors may exclude it.
    candidates = rc.select_candidates(models, max_instances, reference_model=pool[0]['model'])
    return models, candidates


@rr.experiment
def comparison_with_baselines_nonlinear_v3(dataset_name, tolerance, dataset,
        num_linear=1, num_nn_single=1, num_nn_double=1, recourse_budget=2.0,
        folds=1, max_instances=5, seed=42, T=100):
    return _compare(dataset_name, dataset, (num_linear, num_nn_single, num_nn_double),
        recourse_budget, folds, max_instances, seed, T, tolerance=tolerance)


@rr.experiment(settings_factory=lambda mode, seed: campaign.profile_settings(mode,seed,'cost_validity'))
def run_comprehensive_scalability_experiment(dataset_name=None, dataset_data=None,
        model_configs=None, budgets=None, folds=1, max_instances=3, T=50, datasets_dict=None):
    if datasets_dict is not None:
        frames = [run_comprehensive_scalability_experiment(name, data, model_configs, budgets,
                  folds, max_instances, T) for name, data in datasets_dict.items()]
        return pd.concat(frames, ignore_index=True)
    if dataset_name is None or dataset_data is None:
        raise ValueError('Provide a dataset or datasets_dict')
    X, y = dataset_data[:2]
    action = dataset_data[2] if len(dataset_data) > 2 else None
    rows = []
    for counts in model_configs:
        for budget in budgets:
            print(f'{dataset_name}: {counts} competing models, budget {budget:g}', flush=True)
            result = comparison_with_baselines_nonlinear_v2(dataset_name, (X, y), *counts,
                recourse_budget=budget, folds=folds, max_instances=max_instances, seed=42,
                T=T, actionable_indices=action)
            result['Dataset'] = dataset_name
            result['Config'] = f'{sum(counts)} Models ({counts[0]}LR+{counts[1]}NN1+{counts[2]}NN2)'
            result['Total_Models'] = result['Actual Models']
            result['Requested_Models'] = sum(counts)
            result['Num Linear'], result['Num NN1'], result['Num NN2'] = counts
            result['Budget'] = budget
            rows.append(result)
            # Preserve completed configurations and budget points during long sweeps.
            from . import recourse_reporting as rp
            progress = pd.concat(rows, ignore_index=True)
            rp.save_dataset_tables(progress, rc.RESULTS_DIR, 'cost_validity_summary')
    result = pd.concat(rows, ignore_index=True)
    rc.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = str(dataset_name).replace(' ', '_').replace('/', '_')
    destination = rr.current_run().path if rr.current_run() else rc.RESULTS_DIR
    result.to_csv(destination/f'scalability_results_{name}.csv', index=False)
    return result


@rr.experiment
def run_comprehensive_rashomon_violation_experiment(datasets_dict, tolerances, model_configs,
        budgets, folds=1, max_instances=3, T=50):
    rows = []
    for name, data in datasets_dict.items():
        for counts in model_configs:
            for budget in budgets:
                for tolerance in tolerances:
                    result = comparison_with_baselines_nonlinear_v3(name, tolerance, data, *counts,
                        recourse_budget=budget, folds=folds, max_instances=max_instances, seed=42, T=T)
                    result['Dataset'], result['Budget'], result['Tolerance'] = name, budget, tolerance
                    result['Total_Models'] = result['Actual Models']
                    result['Config'] = f'{sum(counts)} Models ({counts[0]}LR+{counts[1]}NN1+{counts[2]}NN2)'
                    rows.append(result)
    return pd.concat(rows, ignore_index=True)


def compare_orpm_adv_gradient_caps(x0, models, budget, model_gradient_budget, T=100):
    """Additional comparison with equal upper limits on model-gradient evaluations.

    Kept separate so the notebook's existing T/step settings are unchanged.
    Actual counts are returned because early stopping/analytic solutions use less.
    """
    n = len(models)
    steps = model_gradient_budget//(n*T)
    if steps < 1:
        raise ValueError('Budget must allow at least one step per ORPM round')
    shared_cap = steps*n*T
    warm = [None]
    def oracle(w, m, x, d, loss):
        point = rc.oracle_gradient_descent(w, m, x, d, loss, num_steps=steps, initial_x=warm[0])
        warm[0] = point.detach()
        return point
    start = time.perf_counter()
    orpm = rc.solve_robust_recourse_with_oracle(x0, models, budget, rc.recourse_loss, oracle, T=T)[0]
    orpm_time = time.perf_counter()-start
    start = time.perf_counter()
    adv = rb.ADV(x0, models, budget, rc.recourse_loss, T=steps*T)[-1]
    adv_time = time.perf_counter()-start
    return pd.DataFrame([{'Method': method, 'Gradient Cap': shared_cap,
        'Gradient Evaluations': point.recourse_info['gradient_evaluations'],
        'Time': elapsed, 'Validity': rc.valid_recourse(point, models, x0, budget),
        'Max Loss': max(rc.recourse_loss(m, point).item() for m in models)}
        for method, point, elapsed in [('ORPM', orpm, orpm_time), ('ADV', adv, adv_time)]])


def oracle_sensitivity(weights, models, x0, budget, configurations, seed=42):
    """Opt-in steps/lr/restarts sweep; never changes a main experiment setting.

    Each configuration supplies num_steps, lr and restarts (including the
    factual start). Random restarts use an independent deterministic generator.
    Reports the best observed weighted loss, not a nonconvex global optimum.
    """
    c = rc.constraints_for(models, x0)
    rows = []
    for config in configurations:
        rng = np.random.default_rng(seed)
        starts = [x0]
        for _ in range(config['restarts']-1):
            direction = rng.normal(size=len(x0))
            direction /= np.linalg.norm(direction)
            radius = budget*rng.random()**(1/len(x0))
            starts.append(rc.project_l2_ball(x0+x0.new_tensor(radius*direction), x0, budget, constraints=c))
        if config['restarts'] < 1:
            raise ValueError('restarts must include at least the factual start')
        best_value, best, gradients = float('inf'), None, 0
        started = time.perf_counter()
        for initial in starts:
            point = rc.oracle_gradient_descent(weights, models, x0, budget, rc.recourse_loss,
                num_steps=config['num_steps'], lr=config['lr'], initial_x=initial, constraints=c)
            gradients += point.recourse_info['gradient_evaluations']
            value = sum(float(w)*rc.recourse_loss(m, point).item() for w, m in zip(weights, models))
            if value < best_value:
                best_value, best = value, point
        rows.append(dict(config, seed=seed, weighted_loss=best_value,
            valid=rc.valid_recourse(best, models, x0, budget, c),
            feasible=rc.feasible(best, x0, budget, c), gradient_evaluations=gradients,
            elapsed_seconds=time.perf_counter()-started,
            oracle_error_upper_bound=best.recourse_info['oracle_error_upper_bound'],
            bound_type=best.recourse_info.get('bound_type', 'analytic_single_affine_solution')))
    return pd.DataFrame(rows)


def compare_compas_actionability(models, candidates, budget, T=100):
    """Appendix A.8 policy: sex fixed, 0 <= age increase <= 5 original years.

    Uses the supplied models, candidates, budget and rounds. Does not change
    experiment settings or silently substitute another model configuration.
    """
    if models.context.name != 'COMPAS':
        raise ValueError('This policy is specific to COMPAS')
    rows = []
    for i, x0 in enumerate(candidates):
        for policy in ['unconstrained', 'compas_appendix']:
            c = rc.constraints_for([], x0) if policy == 'unconstrained' else rc.constraints_for(models, x0, policy=policy)
            warm = [None]
            def oracle(w, m, x, d, loss):
                point = rc.oracle_gradient_descent(w, m, x, d, loss,
                    constraints=c, initial_x=warm[0])
                warm[0] = point.detach()
                return point
            point = rc.solve_robust_recourse_with_oracle(x0, models, budget,
                rc.recourse_loss, oracle, T=T, constraints=c)[0]
            rows.append({'instance': i, 'policy': policy,
                'valid': rc.valid_recourse(point, models, x0, budget, c),
                'feasible': rc.feasible(point, x0, budget, c), 'point': point.tolist()})
    return pd.DataFrame(rows)
