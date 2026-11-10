"""Paired rounds-to-gap study, independent of the ordinary V7 campaigns.

The horizon, solver effort, model counts, budgets, and training protocol do not
change with run mode. Only datasets, seeds, and evaluated individuals change.
"""
from pathlib import Path
import copy
import hashlib
import json
import time
import traceback
from contextlib import contextmanager
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from . import recourse_core as rc, recourse_results as rr
from . import recourse_convergence as gap, recourse_gmc_mixed as mixed
from . import recourse_diversity as dv, recourse_statistics as stats
from . import recourse_reporting as reporting

REVISION = 'v7-rounds-study-1'
PROFILES = {
    'quick': dict(datasets=['Synthetic'], seeds=1, instances=2),
    'intermediate': dict(datasets=['Synthetic', 'COMPAS', 'Polish'], seeds=5, instances=5),
    'full': dict(datasets=['Synthetic', 'COMPAS', 'Polish', 'German',
                          'Give_Me_Some_Credit', 'Folktables'], seeds=10, instances=20),
}
BUDGETS = {name: [.5, 2., 5.] for name in ['Synthetic', 'COMPAS', 'Polish', 'German']}
BUDGETS.update(Give_Me_Some_Credit=[2., 10., 50.], Folktables=[1., 5., 10.])
GROUP = ['dataset', 'family', 'n_models', 'budget']
KEY = GROUP + ['seed', 'instance_id']


def settings(mode='quick', **overrides):
    if mode not in PROFILES:
        raise ValueError('Use quick, intermediate, or full.')
    profile = PROFILES[mode]
    result = dict(revision=REVISION, mode=mode, datasets=list(profile['datasets']),
        seeds=list(range(42, 42+profile['seeds'])), max_instances=profile['instances'],
        max_rounds=200, gap_every=5, primary_threshold=.005,
        thresholds=[.005, .01, .02], stop_on_gap=False, confirmations=2,
        candidate_selection='majority', diversity=True,
        families=['convex_logistic', 'mixed_logistic'], model_counts=[3, 6, 12, 24],
        budgets=copy.deepcopy(BUDGETS), training_epochs=40, training_lr=.01,
        convex_inner_steps=200, mixed_inner_steps=50, dual_seconds=1.,
        diversity_options=dict(accuracy_caliper=.02, candidate_pool_size=32,
                               bank_capacity=8, keep_fractions=[.5, .75]),
        recommendation_fraction=.9,
        cohort='intersection of eligibility across both families and every model count',
        normalization='one fixed loss bound per factual/family across all counts and budgets')
    unknown=set(overrides)-set(result)
    if unknown: raise ValueError(f'Unknown study settings: {sorted(unknown)}')
    result.update(copy.deepcopy(overrides))
    validate(result)
    return result


def validate(c):
    for name in ['max_instances', 'max_rounds', 'gap_every', 'confirmations',
                 'convex_inner_steps', 'mixed_inner_steps', 'training_epochs']:
        if type(c[name]) is not int or c[name]<1: raise ValueError(f'{name} must be a positive integer')
    if not c['seeds'] or len(set(c['seeds']))!=len(c['seeds']): raise ValueError('Require distinct seeds')
    if not c['datasets'] or len(set(c['datasets']))!=len(c['datasets']): raise ValueError('Require distinct datasets')
    if not c['families'] or set(c['families'])-{'convex_logistic','mixed_logistic'}:
        raise ValueError('Supported families: convex_logistic, mixed_logistic')
    counts=c['model_counts']
    if not counts or counts!=sorted(set(counts)) or any(type(n) is not int or n<1 for n in counts):
        raise ValueError('Model counts must be sorted, distinct positive integers')
    if 'mixed_logistic' in c['families'] and any(n%3 for n in counts):
        raise ValueError('Mixed counts must be multiples of three for equal architecture counts')
    rc.candidate_selection_mode(c['candidate_selection'])
    for name in ['diversity', 'stop_on_gap']:
        if type(c[name]) is not bool: raise TypeError(f'{name} must be boolean')
    for v in [c['primary_threshold'], *c['thresholds'], c['dual_seconds']]:
        if not np.isfinite(v) or v<=0: raise ValueError('Thresholds and solver cap must be positive')
    if not 0<c['recommendation_fraction']<=1: raise ValueError('Invalid recommendation fraction')
    for ds in c['datasets']:
        b=c['budgets'][ds]
        if not b or len(set(b))!=len(b) or any(not np.isfinite(v) or v<0 for v in b):
            raise ValueError('Budget grids must contain distinct finite nonnegative values')


def plan(c):
    validate(c)
    count=len(c['model_counts'])*len(c['seeds'])*c['max_instances']
    return pd.DataFrame([dict(dataset=ds, family=f, model_counts=str(c['model_counts']),
        budgets=str(c['budgets'][ds]), seeds=len(c['seeds']), individuals_per_seed=c['max_instances'],
        max_rounds=c['max_rounds'], trajectories=count*len(c['budgets'][ds]),
        max_oracle_rounds=count*len(c['budgets'][ds])*c['max_rounds'],
        gap_checks_per_trajectory=len(gap.checkpoints(c['max_rounds'], c['gap_every'])))
        for ds in c['datasets'] for f in c['families']])


def nested_indices(family, maximum, n):
    if family=='convex_logistic': return list(range(n))
    k=maximum//3
    return [offset+i for offset in [0,k,2*k] for i in range(n//3)]


def eligibility(predictions, rule):
    p=np.asarray(predictions)
    if p.ndim!=2 or not p.shape[1] or not np.isfinite(p).all():
        raise ValueError('Expected finite per-model prediction matrix')
    return p[:,0]<.5 if rule=='reference' else 2*(p<.5).sum(axis=1)>=p.shape[1]


def paired_positions(predictions_by_set, rule, limit, seed):
    masks=[eligibility(p,rule) for p in predictions_by_set]
    common=np.logical_and.reduce(masks)
    selected=np.random.default_rng(seed).permutation(np.flatnonzero(common))[:limit]
    return selected, common, masks


def bound_status(lower, upper, threshold, valid=True):
    if not valid or not np.isfinite([lower,upper]).all() or lower>upper+1e-7:
        return 'invalid_bounds'
    if upper<=threshold: return 'certified_below_threshold'
    if lower>threshold: return 'certified_above_threshold'
    return 'unresolved_bound_interval'


class GapStop:
    def __init__(self, threshold, confirmations=2):
        self.threshold=threshold; self.confirmations=confirmations; self.streak=0

    def update(self, lower, upper, valid=True):
        hit=bound_status(lower,upper,self.threshold,valid)=='certified_below_threshold'
        self.streak=self.streak+1 if hit else 0
        return self.streak>=self.confirmations


def threshold_summary(rows, threshold, max_rounds, confirmations=2):
    """No-hit rounds remain missing; do not replace them with 200 or drop them."""
    rounds=np.asarray([r['round'] for r in rows],int)
    statuses=[bound_status(r['gap_lower_bound'],r['gap_upper_bound'],threshold,
                          r.get('bounds_valid',True)) for r in rows]
    hits=np.asarray([s=='certified_below_threshold' for s in statuses])
    gate=GapStop(threshold,confirmations); confirmed=[]
    for r in rows:
        if gate.update(r['gap_lower_bound'],r['gap_upper_bound'],r.get('bounds_valid',True)):
            confirmed.append(r['round'])
    full=bool(rounds[-1]==max_rounds)
    sustained=None
    if full and hits[-1]:
        last_miss=np.flatnonzero(~hits)
        sustained=int(rounds[last_miss[-1]+1 if len(last_miss) else 0])
    first=int(rounds[hits][0]) if hits.any() else None
    return dict(threshold=threshold, first_certified_round=first,
        first_confirmed_round=confirmed[0] if confirmed else None,
        sustained_certified_round=sustained,
        recrossed_after_first=bool((~hits[np.flatnonzero(hits)[0]:]).any()) if hits.any() else False,
        threshold_reached=bool(hits.any()), confirmed=bool(confirmed),
        final_status=statuses[-1], full_horizon_observed=full,
        rounds_executed=int(rounds[-1]), max_rounds=max_rounds,
        round_resolution='checkpoint grid; not every intermediate round was measured')


def trajectory(models, factual, budget, family, normalizer, config, on_checkpoint=None,
               oracle=None, global_evaluator=None):
    """Online diagnostics at average responses; optional safe early stopping.

    Diagnostics never update algorithm warm starts. eta always uses the fixed
    maximum horizon, even when the trajectory stops early.
    """
    c=config; origin=np.asarray(factual,float); constraints=rc.constraints_for(models,torch.tensor(origin,dtype=torch.float32))
    convex=family=='convex_logistic'
    local=oracle or (gap.ConvexLogitOracle(models,origin,budget,constraints,
        maxiter=c['convex_inner_steps'],normalizer=normalizer) if convex else
        mixed.LocalLogisticOracle(models,origin,budget,constraints,normalizer,c['mixed_inner_steps']))
    evaluator=global_evaluator
    n=len(models); eta=float(np.sqrt(np.log(n)/c['max_rounds']))
    logw=np.full(n,-np.log(n)); warm=origin.copy()
    points=[]; weights=[]; values=[]; oracle_infos=[]; rows=[]
    checkpoints=set(gap.checkpoints(c['max_rounds'],c['gap_every']))
    lower_sum=error_sum=0.; algorithm_seconds=diagnostic_seconds=0.
    gate=GapStop(c['primary_threshold'],c['confirmations'])
    stop_reason='max_rounds'; setup_seconds=0.
    if not convex and evaluator is None:
        tick=time.perf_counter()
        evaluator=mixed.NeuralLogisticBound(models,origin,budget,constraints,normalizer,
                                            c['dual_seconds'],int(c.get('solver_seed',42)))
        setup_seconds=time.perf_counter()-tick
    try:
        for t in range(1,c['max_rounds']+1):
            tick=time.perf_counter(); w=np.exp(logw)
            point,info=local.solve(w,warm); warm=point.copy(); loss=local.losses(point)
            if not np.isfinite(loss).all() or loss.min()<-1e-9 or loss.max()>1+1e-7:
                raise ValueError('Loss normalization failed')
            points.append(point.copy()); weights.append(w.copy()); values.append(loss.copy()); oracle_infos.append(info)
            if convex:
                lower_sum+=info['lower_bound'];error_sum+=info['absolute_gap']
            logw+=eta*loss; logw-=logsumexp(logw)
            algorithm_seconds+=time.perf_counter()-tick
            rr.event('rounds_study_oracle',round=t,weights=w,point=point,losses=loss,oracle=info)
            if t not in checkpoints: continue
            tick=time.perf_counter()
            xb=np.mean(points,axis=0); wb=np.mean(weights,axis=0)
            av=local.losses(xb); mix=np.mean(values,axis=0)
            candidates=[origin,xb,points[int(np.argmin(np.asarray(values)@wb))]]
            start=min(candidates,key=lambda x:float(wb@local.losses(x)))
            response,local_info=local.solve(wb,start)
            if convex:
                evaluation=local_info
                lower=max(lower_sum/t,evaluation['lower_bound'])
                upper=evaluation['objective']; consistent=True
            else:
                response,evaluation=evaluator.solve(wb,response,local)
                lower=evaluation['lower_bound'];upper=evaluation['upper_bound']
                consistent=bool(evaluation['bound_consistent'])
            upper=min(upper,float(wb@av),float(wb@local.losses(origin)),
                      float((np.asarray(values)@wb).min()))
            if lower>upper+1e-7: consistent=False;lower=0.
            interval=gap.gap_interval(float(av.max()),lower,upper)
            randomized=gap.gap_interval(float(mix.max()),lower,upper)
            row=dict(round=t,**interval,bounds_valid=consistent,
                mixed_gap_lower_bound=randomized['gap_lower_bound'],
                mixed_gap_upper_bound=randomized['gap_upper_bound'],
                raw_gap_lower_bound=normalizer*interval['gap_lower_bound'],
                raw_gap_upper_bound=normalizer*interval['gap_upper_bound'],
                normalized_threshold=c['primary_threshold'],
                equivalent_raw_threshold=normalizer*c['primary_threshold'],
                normalizer=normalizer,eta=eta,average_point=xb.copy(),average_weights=wb.copy(),
                dual_response=response,evaluation_oracle=evaluation,local_evaluation=local_info,
                primary_status=bound_status(interval['gap_lower_bound'],interval['gap_upper_bound'],c['primary_threshold'],consistent),
                local_optimizer_success=bool(info['optimizer_success']),
                point_rule='arithmetic_average_of_responses',diagnostic_setup_seconds=setup_seconds,
                algorithm_elapsed_seconds=algorithm_seconds)
            if convex:
                row['mean_oracle_error_upper_bound']=error_sum/t
                row['theory_upper_bound']=gap.regret_bound(n,eta,t,error_sum/t)
                if interval['gap_upper_bound']>row['theory_upper_bound']+1e-6:
                    raise ValueError('Convex gap exceeds regret bound')
            diagnostic_seconds+=time.perf_counter()-tick
            row.update(diagnostic_elapsed_seconds=diagnostic_seconds,
                       total_elapsed_seconds=setup_seconds+algorithm_seconds+diagnostic_seconds)
            rows.append(row)
            trace=dict(points=np.asarray(points),weights=np.asarray(weights),losses=np.asarray(values),
                average_point=xb,average_weights=wb,factual=origin,eta=eta,normalizer=normalizer,
                oracle_optimizer_success=np.asarray([r['optimizer_success'] for r in oracle_infos]),
                constraint_lower=constraints.lower,constraint_upper=constraints.upper,
                immutable_indices=np.asarray(constraints.immutable,dtype=int))
            if on_checkpoint: on_checkpoint(row,trace)
            ready=gate.update(interval['gap_lower_bound'],interval['gap_upper_bound'],consistent)
            if c['stop_on_gap'] and ready:
                stop_reason='confirmed_gap_threshold';break
    finally:
        if evaluator is not None: evaluator.close()
    # Independent feasibility and update checks include every generated response.
    from .recourse_duality import check_feasible
    check_feasible(trace['points'],origin,budget,constraints)
    check_feasible(trace['average_point'],origin,budget,constraints)
    reconstructed=np.full(n,-np.log(n))
    for w,v in zip(trace['weights'],trace['losses']):
        np.testing.assert_allclose(w,np.exp(reconstructed),atol=1e-10)
        reconstructed+=eta*v;reconstructed-=logsumexp(reconstructed)
    return rows,trace,stop_reason


def _hash(value):
    return hashlib.sha256(json.dumps(rr.jsonable(value),sort_keys=True).encode()).hexdigest()


def _block(root, dataset, seed, config):
    """Train once per family; reuse exact models and factuals for all settings."""
    c=config; X,y,action=rc.load_data_and_normalize(dataset)
    run=rr.RunStore(root/'blocks'/dataset/f'seed_{seed}','rounds_study',dict(c,dataset=dataset,seed=seed))
    run.manifest['notebook_version']='V7'
    token=rr._ACTIVE_RUN.set(run); st=rr._ACTIVE_SETTINGS.set(c)
    try:
        run.save_dataset(X,y);maximum=max(c['model_counts']);banks={};sets={}
        for fi,family in enumerate(c['families']):
            rr.set_scope(seed=seed,fold=fi,family=family)
            counts=(maximum,0,0) if family=='convex_logistic' else (maximum//3,)*3
            banks[family],_=rc.train_and_select_candidates_diverse(X,y,*counts,max_instances=0,
                seed=seed,dataset_name=dataset,actionable_indices=action,
                training_epochs=c['training_epochs'],training_lr=c['training_lr'])
            for n in c['model_counts']:
                ids=nested_indices(family,maximum,n)
                sets[family,n]=rc.ModelSet([banks[family][i] for i in ids],
                    context=copy.copy(banks[family].context))
                sets[family,n].metadata={'bank_indices':ids}
        reference=next(iter(banks.values())).context
        for bank in banks.values():
            np.testing.assert_array_equal(bank.context.test_ids,reference.test_ids)
            np.testing.assert_allclose(bank.context.X_test,reference.X_test)
        predictions=[dv.probabilities(m,m.context.X_test) for m in sets.values()]
        selected,common,masks=paired_positions(predictions,c['candidate_selection'],c['max_instances'],seed)
        ids=reference.test_ids[selected].tolist()
        audit=dict(dataset=dataset,seed=seed,rule=c['candidate_selection'],cohort=c['cohort'],
            requested_instances=c['max_instances'],selected_instances=len(selected),selected_ids=ids,
            eligible_in_all_sets=int(common.sum()),
            sets=[dict(family=f,n_models=n,eligible=int(mask.sum())) for (f,n),mask in zip(sets,masks)])
        rr.atomic_json(run.path/'paired_cohort.json',audit)
        run.save_arrays('paired_cohort.npz',test_ids=reference.test_ids,selected_positions=selected,
            eligibility_by_set=np.stack(masks),rejection_counts=np.stack([(p<.5).sum(axis=1) for p in predictions]))
        facts=[torch.tensor(reference.X_test[i],dtype=torch.float32) for i in selected]
        for fi,(family,bank) in enumerate(banks.items()):
            bank.context.candidate_ids=ids;bank.context.candidate_selection=audit
            bank.metadata['candidate_selection']=audit
            run.save_seed(bank,facts,fi,seed,
                nested_indices={str(n):nested_indices(family,maximum,n) for n in c['model_counts']})
        for models in sets.values():
            models.context.candidate_ids=ids;models.context.candidate_selection=audit
        rr.atomic_json(run.path/'set_diagnostics.json',[
            dict(family=f,n_models=n,bank_indices=m.metadata['bank_indices'],**dv.describe_set(m))
            for (f,n),m in sets.items()])
        rows=[];endpoints=[];normalizers=[]
        for fi,(family,bank) in enumerate(banks.items()):
            for pos,x0 in enumerate(facts):
                origin=x0.numpy().astype(float);constraints=rc.constraints_for(bank,x0)
                bmax=max(c['budgets'][dataset])
                normalizer=(gap.ConvexLogitOracle(bank,origin,bmax,constraints).normalizer
                    if family=='convex_logistic' else mixed.common_normalizer([bank],origin,bmax,constraints))
                # Allow tiny differences in independently recomputed float bounds.
                normalizer*=1+1e-12
                normalizers.append(dict(family=family,instance_id=ids[pos],normalizer=normalizer,
                    largest_budget=bmax,largest_set=maximum,raw_threshold=normalizer*c['primary_threshold']))
                for n in c['model_counts']:
                    models=sets[family,n]
                    for budget in c['budgets'][dataset]:
                        identity=dict(dataset=dataset,family=family,n_models=n,budget=budget,
                                      seed=seed,instance_id=ids[pos])
                        rr.set_scope(**identity)
                        directory=run.path/f'{family}/instance_{ids[pos]}/n_{n}/budget_{budget:g}'
                        directory.mkdir(parents=True,exist_ok=True)
                        print(f'{dataset} seed {seed} | {family} | {n} models | budget {budget:g} | individual {pos+1}/{len(facts)}',flush=True)
                        def checkpoint(row,trace):
                            rr._append(directory/'checkpoints.jsonl',dict(identity,**row))
                            rr._npz(directory/'trace.npz',trace)
                            rr.atomic_json(root/'progress.json',dict(status='running',**identity,
                                round=row['round'],gap_interval=[row['gap_lower_bound'],row['gap_upper_bound']],
                                gap_status=row['primary_status'],block=str(run.path)))
                            if row['round']==1 or row['round']%50==0 or row['round']==c['max_rounds']:
                                print(f'  round {row["round"]}/{c["max_rounds"]}: '
                                      f'gap [{row["gap_lower_bound"]:.5g}, {row["gap_upper_bound"]:.5g}] '
                                      f'({row["primary_status"]})',flush=True)
                        try:
                            measured,trace,reason=trajectory(models,origin,budget,family,normalizer,
                                dict(c,solver_seed=seed+pos),checkpoint)
                            rr._npz(directory/'trace.npz',trace)
                            rows.extend(dict(identity,**gap.scalar_row(r)) for r in measured)
                            for threshold in sorted(set(c['thresholds']+[c['primary_threshold']])):
                                end=dict(identity,**threshold_summary(measured,threshold,c['max_rounds'],c['confirmations']),
                                    status='completed',stop_reason=reason,normalizer=normalizer,
                                    final_gap_lower=measured[-1]['gap_lower_bound'],
                                    final_gap_upper=measured[-1]['gap_upper_bound'],
                                    final_mixed_gap_upper=measured[-1]['mixed_gap_upper_bound'],
                                    final_dual_uncertainty=measured[-1]['dual_solve_uncertainty'],
                                    algorithm_seconds=measured[-1]['algorithm_elapsed_seconds'],
                                    diagnostic_seconds=measured[-1]['diagnostic_elapsed_seconds'],
                                    diagnostic_setup_seconds=measured[-1]['diagnostic_setup_seconds'],
                                    local_optimizer_failure_fraction=float(1-trace['oracle_optimizer_success'].mean()),
                                    trace_file=str(directory/'trace.npz'))
                                endpoints.append(end);run.commit(end)
                        except Exception as exc:
                            error=dict(identity,status='error',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
                            rr.atomic_json(directory/'failure.json',error);run.commit(error);endpoints.append(error)
                            print('  Saved failed trajectory:',error['error'],flush=True)
                        pd.DataFrame(rows).to_csv(run.path/'curves.csv',index=False)
                        pd.DataFrame(endpoints).to_csv(run.path/'outcomes.csv',index=False)
        rr.atomic_json(run.path/'normalizers.json',normalizers)
        run.manifest.update(status='completed_with_errors' if any(r['status']=='error' for r in endpoints) else 'completed',
            selected_instances=len(facts))
        rr.atomic_json(run.path/'manifest.json',run.manifest)
        return run.path
    except BaseException as exc:
        run.manifest.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc))
        rr.atomic_json(run.path/'manifest.json',run.manifest);raise
    finally:
        rr._ACTIVE_SETTINGS.reset(st);rr._ACTIVE_RUN.reset(token)


def run_study(config=None, resume=True, results_root=None, show=True):
    c=copy.deepcopy(config or settings());validate(c)
    torch.set_num_threads(1)
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('recourse_*.py')}
    manifest=json.loads((rc.PROJECT_ROOT/'dataset/manifest.json').read_text())
    for entry in manifest['files']:
        if hashlib.sha256((rc.PROJECT_ROOT/'dataset'/entry['file']).read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError(f'Dataset checksum differs: {entry["file"]}')
    import importlib.metadata
    packages={}
    for package in ['numpy','torch','scipy','scikit-learn','gurobipy']:
        try: packages[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: packages[package]=None
    descriptor=dict(config=c,sources=source,dataset_manifest=manifest,packages=packages)
    fingerprint=_hash(descriptor)
    base=Path(results_root or rc.PROJECT_ROOT/'results')/'V7'/c['mode']/c['candidate_selection']/(
        'diversity_on' if c['diversity'] else 'diversity_off')/'rounds_to_gap'
    root=None
    if resume and base.exists():
        for path in sorted(base.iterdir(),reverse=True):
            if (path/'protocol.json').exists() and json.loads((path/'protocol.json').read_text()).get('fingerprint')==fingerprint:
                root=path;break
    if root is None:
        root=base/rr._stamp();root.mkdir(parents=True)
        rr.atomic_json(root/'protocol.json',dict(fingerprint=fingerprint,**descriptor))
    print('Rounds study results:',root,flush=True)
    with _exclusive(root):
        return _execute_study(root,c,resume,show)


@contextmanager
def _exclusive(root):
    lock=root/'execution.lock'
    try:
        with lock.open('x') as stream:
            stream.write('This study is in use. Do not run two launchers into it concurrently.\n')
    except FileExistsError:
        raise RuntimeError(f'Study has an execution lock: {lock}. If a kernel crashed, '
                           'use resume=False for a separate run; existing artifacts are preserved.')
    try: yield
    finally: lock.unlink(missing_ok=True)


def _execute_study(root,c,resume,show):
    blocks=json.loads((root/'blocks.json').read_text()) if (root/'blocks.json').exists() else {}
    errors=[]
    # Existing V7 kernel variables and standard campaign settings are preserved.
    if 'mixed_logistic' in c['families']:
        from .recourse_runtime import solver_check
        solver_check(require_full_size=True)
    try:
        with threadpool_limits(limits=1):
            for ds in c['datasets']:
                for seed in c['seeds']:
                    key=f'{ds}:{seed}'
                    if resume and key in blocks and (Path(blocks[key])/'manifest.json').is_file():
                        saved=json.loads((Path(blocks[key])/'manifest.json').read_text())
                        if saved['status']=='completed':
                            print('Reusing completed dataset/seed:',key,flush=True);continue
                    try:
                        blocks[key]=str(_block(root,ds,seed,c))
                        rr.atomic_json(root/'blocks.json',blocks)
                    except Exception as exc:
                        errors.append(dict(dataset=ds,seed=seed,error=repr(exc),traceback=traceback.format_exc()))
                        rr.atomic_json(root/'failures.json',errors)
                        print('Dataset/seed failed; partial artifacts retained:',key,repr(exc),flush=True)
    except KeyboardInterrupt:
        rr.atomic_json(root/'progress.json',dict(status='interrupted',completed_blocks=len(blocks),results=str(root)))
        raise
    result=analyze(root,show=show)
    rr.atomic_json(root/'progress.json',dict(status='completed_with_errors' if errors or result['failed_trajectories'] else 'completed',
        completed_blocks=len(blocks),requested_blocks=len(c['datasets'])*len(c['seeds']),results=str(root)))
    return result


def _read_frames(paths, name):
    frames=[]
    for path in paths:
        p=Path(path)/name
        if p.is_file() and p.stat().st_size>1:
            frames.append(pd.read_csv(p))
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def analyze(root, show=False):
    """Read saved results only; never train a model or rerun a recourse solver."""
    root=Path(root);c=json.loads((root/'protocol.json').read_text())['config']
    blocks=json.loads((root/'blocks.json').read_text()) if (root/'blocks.json').exists() else {}
    curves=_read_frames(blocks.values(),'curves.csv');outcomes=_read_frames(blocks.values(),'outcomes.csv')
    cohorts=[json.loads((Path(p)/'paired_cohort.json').read_text()) for p in blocks.values()]
    pd.DataFrame(cohorts).to_csv(root/'cohorts.csv',index=False)
    failed=int(outcomes.status.eq('error').sum()) if not outcomes.empty else 0
    summary=pd.DataFrame();recommendations=[];attainment=[]
    if not curves.empty:
        curves.to_csv(root/'all_checkpoints.csv',index=False)
        outcomes.to_csv(root/'all_outcomes.csv',index=False)
        good=outcomes[outcomes.status.eq('completed')].copy()
        good['final_below_pct']=good.final_status.eq('certified_below_threshold')*100.
        good['final_above_pct']=good.final_status.eq('certified_above_threshold')*100.
        good['unresolved_pct']=good.final_status.eq('unresolved_bound_interval')*100.
        good['invalid_pct']=good.final_status.eq('invalid_bounds')*100.
        good['ever_certified_pct']=good.threshold_reached.astype(float)*100
        good['recrossed_pct']=good.recrossed_after_first.astype(float)*100
        metrics=['final_below_pct','final_above_pct','unresolved_pct','invalid_pct','ever_certified_pct',
                 'recrossed_pct','final_gap_lower','final_gap_upper','final_dual_uncertainty','rounds_executed',
                 'algorithm_seconds','diagnostic_seconds','diagnostic_setup_seconds','final_mixed_gap_upper',
                 'local_optimizer_failure_fraction']
        seed=good.groupby(GROUP+['threshold','seed'])[metrics].mean().reset_index()
        summary=stats.summarize_seed_frame(seed,GROUP+['threshold'],metrics)
        seed.to_csv(root/'seed_summary.csv',index=False);summary.to_csv(root/'summary.csv',index=False)
        for key,g in good.groupby(GROUP+['threshold']):
            identity=dict(zip(GROUP+['threshold'],key))
            grid=gap.checkpoints(c['max_rounds'],c['gap_every'])
            for t in grid:
                first=g.assign(value=(g.first_certified_round.le(t)&g.first_certified_round.notna()).astype(float))
                fraction=stats.mean_ci(first.groupby('seed').value.mean(),(0,1))
                attainment.append(dict(identity,round=t,**fraction))
            stable=[]
            failures=outcomes[outcomes.status.eq('error')]
            for column,value in identity.items():
                if column!='threshold': failures=failures[failures[column].eq(value)]
            enough_data=not len(failures) and g.seed.nunique()==len(c['seeds'])
            complete=bool(g.full_horizon_observed.all()) and enough_data
            if complete:
                for t in grid:
                    v=(g.sustained_certified_round.le(t)&g.sustained_certified_round.notna()).astype(float)
                    fraction=v.groupby(g.seed).mean().mean()
                    if fraction>=c['recommendation_fraction']:stable.append(t)
            recommendations.append(dict(identity,suggested_cap=stable[0] if stable else None,
                target_fraction=c['recommendation_fraction'],actual_seeds=g.seed.nunique(),actual_individuals=len(g),
                failed_trajectories=len(failures),
                status='exploratory_observed_cap' if stable and g.seed.nunique()<5 else
                       'observed_cap' if stable else 'incomplete_trajectories' if not enough_data else
                       'early_stop_prevents_stability_audit' if not complete else
                       'target_not_established_within_horizon',
                note='Observed sampled checkpoints only; one fixed-200 learning-rate schedule; failures excluded and reported separately.'))
        recommendation=pd.DataFrame(recommendations)
        reach=pd.DataFrame(attainment)
        recommendation.to_csv(root/'recommended_round_caps.csv',index=False)
        reach.to_csv(root/'threshold_attainment.csv',index=False)
        for ds in good.dataset.unique():
            reporting.save_table(good[good.dataset.eq(ds)],root,ds,'rounds_instance_outcomes')
            reporting.save_table(summary[summary.dataset.eq(ds)],root,ds,'rounds_seed_summary')
            reporting.save_table(recommendation[recommendation.dataset.eq(ds)],root,ds,'rounds_recommendations')
            _plots(curves[curves.dataset.eq(ds)],reach[reach.dataset.eq(ds)],root,ds,c,show)
    else:
        recommendation=pd.DataFrame()
    note=(f'Primary normalized threshold: {c["primary_threshold"]}; fixed horizon: {c["max_rounds"]}.\n'
          f'Completed dataset/seed artifacts: {len(blocks)}; failed trajectories: {failed}.\n'
          'No-hit rounds are missing/censored, not assigned the horizon. Final above-threshold and unresolved intervals are distinct.\n'
          'A small gap upper bound supports convergence at that checkpoint; a large upper bound alone does not prove failure.\n'
          'Majority eligibility is intersected across all model counts and families before any recourse search.\n'
          'The shared normalizer is constant across counts/budgets within a factual/family; raw cross-entropy gaps are also saved.\n'
          'This more restrictive common cohort can differ from ordinary V7 cohorts; conclusions are conditional on it.\n'
          'Recommendations require a full horizon and the chosen fraction to stay certified at every remaining sampled checkpoint.\n'
          'They are exploratory on small runs and are not convergence guarantees for new individuals.\n'
          'Changing T also changes a horizon-tuned learning rate. Recommended caps apply to prefixes of the fixed-200 schedule.\n'
          'The tested loss is cross-entropy for both families; caps do not directly establish convergence for the squared-loss comparison experiments.\n'
          'Confidence intervals use seed means. One-seed quick runs do not have confidence intervals.\n')
    (root/'report.txt').write_text(note)
    return dict(directory=str(root),summary=summary,recommendations=recommendation,
                curves=curves,outcomes=outcomes,failed_trajectories=failed)


def _plots(curves, reach, root, dataset, config, show):
    import matplotlib.pyplot as plt
    for family,data in curves.groupby('family'):
        budgets=sorted(data.budget.unique())
        for prefix,label,name in [('', 'Normalized duality gap','normalized'),
                                  ('mixed_', 'Uniform-response duality gap','uniform_response'),
                                  ('raw_','Raw cross-entropy duality gap','raw')]:
            fig,axes=plt.subplots(1,len(budgets),figsize=(5*len(budgets),4),squeeze=False)
            for ax,budget in zip(axes[0],budgets):
                for n,g in data[data.budget.eq(budget)].groupby('n_models'):
                    seed=g.groupby(['seed','round'])[[prefix+'gap_lower_bound',prefix+'gap_upper_bound']].mean().reset_index()
                    table=stats.summarize_seed_frame(seed,['round'],[prefix+'gap_lower_bound',prefix+'gap_upper_bound'])
                    hi=table[table.metric.eq(prefix+'gap_upper_bound')].sort_values('round')
                    lo=table[table.metric.eq(prefix+'gap_lower_bound')].sort_values('round')
                    line,=ax.plot(hi['round'],hi['mean'],label=f'{n} models, upper bound')
                    ax.plot(lo['round'],lo['mean'],'--',color=line.get_color())
                    ax.fill_between(hi['round'].to_numpy(),hi.ci_low.to_numpy(),hi.ci_high.to_numpy(),color=line.get_color(),alpha=.12)
                if prefix!='raw_': ax.axhline(config['primary_threshold'],color='black',ls=':',label='Gap threshold')
                ax.set(xlabel='ORPM round',ylabel=label,title=f'Budget {budget:g}',ylim=(0,None))
                ax.grid(alpha=.2);ax.legend(fontsize=7)
            fig.suptitle(f'{dataset} | {family.replace("_"," ")} | solid: upper; dashed: lower\n'
                         'Shading: pointwise 95% intervals over seed means; absent for one seed',fontsize=10)
            fig.tight_layout();reporting.save_figure(fig,root,dataset,f'rounds_{family}_{name}_gaps',show)
        part=reach[reach.family.eq(family)&np.isclose(reach.threshold,config['primary_threshold'])]
        fig,axes=plt.subplots(1,len(budgets),figsize=(5*len(budgets),4),squeeze=False)
        for ax,budget in zip(axes[0],budgets):
            for n,g in part[part.budget.eq(budget)].groupby('n_models'):
                g=g.sort_values('round');line,=ax.plot(g['round'],100*g['mean'],label=f'{n} models')
                ax.fill_between(g['round'].to_numpy(),100*g.ci_low.to_numpy(),100*g.ci_high.to_numpy(),color=line.get_color(),alpha=.12)
            ax.set(xlabel='ORPM round',ylabel='Individuals ever certified below threshold (%)',
                   title=f'Budget {budget:g}',ylim=(0,103));ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle(f'{dataset} | {family.replace("_"," ")} | gap upper bound ≤ {config["primary_threshold"]}\n'
                     'First attainment; later recrossings are reported separately',fontsize=10)
        fig.tight_layout();reporting.save_figure(fig,root,dataset,f'rounds_{family}_attainment',show)
