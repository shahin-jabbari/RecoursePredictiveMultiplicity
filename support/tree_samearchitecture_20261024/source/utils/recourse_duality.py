"""V7 duality gaps at averaged recourses and uniform response distributions.

Optimization lower/upper bounds and statistical seed intervals are separate.
All diagnostics use average pre-update weights; only the affine-logit family
has an averaged-feature convergence theorem.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
from . import recourse_core as rc, recourse_results as rr, recourse_campaign as campaign
from . import recourse_convergence as gap, recourse_statistics as rs, recourse_reporting as rp
from . import recourse_gmc_mixed as mixed, recourse_trees as rt, recourse_gmc_treediversity as td

def check_feasible(points, origin, budget, constraints):
    points = np.atleast_2d(np.asarray(points, float))
    assert np.isfinite(points).all()
    assert np.max(np.linalg.norm(points-origin, axis=1)) <= budget+1e-5
    assert np.all(points >= constraints.lower-1e-5)
    assert np.all(points <= constraints.upper+1e-5)
    if len(constraints.immutable):
        np.testing.assert_allclose(points[:, constraints.immutable],
            np.broadcast_to(origin[constraints.immutable], (len(points), len(constraints.immutable))),
            atol=1e-6, rtol=0)

def average_diagnostics(rows, trace, loss_function, constraints, origin, budget, family):
    """Reuse valid dual bounds; evaluate the arithmetic mean in feature space.

The gap of a distribution over responses remains a separate diagnostic.
For trees, prediction follows the saved float32 tree inference convention.
"""
    points = np.asarray(trace['points'], float)
    weights = np.asarray(trace['weights'], float)
    losses = np.asarray(trace['losses'], float)
    check_feasible(points, origin, budget, constraints)
    assert weights.shape == losses.shape
    assert np.isfinite(losses).all() and losses.min() >= 0 and losses.max() <= 1+1e-8
    np.testing.assert_allclose(weights.sum(axis=1), 1., atol=1e-12, rtol=0)
    logw = np.full(weights.shape[1], -np.log(weights.shape[1]))
    for w, v, p in zip(weights, losses, points):
        np.testing.assert_allclose(w, np.exp(logw), atol=1e-11, rtol=1e-10)
        np.testing.assert_allclose(v, loss_function(p), atol=1e-10, rtol=1e-9)
        logw += float(trace['eta'])*v
        logw -= logsumexp(logw)
    averaged = np.cumsum(points, axis=0)/np.arange(1, len(points)+1)[:, None]
    check_feasible(averaged, origin, budget, constraints)
    trace['average_points'] = averaged
    trace['average_point'] = averaged[-1]
    trace['average_weights'] = weights.mean(axis=0)
    for row in rows:
        t = row['round']
        point = averaged[t-1]
        wb = weights[:t].mean(axis=0)
        np.testing.assert_allclose(row['average_weights'], wb, atol=1e-12, rtol=0)
        if family != 'convex_logistic':
            row.update(best_gap_lower_bound=row['gap_lower_bound'],
                       best_gap_upper_bound=row['gap_upper_bound'], best_primal=row['primal'])
        v = loss_function(point)
        lower = row['dual_lower_bound']
        upper = min(row['dual_upper_bound'], float(wb@v))
        # A verified feasible average can improve the dual upper bound. Never
        # retain a contradictory solver lower bound or present it as a zero gap.
        if lower > upper+1e-7:
            row['average_bound_inconsistency'] = float(lower-upper)
            lower = 0.
            row['certificate_status'] = 'bound_inconsistency_fallback'
        else:
            row['average_bound_inconsistency'] = 0.
        row.update(gap.gap_interval(float(v.max()), lower, upper))
        mixture = gap.gap_interval(float(losses[:t].mean(axis=0).max()), lower, upper)
        row.update(average_point=point, average_weights=wb,
            mixed_primal=mixture['primal'], mixed_gap_lower_bound=mixture['gap_lower_bound'],
            mixed_gap_upper_bound=mixture['gap_upper_bound'],
            point_rule='arithmetic_average_of_oracle_responses',
            gap_kind=family+'_averaged_point', verified=True)
        if family == 'convex_logistic':
            row['mixed_theory_upper_bound'] = row['theory_upper_bound']
            assert row['gap_upper_bound'] <= row['theory_upper_bound']+1e-6
        normalizer = float(trace['normalizer'])
        row.update(normalizer=normalizer, raw_gap_lower_bound=normalizer*row['gap_lower_bound'],
            raw_gap_upper_bound=normalizer*row['gap_upper_bound'])
    return averaged[-1], rows, trace


def trajectory(models, x0, budget, family, T, every=5, seed=42,
               tree_options=None, artifact_dir=None):
    origin=x0.numpy().astype(float)
    c=rc.constraints_for(models,x0)
    if family=='convex_logistic':
        evaluator=gap.ConvexLogitOracle(models,origin,budget,c)
        _,rows,trace=gap.convex_trajectory(models,origin,budget,T=T,every=every,
            constraints=c,maxiter=200,normalizer=evaluator.normalizer)
        loss_function=evaluator.losses
    elif family=='mixed_logistic':
        normalizer=mixed.common_normalizer([models],origin,budget,c)
        _,rows,trace=mixed.trajectory(models,origin,budget,normalizer,T=T,every=every,
            maxiter=50,dual_seconds=1.,seed=seed)
        evaluator=mixed.LocalLogisticOracle(models,origin,budget,c,normalizer,50)
        loss_function=evaluator.losses
    elif family=='decision_tree':
        options=tree_options or campaign.tree_options('quick')
        best,rounds=rt.tree_orpm(models,x0,budget,T=T,time_limit=options['time_limit'],
            abs_gap=1e-5,constraints=c,artifact_dir=artifact_dir,seed=seed)
        rows=gap.tree_gap_trajectory(models,x0,budget,rounds,c,every=every,
            time_limit=options['time_limit'],abs_gap=1e-6,seed=seed,
            artifact_dir=Path(artifact_dir)/'gap' if artifact_dir else None)
        trace=dict(best.recourse_trace,losses=np.asarray([r['losses'] for r in rounds]),
            normalizer=1.,eta=float(np.sqrt(np.log(len(models))/T)),
            oracle_lower_bounds=np.asarray([r['oracle']['lower_bound'] or 0. for r in rounds]),
            oracle_statuses=np.asarray([r['oracle']['solver_status'] for r in rounds]))
        loss_function=lambda x:(1-rt.probabilities(models,torch.tensor(x,dtype=torch.float32)))**2
    else:
        raise ValueError(f'Unknown duality-gap family: {family}')
    return average_diagnostics(rows,trace,loss_function,c,origin,budget,family)


@rr.experiment(settings_factory=lambda mode,seed:campaign.profile_settings(mode,seed,'duality_gap'))
def run_duality_experiment(dataset_name,dataset,family='convex_logistic',counts=(3,0,0),
        recourse_budget=2.,actionable_indices=None,folds=2,max_instances=3,seed=42,T=20,
        gap_every=5,tree_options=None):
    run=rr.current_run();settings=rr._ACTIVE_SETTINGS.get()
    records=[];curves=[];X,y=dataset
    rr.record_iteration_limits(T,T)
    rr.atomic_json(run.path/'duality_protocol.json',dict(family=family,
        candidate_selection=settings['candidate_selection'],counts=counts,
        T=T,gap_every=gap_every,budget=recourse_budget,
        deterministic_output='arithmetic average of oracle responses',
        randomized_output='uniform distribution over responses; average per-model losses',
        adversary='average pre-update weights',
        convex_guarantee=family=='convex_logistic',
        global_bounds='convex supporting plane for affine logits; Gurobi bounds for trees and neural loss minorants',
        normalization='set-specific conservative loss bound per factual; compare raw cross-entropy gaps across different model sets',
        feature_domain='continuous box/L2/immutable relaxation',
        statistics='equal-weight seed means; pointwise 95% Student-t confidence intervals'))
    for fold in range(folds):
        current_seed=seed+fold
        rr.set_scope(seed=current_seed,fold=fold)
        print(f'{dataset_name}/{family}: preparing seed {current_seed} ({fold+1}/{folds})',flush=True)
        if family=='decision_tree':
            options=tree_options or campaign.tree_options(settings['mode'])
            models,_,facts=rt.train_tree_pool(X,y,dataset_name,current_seed,
                family='decision_tree',num_models=options['num_models'],num_unseen=0,
                max_instances=max_instances,max_depth=options['max_depth'],
                min_samples_leaf=options['min_samples_leaf'],actionable_indices=actionable_indices)
            models.metadata.update(diversity=False,**td.describe(models))
            if settings.get('diversity'):
                reference=models
                reference_metadata=dict(reference.metadata)
                models=td.matched_set(reference,
                    settings['diversity_options']['candidate_pool_size'],
                    settings['diversity_options']['accuracy_caliper'],options['max_depth'])
                # Reselect AFTER masking. Majority must query the final competing set.
                facts=rc.select_candidates(models,max_instances,reference_model=reference[0],order='test')
                models.metadata['reference_set_diagnostics']=reference_metadata
            td.save_set(run,models,facts,current_seed,fold)
        else:
            models,facts=rc.train_and_select_candidates_diverse(X,y,*counts,max_instances=max_instances,
                seed=current_seed,actionable_indices=actionable_indices,dataset_name=dataset_name)
            run.save_seed(models,facts,fold,current_seed)
        if not facts:
            run.commit(dict(dataset=dataset_name,family=family,seed=current_seed,fold=fold,
                method='ORPM',status='no_eligible_factuals',candidate_selection=settings['candidate_selection']))
        for position,x0 in enumerate(facts):
            before=time.perf_counter();iid=models.context.candidate_ids[position]
            identity=dict(dataset=dataset_name,family=family,seed=current_seed,fold=fold,
                instance_id=iid,method='ORPM',budget=recourse_budget,actual_models=len(models),
                candidate_selection=settings['candidate_selection'])
            rr.set_scope(seed=current_seed,fold=fold,instance_id=iid,method='ORPM')
            directory=f'seed_{current_seed}_fold_{fold}/instance_{iid}'
            point,rows,trace=trajectory(models,x0,recourse_budget,family,T,every=gap_every,
                seed=current_seed+position,tree_options=tree_options,artifact_dir=run.path/directory/'solver')
            c=rc.constraints_for(models,x0)
            trace.update(factual=x0.numpy(),constraint_lower=c.lower,constraint_upper=c.upper,
                immutable_indices=np.asarray(c.immutable,dtype=int))
            run.save_arrays(directory+'/duality_trace.npz',**trace)
            for row in rows:
                rr._append(run.path/'duality_checkpoints.jsonl',rr.jsonable(dict(identity,**row)))
                curves.append(dict(identity,**gap.scalar_row(row)))
            p=torch.tensor(point,dtype=torch.float32)
            assert rc.feasible(p,x0,recourse_budget,c)
            record=dict(identity,**gap.scalar_row(rows[-1]),status='completed',point=point,
                factual=x0.numpy(),valid=rc.valid_recourse(p,models,x0,recourse_budget,c),
                feasible=True,cost=float(np.linalg.norm(point-x0.numpy())),max_loss=rows[-1]['primal'],
                elapsed_seconds=time.perf_counter()-before,trajectory_file=directory+'/duality_trace.npz')
            records.append(record);run.commit(record)
            pd.DataFrame(curves).to_csv(run.path/'duality_checkpoints.csv',index=False)
            print(f'  {position+1}/{len(facts)} factuals: average-point gap '
                  f'[{record["gap_lower_bound"]:.5g}, {record["gap_upper_bound"]:.5g}]',flush=True)
    actual=tree_options['num_models'] if family=='decision_tree' and tree_options else sum(counts)
    result=dict(Dataset=dataset_name,Family=family,Method='ORPM',Budget=recourse_budget,
        Config=f'{actual} models',**{'Actual Models':actual,'Requested Seeds':folds,
        'Completed Seeds':len({r['seed'] for r in records}),'Evaluated Instances':len(records),
        'ORPM T':T,'Gap Definition':'averaged point and average pre-update weights',
        'Run Directory':str(run.path)})
    for label,metric,scale,bounds in [('Gap Lower Bound','gap_lower_bound',1.,(0.,np.inf)),
        ('Gap Upper Bound','gap_upper_bound',1.,(0.,np.inf)),
        ('Randomized Gap Lower Bound','mixed_gap_lower_bound',1.,(0.,np.inf)),
        ('Randomized Gap Upper Bound','mixed_gap_upper_bound',1.,(0.,np.inf)),
        ('Raw Gap Upper Bound','raw_gap_upper_bound',1.,(0.,np.inf)),
        ('Validity','valid',100.,(0.,100.)),('Cost','cost',1.,(0.,np.inf))]:
        scaled=[dict(r,**{metric:float(r[metric])*scale}) for r in records]
        result.update(rs.columns(label,rs.metric_summary(scaled,metric,bounds=bounds)))
    return pd.DataFrame([result])


def plot_duality(raw,output_dir,show=False):
    """Separate averaged-feature and randomized-strategy gaps by dataset/family."""
    import matplotlib.pyplot as plt
    if raw.empty:return []
    outputs=[]
    metrics=['gap_lower_bound','gap_upper_bound','mixed_gap_lower_bound','mixed_gap_upper_bound',
        'raw_gap_lower_bound','raw_gap_upper_bound','dual_solve_uncertainty','primal','mixed_primal']
    for (ds,family,selection,budget),data in raw.groupby(['dataset','family','candidate_selection','budget']):
        seed=data.groupby(['seed','round'])[metrics].mean().reset_index()
        stats=rs.summarize_seed_frame(seed,['round'],metrics,{m:(0.,np.inf) for m in metrics})
        name=f'duality_gap_{family}_{selection}_budget_{budget:g}'
        rp.save_table(data,output_dir,ds,name+'_instances')
        rp.save_table(seed,output_dir,ds,name+'_seed_means')
        rp.save_table(stats,output_dir,ds,name+'_summary')
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for ax,prefix,title in zip(axes,['','mixed_'],['Average feature vector','Uniform random oracle response']):
            for bound,style in [('upper','-'),('lower','--')]:
                part=stats[stats.metric.eq(prefix+f'gap_{bound}_bound')].sort_values('round')
                line,=ax.plot(part['round'],part['mean'],style,label=f'Gap {bound} bound')
                ax.fill_between(part['round'].to_numpy(),part.ci_low.to_numpy(),part.ci_high.to_numpy(),
                    color=line.get_color(),alpha=.13)
            ax.set(xlabel='ORPM round',ylabel='Duality gap (squared loss)' if family=='decision_tree'
                else 'Duality gap (normalized cross entropy)',title=title,ylim=(0,None))
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle(f'{rp.dataset_label(ds)} | {family.replace("_"," ")} | {selection} selection | budget {budget:g}\n'
            'Lines: optimization bounds. Shading: pointwise 95% confidence intervals across seed means.',fontsize=10)
        fig.tight_layout();outputs.extend(rp.save_figure(fig,output_dir,ds,name,show))
    return outputs
