"""Paired tree/forest convergence, comparison, and cost-validity experiments."""
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import recourse_core as rc
from . import recourse_results as rr
from . import recourse_statistics as rs
from . import recourse_reporting as rp
from . import recourse_trees as rt
from . import recourse_convergence as rg

DATASET_BUDGETS = {
    'Synthetic':[.5,1.,2.,5.,10.], 'COMPAS':[.5,1.,2.,5.,10.],
    'German':[.5,1.,2.,5.,10.], 'Polish':[.5,1.,2.,5.,10.],
    'Folktables':[1.,2.,5.,10.,20.], 'Give_Me_Some_Credit':[2.,5.,10.,20.,50.],
}
DATASET_LABELS = {'Synthetic':'Synthetic', 'COMPAS':'COMPAS', 'German':'German Credit',
    'Polish':'Polish Bankruptcy', 'Folktables':'FolkTables ACSIncome',
    'Give_Me_Some_Credit':'Give Me Some Credit'}


def tree_experiment_settings(Quick=True, seed=42):
    from .recourse_campaign import profile_settings
    return profile_settings(Quick,seed,'tree_comparison')


def default_tree_options(Quick=True):
    from .recourse_campaign import tree_options
    return tree_options(Quick)


def tree_workload_plan(kind, model_specs, datasets, settings, options, reference_budgets):
    """Count requested ORPM solves; capped solver time is not a wall-clock estimate."""
    rows = []
    for dataset in datasets:
        name = rc.canonical_name(dataset)
        from .recourse_campaign import budget_grid
        budgets = budget_grid(name,settings['mode'],tree=True) if kind == 'tradeoff' else [reference_budgets[name]]
        for spec in model_specs:
            rounds = settings['CONVERGENCE_T'] if kind == 'convergence' else settings['TREE_ORPM_T']
            calls = settings['folds'] * settings['max_instances'] * len(budgets) * rounds
            methods = options.get('methods')
            if methods is not None and 'ORPM-tree' not in methods:
                calls = 0
            rows.append(dict(dataset=name, **spec, budgets=budgets,
                requested_seeds=settings['folds'], max_instances_per_seed=settings['max_instances'],
                orpm_rounds=rounds, orpm_oracle_calls=calls,
                seconds_per_oracle_solve=options.get('time_limit',5.),
                orpm_capped_solver_hours=calls*options.get('time_limit',5.)/3600))
    return rows


def _single_solve(models,x0,budget,c,options,artifact_dir,objective,threshold=.5):
    solver=rt.TreeOracle(models,x0,budget,c,artifact_dir=artifact_dir,**options)
    try: point=solver.solve(objective=objective,threshold=threshold)
    finally: solver.close()
    point.recourse_trace={'points':np.asarray([x0.numpy(),point.numpy()])}
    return point


def _summary(records,methods,settings,dataset_name,family,aggregation,budgets):
    completed=[r for r in records if r.get('instance_id') is not None]
    rows=[]
    for budget in budgets:
        for method in methods:
            subset=[dict(r) for r in completed if r['budget']==budget and r['method']==method]
            for row in subset:
                row['successful_cost']=row['cost'] if row['valid'] else np.nan
            base={'Dataset':dataset_name,'Family':family,'Aggregation':aggregation,
                'Budget':budget,'Method':method,'Mode':settings['mode'],'T':settings['T'],
                'Tree Profile':settings.get('tree_profile','explicit_fixture'),
                'Requested Seeds':settings['folds'],'Requested Instances Per Seed':settings['max_instances'],
                'Evaluated Instances':len(subset),'Completed Seeds':len({r['seed'] for r in subset})}
            for prefix,metric,scale,bounds,unit in [
                ('Validity','valid',100.,(0.,100.),''),
                ('Unseen Validity','unseen_all_valid',100.,(0.,100.),''),
                ('Unseen Fraction','unseen_fraction_valid',100.,(0.,100.),''),
                ('Cost','cost',1.,(0.,np.inf),''),
                ('Successful Cost','successful_cost',1.,(0.,np.inf),''),
                ('Time','elapsed_seconds',1.,(0.,np.inf),'s'),
                ('Worst Loss','max_loss',1.,(0.,1.),'')]:
                base.update(rs.columns(prefix,rs.metric_summary(subset,metric,scale,bounds),unit))
            base['Mean Cost']=base['Cost Mean']
            base['Mean Successful Cost']=base['Successful Cost Mean']
            base['Time Instance SD (s)']=np.std([r['elapsed_seconds'] for r in subset],ddof=1) if len(subset)>1 else np.nan
            base['Failure Count']=sum(r['status'] in {'solver_error','infeasible','no_incumbent',
                'incumbent_failed_validation','no_stable_candidate_found'} for r in subset)
            base['Duality Evaluation Failures']=sum(bool(r.get('duality_diagnostic_error')) for r in subset)
            for label,metric in [('Duality Gap Lower Bound','duality_gap_lower_bound'),
                                 ('Duality Gap Upper Bound','duality_gap_upper_bound'),
                                 ('Duality Diagnostic Time','duality_diagnostic_seconds')]:
                base.update(rs.columns(label,rs.metric_summary(subset,metric,bounds=(0.,np.inf))))
            rows.append(base)
    return pd.DataFrame(rows),rs.per_seed_table(completed)


@rr.experiment(settings_factory=tree_experiment_settings)
def run_tree_experiment(dataset_name,dataset_data,family='decision_tree',kind='comparison',
        budgets=(2.0,),num_models=5,num_unseen=5,aggregation='probability',max_depth=4,
        n_estimators=25,min_samples_leaf=5,epsilon=.05,pool_multiplier=4,seed=42,
        folds=2,T=50,max_instances=20,time_limit=5.,abs_gap=1e-5,methods=None,
        include_direct_reference=True,temperature=.1,forest_risk=.1,forest_confidence=.95,
        robx_options=None,max_pool_multiplier=None,duality_gap_every=5,duality_time_limit=None):
    if kind not in {'convergence','comparison','tradeoff'}:
        raise ValueError('Unknown experiment kind')
    budgets=sorted(set(float(b) for b in budgets))
    if not budgets or any(not np.isfinite(b) or b<0 for b in budgets):
        raise ValueError('Provide finite, nonnegative budgets')
    if methods is None:
        methods=['ORPM-tree'] if kind=='convergence' else [
            'ORPM-tree','ADV-surrogate','ROAR-surrogate','RobX-constrained','OCEAN-style']
        if include_direct_reference:methods+=['Joint-minimax']
        if kind!='convergence' and family=='random_forest' and aggregation=='hard_vote':
            methods+=['RobustCF4RF-DirectSAA','RobustCF4RF-RobustSAA']
    methods=list(dict.fromkeys(methods))
    supported={'ORPM-tree','ADV-surrogate','ROAR-surrogate','RobX-constrained','OCEAN-style',
               'Joint-minimax','RobustCF4RF-DirectSAA','RobustCF4RF-RobustSAA'}
    if not methods or set(methods)-supported:raise ValueError('Unknown or empty method selection')
    if any(m.startswith('RobustCF4RF') for m in methods) and (family!='random_forest' or aggregation!='hard_vote'):
        raise ValueError('RobustCF4RF requires a matched hard-voting random-forest experiment.')
    run=rr.current_run(); settings=rr._ACTIVE_SETTINGS.get()
    orpm_T=rr.orpm_rounds(T,kind=kind,family=family)
    rr.record_iteration_limits(orpm_T,T)
    X,y=dataset_data[:2]
    actionable=dataset_data[2] if len(dataset_data)>2 else None
    records=[]; convergence=[]; gap_records=[]
    rr.atomic_json(run.path/'tree_protocol.json',{
        'model_selection':'train only; admit using validation; held-out factuals from deployed-model rejection',
        'unseen_models':'separate model seeds; never queried by recourse optimizers',
        'budget_pairing':'one model set and factual set per seed reused across every budget/method',
        'training_scope':'same training sample; randomized tree/forest fitting',
        'feature_domain':'continuous actionability relaxation represented in float32',
        'return_rule':'ORPM best feasible candidate; no averaging or convex theorem',
        'duality_gap':'max loss at best point minus minimum weighted loss at average pre-update weights',
        'duality_checkpoints':rg.checkpoints(orpm_T,duality_gap_every) if kind=='convergence' else [],
        'duality_time_limit':time_limit if duality_time_limit is None else duality_time_limit,
        'duality_evaluation':'separate post-hoc solver; does not influence the ORPM trajectory',
        'mixed_gap':'max mean per-model iterate loss minus the same dual objective; not averaged features',
        'baselines':'OCEAN-style and RobustCF4RF are formulation reimplementations; ADV/ROAR/RobX adaptations labeled',
        'score_threshold':.5,'aggregation':aggregation,'methods':methods,
        'forest_confidence_semantics':'central normal coverage for positive Agresti-Coull inflation',
        'citations':{
            'OCEAN':'https://proceedings.mlr.press/v139/parmentier21a.html',
            'RobX':'https://proceedings.mlr.press/v162/dutta22a.html',
            'RobustCF4RF':'https://arxiv.org/abs/2205.14116',
            'FOCUS':'https://ojs.aaai.org/index.php/AAAI/article/view/20468'}})
    for fold in range(folds):
        current_seed=seed+fold
        rr.set_scope(fold=fold,seed=current_seed)
        print(f'{dataset_name}/{family}/{aggregation}: training seed {current_seed} '
              f'({fold+1}/{folds}), ORPM cap {orpm_T}, up to {max_instances} factuals', flush=True)
        try:
            models,unseen,candidates=rt.train_tree_pool(X,y,dataset_name,current_seed,family,
                num_models,num_unseen,max_instances,aggregation,max_depth,n_estimators,
                min_samples_leaf,epsilon,pool_multiplier,actionable,max_pool_multiplier)
        except Exception as exc:
            row={'dataset':dataset_name,'family':family,'aggregation':aggregation,'seed':current_seed,
                 'fold':fold,'status':'model_generation_failed','error':f'{type(exc).__name__}: {exc}'}
            run.commit(row);records.append(row)
            print(f'{dataset_name}/{family}/{aggregation}, seed {current_seed}: {row["error"]}')
            continue
        if not candidates:
            row={'dataset':dataset_name,'seed':current_seed,'fold':fold,'status':'no_eligible_factuals'}
            run.commit(row);records.append(row)
        print(f'  Seed {current_seed}: {len(candidates)} eligible factuals; '
              f'{len(models)} optimization models, {len(unseen)} unseen models', flush=True)
        for position,x0 in enumerate(candidates):
            instance_id=models.context.candidate_ids[position]
            c=rc.constraints_for(models,x0)
            for budget in budgets:
                instance_start=time.perf_counter()
                for method in methods:
                    rr.set_scope(fold=fold,seed=current_seed,instance_id=instance_id,budget=budget,method=method)
                    key=f'{method}_budget_{budget:g}'
                    artifact_dir=run.path/f'seed_{current_seed}_fold_{fold}'/f'instance_{instance_id}'/rr._safe(key)/'solver'
                    options={'time_limit':time_limit,'abs_gap':abs_gap,'seed':current_seed+position}
                    start=time.perf_counter(); error=None; rounds=[]
                    try:
                        if method=='ORPM-tree':
                            point,rounds=rt.tree_orpm(models,x0,budget,T=orpm_T,constraints=c,
                                                     artifact_dir=artifact_dir,**options)
                        elif method=='ADV-surrogate':
                            point=rt.adv_surrogate(models,x0,budget,T,temperature,c)
                        elif method=='ROAR-surrogate':
                            point=rt.roar_surrogate(models,x0,budget,T,current_seed+position,c)
                        elif method=='RobX-constrained':
                            point=rt.robx_constrained(models[0],x0,budget,models.context.X_train,
                                T=T,constraints=c,artifact_dir=artifact_dir,**options,**(robx_options or {}))
                        elif method=='Joint-minimax':
                            point=_single_solve(models,x0,budget,c,options,artifact_dir,'minimax')
                        else:
                            threshold=.5
                            if method.startswith('RobustCF4RF'):
                                confidence=forest_confidence if method.endswith('RobustSAA') else None
                                threshold=rt.forest_robust_threshold(len(models[0].trees),forest_risk,confidence)
                            point=_single_solve([models[0]],x0,budget,c,options,artifact_dir,'min_cost',threshold)
                            point.recourse_info.update(formulation=method,risk=forest_risk if method.startswith('RobustCF4RF') else None)
                    except Exception as exc:
                        error=f'{type(exc).__name__}: {exc}'
                        point=rc._tag(x0.clone(),status='solver_error',error=error,gradient_evaluations=0)
                    elapsed=time.perf_counter()-start
                    gap_rows=[]; diagnostic_error=None
                    if kind=='convergence' and method=='ORPM-tree' and rounds:
                        try:
                            gap_rows=rg.tree_gap_trajectory(models,x0,budget,rounds,c,
                                every=duality_gap_every,
                                time_limit=time_limit if duality_time_limit is None else duality_time_limit,
                                abs_gap=abs_gap,seed=current_seed+position,
                                artifact_dir=artifact_dir/'duality_evaluation')
                        except Exception as exc:
                            diagnostic_error=f'{type(exc).__name__}: {exc}'
                            rr.event('tree_duality_failed',error=diagnostic_error)
                            print(f'  Duality evaluation failed: {diagnostic_error}',flush=True)
                        for gap_row in gap_rows:
                            gap_records.append(dict(dataset=dataset_name,family=family,aggregation=aggregation,
                                config=f'{len(models)} {family} models',seed=current_seed,instance_id=instance_id,
                                budget=budget,**rg.scalar_row(gap_row)))
                        if gap_records:pd.DataFrame(gap_records).to_csv(run.path/'duality_convergence.csv',index=False)
                    p=rt.probabilities(models,point); eval_p=rt.probabilities(unseen,point) if unseen else np.array([])
                    feasible=rc.feasible(point,x0,budget,c)
                    row={'dataset':dataset_name,'family':family,'aggregation':aggregation,
                        'method':method,'artifact_key':key,'fold':fold,'seed':current_seed,'instance_id':instance_id,
                        'budget':budget,'T':orpm_T if method=='ORPM-tree' else T,'ORPM_T':orpm_T,'baseline_T':T,'actual_models':len(models),'actual_unseen_models':len(unseen),
                        'tree_profile':settings.get('tree_profile','explicit_fixture'),
                        'status':point.recourse_info['status'],'solver':point.recourse_info,
                        'valid':bool(feasible and (p>=.5).all() and error is None),
                        'deployed_valid':bool(feasible and p[0]>=.5 and error is None),
                        'unseen_all_valid':bool(feasible and (eval_p>=.5).all() and error is None) if unseen else None,
                        'unseen_fraction_valid':float((eval_p>=.5).mean()) if feasible and unseen and error is None else 0. if unseen else None,
                        'feasible':feasible,'max_loss':float(((1-p)**2).max()),
                        'cost':float(torch.linalg.vector_norm(point-x0)),
                        'elapsed_seconds':elapsed,'training_seconds_per_seed':models.metadata['training_seconds'],
                        'duality_diagnostic_seconds':gap_rows[-1]['diagnostic_elapsed_seconds'] if gap_rows else None,
                        'duality_diagnostic_error':diagnostic_error,
                        'duality_gap_lower_bound':gap_rows[-1]['gap_lower_bound'] if gap_rows else None,
                        'duality_gap_upper_bound':gap_rows[-1]['gap_upper_bound'] if gap_rows else None,
                        'gradient_evaluations':point.recourse_info.get('gradient_evaluations',0),
                        'probabilities':p.tolist(),'unseen_probabilities':eval_p.tolist(),
                        'point':point.tolist(),'factual':x0.tolist(),'feature_domain':'continuous_relaxation_float32',
                        'error':error}
                    records.append(run.commit(row,point,models,x0,c))
                    for r in rounds:
                        convergence.append({'dataset':dataset_name,'family':family,'aggregation':aggregation,
                            'seed':current_seed,'instance_id':instance_id,'budget':budget,
                            **{k:r[k] for k in ['round','current_worst_loss','best_worst_loss',
                                'minimax_lower_bound','minimax_gap_upper_bound','weighted_oracle_gap','elapsed_seconds']}})
                # Incremental tables in addition to the canonical per-method JSONL.
                summary,per_seed=_summary(records,methods,settings,dataset_name,family,aggregation,budgets)
                summary.to_csv(run.path/'tree_summary.csv',index=False)
                per_seed.to_csv(run.path/'tree_per_seed.csv',index=False)
                if convergence:pd.DataFrame(convergence).to_csv(run.path/'tree_convergence.csv',index=False)
                print(f'  Seed {current_seed}: factual {position+1}/{len(candidates)}, '
                      f'budget {budget:g} completed in {time.perf_counter()-instance_start:.1f}s',flush=True)
    summary,per_seed=_summary(records,methods,settings,dataset_name,family,aggregation,budgets)
    summary['Run Directory']=str(run.path)
    summary['ORPM T']=orpm_T
    summary['Baseline T']=T
    summary.to_csv(run.path/'tree_summary.csv',index=False)
    per_seed.to_csv(run.path/'tree_per_seed.csv',index=False)
    summary.attrs['convergence_file']=str(run.path/'tree_convergence.csv')
    if gap_records:
        rg.plot_gap_convergence(pd.DataFrame(gap_records),run.path,show=False)
    return summary


def run_tree_convergence_experiment(dataset_name,dataset_data,**kwargs):
    return run_tree_experiment(dataset_name,dataset_data,kind='convergence',**kwargs)


def compare_tree_recourse_methods(dataset_name,dataset_data,**kwargs):
    return run_tree_experiment(dataset_name,dataset_data,kind='comparison',**kwargs)


def run_tree_cost_validity_experiment(dataset_name,dataset_data,budgets=None,**kwargs):
    budgets=DATASET_BUDGETS[rc.canonical_name(dataset_name)] if budgets is None else budgets
    return run_tree_experiment(dataset_name,dataset_data,kind='tradeoff',budgets=budgets,**kwargs)


def plot_tree_convergence(result,show=True,output_dir=None):
    run=Path(result['Run Directory'].iloc[0]); source=run/'tree_convergence.csv'
    gap_source=run/'duality_convergence.csv'
    if gap_source.exists():
        rg.plot_gap_convergence(pd.read_csv(gap_source),output_dir or run,show=show)
    if not source.exists():
        print('No convergence records to plot:',run);return None
    raw=pd.read_csv(source)
    # Each seed contributes one mean; then summarize across seeds.
    metrics=['current_worst_loss','best_worst_loss','minimax_lower_bound',
             'minimax_gap_upper_bound','weighted_oracle_gap','elapsed_seconds']
    seed=raw.groupby(['budget','seed','round'])[metrics].mean().reset_index()
    bounds={metric:(0.,1.) for metric in ['current_worst_loss','best_worst_loss','minimax_lower_bound']}
    bounds.update({metric:(0.,np.inf) for metric in ['minimax_gap_upper_bound','weighted_oracle_gap','elapsed_seconds']})
    stats=rs.summarize_seed_frame(seed,['budget','round'],metrics,bounds)
    stats.to_csv(run/'convergence_plot_data.csv',index=False)
    seed.to_csv(run/'convergence_plot_seed_means.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for budget in sorted(raw.budget.unique()):
        def series(metric):
            return stats[(stats.budget==budget)&(stats.metric==metric)].sort_values('round')
        for metric,style in [('current_worst_loss',':'),('best_worst_loss','-'),('minimax_lower_bound','--')]:
            sub=series(metric)
            line,=axes[0].plot(sub['round'],sub['mean'],style,label=f'{metric.replace("_"," ")}; b={budget:g}')
            if metric=='best_worst_loss':
                axes[0].fill_between(sub['round'].to_numpy(),sub.ci_low.to_numpy(),sub.ci_high.to_numpy(),color=line.get_color(),alpha=.12)
        for metric,style,label in [('minimax_gap_upper_bound','-','Minimax bound gap'),('weighted_oracle_gap','--','Oracle gap')]:
            sub=series(metric)
            axes[1].plot(sub['round'],sub['mean'],style,label=f'{label}; b={budget:g}')
        axes[2].plot(series('elapsed_seconds')['mean'],series('best_worst_loss')['mean'],label=f'b={budget:g}')
    axes[0].set(xlabel='Round',ylabel='Worst-model MSE',title='Original-model loss and lower bound')
    axes[1].set(xlabel='Round',ylabel='Gap',title='Solver bounds; no convex convergence claim')
    axes[2].set(xlabel='Elapsed seconds',ylabel='Best worst-model MSE',title='Progress versus runtime')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle(f"{raw.dataset.iloc[0]} / {raw.family.iloc[0]} / {raw.aggregation.iloc[0]}\nPointwise 95% seed-t CI; unavailable intervals omitted; <5 seeds exploratory")
    dataset=raw.dataset.iloc[0]
    root=Path(output_dir) if output_dir is not None else run
    name=f'tree_convergence_{raw.family.iloc[0]}_{raw.aggregation.iloc[0]}'
    rp.save_table(stats,root,dataset,name+'_data')
    rp.save_table(seed,root,dataset,name+'_seed_means')
    fig.tight_layout()
    rp.save_figure(fig,root,dataset,name,show)
    return fig


def plot_tree_tradeoffs(results,output_dir,show=True):
    if results.empty:return []
    paths=[]
    for (dataset,family,aggregation),group in results.groupby(['Dataset','Family','Aggregation'],sort=False):
        for prefix,tag,label in [('Validity','optimization','Valid recourse across all competing models (%)'),
                                 ('Unseen Validity','unseen','Valid recourse across all unseen models (%)')]:
            data=group.rename(columns={prefix+' Mean':'Mean',prefix+' CI95 Low':'CI95 Low',
                prefix+' CI95 High':'CI95 High',prefix+' N Seeds':'N Seeds',prefix+' CI Status':'CI Status'})
            name=f'tree_validity_vs_budget_{family}_{aggregation}_{tag}'
            fig=rp._plot_lines(data,'Budget','Recourse budget (standardized L2 distance; log scale)',
                label,f'{rp.dataset_label(dataset)}\n{family.replace("_"," ")} / {aggregation.replace("_"," ")} — {tag}',True)
            rp.save_table(group.reset_index(drop=True),output_dir,dataset,name+'_data')
            rp.save_table(rp._compact_table(data,'Budget').reset_index(drop=True),output_dir,dataset,name,latex=True)
            paths.extend(rp.save_figure(fig,output_dir,dataset,name,show))
    return paths


def plot_tree_comparison(results,output_dir,show=True):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    if results.empty:return []
    paths=[]
    for (dataset,family,aggregation,budget),group in results.groupby(['Dataset','Family','Aggregation','Budget']):
        fig,axes=plt.subplots(1,3,figsize=(16,4.5))
        for ax,metric,low_col,high_col,label in [
            (axes[0],'Validity Mean','Validity CI95 Low','Validity CI95 High','All-model validity (%)'),
            (axes[1],'Mean Successful Cost','Successful Cost CI95 Low','Successful Cost CI95 High','Mean successful cost (standardized L2)'),
            (axes[2],'Time Mean (s)','Time CI95 Low (s)','Time CI95 High (s)','Mean runtime (seconds)')]:
            values=group[metric].to_numpy(float);positions=np.arange(len(group))
            ax.bar(positions,values)
            low=group[low_col].to_numpy(float);high=group[high_col].to_numpy(float)
            keep=np.isfinite(low)&np.isfinite(high)&np.isfinite(values)
            if keep.any():
                ax.errorbar(positions[keep],values[keep],yerr=np.stack([values[keep]-low[keep],high[keep]-values[keep]]),fmt='none',color='black',capsize=3)
            ax.set_xticks(positions);ax.set_xticklabels(group.Method,rotation=45,ha='right',fontsize=8)
            ax.set_ylabel(label);ax.set_xlabel('Recourse method');ax.grid(axis='y',alpha=.2)
        axes[0].set_ylim(0,105)
        fig.suptitle(f'{dataset} / {family} / {aggregation}; budget {budget:g}\nPointwise 95% seed-t CI; unavailable intervals omitted; <5 seeds exploratory')
        fig.tight_layout()
        name=f'tree_comparison_{dataset}_{family}_{aggregation}_b{budget:g}'
        rp.save_table(group.reset_index(drop=True),output_dir,dataset,name)
        paths.extend(rp.save_figure(fig,output_dir,dataset,name,show))
    return paths


def run_tree_batch(kind,model_specs,datasets=None,Quick=True,options=None,
                   reference_budgets=None,show=True):
    """Execute one complete notebook section; preserve partial batch summaries."""
    from . import recourse_campaign as campaign
    datasets=campaign.profile_settings(Quick)['datasets'] if datasets is None else list(datasets)
    options={**default_tree_options(Quick), **dict(options or {})}
    seed=int(options.get('seed',42))
    settings=campaign.profile_settings(Quick,seed,'tree_'+kind)
    batch_dir=rr.start_notebook_experiment('tree_'+kind,Quick,seed,settings_factory=lambda *_:settings)
    reference_budgets=reference_budgets or {
        'Synthetic':2.,'COMPAS':2.,'German':2.,'Polish':2.,'Folktables':5.,'Give_Me_Some_Credit':10.}
    rr.atomic_json(batch_dir/'batch_settings.json',{
        'kind':kind,'model_specs':model_specs,'datasets':datasets,'options':options,
        'mode':settings,'reference_budgets':reference_budgets,
        'tradeoff_budgets':DATASET_BUDGETS})
    plan=tree_workload_plan(kind,model_specs,datasets,settings,options,reference_budgets)
    rr.atomic_json(batch_dir/'workload_plan.json',{'configurations':plan,
        'interpretation':'Requested maximum counts. Capped solver hours assume every ORPM solve reaches its time limit; exclude baselines, training, model construction and I/O. This is not a wall-clock bound or forecast.'})
    calls=sum(row['orpm_oracle_calls'] for row in plan)
    capped=sum(row['orpm_capped_solver_hours'] for row in plan)
    print(f'Tree profile: {settings["tree_profile"]}; {len(plan)} dataset/model configurations; '
          f'up to {calls:,} ORPM oracle calls. '
          f'If all calls reach the solver limit: {capped:.2f} solver hours '
          '(plus training, baselines, construction and I/O).',flush=True)
    dispatch={'convergence':run_tree_convergence_experiment,'comparison':compare_tree_recourse_methods,
              'tradeoff':run_tree_cost_validity_experiment}
    if kind not in dispatch:raise ValueError('Unknown experiment kind')
    frames=[];errors=[];combined=pd.DataFrame()
    for dataset in datasets:
        try:data=rc.load_data_and_normalize(dataset)
        except Exception as exc:
            errors.append({'dataset':dataset,'stage':'loading','error':f'{type(exc).__name__}: {exc}'})
            rr.atomic_json(batch_dir/'batch_errors.json',errors)
            print(f'Unable to load {dataset}: {exc}');continue
        for spec in model_specs:
            print(f"Running {kind}: {dataset} / {spec['family']} / {spec['aggregation']}")
            try:
                kwargs={**options,**spec,'Quick':Quick}
                if kind!='tradeoff':kwargs['budgets']=[reference_budgets[rc.canonical_name(dataset)]]
                else:kwargs['budgets']=campaign.budget_grid(rc.canonical_name(dataset),Quick,tree=True)
                active=rr._ACTIVE_SETTINGS.get()
                token=rr._ACTIVE_SETTINGS.set(active or settings)
                try:result=dispatch[kind](dataset,data,**kwargs)
                finally:rr._ACTIVE_SETTINGS.reset(token)
                frame=result.copy();frame.attrs={};frames.append(frame)
                combined=pd.concat(frames,ignore_index=True)
                combined.to_csv(batch_dir/f'tree_{kind}_all_results.csv',index=False)
                rp.save_dataset_tables(combined,batch_dir,f'tree_{kind}_summary')
            except Exception as exc:
                errors.append({'dataset':dataset,**spec,'stage':'experiment','error':f'{type(exc).__name__}: {exc}'})
                rr.atomic_json(batch_dir/'batch_errors.json',errors)
                print(f'Experiment failed: {exc}');continue
            if kind=='convergence':
                try:plot_tree_convergence(result,show,output_dir=batch_dir)
                except Exception as exc:
                    errors.append({'dataset':dataset,**spec,'stage':'plotting','error':f'{type(exc).__name__}: {exc}'})
    rr.atomic_json(batch_dir/'batch_errors.json',errors)
    if not combined.empty:
        if kind=='comparison':plot_tree_comparison(combined,batch_dir,show)
        if kind=='tradeoff':plot_tree_tradeoffs(combined,batch_dir,show)
    print('Batch outputs:',batch_dir)
    return combined
