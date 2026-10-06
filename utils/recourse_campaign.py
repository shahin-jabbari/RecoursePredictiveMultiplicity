"""V6 replication profiles, progress records, and compatible-job resume."""
import hashlib
import json
from pathlib import Path
import pickle
import time
import numpy as np
import pandas as pd

PROFILE_REVISION='v6-replication-v1'
DATASETS=['Synthetic','COMPAS','German','Give_Me_Some_Credit','Polish','Folktables']
BUDGETS={'Synthetic':[.5,1.,2.,5.,10.], 'COMPAS':[.5,1.,2.,5.,10.],
         'German':[.5,1.,2.,5.,10.], 'Polish':[.5,1.,2.,5.,10.],
         'Folktables':[1.,2.,5.,10.,20.], 'Give_Me_Some_Credit':[2.,5.,50.,100.,200.]}
REFERENCE={'Synthetic':2.,'COMPAS':2.,'German':2.,'Polish':2.,'Folktables':5.,'Give_Me_Some_Credit':10.}
# The latest tree analysis motivated a finer GMC grid; retain non-tree dataset grids.
TREE_BUDGETS={**BUDGETS,'Give_Me_Some_Credit':[2.,5.,10.,20.,50.]}
GAP_EVERY=5
MIXED_CONFIGS=[(2,2,2),(4,4,4),(8,8,8)]
TREE_SPECS=[dict(family='decision_tree',aggregation='probability'),
            dict(family='random_forest',aggregation='probability'),
            dict(family='random_forest',aggregation='hard_vote')]
TREE_CONVERGENCE_SPECS=[dict(TREE_SPECS[0])]
SECTIONS=['convergence','convergence_duality','linear_baselines','generalization','nonlinear_baselines','cost_validity',
          'tree_convergence','tree_comparison','tree_tradeoff']
PROFILES={
 'quick':dict(datasets=['Synthetic','COMPAS'],seeds=2,instances=3,baseline_T=20,ORPM_T=10,
     convergence_instances=3,convergence_T=20,gap_T=20,convergence_configs=[(1,1,1)],
     tradeoff_instances=3,budget_indices=[0,2,3],tree_instances=3,tree_T=5,tree_convergence_T=10,
     inner_oracle_steps=25,training_epochs=40,lime_samples=500,miqcp_time_limit=1.,include_miqcp=False),
 'intermediate':dict(datasets=['Synthetic','COMPAS','Give_Me_Some_Credit','Folktables'],
     seeds=6,instances=15,baseline_T=75,ORPM_T=35,convergence_instances=10,convergence_T=50,
     gap_T=100,convergence_configs=[(2,2,2)],tradeoff_instances=8,budget_indices=[0,2,3,4],
     tree_instances=10,tree_T=15,tree_convergence_T=50,inner_oracle_steps=200,
     training_epochs=100,lime_samples=5000,miqcp_time_limit=10.,include_miqcp=True),
 'full':dict(datasets=list(DATASETS),seeds=10,instances=20,baseline_T=100,ORPM_T=40,
     convergence_instances=20,convergence_T=75,gap_T=200,convergence_configs=[(2,2,2)],
     tradeoff_instances=8,budget_indices=[0,1,2,3,4],tree_instances=20,tree_T=15,tree_convergence_T=100,
     inner_oracle_steps=200,training_epochs=100,lime_samples=5000,miqcp_time_limit=10.,include_miqcp=True),
}


def datasets_for(mode, section=None):
    name=mode_name(mode)
    datasets=list(PROFILES[name]['datasets'])
    if name=='full' and section=='convergence_duality':
        return ['Synthetic','COMPAS','German','Give_Me_Some_Credit']
    if section and section.startswith('tree_'):
        return [d for d in datasets if d in ['Synthetic','COMPAS','Give_Me_Some_Credit']]
    if name=='full' and section=='generalization':
        return ['Synthetic','COMPAS','German']
    return datasets



def mode_name(mode):
    if type(mode) is bool:return 'quick' if mode else 'full'
    if isinstance(mode,str) and mode in PROFILES:return mode
    raise TypeError("Mode must be 'quick', 'intermediate', 'full', True, or False")


def profile_settings(mode=True,seed=42,section=None):
    name=mode_name(mode);p=PROFILES[name]
    instances=p['instances'];T=p['baseline_T'];tree=bool(section and section.startswith('tree_'))
    if section in {'convergence','convergence_duality'}:
        instances=p['convergence_instances'];T=p['gap_T'] if section=='convergence_duality' else p['convergence_T']
    if section=='cost_validity':instances=p['tradeoff_instances']
    if tree:instances=p['tree_instances'];T=p['tree_convergence_T'] if section=='tree_convergence' else p['tree_T']
    from . import recourse_core as rc
    result=dict(Quick=mode,mode=name,folds=p['seeds'],T=T,max_instances=instances,
        seeds=list(range(int(seed),int(seed)+p['seeds'])),ORPM_T=p['tree_T'] if tree else p['ORPM_T'],
        CONVERGENCE_T=p['tree_convergence_T'] if tree else T if section=='convergence_duality' else p['convergence_T'],
        TREE_ORPM_T=p['tree_T'],profile_revision=PROFILE_REVISION,section=section,
        datasets=datasets_for(mode,section),gap_every=5 if name=='quick' else 10,
        target_mapping=dict(rc.FAVORABLE_LABEL),label_provenance=dict(rc.LABEL_PROVENANCE),
        **{k:p[k] for k in ['inner_oracle_steps','training_epochs','lime_samples','miqcp_time_limit','include_miqcp']})
    # Retain the completed three-seed exploratory generalization protocol in full mode.
    if name=='full' and section=='generalization':
        result.update(folds=3,seeds=list(range(seed,seed+3)),max_instances=10,T=50,ORPM_T=25)
    if tree:result['tree_profile']='v6-small' if name=='quick' else 'depth-four'
    return result


def tree_options(mode=True):
    name=mode_name(mode)
    options=dict(num_models=5,num_unseen=5,max_depth=4,n_estimators=15,min_samples_leaf=5,
        epsilon=.05,pool_multiplier=4,max_pool_multiplier=32,seed=42,time_limit=1.,abs_gap=1e-5,include_direct_reference=True,
        temperature=.1,forest_risk=.1,forest_confidence=.95)
    if name=='quick':options.update(num_models=3,num_unseen=3,max_depth=2,n_estimators=5,time_limit=.25)
    return options


def budget_grid(dataset,mode,tree=False):
    return [(TREE_BUDGETS if tree else BUDGETS)[dataset][i] for i in PROFILES[mode_name(mode)]['budget_indices']]


def jobs_for(section,mode,seed=42):
    if section not in SECTIONS:raise ValueError(f'Unknown section: {section}')
    name=mode_name(mode);p=PROFILES[name];jobs=[]
    for ds in datasets_for(mode,section):
        if section=='convergence':
            # Preserve the original convergence budgets in full mode.
            for counts in p['convergence_configs']:
                for budget in ([2.,5.] if name=='full' else [2.]):
                    jobs.append(dict(dataset=ds,counts=list(counts),budget=budget))
        elif section=='convergence_duality':
            for budget in [2.]:
                jobs.append(dict(dataset=ds,num_models=3 if name=='quick' else 6,budget=budget,
                    gap_every=profile_settings(mode,seed,section)['gap_every'],loss='normalized_affine_logit_bce'))
        elif section=='cost_validity':
            for counts in MIXED_CONFIGS:
                for budget in budget_grid(ds,mode):jobs.append(dict(dataset=ds,counts=list(counts),budget=budget))
        elif section=='linear_baselines':jobs.append(dict(dataset=ds,budget=5.))
        elif section=='generalization':jobs.append(dict(dataset=ds,budget=2.,num_opt_models=3 if name=='quick' else 8,num_eval_models=5 if name=='quick' else 50))
        elif section=='nonlinear_baselines':
            budgets={'Synthetic':[5.],'Polish':[5.],'Give_Me_Some_Credit':[10.,50.]}.get(ds,[5.,8.])
            for budget in (budgets[:1] if name=='quick' else budgets):jobs.append(dict(dataset=ds,budget=budget,num_models=3 if name=='quick' else 5))
        else:
            specs=TREE_CONVERGENCE_SPECS if section=='tree_convergence' else TREE_SPECS
            for spec in specs:
                jobs.append(dict(dataset=ds,**spec,budgets=budget_grid(ds,mode,tree=True) if section=='tree_tradeoff' else [REFERENCE[ds]],
                    **({'gap_every':GAP_EVERY} if section=='tree_convergence' else {})))
    return jobs


def workload_plan(mode='quick',seed=42):
    """Count real jobs and oracle calls; do not present solver caps as runtime predictions."""
    rows=[]
    for section in SECTIONS:
        s=profile_settings(mode,seed,section);jobs=jobs_for(section,mode,seed)
        facts=sum(len(j.get('budgets',[j.get('budget')])) for j in jobs)*s['folds']*s['max_instances']
        rounds=s['CONVERGENCE_T'] if section in {'convergence','convergence_duality','tree_convergence'} else s['ORPM_T']
        # The full profile retains a larger ORPM cap for full GMC mixed-model high budgets.
        calls=sum(len(j.get('budgets',[j.get('budget')]))*s['folds']*s['max_instances']*
            (100 if mode_name(mode)=='full' and section=='cost_validity' and j['dataset']=='Give_Me_Some_Credit' and j['budget']>=50 else rounds)
            for j in jobs)
        from .recourse_convergence import checkpoints
        diagnostic_calls=facts*len(checkpoints(rounds,GAP_EVERY)) if section in {'convergence_duality','tree_convergence'} else 0
        rows.append(dict(section=section,datasets=len(s['datasets']),seeds=s['folds'],
            instances_per_seed=s['max_instances'],jobs=len(jobs),budget_configurations=sum(len(j.get('budgets',[None])) for j in jobs),
            ORPM_rounds=rounds,baseline_cap=s['T'],max_factual_configuration_evaluations=facts,
            max_oracle_calls=calls,max_gap_evaluation_calls=diagnostic_calls,
            gradient_oracle_inner_steps=0 if section.startswith('tree_') or section=='convergence_duality' else calls*s['inner_oracle_steps'],
            tree_ORPM_capped_solver_hours=calls*tree_options(mode)['time_limit']/3600 if section.startswith('tree_') else 0.,
            tree_gap_capped_solver_hours=diagnostic_calls*tree_options(mode)['time_limit']/3600 if section.startswith('tree_') else 0.))
    return pd.DataFrame(rows)


def _hash(value):
    from . import recourse_results as rr
    return hashlib.sha256(json.dumps(rr.jsonable(value),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def source_hashes():
    root=Path(__file__).resolve().parent
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('recourse_*.py'))}


def data_catalog():
    """A changed source file or data location starts a fresh section namespace.

    Actual loaded values are independently hashed for every cached job.
    """
    from . import recourse_core as rc
    root=Path(rc.PROJECT_ROOT).resolve();files=[]
    for name in ['dataset']:
        folder=root/name
        if folder.exists():
            for p in sorted(folder.rglob('*')):
                if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('.'):
                    stat=p.stat();files.append((str(p.relative_to(root)),stat.st_size,stat.st_mtime_ns))
    return dict(project_root=str(root),files=files)


def data_hash(data):
    h=hashlib.sha256()
    for x in data[:2]:
        frame=pd.DataFrame(x)
        h.update(pd.util.hash_pandas_object(frame,index=True).to_numpy().tobytes())
        h.update(str(list(frame.columns)).encode());h.update(str(list(frame.dtypes)).encode())
    h.update(str(data[2] if len(data)>2 else None).encode())
    return h.hexdigest()


def _atomic_pickle(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:pickle.dump(value,f,pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def start_section(section,mode,seed=42,resume=True):
    from . import recourse_results as rr
    from . import recourse_core as rc
    settings=profile_settings(mode,seed,section)
    import importlib.metadata as metadata
    packages={}
    for name in ['numpy','pandas','torch','scipy','scikit-learn','gurobipy','lime']:
        try:packages[name]=metadata.version(name)
        except metadata.PackageNotFoundError:packages[name]=None
    descriptor=dict(section=section,settings=settings,jobs=jobs_for(section,mode,seed),sources=source_hashes(),packages=packages,data_catalog=data_catalog())
    # Solver options affect results even if changed in memory rather than in a file.
    if section.startswith('tree_'):descriptor['tree_options']=tree_options(mode)
    # Boolean/string aliases are semantically equivalent for resume.
    descriptor['settings']=dict(settings,Quick=settings['mode'])
    fingerprint=_hash(descriptor)
    base=Path(rr.RESULTS_ROOT)/rr.NOTEBOOK_VERSION/settings['mode']/section
    if resume and base.exists():
        for path in sorted(base.iterdir(),reverse=True):
            metadata=path/'campaign.json'
            if metadata.exists() and json.loads(metadata.read_text()).get('fingerprint')==fingerprint:
                rc.RESULTS_DIR=path
                print('Resuming compatible session:',path,flush=True)
                return path,settings,descriptor['jobs']
    path=rr.start_notebook_experiment(section,mode,seed,settings_factory=lambda *_:settings)
    rr.atomic_json(path/'campaign.json',dict(fingerprint=fingerprint,**descriptor))
    return path,settings,descriptor['jobs']


def execute_job(path,job,settings,data,call,resume=True):
    """Reuse one completed job only with matching source, parameters, and input data."""
    from . import recourse_results as rr
    key=_hash(dict(job=job,settings=dict(settings,Quick=settings['mode']),data_hash=data_hash(data)))
    folder=Path(path)/'checkpoints'/key
    metadata=folder/'complete.json';result_path=folder/'result.pkl'
    if resume and metadata.exists() and result_path.exists():
        meta=json.loads(metadata.read_text())
        if meta.get('result_sha256')==hashlib.sha256(result_path.read_bytes()).hexdigest():
            result=pd.read_pickle(result_path)
            # Do not return cached paths pointing at deleted/missing underlying records.
            dirs=result.get('Run Directory',pd.Series(dtype=str)).dropna().unique()
            if all(Path(d,'manifest.json').exists() and Path(d,'instances.jsonl').exists() for d in dirs):
                print('  Reused completed job',flush=True)
                return result,True
    folder.mkdir(parents=True,exist_ok=True)
    rr.atomic_json(folder/'started.json',dict(job=job,settings=settings,input_hash=data_hash(data)))
    token=rr._ACTIVE_SETTINGS.set(settings)
    try:result=call()
    finally:rr._ACTIVE_SETTINGS.reset(token)
    _atomic_pickle(result_path,result)
    rr.atomic_json(metadata,dict(job=job,result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
        status='completed',note='Completed model/solver failures are retained; inspect contributing counts.'))
    return result,False


def _dispatch(section,job,data,settings,mode,seed):
    from . import recourse_experiments as re
    from . import recourse_tree_experiments as rte
    X,y=data[:2];ds=job['dataset'];base=dict(dataset_name=ds,seed=seed,Quick=mode)
    if section.startswith('tree_'):
        options=dict(tree_options(mode),seed=seed)
        if section=='tree_convergence':options['duality_gap_every']=job['gap_every']
        return rte.run_tree_experiment(dataset_name=ds,dataset_data=data,kind=section.removeprefix('tree_'),
            family=job['family'],aggregation=job['aggregation'],budgets=job['budgets'],Quick=mode,**options)
    base['dataset']=(X,y)
    if section=='convergence_duality':
        from . import recourse_convergence as rg
        return rg.run_convex_duality_experiment(**base,num_models=job['num_models'],
            recourse_budget=job['budget'],actionable_indices=data[2],gap_every=job['gap_every'])
    if section=='convergence':
        a,b,c=job['counts']
        return re.run_convergence_experiment(**base,num_linear=a,num_nn_single_layer=b,num_nn_double_layer=c,
            recourse_budget=job['budget'],actionable_indices=data[2])
    if section=='linear_baselines':return re.comparison_with_baselines_linear(**base,recourse_budget=job['budget'])
    if section=='generalization':return re.run_generalization_experiment(**base,recourse_budget=job['budget'],num_opt_models=job['num_opt_models'],num_eval_models=job['num_eval_models'])
    if section=='nonlinear_baselines':return re.comparison_with_baselines_nonlinear(**base,recourse_budget=job['budget'],num_models=job['num_models'],include_miqcp=settings['include_miqcp'])
    a,b,c=job['counts']
    result=re.comparison_with_baselines_nonlinear_v2(**base,num_linear=a,num_nn_single=b,num_nn_double=c,
        recourse_budget=job['budget'],actionable_indices=data[2])
    result['Config']=f'{a+b+c} Models ({a}LR+{b}NN1+{c}NN2)'
    result['Num Linear'],result['Num NN1'],result['Num NN2']=a,b,c
    result['Total_Models']=result['Actual Models'];result['Requested_Models']=a+b+c
    return result


def run_section(section,mode='quick',seed=42,resume=True,show=True):
    from . import recourse_core as rc
    from . import recourse_results as rr
    from . import recourse_reporting as rp
    from . import recourse_tree_experiments as rte
    path,settings,jobs=start_section(section,mode,seed,resume)
    rr.atomic_json(path/'workload_plan.json',dict(settings=settings,jobs=jobs,
        runtime_note='Iteration and solver-call caps; actual runtime depends on hardware.'))
    frames=[];trajectories=[];errors=[];done=0;started=time.perf_counter()
    for dataset in settings['datasets']:
        selected=[j for j in jobs if j['dataset']==dataset];dataset_frames=[]
        print(f'[{section}] Loading {dataset}; {len(selected)} jobs; {settings["folds"]} seeds; '
              f'up to {settings["max_instances"]} factuals/seed',flush=True)
        try:
            data=(*rc.get_data_synthetic(2000 if settings['mode']=='quick' else 20000),[0,1]) if dataset=='Synthetic' and section=='nonlinear_baselines' else rc.load_data_and_normalize(dataset)
            if data[0] is None:raise ValueError('Dataset loader returned no data')
        except Exception as exc:
            errors.append(dict(dataset=dataset,stage='loading',error=f'{type(exc).__name__}: {exc}'))
            rr.atomic_json(path/'campaign_errors.json',errors);raise RuntimeError(f'Dataset loading failed: {dataset}') from exc
        for job in selected:
            done+=1
            print(f'[{section}] Job {done}/{len(jobs)} | {job} | elapsed {(time.perf_counter()-started)/60:.1f} min',flush=True)
            try:
                result,reused=execute_job(path,job,settings,data,
                    lambda:_dispatch(section,job,data,settings,mode,seed),resume)
            except Exception as exc:
                errors.append(dict(job=job,stage='experiment',error=f'{type(exc).__name__}: {exc}'))
                rr.atomic_json(path/'campaign_errors.json',errors);raise RuntimeError(f'Experiment failed: {job}') from exc
            if 'trajectory' in result.attrs:trajectories.append(result.attrs['trajectory'])
            frame=result.copy();frame.attrs={};frames.append(frame);dataset_frames.append(frame)
            combined=pd.concat(frames,ignore_index=True)
            combined.to_csv(path/'all_results.csv',index=False)
            rp.save_dataset_tables(combined,path,section+'_summary')
            if section=='convergence':_atomic_pickle(path/'all_trajectories.pkl',trajectories)
            if section=='tree_convergence':rte.plot_tree_convergence(result,show=show,output_dir=path)
            if section=='convergence_duality':
                from . import recourse_convergence as rg
                source=Path(result['Run Directory'].iloc[0])/'duality_convergence.csv'
                if source.exists():rg.plot_gap_convergence(pd.read_csv(source),path,show=show)
            rr.atomic_json(path/'progress.json',dict(section=section,mode=settings['mode'],jobs_visited=done,
                total_jobs=len(jobs),summary_rows=len(combined),last_job=job,reused=reused,
                elapsed_seconds=time.perf_counter()-started,errors=len(errors)))
        if dataset_frames:
            dataset_result=pd.concat(dataset_frames,ignore_index=True)
            if section=='cost_validity':rp.plot_tradeoffs(dataset_result,path,show=show)
            if section=='tree_comparison':rte.plot_tree_comparison(dataset_result,path,show=show)
            if section=='tree_tradeoff':rte.plot_tree_tradeoffs(dataset_result,path,show=show)
    rr.atomic_json(path/'campaign_errors.json',errors)
    result=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    result.attrs.update(trajectories=trajectories,campaign_directory=str(path),campaign_errors=errors)
    print(f'[{section}] Finished; outputs: {path}; failed jobs/datasets: {len(errors)}',flush=True)
    return result
