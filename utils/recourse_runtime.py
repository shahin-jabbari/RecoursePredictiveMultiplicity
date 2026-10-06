"""Portable setup, dataset verification, and a single-notebook experiment runner."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import time

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from . import recourse_core as rc
from . import recourse_campaign as campaign
from . import recourse_results as rr
from . import recourse_reporting as rp
from . import recourse_multiplicity as burden

_THREAD_LIMIT = None
_RUN_INDEX = []


def dataset_inventory(verify=True):
    """Validate bundled inputs; never download data or read prior experiments."""
    manifest = json.loads((rc.PROJECT_ROOT/'dataset'/'manifest.json').read_text())
    if verify:
        for entry in manifest['files']:
            path = rc.PROJECT_ROOT/'dataset'/entry['file']
            if not path.is_file():
                raise FileNotFoundError(f'Missing bundled dataset: {path}')
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError(f'Bundled data checksum mismatch: {path.name}')
    rows=[]
    for name in campaign.DATASETS:
        X,y,action=rc.load_data_and_normalize(name)
        if not np.isfinite(np.asarray(X)).all() or set(np.unique(y)) != {0,1}:
            raise ValueError(f'{name}: nonfinite features or unexpected target labels')
        rows.append(dict(dataset=name,observations=len(y),features=X.shape[1],
            modifiable_coordinates=len(action),recorded_target=rc.FAVORABLE_LABEL[name],
            target_note=rc.LABEL_PROVENANCE.get(name,'Target 1 is favorable')))
    return pd.DataFrame(rows)


def solver_check(require_full_size=False):
    """Fail before a campaign if the requested Gurobi workload is unavailable."""
    try:
        import gurobipy as gp
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag',0);env.start()
            with gp.Model('v6_preflight',env=env) as model:
                count=201 if require_full_size else 2
                x=model.addVars(count,lb=0,ub=1)
                model.addConstr(x[0]>=.5)
                model.setObjective(gp.quicksum(x[i]*x[i] for i in range(count)))
                model.Params.TimeLimit=2
                model.optimize()
                if model.SolCount == 0:raise RuntimeError('Solver probe produced no feasible point')
        return dict(available=True,version='.'.join(map(str,gp.gurobi.version())),
                    full_size_probe=require_full_size)
    except Exception as exc:
        raise RuntimeError('Gurobi preflight failed. Install gurobipy and configure a suitable license. '
            'Intermediate/full tree profiles and the neural MIQCP reference need a full-size license. '
            'For a non-tree-only run, set RUN_TREES=False and INCLUDE_NEURAL_MIQCP=False. '
            f'Original error: {exc}') from exc


def initialize(mode='quick',seed=42,run_trees=True,include_neural_miqcp=None,root=None):
    global _THREAD_LIMIT
    expected=rc.PROJECT_ROOT.resolve()
    if root is not None and Path(root).resolve()!=expected:
        raise RuntimeError('The imported utils package belongs to a different folder. Restart the '
                           'kernel and open this V6 notebook from its extracted directory.')
    name=campaign.mode_name(mode)
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    _THREAD_LIMIT=threadpool_limits(limits=1)
    if include_neural_miqcp is not None:
        campaign.PROFILES[name]['include_miqcp']=bool(include_neural_miqcp)
    include_miqcp=campaign.PROFILES[name]['include_miqcp']
    inventory=dataset_inventory()
    solver=solver_check(require_full_size=(name!='quick' or include_miqcp)) if run_trees or include_miqcp else {'available':'not requested'}
    session=rr.configure_notebook('v6',expected/'results')
    versions={p:importlib.metadata.version(p) for p in
              ['numpy','pandas','scipy','torch','scikit-learn','matplotlib','lime']}
    rr.atomic_json(session/'environment.json',dict(mode=name,seed=seed,packages=versions,
        solver=solver,run_trees=run_trees,include_neural_miqcp=include_miqcp,
        dataset_inventory=inventory.to_dict('records'),
        dataset_manifest=json.loads((expected/'dataset'/'manifest.json').read_text())))
    _RUN_INDEX.clear()
    print(f'V6 / {name}; outputs: {expected / "results" / "v6" / name}',flush=True)
    print('All six datasets are local. German/Polish retain the recorded targets for numerical '
          'replication and carry target-audit flags; see dataset/README.txt.',flush=True)
    print('Neural MIQCP reference:', 'enabled' if include_miqcp else 'disabled for this run')
    return inventory,campaign.workload_plan(mode,seed)


def run_section(section,mode='quick',seed=42,resume=True,show=True):
    started=time.perf_counter()
    result=campaign.run_section(section,mode,seed,resume,show)
    if result.empty:
        raise RuntimeError(f'{section} produced no summary rows; inspect its saved diagnostics.')
    if 'Model Generation Failures' in result and (result['Model Generation Failures']>0).any():
        raise RuntimeError(f'{section} has model-generation failures; results were saved for inspection.')
    if section=='convergence':
        rp.plot_convergence(result.attrs.get('trajectories',[]),
                            result.attrs['campaign_directory'],show=show)
    _record_section(section,mode,result,time.perf_counter()-started)
    return result


def run_burden(source,mode='quick',resume=True,show=True,name='multiplicity_burden'):
    """Use the trade-offs created earlier in this notebook, not a previous version."""
    if source is None or (isinstance(source,pd.DataFrame) and source.empty):
        raise ValueError('Run the matching cost-validity experiment cell before multiplicity burden.')
    started=time.perf_counter()
    result=burden.run_from_saved(source,Quick=mode,results_root=rr.RESULTS_ROOT,resume=resume,show=show)
    if result.attrs.get('search_errors'):
        raise RuntimeError('Individual-recourse searches reported errors; see the saved errors.json.')
    _record_section(name,mode,result,time.perf_counter()-started)
    return result


def _record_section(section,mode,result,elapsed):
    _RUN_INDEX.append(dict(section=section,mode=campaign.mode_name(mode),elapsed_seconds=elapsed,
        summary_rows=len(result),directory=result.attrs.get('campaign_directory')))
    rr.atomic_json(rc.PROJECT_ROOT/'results'/'v6'/'latest_notebook_index.json',_RUN_INDEX)


def compact_summary(frame):
    if frame is None:return pd.DataFrame()
    keys=['Dataset','Family','Aggregation','Method','Config','Budget','Actual Models','Mode']
    columns=[c for c in keys if c in frame]
    columns += [c for c in frame if c.endswith('Validity Mean') or c.endswith('Failure Count')]
    if not columns:return frame
    return frame[columns]


def run_index():
    return pd.DataFrame(_RUN_INDEX)
