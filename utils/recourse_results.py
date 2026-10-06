"""Experiment modes and incremental, self-describing research artifacts."""
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import time
import uuid

import numpy as np
import pandas as pd
import torch
from . import recourse_statistics as rs

_ACTIVE_RUN = ContextVar('recourse_run', default=None)
_ACTIVE_SETTINGS = ContextVar('recourse_settings', default=None)
SCHEMA_VERSION = 3
NOTEBOOK_VERSION = 'unversioned'
RESULTS_ROOT = None


def configure_notebook(version, results_root='results'):
    """Select a version namespace; never move, delete, or overwrite previous runs."""
    global NOTEBOOK_VERSION, RESULTS_ROOT
    from . import recourse_core as rc
    if not isinstance(version, str) or not version or _safe(version) != version:
        raise ValueError('Use a simple version name such as v4.')
    NOTEBOOK_VERSION = version
    RESULTS_ROOT = Path(results_root).expanduser().resolve()
    rc.RESULTS_DIR = RESULTS_ROOT / NOTEBOOK_VERSION / 'setup' / _stamp()
    rc.RESULTS_DIR.mkdir(parents=True, exist_ok=False)
    atomic_json(rc.RESULTS_DIR / 'session.json', {'notebook_version': version})
    return rc.RESULTS_DIR


def start_notebook_experiment(name, Quick=True, seed=42, *, settings_factory=None):
    """Start a fresh namespace for all artifacts produced by one execution cell."""
    from . import recourse_core as rc
    if current_run() is not None:
        raise RuntimeError('Start a notebook experiment outside an active experiment call.')
    if RESULTS_ROOT is None:
        raise RuntimeError('Run configure_notebook(version, results_root) in setup first.')
    settings = (settings_factory or experiment_settings)(Quick, seed)
    rc.RESULTS_DIR = RESULTS_ROOT / NOTEBOOK_VERSION / settings['mode'] / _safe(name) / _stamp()
    rc.RESULTS_DIR.mkdir(parents=True, exist_ok=False)
    atomic_json(rc.RESULTS_DIR / 'session.json', {
        'notebook_version': NOTEBOOK_VERSION, 'experiment': name, **settings})
    print('Results directory:', rc.RESULTS_DIR)
    return rc.RESULTS_DIR


def experiment_settings(Quick=True, seed=42):
    """Resolve the suite profile; booleans remain aliases for quick/full."""
    from .recourse_campaign import profile_settings
    return profile_settings(Quick, seed)


def orpm_rounds(baseline_T, dataset_name=None, budget=None, counts=None,
                kind='comparison', family='differentiable'):
    """Separate outer ORPM rounds from baseline iterations and inner-oracle steps."""
    settings = _ACTIVE_SETTINGS.get()
    if settings is not None and settings.get('profile_revision'):
        if kind == 'convergence':
            return int(settings['CONVERGENCE_T'])
        if family in {'decision_tree', 'random_forest'}:
            return int(settings['TREE_ORPM_T'])
        name = str(dataset_name).lower().replace('_','').replace(' ','')
        mixed = counts is not None and counts[0] > 0 and sum(counts[1:]) > 0
        if settings['mode'] == 'full' and name == 'givemesomecredit' and mixed and budget is not None and budget >= 50:
            return 100
        return int(settings['ORPM_T'])
    if settings is None or settings.get('mode') != 'full':
        return int(baseline_T)
    if kind == 'convergence':
        return int(settings.get('CONVERGENCE_T', 100))
    if family in {'decision_tree', 'random_forest'}:
        return int(settings.get('TREE_ORPM_T', 100))
    name = str(dataset_name).lower().replace('_','').replace(' ','')
    mixed = counts is not None and counts[0] > 0 and sum(counts[1:]) > 0
    if name == 'givemesomecredit' and mixed and budget is not None and budget >= 50:
        return int(settings.get('CONVERGENCE_T', 100))
    return int(settings.get('ORPM_T', 50))


def record_iteration_limits(orpm_T, baseline_T):
    run = current_run()
    if run is not None:
        run.manifest['iteration_limits'] = {'ORPM_T': int(orpm_T), 'baseline_T': int(baseline_T),
            'inner_oracle_steps': 'unchanged from the selected oracle',
            'early_stopping': 'no new plateau-based early stopping'}
        atomic_json(run.path/'manifest.json', run.manifest)


def current_run():
    return _ACTIVE_RUN.get()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return jsonable(value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'_'+uuid.uuid4().hex[:10]


def _safe(name):
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in str(name))[:120]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('w') as stream:
        json.dump(jsonable(value), stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def _npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def _append(path, record):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(jsonable(record), allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())


def set_scope(**scope):
    run = current_run()
    if run:
        run.scope = scope


def event(kind, **data):
    run = current_run()
    if run:
        run.event(kind, **data)


def attach_trace(point, **trace):
    if current_run() is not None:
        point.recourse_trace = trace
    return point


def log_notebook_error(original_cell, error):
    """Keep dataset-loading or orchestration failures outside an experiment call."""
    from . import recourse_core as rc
    import traceback
    path = rc.RESULTS_DIR/'notebook_errors'/f'cell_{original_cell}_{_stamp()}.json'
    atomic_json(path, {'original_cell': original_cell, 'status': 'failed',
        'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()})


def _descriptor(model):
    from . import recourse_core as rc
    mask = None
    if isinstance(model, rc.FeatureMaskedModel):
        mask = model.feature_mask.detach().cpu().tolist()
        base = model.base_model
    else:
        base = model
    layers = [base.linear] if isinstance(base, rc.LogisticRegressionModel) else list(base.net)
    dims = [(layer.in_features, layer.out_features) for layer in layers if isinstance(layer, torch.nn.Linear)]
    return {'class': type(base).__name__, 'linear_dimensions': dims, 'feature_mask': mask,
            'training_settings': getattr(model, 'training_settings', {}),
            'training_loss': getattr(model, 'training_loss_history', []),
            'training_artifact': getattr(model, 'training_artifact', None)}


def load_models(seed_directory):
    """Rebuild saved model weights without unpickling an arbitrary model object."""
    from . import recourse_core as rc
    directory = Path(seed_directory)
    metadata = json.loads((directory/'models.json').read_text())
    result = []
    for i, desc in enumerate(metadata['models']):
        dims = desc['linear_dimensions']; d = dims[0][0]
        if desc['class'] == 'LogisticRegressionModel':
            model = rc.LogisticRegressionModel(d)
        elif desc['class'] == 'NeuralNetworkModelSingleLayer':
            model = rc.NeuralNetworkModelSingleLayer(d, dims[0][1])
        elif desc['class'] == 'NeuralNetworkModelDoubleLayers':
            model = rc.NeuralNetworkModelDoubleLayers(d, dims[0][1], dims[1][1])
        else:
            raise ValueError(f"Unsupported saved model class: {desc['class']}")
        if desc['feature_mask'] is not None:
            model = rc.FeatureMaskedModel(model, desc['feature_mask'])
        with np.load(directory/f'model_{i:04d}.npz', allow_pickle=False) as saved:
            state = {k: torch.tensor(saved[k]) for k in saved.files}
        model.load_state_dict(state); model.eval(); result.append(model)
    return result


class RunStore:
    def __init__(self, root, experiment, config, parent=None):
        self.path = Path(root)/(_stamp()+'_'+_safe(experiment))
        self.path.mkdir(parents=True, exist_ok=False)
        self.scope, self.training_count, self.record_count = {}, 0, 0
        self.errors = 0
        self.event_write_seconds = 0.0
        self.manifest = {'schema_version': SCHEMA_VERSION, 'run_id': self.path.name,
            'notebook_version': NOTEBOOK_VERSION,
            'experiment': experiment, 'configuration': config, 'status': 'running',
            'started_utc': datetime.now(timezone.utc).isoformat(),
            'parent_run': str(parent.path) if parent else None,
            'python': sys.version, 'platform': platform.platform(),
            'packages': {}, 'source_sha256': {},
            'instance_count_semantics': 'maximum eligible factual instances per seed',
            'uncertainty_protocol': {
                'unit': 'equally weighted seed means; conditional on dataset and protocol',
                'interval': 'approximate pointwise 95% Student-t, df=S-1, sample SD ddof=1',
                'pairing': 'same seed and factual, then average differences within seed',
                'missing': 'exclude missing observations; report actual counts',
                'few_seeds': 'fewer than five is exploratory',
                'unavailable': 'fewer than two seeds or zero observed seed variance',
                'bounds': 'display endpoints intersect metric range; raw endpoints retained'}}
        for package in ['numpy', 'pandas', 'torch', 'scipy', 'scikit-learn', 'lime', 'gurobipy']:
            try:
                self.manifest['packages'][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                self.manifest['packages'][package] = None
        source_dir = Path(__file__).resolve().parent
        for filename in ['recourse_core.py', 'recourse_baselines.py', 'recourse_experiments.py', 'recourse_results.py',
                         'recourse_trees.py', 'recourse_tree_experiments.py', 'recourse_statistics.py', 'recourse_reporting.py', 'recourse_campaign.py', 'recourse_convergence.py', 'recourse_multiplicity.py', 'recourse_runtime.py', '__init__.py']:
            source = source_dir/filename
            if source.exists():
                data = source.read_bytes()
                self.manifest['source_sha256'][filename] = hashlib.sha256(data).hexdigest()
                destination = self.path/'source'/filename
                destination.parent.mkdir(exist_ok=True); destination.write_bytes(data)
        atomic_json(self.path/'manifest.json', self.manifest)
        (self.path/'instances.jsonl').touch()
        self.event('run_started')

    def event(self, kind, **data):
        started = time.perf_counter()
        if kind.endswith('failed') or kind.endswith('error'):
            self.errors += 1
        _append(self.path/'events.jsonl', {'event': kind, 'utc': datetime.now(timezone.utc).isoformat(),
                                         **self.scope, **data})
        self.event_write_seconds += time.perf_counter()-started

    def save_dataset(self, X, y, label='dataset'):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        _npz(self.path/f'{label}.npz', {'X_raw': np.asarray(X), 'y_raw': np.asarray(y)})
        atomic_json(self.path/f'{label}.json', {'feature_names': list(frame.columns),
            'row_index': frame.index.tolist(), 'shape': frame.shape,
            'row_ids_in_splits': 'zero-based positions into this saved input snapshot'})

    def save_training(self, model):
        index = self.training_count; self.training_count += 1
        name = f'training/model_{index:05d}'
        _npz(self.path/(name+'.npz'), {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
        atomic_json(self.path/(name+'.json'), {**self.scope, **_descriptor(model)})
        model.training_artifact = name
        self.event('model_trained', artifact=name, settings=model.training_settings)

    def save_seed(self, models, candidates, fold, seed, **extra):
        ctx = models.context
        directory = self.path/f'seed_{seed}_fold_{fold}'
        directory.mkdir(exist_ok=True)
        arrays = {name: getattr(ctx, name) for name in ['X_train', 'y_train', 'X_validation',
            'y_validation', 'X_test', 'y_test', 'train_ids', 'validation_ids', 'test_ids']}
        arrays['candidate_ids'] = np.asarray(ctx.candidate_ids, dtype=np.int64)
        arrays['candidates'] = np.asarray([x.detach().cpu().numpy() for x in candidates], dtype=np.float32).reshape(-1, len(ctx.columns))
        with torch.no_grad():
            for split in ['train', 'validation', 'test']:
                data = torch.tensor(getattr(ctx, 'X_'+split), dtype=torch.float32)
                arrays['probabilities_'+split] = np.stack([
                    np.concatenate([m(chunk).reshape(-1).cpu().numpy() for chunk in data.split(4096)]) for m in models], axis=1)
        _npz(directory/'split_and_predictions.npz', arrays)
        atomic_json(directory/'preprocessing.json', {'feature_names': ctx.columns, 'dataset': ctx.name,
            'mean': ctx.scaler.mean_, 'scale': ctx.scaler.scale_, 'variance': ctx.scaler.var_,
            'training_samples_seen': ctx.scaler.n_samples_seen_,
            'actionable_indices': ctx.actionable_indices, 'actionability_policy': ctx.actionability_policy,
            'epsilon': ctx.epsilon, 'seed': ctx.seed, **extra})
        for i, model in enumerate(models):
            _npz(directory/f'model_{i:04d}.npz', {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
        atomic_json(directory/'models.json', {'models': [_descriptor(m) for m in models],
            'selection_metadata': models.metadata, **extra})
        self.event('seed_prepared', fold=fold, seed=seed, actual_models=len(models),
                   available_instances=len(candidates), artifact=str(directory.relative_to(self.path)))
        return directory

    def save_split(self, ctx):
        directory = self.path/f"seed_{ctx.seed}_fold_{self.scope.get('fold', 0)}"
        directory.mkdir(exist_ok=True)
        _npz(directory/'split.npz', {name: getattr(ctx, name) for name in ['X_train', 'y_train',
            'X_validation', 'y_validation', 'X_test', 'y_test', 'train_ids', 'validation_ids', 'test_ids']})
        atomic_json(directory/'preprocessing.json', {'feature_names': ctx.columns, 'dataset': ctx.name,
            'mean': ctx.scaler.mean_, 'scale': ctx.scaler.scale_, 'variance': ctx.scaler.var_,
            'training_samples_seen': ctx.scaler.n_samples_seen_, 'seed': ctx.seed,
            'actionable_indices': ctx.actionable_indices, 'actionability_policy': ctx.actionability_policy})
        self.event('split_prepared', seed=ctx.seed, artifact=str(directory.relative_to(self.path)))

    def save_arrays(self, relative_path, **arrays):
        _npz(self.path/relative_path, {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
                                      for k, v in arrays.items()})

    def commit(self, record, point=None, models=None, x0=None, constraints=None):
        from . import recourse_core as rc
        row = dict(record, run_id=self.path.name)
        if point is not None:
            directory = self.path/f"seed_{row['seed']}_fold_{row['fold']}"/f"instance_{row['instance_id']}"/_safe(row.get('artifact_key', row['method']))
            directory.mkdir(parents=True, exist_ok=True)
            arrays = {'factual': x0.detach().cpu().numpy(), 'returned_point': point.detach().cpu().numpy()}
            trace = getattr(point, 'recourse_trace', {})
            def flatten(value, key):
                if isinstance(value, dict):
                    for k, v in value.items(): flatten(v, key+'/'+str(k))
                elif isinstance(value, (list, tuple)) and value and isinstance(value[0], dict):
                    for i, v in enumerate(value): flatten(v, key+'/'+str(i))
                elif value is not None:
                    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
                    if array.dtype == object:
                        raise TypeError(f'Object array cannot be saved safely: {key}')
                    arrays[key] = array
            flatten(trace, 'trace')
            points = trace.get('average_points', trace.get('points'))
            if points is not None and len(points):
                points = torch.as_tensor(np.asarray(points), dtype=x0.dtype)
                with torch.no_grad():
                    arrays['evaluation/probabilities'] = np.stack([m(points).reshape(-1).cpu().numpy() for m in models], axis=1)
                arrays['evaluation/losses'] = (arrays['evaluation/probabilities']-1)**2
                arrays['evaluation/costs'] = torch.linalg.vector_norm(points-x0, dim=1).cpu().numpy()
                arrays['evaluation/feasible'] = np.asarray([rc.feasible(p, x0, row.get('budget', float('inf')), constraints) for p in points])
                arrays['evaluation/valid'] = arrays['evaluation/feasible'] & (arrays['evaluation/probabilities'] >= .5).all(axis=1)
            ctx = getattr(models, 'context', None)
            if ctx is not None:
                arrays['factual_original_units'] = ctx.scaler.inverse_transform(arrays['factual'][None])[0]
                arrays['returned_original_units'] = ctx.scaler.inverse_transform(arrays['returned_point'][None])[0]
            if constraints is not None:
                arrays['constraint_lower'], arrays['constraint_upper'] = constraints.lower, constraints.upper
                arrays['immutable_indices'] = np.asarray(constraints.immutable, dtype=int)
            _npz(directory/'trajectory.npz', arrays)
            row['trajectory_file'] = str((directory/'trajectory.npz').relative_to(self.path))
            atomic_json(directory/'outcome.json', row)
        _append(self.path/'instances.jsonl', row)
        self.record_count += 1
        if row.get('status') in {'solver_error', 'model_generation_failed'}:
            self.errors += 1
        self.event('instance_finished', method=row.get('method'), status=row.get('status'), record_count=self.record_count)
        return row

    def finish(self, result=None, error=None):
        if isinstance(result, pd.DataFrame):
            tmp = self.path/'summary.csv.tmp'; result.to_csv(tmp, index=True); os.replace(tmp, self.path/'summary.csv')
        # Keep JSONL canonical; these tables are convenience views for analysis.
        rows = read_records(self.path)
        if rows:
            rs.save_statistics(rows, self.path)
            table = pd.json_normalize(rows)
            table.to_csv(self.path/'instances.csv', index=False)
            if all(name in table.columns for name in ['seed', 'method', 'valid']):
                valid = table[table['valid'].notna()]
                if len(valid):
                    groups = [c for c in ['seed', 'family', 'aggregation', 'budget', 'method'] if c in valid]
                    summary = valid.groupby(groups, dropna=False).agg(
                        instances=('valid', 'size'), validity=('valid', 'mean'),
                        mean_cost=('cost', 'mean'), mean_max_loss=('max_loss', 'mean'),
                        mean_seconds=('elapsed_seconds', 'mean'))
                    summary.to_csv(self.path/'per_seed_summary.csv')
        self.manifest.update(status=('interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else
            'failed' if error else 'completed_with_errors' if self.errors else 'completed'),
            finished_utc=datetime.now(timezone.utc).isoformat(), records=self.record_count,
            event_log_write_seconds=self.event_write_seconds,
            error=None if error is None else f'{type(error).__name__}: {error}')
        atomic_json(self.path/'manifest.json', self.manifest)


def read_records(run_directory):
    """Read completed records; tolerate only a truncated final line after a crash."""
    path = Path(run_directory)/'instances.jsonl'
    if not path.exists(): return []
    raw = path.read_text()
    lines = raw.splitlines(); rows = []
    for i, line in enumerate(lines):
        try: rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines)-1 or raw.endswith('\n'): raise
    return rows


def experiment(function=None, *, settings_factory=None):
    """Apply the selected mode to every nested experiment and save each invocation."""
    if function is None:
        return lambda target: experiment(target, settings_factory=settings_factory)
    signature = inspect.signature(function)
    @wraps(function)
    def wrapped(*args, Quick=True, **kwargs):
        from . import recourse_core as rc
        bound = signature.bind(*args, **kwargs); bound.apply_defaults()
        settings = _ACTIVE_SETTINGS.get() or (settings_factory or experiment_settings)(Quick, bound.arguments.get('seed', 42))
        settings = dict(settings)
        base_seed = int(bound.arguments.get('seed', settings['seeds'][0]))
        settings['seeds'] = list(range(base_seed, base_seed+settings['folds']))
        for key in ['folds', 'T', 'max_instances']:
            if key in signature.parameters: bound.arguments[key] = settings[key]
        config = {k: v for k, v in bound.arguments.items() if k not in {'dataset', 'dataset_data', 'datasets_dict'}}
        config.update(settings)
        parent = current_run()
        root = parent.path/'children' if parent else rc.RESULTS_DIR/'runs'
        run = RunStore(root, function.__name__, config, parent)
        rt = _ACTIVE_RUN.set(run); st = _ACTIVE_SETTINGS.set(settings)
        try:
            data = bound.arguments.get('dataset', bound.arguments.get('dataset_data'))
            if data is not None: run.save_dataset(data[0], data[1])
            result = function(*bound.args, **bound.kwargs)
            if isinstance(result, pd.DataFrame):
                if 'Run Directory' not in result:
                    result['Run Directory'] = str(run.path)
                result['Summary Run Directory'] = str(run.path)
                result['Quick'] = settings['Quick']
                result['Mode'] = settings['mode']
                result['Profile Revision'] = settings.get('profile_revision','legacy')
                result['Notebook Version'] = NOTEBOOK_VERSION
                if 'tree_profile' in settings:
                    result['Tree Profile'] = settings['tree_profile']
                if 'T' not in result:
                    result['T'] = settings['T']
                result['Baseline T'] = settings['T']
                limits = run.manifest.get('iteration_limits', {})
                if 'ORPM T' not in result and 'ORPM_T' in limits:
                    result['ORPM T'] = limits['ORPM_T']
                result['Uncertainty'] = 'pointwise 95% Student-t CI across equally weighted seed means'
                result['CI Caution'] = 'Exploratory: fewer than 5 requested seeds' if settings['folds'] < 5 else 'Check metric-specific completed seed counts and CI status'
                result['Requested Instances Per Seed'] = settings['max_instances']
            if isinstance(result, pd.DataFrame):
                if 'Dataset' not in result and bound.arguments.get('dataset_name') is not None:
                    result['Dataset'] = bound.arguments['dataset_name']
                if 'Budget' not in result and bound.arguments.get('recourse_budget') is not None:
                    result['Budget'] = bound.arguments['recourse_budget']
                if 'Config' not in result and not isinstance(result.index, pd.RangeIndex):
                    result['Config'] = result.index.astype(str)
            run.finish(result)
            if parent:
                parent.errors += bool(run.errors)
                parent.event('child_completed', child_run=str(run.path), status=run.manifest['status'])
            return result
        except BaseException as exc:
            run.finish(error=exc)
            raise
        finally:
            _ACTIVE_RUN.reset(rt); _ACTIVE_SETTINGS.reset(st)
    wrapped.__signature__ = signature.replace(parameters=[*signature.parameters.values(),
        inspect.Parameter('Quick', inspect.Parameter.KEYWORD_ONLY, default=True)])
    return wrapped
