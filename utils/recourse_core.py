"""Recourse implementation. Existing experiment settings remain in the notebook.

Default MSE experiments are heuristic. Use make_convex_loss and
convex_gap_certificate for a separate convex logistic benchmark.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.nn import functional as F
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from . import recourse_results as rr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / 'results'
# Historical target encodings are retained for numerical replication.
# German y=1 is bad credit; Polish orientation has not been independently verified.
FAVORABLE_LABEL = {'Synthetic': 1, 'Polish': 1, 'German': 1, 'COMPAS': 0,
                   'Folktables': 1, 'Give_Me_Some_Credit': 0}
LABEL_PROVENANCE = {'Polish': 'Recorded target y=1; semantic orientation unverified',
                    'German': 'Recorded target y=1 is bad credit; historical replication only',
                    'COMPAS': 'two_year_recid=0 is favorable',
                    'Give_Me_Some_Credit': 'SeriousDlqin2yrs=0 is favorable'}
SCHEMAS = {}
RUN_RECORDS = []


def canonical_name(name):
    key = str(name).lower().replace('_', '').replace(' ', '')
    return {'folktables': 'Folktables', 'givemesomecredit': 'Give_Me_Some_Credit',
            'german': 'German', 'compas': 'COMPAS', 'polish': 'Polish',
            'synthetic': 'Synthetic'}.get(key, str(name))


def favorable_labels(y, name='Synthetic'):
    y = np.asarray(y)
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Expected explicitly encoded binary labels 0/1.')
    return (y == FAVORABLE_LABEL.get(canonical_name(name), 1)).astype(np.int64)


def _frame(X, name):
    name = canonical_name(name)
    if not isinstance(X, pd.DataFrame):
        columns = SCHEMAS.get(name)
        if columns is None:
            if name in {'COMPAS', 'German', 'Polish'}:
                raw, _ = load_dataset(name)
                columns = list(raw.columns)
            elif name == 'Give_Me_Some_Credit':
                df, _, _ = load_give_me_some_credit()
                columns = list(df.drop(columns='SeriousDlqin2yrs').columns)
        if columns is None:
            columns = [f'feature_{i}' for i in range(np.asarray(X).shape[1])]
        if len(columns) != np.asarray(X).shape[1]:
            raise ValueError(f'{name}: data dimensions disagree with feature schema.')
        X = pd.DataFrame(X, columns=columns)
    X = X.copy()
    X.attrs['dataset_name'] = name
    SCHEMAS[name] = list(X.columns)
    return X


def get_data_synthetic(num_samples=1000, seed=0):
    if seed == 0 and num_samples in {2000, 20000}:
        filename = 'Synthetic.csv' if num_samples == 2000 else 'Synthetic_nonlinear.csv'
        df = pd.read_csv(PROJECT_ROOT / 'dataset' / filename, float_precision='round_trip')
        return _frame(df.drop(columns='target'), 'Synthetic'), df['target'].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    n0 = num_samples // 2
    X = np.vstack([rng.multivariate_normal([-3, -3], 5*np.eye(2), n0),
                   rng.multivariate_normal([3, 3], 5*np.eye(2), num_samples-n0)])
    y = np.r_[np.zeros(n0), np.ones(num_samples-n0)].astype(int)
    order = rng.permutation(num_samples)
    return _frame(X[order], 'Synthetic'), y[order]


def immutable_names(name, columns):
    name = canonical_name(name)
    if name == 'COMPAS':
        return [c for c in columns if c == 'age' or c.startswith(('sex', 'race'))]
    if name == 'German':
        # UCI encoded columns: age=13, personal status/sex=9, foreign worker=20,
        # dependents=18. Every encoded column of an immutable group is frozen.
        return [c for c in columns if c in {'Attribute13', 'Attribute18'}
                or c.startswith(('Attribute9_', 'Attribute20_'))]
    names = {'Folktables': {'AGEP', 'SEX', 'RAC1P', 'MAR', 'RELP', 'POBP'},
             'Give_Me_Some_Credit': {'age', 'NumberOfDependents'}}.get(name, set())
    return [c for c in columns if c in names]


def feature_partition(name, columns):
    imm = immutable_names(name, columns)
    return [c for c in columns if c not in imm], imm


def _balanced(X, y, seed=42):
    rng = np.random.default_rng(seed)
    labels, counts = np.unique(y, return_counts=True)
    ids = np.concatenate([rng.choice(np.flatnonzero(y == c), counts.min(), replace=False)
                          for c in labels])
    rng.shuffle(ids)
    return X.iloc[ids].reset_index(drop=True), np.asarray(y)[ids]


def load_dataset(name, seed=42):
    name = canonical_name(name)
    files = {'COMPAS': ('compas.csv', 'two_year_recid'),
             'German': ('germanc.csv', 'y'),
             'Polish': ('polish-companies_clean_uncut.csv', 'y')}
    filename, target = files[name]
    df = pd.read_csv(PROJECT_ROOT/'dataset'/filename)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')].apply(pd.to_numeric, errors='coerce').dropna()
    X = _frame(df.drop(columns=target), name)
    X, y = _balanced(X, df[target].to_numpy(dtype=int), seed)
    return _frame(X, name), y


def load_give_me_some_credit(filepath='GiveMeSomeCredit.csv'):
    path = Path(filepath)
    if not path.is_absolute():
        path = PROJECT_ROOT/'dataset'/path
    df = pd.read_csv(path, index_col=0).dropna()
    features = list(df.drop(columns='SeriousDlqin2yrs').columns)
    SCHEMAS['Give_Me_Some_Credit'] = features
    act, imm = feature_partition('Give_Me_Some_Credit', features)
    return df, act, imm


def load_folktables_acs_income(state='CA', year='2018'):
    if state != 'CA' or str(year) != '2018':
        raise ValueError('The bundled ACSIncome task is California, 2018, one-year ACS.')
    df = pd.read_csv(PROJECT_ROOT/'dataset'/'ACSIncome_CA_2018.csv.gz')
    features = list(df.drop(columns='target').columns)
    SCHEMAS['Folktables'] = features
    act, imm = feature_partition('Folktables', features)
    return df, act, imm


def load_data_and_normalize(dataset_name):
    """Compatibility name: return RAW data. Scaling is fitted within each split."""
    name = canonical_name(dataset_name)
    if name == 'Synthetic':
        X, y = get_data_synthetic(2000)
    elif name in {'COMPAS', 'Polish', 'German'}:
        X, y = load_dataset(name)
    elif name == 'Folktables':
        df, _, _ = load_folktables_acs_income()
        X, y = _frame(df.drop(columns='target'), name), df['target'].to_numpy()
    elif name == 'Give_Me_Some_Credit':
        df, _, _ = load_give_me_some_credit()
        X, y = _frame(df.drop(columns='SeriousDlqin2yrs'), name), df['SeriousDlqin2yrs'].to_numpy()
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')
    act, _ = feature_partition(name, list(X.columns))
    return X, y, [X.columns.get_loc(c) for c in act]


dl = SimpleNamespace(load_compas_notebook=lambda: load_dataset('COMPAS'),
                     load_polish_notebook=lambda: load_dataset('Polish'),
                     load_germanc_notebook=lambda: load_dataset('German'))


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.linear = nn.Linear(input_dim, 1)

    def logits(self, x):
        return self.linear(x)

    def forward(self, x):
        return torch.sigmoid(self.logits(x))


class NeuralNetworkModelSingleLayer(nn.Module):
    def __init__(self, input_dim, hidden_nodes=20, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_nodes), nn.ReLU(),
                                 nn.Linear(hidden_nodes, 1), nn.Sigmoid())

    def logits(self, x):
        return self.net[:-1](x)

    def forward(self, x):
        return torch.sigmoid(self.logits(x))


class NeuralNetworkModelDoubleLayers(nn.Module):
    def __init__(self, input_dim, hidden_nodes1=50, hidden_nodes2=100, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_nodes1), nn.ReLU(),
            nn.Linear(hidden_nodes1, hidden_nodes2), nn.ReLU(), nn.Linear(hidden_nodes2, 1), nn.Sigmoid())

    def logits(self, x):
        return self.net[:-1](x)

    def forward(self, x):
        return torch.sigmoid(self.logits(x))


class FeatureMaskedModel(nn.Module):
    def __init__(self, base_model, feature_mask):
        super().__init__()
        self.base_model = base_model
        self.register_buffer('feature_mask', torch.tensor(feature_mask, dtype=torch.float32))

    def logits(self, x):
        return self.base_model.logits(x*self.feature_mask)

    def forward(self, x):
        return torch.sigmoid(self.logits(x))


_TRAINING_CACHE = {}


def train_model(model, X, y, epochs=100, lr=0.01, seed=None):
    settings = rr._ACTIVE_SETTINGS.get() or {}
    epochs = int(settings.get('training_epochs', epochs))
    if seed is not None:
        torch.manual_seed(seed)
    # Reuse only byte-identical training inputs, initialization and hyperparameters.
    # Test/validation data and recourse outputs never enter this cache.
    digest = hashlib.sha256(repr((type(model).__name__, epochs, lr, seed)).encode())
    for array in [np.asarray(X), np.asarray(y)]:
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    cache_key = digest.hexdigest()
    if cache_key in _TRAINING_CACHE:
        weights, history = _TRAINING_CACHE[cache_key]
        model.load_state_dict(weights); model.eval()
        model.training_settings = {'epochs': epochs, 'lr': lr, 'seed': seed, 'cache_hit': True}
        model.training_loss_history = list(history)
        if rr.current_run(): rr.current_run().save_training(model)
        return model
    if seed is not None:
        torch.manual_seed(seed)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)
    model.train()
    loss_history = []
    for _ in range(epochs):
        optimizer.zero_grad()
        # Same BCE training objective, evaluated stably from logits.
        loss = F.binary_cross_entropy_with_logits(model.logits(xt), yt)
        loss_history.append(loss.item())
        loss.backward()
        optimizer.step()
    model.eval()
    model.training_settings = {'epochs': epochs, 'lr': lr, 'seed': seed}
    with torch.no_grad():
        loss_history.append(F.binary_cross_entropy_with_logits(model.logits(xt), yt).item())
    model.training_loss_history = loss_history
    if len(_TRAINING_CACHE) >= 256:
        _TRAINING_CACHE.pop(next(iter(_TRAINING_CACHE)))
    _TRAINING_CACHE[cache_key] = (copy.deepcopy(model.state_dict()), list(loss_history))
    if rr.current_run():
        rr.current_run().save_training(model)
    return model


def recourse_loss(model, x):
    return ((model(x).reshape(-1)-1)**2).mean()


recourse_loss.kind = 'mse_probability'
recourse_loss.bound = 1.0


def linear_parameters(model):
    if isinstance(model, FeatureMaskedModel):
        a, b = linear_parameters(model.base_model)
        return a*model.feature_mask, b
    if not isinstance(model, LogisticRegressionModel):
        raise TypeError('A linear logit model is required.')
    return model.linear.weight.detach().reshape(-1), model.linear.bias.detach().reshape(())


def is_linear(model):
    try:
        linear_parameters(model)
        return True
    except TypeError:
        return False


@dataclass
class SplitContext:
    name: str
    columns: list
    scaler: StandardScaler
    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_ids: np.ndarray
    validation_ids: np.ndarray
    test_ids: np.ndarray
    seed: int
    actionable_indices: object = None
    epsilon: float = 0.05
    candidate_ids: list = field(default_factory=list)
    actionability_policy: str = 'immutable'


class ModelSet(list):
    def __init__(self, values=(), context=None):
        super().__init__(values)
        self.context = context
        self.metadata = {}


def prepare_split(X, y, seed=42, dataset_name=None, actionable_indices=None):
    name = canonical_name(dataset_name or getattr(X, 'attrs', {}).get('dataset_name', 'Synthetic'))
    X = _frame(X, name)
    y = favorable_labels(y, name)
    if actionable_indices is None:
        actionable, _ = feature_partition(name, list(X.columns))
        actionable_indices = [X.columns.get_loc(c) for c in actionable]
    ids = np.arange(len(y))
    development, test = train_test_split(ids, test_size=0.2, random_state=seed, stratify=y)
    train, validation = train_test_split(development, test_size=0.2, random_state=seed+1,
                                         stratify=y[development])
    scaler = StandardScaler().fit(X.iloc[train])
    data = scaler.transform(X).astype(np.float32)
    context = SplitContext(name, list(X.columns), scaler, data[train], y[train],
        data[validation], y[validation], data[test], y[test], train, validation, test,
        seed, actionable_indices=actionable_indices)
    if rr.current_run(): rr.current_run().save_split(context)
    return context


def _accuracy(model, X, y):
    with torch.no_grad():
        p = model(torch.as_tensor(X, dtype=torch.float32)).reshape(-1)
    return float(((p >= 0.5).numpy() == np.asarray(y)).mean())


def set_model_weights(model, theta):
    with torch.no_grad():
        model.linear.weight.copy_(theta[:-1].reshape(1, -1))
        model.linear.bias.copy_(theta[-1:].reshape(1))


def get_hessian(model, X, y=None, lambda_reg=1e-5):
    X = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    a, b = linear_parameters(model)
    p = torch.sigmoid(X@a.double()+b.double())
    aug = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], dim=1)
    return (aug.T*(p*(1-p)))@aug/len(X)+lambda_reg*torch.eye(aug.shape[1], dtype=X.dtype)


def sample_from_ellipsoid(central_theta, H, epsilon, num_samples, seed=42):
    if epsilon < 0:
        raise ValueError('epsilon must be nonnegative')
    values, vectors = torch.linalg.eigh(H.double())
    if values.min() <= 0:
        raise ValueError('H must be positive definite')
    transform = (vectors/torch.sqrt(values))@vectors.T
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(num_samples):
        direction = rng.normal(size=len(central_theta))
        direction /= np.linalg.norm(direction)
        radius = np.sqrt(2*epsilon)*rng.random()**(1/len(direction))
        result.append((central_theta.double()+transform@torch.tensor(radius*direction)).float())
    return result


def select_candidates(models, max_instances=10, reference_model=None):
    ctx = models.context
    reference_model = reference_model or models[0]
    with torch.no_grad():
        p = reference_model(torch.tensor(ctx.X_test)).reshape(-1).numpy()
    eligible = np.flatnonzero(p < 0.5)
    # Independent RNG: selection does not depend on how many competing models were trained.
    selected = np.random.default_rng(ctx.seed).permutation(eligible)[:max_instances]
    ctx.candidate_ids = ctx.test_ids[selected].tolist()
    return [torch.tensor(ctx.X_test[i]) for i in selected]


def train_and_select_candidates(X, y, num_linear=1, num_nn_single_layer=1,
        num_nn_double_layer=1, max_instances=10, seed=42, epsilon=0.05,
        actionable_indices=None, diverse=False, dataset_name=None, context=None,
        training_epochs=100, training_lr=0.01):
    ctx = context or prepare_split(X, y, seed, dataset_name, actionable_indices)
    ctx.epsilon = epsilon
    d = len(ctx.columns)
    base = train_model(LogisticRegressionModel(d, seed), ctx.X_train, ctx.y_train,
                       epochs=training_epochs, lr=training_lr, seed=seed)
    baseline = _accuracy(base, ctx.X_validation, ctx.y_validation)
    center = torch.cat([base.linear.weight.detach().reshape(-1), base.linear.bias.detach()])
    H = get_hessian(base, ctx.X_train)
    models = ModelSet(context=ctx)
    factories = [lambda s: LogisticRegressionModel(d, s),
                 lambda s: NeuralNetworkModelSingleLayer(d, 20, s),
                 lambda s: NeuralNetworkModelDoubleLayers(d, 50, 100, s)]
    requested = [num_linear, num_nn_single_layer, num_nn_double_layer]
    accuracies = []
    for kind, count in enumerate(requested):
        accepted = 0
        # Separate type streams preserve prefixes as model counts increase.
        for attempt in range(count*20):
            if accepted == count:
                break
            model_seed = seed + kind*100 + attempt
            if kind == 0 and attempt == 0:
                model = copy.deepcopy(base)
            elif kind == 0 and not diverse:
                model = factories[kind](model_seed)
                theta = sample_from_ellipsoid(center, H, epsilon, 1, model_seed)[0]
                set_model_weights(model, theta)
            else:
                model = factories[kind](model_seed)
                mask = np.ones(d)
                if diverse and attempt > 0 and actionable_indices is not None:
                    rng = np.random.default_rng(model_seed)
                    for i in range(d):
                        if i not in actionable_indices and rng.random() < 0.3:
                            mask[i] = 0
                    model = FeatureMaskedModel(model, mask)
                train_model(model, ctx.X_train, ctx.y_train,
                            epochs=training_epochs, lr=training_lr, seed=model_seed)
            model.eval()
            acc = _accuracy(model, ctx.X_validation, ctx.y_validation)
            rr.event('model_admission', architecture_kind=kind, attempt=attempt, model_seed=model_seed,
                validation_accuracy=acc, baseline_validation_accuracy=baseline,
                accepted=baseline-acc <= epsilon,
                training_artifact=getattr(model, 'training_artifact', None),
                sampled_linear_parameters=([*linear_parameters(model)[0].tolist(), float(linear_parameters(model)[1])]
                    if kind == 0 else None))
            if baseline-acc <= epsilon:
                model.recourse_context = ctx
                models.append(model)
                accuracies.append(acc)
                accepted += 1
        if accepted != count:
            raise RuntimeError(f'Insufficient accepted models of type {kind}: {accepted}/{count}; '
                               'requested counts are not silently relabeled.')
    if not models:
        raise ValueError('At least one model is required')
    models.metadata = {'requested_counts': requested, 'actual_count': len(models),
                       'validation_accuracies': accuracies, 'baseline_validation_accuracy': baseline,
                       'split_seed': ctx.seed, 'selection_set': 'validation'}
    return models, select_candidates(models, max_instances)


def train_and_select_candidates_diverse(X, y, num_linear=1, num_nn_single_layer=1,
        num_nn_double_layer=1, max_instances=10, seed=42, epsilon=0.05, actionable_indices=None,
        **kwargs):
    return train_and_select_candidates(X, y, num_linear, num_nn_single_layer,
        num_nn_double_layer, max_instances, seed, epsilon, actionable_indices,
        diverse=True, **kwargs)


def train_and_select_candidates_rashomon(X, y, max_instances=10, seed=42):
    # Preserve the legacy helper's 500 epochs, .01 learning rate, four models,
    # and five repetitions. Its protocol is now explicitly repeated holdout.
    result = None
    for repetition in range(5):
        result = train_and_select_candidates(X, y, 4, 0, 0, max_instances, seed+repetition,
                                            training_epochs=500, training_lr=0.01)
    return result


@dataclass
class Constraints:
    lower: np.ndarray
    upper: np.ndarray
    immutable: list = field(default_factory=list)
    policy: str = 'continuous_l2'


def constraints_for(models, x0, immutable_indices=None, policy=None):
    x = x0.detach().cpu().numpy().astype(float)
    lo, hi = np.full(len(x), -np.inf), np.full(len(x), np.inf)
    ctx = getattr(models, 'context', None)
    if ctx is None and len(models):
        ctx = getattr(models[0], 'recourse_context', None)
    indices = list(immutable_indices or [])
    if ctx is not None and ctx.actionable_indices is not None:
        indices = sorted(set(indices) | (set(range(len(x)))-set(ctx.actionable_indices)))
    policy = policy or (ctx.actionability_policy if ctx is not None else 'immutable')
    if ctx is not None and ctx.name == 'COMPAS' and policy == 'compas_appendix':
        age = ctx.columns.index('age')
        indices = [i for i, c in enumerate(ctx.columns) if c.startswith('sex')]
        lo[age], hi[age] = x[age], x[age]+5.0/ctx.scaler.scale_[age]
    lo[indices], hi[indices] = x[indices], x[indices]
    return Constraints(lo, hi, indices, policy)


def project_l2_ball(x, x0, delta, immutable_indices=None, constraints=None):
    if delta < 0:
        raise ValueError('delta must be nonnegative')
    c = constraints or constraints_for([], x0, immutable_indices)
    diff = x-x0
    lo = torch.as_tensor(c.lower, device=x.device, dtype=x.dtype)-x0
    hi = torch.as_tensor(c.upper, device=x.device, dtype=x.dtype)-x0
    if torch.any(lo > 0) or torch.any(hi < 0):
        raise ValueError('The constraint box must contain x0')
    clipped = torch.maximum(lo, torch.minimum(hi, diff))
    if delta == 0:
        return x0.clone()
    if torch.linalg.vector_norm(clipped) <= delta:
        return x0+clipped
    # Exact Euclidean projection onto a box intersected with a centered L2 ball.
    left, right = 0.0, max(1.0, float(torch.linalg.vector_norm(diff)/delta))
    for _ in range(60):
        multiplier = (left+right)/2
        candidate = torch.maximum(lo, torch.minimum(hi, diff/(1+multiplier)))
        if torch.linalg.vector_norm(candidate) > delta:
            left = multiplier
        else:
            right = multiplier
    return x0+torch.maximum(lo, torch.minimum(hi, diff/(1+right)))


def feasible(x, x0, budget, constraints=None, atol=1e-5):
    x = x.detach().cpu().numpy()
    original = x0.detach().cpu().numpy()
    c = constraints or constraints_for([], x0)
    return bool(np.isfinite(x).all() and np.linalg.norm(x-original) <= budget+atol
                and np.all(x >= c.lower-atol) and np.all(x <= c.upper+atol))


def valid_recourse(x, models, x0=None, budget=None, constraints=None):
    if x0 is not None and not feasible(x, x0, budget, constraints):
        return False
    with torch.no_grad():
        probabilities = torch.stack([m(x).reshape(()) for m in models])
    return bool(torch.isfinite(probabilities).all() and (probabilities >= 0.5).all())


def _tag(x, **info):
    x.recourse_info = info
    return x


def _linear_support(a, x0, delta, c):
    """Maximize a.x over the action set (analytic without directional bounds)."""
    a = np.asarray(a, dtype=float).copy()
    a[c.immutable] = 0
    start = x0.detach().cpu().numpy().astype(float)
    norm = np.linalg.norm(a)
    if norm == 0 or delta == 0:
        return start, True
    candidate = start+delta*a/norm
    if np.all(candidate >= c.lower) and np.all(candidate <= c.upper):
        return candidate, True
    result = minimize(lambda x: -float(a@x), start, jac=lambda x: -a,
        bounds=list(zip(c.lower, c.upper)), method='SLSQP',
        constraints=[{'type': 'ineq', 'fun': lambda x: delta**2-np.sum((x-start)**2),
                      'jac': lambda x: -2*(x-start)}], options={'ftol': 1e-12, 'maxiter': 200})
    # Numerical solver success is not an analytic global certificate.
    return result.x, False


def make_convex_loss(models, x0, delta):
    if not all(is_linear(m) for m in models):
        raise TypeError('The convex benchmark requires affine logits for every model')
    bounds = []
    for m in models:
        a, b = linear_parameters(m)
        bounds.append(F.softplus(-a@x0-b+delta*torch.linalg.vector_norm(a)).item())
    scale = max(max(bounds), np.finfo(float).eps)
    def loss(model, x):
        return F.softplus(-model.logits(x)).mean()/scale
    loss.kind, loss.bound, loss.normalizer = 'normalized_logit_bce', 1.0, scale
    return loss


def convex_lower_bound(loss, weights, models, x, x0, delta, c):
    z = x.detach().clone().requires_grad_(True)
    value = sum(float(w)*loss(m, z) for w, m in zip(weights, models))
    grad = torch.autograd.grad(value, z)[0]
    grad[c.immutable] = 0
    # Supporting hyperplane minimized over a superset of the box-constrained ball.
    lower = value.item()+float(grad@(x0-z))-delta*float(torch.linalg.vector_norm(grad))
    return max(0.0, lower), value.item()


def oracle_gradient_descent(weights, models, x0, delta, loss_func, lr=0.01,
        tol=1e-4, num_steps=200, cost_func=None, immutable_indices=None,
        constraints=None, initial_x=None, return_info=False):
    settings = rr._ACTIVE_SETTINGS.get() or {}
    num_steps = min(num_steps, int(settings.get('inner_oracle_steps', num_steps)))
    if cost_func is not None:
        raise NotImplementedError('This oracle implements L2 cost only; arbitrary costs need a projector.')
    if num_steps < 1:
        raise ValueError('num_steps must be positive')
    w = np.asarray(weights, dtype=float)
    if len(w) != len(models) or not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1):
        raise ValueError('weights must be a probability vector matching models')
    c = constraints or constraints_for(models, x0, immutable_indices)
    kind = getattr(loss_func, 'kind', None)
    convex = kind == 'normalized_logit_bce' and all(is_linear(m) for m in models)
    if len(models) == 1 and is_linear(models[0]) and kind in {'mse_probability', 'normalized_logit_bce'}:
        a, _ = linear_parameters(models[0])
        point, success = _linear_support(a.cpu().numpy(), x0, delta, c)
        x = torch.as_tensor(point, device=x0.device, dtype=x0.dtype)
        info = {'status': 'linear_monotone_solution' if success else 'solver_incomplete',
                'global_certificate': success and c.policy != 'compas_appendix',
                'oracle_error_upper_bound': 0.0 if success and c.policy != 'compas_appendix' else None,
                'gradient_evaluations': 0, 'inner_steps': 0, 'convex_loss': convex}
        x = _tag(x, **info)
        rr.attach_trace(x, points=x.detach().cpu().numpy()[None],
                        objectives=np.asarray([sum(float(wi)*loss_func(m, x).item() for wi, m in zip(w, models))]))
        return (x, info) if return_info else x
    x = project_l2_ball(initial_x if initial_x is not None else x0, x0, delta, constraints=c)
    x = x.detach().clone().requires_grad_(True)
    optimizer = optim.Adam([x], lr=lr)
    best, best_value = x.detach().clone(), float('inf')
    status, steps = 'step_limit', 0
    trace_points, trace_values, trace_gradients, trace_stationarity = [], [], [], []
    for step in range(num_steps):
        value = sum(float(wi)*loss_func(m, x) for wi, m in zip(w, models))
        if not torch.isfinite(value):
            raise FloatingPointError('Non-finite oracle objective')
        if value.item() < best_value:
            best, best_value = x.detach().clone(), value.item()
        grad = torch.autograd.grad(value, x)[0]
        if not torch.isfinite(grad).all():
            raise FloatingPointError('Non-finite oracle gradient')
        # A step or gradient tolerance is only a local stopping diagnostic.
        projected = project_l2_ball(x.detach()-grad, x0, delta, constraints=c)
        stationarity = float(torch.linalg.vector_norm(x.detach()-projected))
        if rr.current_run():
            trace_points.append(x.detach().cpu().numpy().copy())
            trace_values.append(value.item())
            trace_gradients.append(float(torch.linalg.vector_norm(grad)))
            trace_stationarity.append(stationarity)
        if tol is not None and stationarity < tol:
            status = 'local_stationarity_only'
            if convex:
                lower, upper = convex_lower_bound(loss_func, w, models, x, x0, delta, c)
                if upper-lower <= tol:
                    status = 'convex_oracle_error_certified'
                    break
            # Do not stop a nonconvex search merely because sigmoid gradients are small.
        optimizer.zero_grad()
        x.grad = grad
        optimizer.step()
        with torch.no_grad():
            x.copy_(project_l2_ball(x, x0, delta, constraints=c))
        steps += 1
    final_value = sum(float(wi)*loss_func(m, x).item() for wi, m in zip(w, models))
    if final_value < best_value:
        best, best_value = x.detach().clone(), final_value
    lower = 0.0
    if convex:
        lower, _ = convex_lower_bound(loss_func, w, models, best, x0, delta, c)
    info = {'status': status, 'global_certificate': convex,
            'oracle_error_upper_bound': max(0.0, best_value-lower),
            'bound_type': 'convex_supporting_plane' if convex else 'trivial_nonnegative_loss_bound',
            'gradient_evaluations': (step+1)*len(models), 'inner_steps': steps,
            'convex_loss': convex}
    best = _tag(best, **info)
    rr.attach_trace(best, points=np.asarray(trace_points), objectives=np.asarray(trace_values),
        gradient_norms=np.asarray(trace_gradients), stationarity=np.asarray(trace_stationarity),
        final_iterate=x.detach().cpu().numpy(), final_objective=final_value)
    return (best, info) if return_info else best


def solve_robust_recourse_with_oracle(x0, models, delta, loss_func, oracle_func,
        T=100, eta=None, cost_func=None, B=1, return_diagnostics=False, constraints=None):
    event_time_start = rr.current_run().event_write_seconds if rr.current_run() else 0.0
    if T < 1 or len(models) == 0 or B <= 0 or delta < 0:
        raise ValueError('Require T>=1, nonempty models, B>0 and delta>=0')
    if cost_func is not None:
        raise NotImplementedError('Only L2 cost is implemented')
    n = len(models)
    eta = np.sqrt(np.log(n)/T)/B if eta is None else eta
    logw = np.full(n, -np.log(n))
    x_sum = torch.zeros_like(x0)
    history, weight_history, updates, infos = [], [], [], []
    warm = None
    responses, loss_history, oracle_traces = [], [], []
    c = constraints or constraints_for(models, x0)
    for _ in range(T):
        w = np.exp(logw)
        weight_history.append(w.copy())
        if oracle_func is oracle_gradient_descent:
            xt = oracle_func(w, models, x0, delta, loss_func, initial_x=warm, constraints=c)
        else:
            xt = oracle_func(w, models, x0, delta, loss_func)
        if not feasible(xt, x0, delta, c):
            raise ValueError('Oracle returned a non-finite or infeasible recourse')
        warm = xt.detach()
        infos.append(getattr(xt, 'recourse_info', {'status': 'external_oracle_unverified'}))
        x_sum += xt.detach()
        history.append((x_sum/(len(history)+1)).clone())
        losses = np.array([loss_func(m, xt).item() for m in models])
        if not np.isfinite(losses).all() or (losses < -1e-7).any() or (losses > B+1e-6).any():
            raise ValueError('Observed losses violate the supplied bound B')
        updates.append(eta*losses)
        logw += eta*losses
        maximum = logw.max()
        logw -= maximum + np.log(np.exp(logw-maximum).sum())
        if rr.current_run():
            responses.append(xt.detach().cpu().numpy().copy())
            loss_history.append(losses.copy())
            oracle_traces.append(getattr(xt, 'recourse_trace', {}))
            rr.event('orpm_round', round=len(history), point=xt, average_point=history[-1],
                weights=w, next_weights=np.exp(logw), losses=losses, oracle=infos[-1])
    xbar, wbar = x_sum/T, np.mean(weight_history, axis=0)
    convex = getattr(loss_func, 'kind', '') == 'normalized_logit_bce' and all(is_linear(m) for m in models)
    info = {'status': 'convex_run' if convex else 'nonconvex_heuristic',
            'oracle_diagnostics': infos, 'last_weights': np.exp(logw).tolist(),
            'gradient_evaluations': sum(i.get('gradient_evaluations', 0) for i in infos),
            'rounds': T, 'convex_loss': convex}
    info['round_log_write_seconds'] = rr.current_run().event_write_seconds-event_time_start if rr.current_run() else 0.0
    xbar = _tag(xbar, **info)
    rr.attach_trace(xbar, points=np.asarray(responses),
        average_points=np.asarray([x.detach().cpu().numpy() for x in history]),
        weights=np.asarray(weight_history), losses=np.asarray(loss_history),
        updates=np.asarray(updates), average_weights=wbar, final_weights=np.exp(logw),
        oracle_traces=oracle_traces)
    result = (xbar, wbar, history, weight_history, updates)
    return result+(info,) if return_diagnostics else result


def convex_gap_certificate(xbar, wbar, models, x0, delta, loss_func, num_steps=200):
    if getattr(loss_func, 'kind', '') != 'normalized_logit_bce' or not all(is_linear(m) for m in models):
        raise ValueError('A convex gap certificate requires normalized affine-logit BCE')
    c = constraints_for(models, x0)
    if not feasible(xbar, x0, delta, c):
        raise ValueError('xbar must be feasible')
    response = oracle_gradient_descent(wbar, models, x0, delta, loss_func,
                                      num_steps=num_steps, initial_x=xbar, constraints=c)
    lower, upper = convex_lower_bound(loss_func, wbar, models, response, x0, delta, c)
    primal = max(loss_func(m, xbar).item() for m in models)
    return {'primal': primal, 'dual_lower_bound': lower, 'dual_upper_bound': upper,
            'gap_lower_bound': max(0.0, primal-upper),
            'gap_upper_bound': max(0.0, primal-lower), 'certificate': 'convex_supporting_plane'}


def run_convex_convergence_benchmark(x0, models, delta, T=100):
    loss = make_convex_loss(models, x0, delta)
    xb, wb, xhist, whist, _ = solve_robust_recourse_with_oracle(
        x0, models, delta, loss, oracle_gradient_descent, T=T, B=1)
    rows = []
    for t, x in enumerate(xhist, 1):
        row = convex_gap_certificate(x, np.mean(whist[:t], axis=0), models, x0, delta, loss)
        row['round'] = t
        rows.append(row)
    return pd.DataFrame(rows)
