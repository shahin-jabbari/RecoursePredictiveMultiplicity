"""Validation-matched feature-subset variation; no test outcomes guide selection.

The legacy models are paired controls, not a claimed low-diversity optimum.
Candidates omit whole predictor groups (including actionable groups). Each
replacement is within the declared accuracy caliper of its control model and
passes the same admission floor. Unmatched slots retain their control model.
"""
import copy
import hashlib
import json
import re

import numpy as np
import torch

from . import recourse_core as rc
from . import recourse_results as rr

DEFAULT_OPTIONS = dict(accuracy_caliper=.02, candidate_pool_size=32,
                       bank_capacity=8, keep_fractions=[.5, .75])
_BANK_CACHE = {}


def feature_groups(columns):
    """Keep encoded categories of the same original predictor together."""
    groups = {}
    for index, column in enumerate(columns):
        name = str(column)
        match = re.match(r'^(Attribute\d+)_', name)
        group = match.group(1) if match else name.split('=', 1)[0]
        groups.setdefault(group, []).append(index)
    return groups


def feature_mask(groups, actionable_indices, rng, keep_fraction):
    names = list(groups)
    if len(names) < 2:
        raise ValueError('Feature diversity needs at least two predictor groups.')
    actions = set(actionable_indices)
    actionable_groups = [i for i, name in enumerate(names) if actions.intersection(groups[name])]
    if not actionable_groups:
        raise ValueError('Feature diversity requires at least one actionable group.')
    drop_count = min(len(names)-1, max(1, int(round(len(names)*(1-keep_fraction)))))
    # At least one omitted group can otherwise be modified by recourse.
    first = int(rng.choice(actionable_groups))
    dropped = {first}
    for i in rng.permutation(len(names)):
        if len(dropped) >= drop_count:
            break
        proposal = dropped | {int(i)}
        if len(actionable_groups) > 1 and set(actionable_groups).issubset(proposal):
            continue
        dropped = proposal
    mask = np.ones(sum(len(v) for v in groups.values()), dtype=np.float32)
    for i in dropped:
        mask[groups[names[i]]] = 0
    return mask


def probabilities(models, X):
    with torch.no_grad():
        x = torch.as_tensor(X, dtype=torch.float32)
        return np.stack([np.concatenate([m(part).reshape(-1).numpy()
            for part in x.split(4096)]) for m in models], axis=1)


def masks_for(models, d):
    return np.stack([m.feature_mask.detach().numpy() if isinstance(m, rc.FeatureMaskedModel)
        else np.ones(d) for m in models])


def model_hash(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()


def describe_set(models):
    ctx = models.context
    p = probabilities(models, ctx.X_validation)
    masks = masks_for(models, len(ctx.columns))
    pairs = [(i,j) for i in range(len(models)) for j in range(i+1,len(models))]
    groups = list(feature_groups(ctx.columns).values())
    gm = np.array([[bool(mask[g].any()) for g in groups] for mask in masks])
    disagreement = [np.mean((p[:,i]>=.5)!=(p[:,j]>=.5)) for i,j in pairs]
    mask_distance = [1-np.sum(gm[i]&gm[j])/np.sum(gm[i]|gm[j]) for i,j in pairs]
    return dict(validation_mean_accuracy=float(np.mean((p>=.5)==ctx.y_validation[:,None])),
        validation_pairwise_disagreement=float(np.mean(disagreement)) if pairs else 0.,
        validation_probability_sd=float(np.std(p,axis=1).mean()),
        mean_feature_group_jaccard_distance=float(np.mean(mask_distance)) if pairs else 0.,
        distinct_feature_masks=len({tuple(m) for m in masks}),
        models_omitting_actionable_features=int(np.sum((masks[:,ctx.actionable_indices]==0).any(axis=1))),
        mean_predictors_used=float(masks.sum(axis=1).mean()),
        model_hashes=[model_hash(m) for m in models],
        diversity_measurement_split='validation')


def select_matched(reference_probabilities, candidate_probabilities, candidate_masks,
                   y_validation, reference_accuracies, admission_floor, caliper,
                   pinned_first=False):
    """Greedy validation disagreement, fixed order, unique masked candidates.

    A reference fallback is reserved for every slot. There is no relaxed caliper
    or test-based reselection when a masked candidate cannot be matched.
    """
    accuracies = np.mean((candidate_probabilities>=.5)==np.asarray(y_validation)[:,None], axis=0)
    labels = candidate_probabilities>=.5
    chosen, used, selected_labels, selected_masks = [], set(), [], []
    for slot, reference_accuracy in enumerate(reference_accuracies):
        eligible = [i for i,a in enumerate(accuracies)
                    if i not in used and a >= admission_floor-1e-12
                    and abs(a-reference_accuracy) <= caliper+1e-12]
        if pinned_first and slot == 0:
            eligible = []
        def score(i):
            comparisons = selected_labels or [reference_probabilities[:,slot]>=.5]
            disagreement = np.mean([np.mean(labels[:,i]!=v) for v in comparisons])
            mask_distance = np.mean([np.mean(candidate_masks[i]!=v) for v in selected_masks]) if selected_masks else 0.
            return (float(disagreement), float(mask_distance), -abs(accuracies[i]-reference_accuracy), -i)
        winner = max(eligible,key=score) if eligible else None
        chosen.append(winner)
        if winner is None:
            selected_labels.append(reference_probabilities[:,slot]>=.5)
            selected_masks.append(np.ones(candidate_masks.shape[1]))
        else:
            used.add(winner)
            selected_labels.append(labels[:,winner])
            selected_masks.append(candidate_masks[winner])
    return chosen, accuracies


def _cache_key(ctx, kind, capacity, legacy_diverse, epsilon, epochs, lr, options, original_action, pinned_first):
    digest = hashlib.sha256(json.dumps(dict(seed=ctx.seed,kind=kind,capacity=capacity,
        legacy_diverse=legacy_diverse,epsilon=epsilon,epochs=epochs,lr=lr,options=options,
        columns=ctx.columns,actionable=ctx.actionable_indices,original_action=original_action,pinned_first=pinned_first),
        sort_keys=True).encode())
    # Training and validation only. Selection never accesses test labels/outcomes.
    for a in [ctx.X_train,ctx.y_train,ctx.X_validation,ctx.y_validation]:
        digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()


def _bank(X,y,ctx,kind,capacity,legacy_diverse,epsilon,epochs,lr,options,original_action,pinned_first):
    key = _cache_key(ctx,kind,capacity,legacy_diverse,epsilon,epochs,lr,options,original_action,pinned_first)
    if key in _BANK_CACHE:
        return _BANK_CACHE[key]
    counts = [0,0,0]; counts[kind] = capacity
    reference,_ = rc.train_and_select_candidates(X,y,*counts,max_instances=0,seed=ctx.seed,
        epsilon=epsilon,actionable_indices=original_action,diverse=legacy_diverse,
        dataset_name=ctx.name,context=ctx,training_epochs=epochs,training_lr=lr,diversity=False)
    d = len(ctx.columns)
    factories = [lambda s:rc.LogisticRegressionModel(d,s),
                 lambda s:rc.NeuralNetworkModelSingleLayer(d,20,s),
                 lambda s:rc.NeuralNetworkModelDoubleLayers(d,50,100,s)]
    groups = feature_groups(ctx.columns)
    candidates,metadata = [],[]
    print(f'  Diversity bank: {ctx.name}, seed {ctx.seed}, architecture {kind}; '
          f'{options["candidate_pool_size"]} masked candidates',flush=True)
    for attempt in range(options['candidate_pool_size']):
        model_seed = ctx.seed+10000*(kind+1)+attempt
        fraction = options['keep_fractions'][attempt%len(options['keep_fractions'])]
        mask = feature_mask(groups,ctx.actionable_indices,np.random.default_rng(model_seed),fraction)
        model = rc.FeatureMaskedModel(factories[kind](model_seed),mask)
        rc.train_model(model,ctx.X_train,ctx.y_train,epochs=epochs,lr=lr,seed=model_seed)
        candidates.append(model)
        metadata.append(dict(candidate_index=attempt,model_seed=model_seed,keep_fraction=fraction,
            excluded_groups=[name for name,ids in groups.items() if not mask[ids].any()]))
    rp = probabilities(reference,ctx.X_validation)
    cp = probabilities(candidates,ctx.X_validation)
    acc = np.mean((rp>=.5)==ctx.y_validation[:,None],axis=0)
    selected,ca = select_matched(rp,cp,masks_for(candidates,d),ctx.y_validation,acc,
        reference.metadata['baseline_validation_accuracy']-epsilon,options['accuracy_caliper'],
        pinned_first=pinned_first)
    for row,a in zip(metadata,ca):row['validation_accuracy']=float(a)
    bank = dict(key=key,reference=reference,candidates=candidates,selection=selected,
        reference_probabilities=rp,candidate_probabilities=cp,reference_accuracies=acc,
        candidate_accuracies=ca,metadata=metadata,baseline=reference.metadata['baseline_validation_accuracy'],
        pinned_first=pinned_first,artifact=None)
    _BANK_CACHE[key] = bank
    return bank


def _save_bank(bank,ctx,kind,options):
    run = rr.current_run()
    if run is None:return None
    if bank['artifact'] is not None:return bank['artifact']
    path = run.path/f'diversity_banks/seed_{ctx.seed}_kind_{kind}_{bank["key"][:12]}'
    path.mkdir(parents=True,exist_ok=True)
    for label in ['reference','candidates']:
        for i,model in enumerate(bank[label]):
            rr._npz(path/f'{label}_{i:04d}.npz',{k:v.detach().numpy() for k,v in model.state_dict().items()})
        rr.atomic_json(path/f'{label}.json',[rr._descriptor(m) for m in bank[label]])
    rr._npz(path/'validation_predictions.npz',dict(reference=bank['reference_probabilities'],
        candidates=bank['candidate_probabilities'],labels=ctx.y_validation,validation_ids=ctx.validation_ids))
    rr.atomic_json(path/'selection.json',dict(key=bank['key'],options=options,
        candidate_metadata=bank['metadata'],selected_candidate_indices=bank['selection'],
        reference_accuracies=bank['reference_accuracies'],baseline_accuracy=bank['baseline'],
        pinned_first=bank['pinned_first'],
        null_selection='Pinned deployed model or no eligible masked replacement'))
    bank['artifact'] = str(path)
    return str(path)


def train_and_select(X,y,counts,max_instances,seed,epsilon,actionable_indices,
                     legacy_diverse,dataset_name,context,training_epochs,training_lr):
    settings = rr._ACTIVE_SETTINGS.get() or {}
    options = dict(DEFAULT_OPTIONS,**settings.get('diversity_options',{}))
    if not 0 <= options['accuracy_caliper'] <= 1:
        raise ValueError('accuracy_caliper must be in [0,1].')
    if options['candidate_pool_size'] < 1 or options['bank_capacity'] < 1:
        raise ValueError('Diversity bank sizes must be positive.')
    if not options['keep_fractions'] or any(not 0 < f < 1 for f in options['keep_fractions']):
        raise ValueError('keep_fractions must be strictly between zero and one.')
    ctx = context or rc.prepare_split(X,y,seed,dataset_name,actionable_indices)
    ctx.epsilon = epsilon
    epochs = int(settings.get('training_epochs',training_epochs))
    models = rc.ModelSet(context=ctx); reference = rc.ModelSet(context=ctx)
    records,artifacts,baselines = [],[],[]
    for kind,count in enumerate(counts):
        if not count:continue
        capacity = max(options['bank_capacity'],count)
        pinned_first=not any(counts[:kind])
        bank = _bank(X,y,ctx,kind,capacity,legacy_diverse,epsilon,epochs,training_lr,options,actionable_indices,pinned_first)
        artifacts.append(_save_bank(bank,ctx,kind,options));baselines.append(bank['baseline'])
        for slot in range(count):
            index = bank['selection'][slot]
            control = bank['reference'][slot]
            model = control if index is None else bank['candidates'][index]
            old_context=getattr(model,'recourse_context',None)
            model = copy.deepcopy(model,{id(old_context):ctx} if old_context is not None else {})
            model.recourse_context = ctx
            models.append(model);reference.append(control)
            accuracy = bank['reference_accuracies'][slot] if index is None else bank['candidate_accuracies'][index]
            records.append(dict(architecture_kind=kind,slot=slot,masked_candidate_index=index,
                reference_validation_accuracy=float(bank['reference_accuracies'][slot]),
                validation_accuracy=float(accuracy),accuracy_difference=float(accuracy-bank['reference_accuracies'][slot]),
                selection_reason='pinned_deployed_model' if pinned_first and slot==0 else
                    'no_accuracy_matched_masked_candidate' if index is None else 'matched_masked_candidate'))
    if not models:raise ValueError('At least one model is required.')
    # This is exactly the reference used by the matching legacy experiment.
    factuals = rc.select_candidates(models,max_instances,reference_model=reference[0])
    models.metadata = dict(requested_counts=list(counts),actual_count=len(models),
        validation_accuracies=[r['validation_accuracy'] for r in records],
        baseline_validation_accuracy=baselines[0],split_seed=ctx.seed,selection_set='validation',
        diversity=True,diversity_protocol='feature_groups_accuracy_matched_v1',
        diversity_options=options,model_matches=records,bank_artifacts=artifacts,
        masked_replacements=sum(r['selection_reason']=='matched_masked_candidate' for r in records),
        unmatched_fallbacks=sum(r['selection_reason']=='no_accuracy_matched_masked_candidate' for r in records),
        candidate_selection=dict(ctx.candidate_selection),
        cohort_reference='pinned legacy model for reference mode; actual competing set for majority mode',
        reference_model_hashes=[model_hash(m) for m in reference],
        reference_set_diagnostics=describe_set(reference),**describe_set(models))
    if rr.current_run():rr.event('diversity_set_selected',**models.metadata)
    return models,factuals
