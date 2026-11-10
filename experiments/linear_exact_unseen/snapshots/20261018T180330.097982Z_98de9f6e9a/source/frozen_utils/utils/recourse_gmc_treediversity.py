"""Paired feature-subset decision trees using original V6 tree controls."""
import copy
import hashlib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from . import recourse_trees as rt
from . import recourse_core as rc
from . import recourse_diversity as dv
from . import recourse_results as rr


def model_hash(model):
    h=hashlib.sha256()
    for tree in model.trees:
        for key in ['left','right','feature','threshold','probability']:
            h.update(np.ascontiguousarray(getattr(tree,key)).tobytes())
    return h.hexdigest()


def describe(models):
    c=models.context;p=np.stack([m.predict_probability(c.X_validation) for m in models],axis=1)
    pairs=[(i,j) for i in range(len(models)) for j in range(i+1,len(models))]
    masks=np.asarray([m.metadata.get('feature_mask',np.ones(len(c.columns))) for m in models])
    return dict(validation_mean_accuracy=float(((p>=.5)==c.y_validation[:,None]).mean()),
        validation_pairwise_disagreement=float(np.mean([np.mean((p[:,i]>=.5)!=(p[:,j]>=.5)) for i,j in pairs])),
        mean_predictors_used=float(masks.sum(axis=1).mean()),model_hashes=[model_hash(m) for m in models],
        models_omitting_actionable_features=int((masks[:,c.actionable_indices]==0).any(axis=1).sum()))


def fit_subset(ctx,mask,seed,max_depth=4):
    active=np.flatnonzero(mask)
    est=DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=5,max_features='sqrt',random_state=seed)
    est.fit(ctx.X_train[:,active],ctx.y_train)
    model=rt.TreeModel.from_estimator(est,metadata=dict(seed=seed,feature_mask=mask.tolist(),parameters=est.get_params()))
    # Map split coordinates back to the common full feature space. The existing
    # tree oracle then encodes exactly the same prediction rule as the subset fit.
    for tree in model.trees:
        branch=tree.left!=-1;tree.feature[branch]=active[tree.feature[branch]]
    model.dimension=len(ctx.columns);model.recourse_context=ctx
    np.testing.assert_allclose(model.predict_probability(ctx.X_validation[:200]),
        est.predict_proba(ctx.X_validation[:200,active])[:,list(est.classes_).index(1)],atol=0,rtol=0)
    return model


def matched_set(reference,pool_size=12,caliper=.02,max_depth=4):
    ctx=reference.context;groups=dv.feature_groups(ctx.columns);candidates=[]
    for k in range(pool_size):
        s=ctx.seed+20000+k
        mask=dv.feature_mask(groups,ctx.actionable_indices,np.random.default_rng(s),[.5,.75][k%2])
        model=fit_subset(ctx,mask,s,max_depth);candidates.append(model)
        if rr.current_run():
            fold=rr.current_run().scope.get('fold',0)
            model.save(rr.current_run().path/f'seed_{ctx.seed}_fold_{fold}/masked_pool/model_{k:04d}')
    rp=np.stack([m.predict_probability(ctx.X_validation) for m in reference],axis=1)
    cp=np.stack([m.predict_probability(ctx.X_validation) for m in candidates],axis=1)
    accuracies=((rp>=.5)==ctx.y_validation[:,None]).mean(axis=0)
    selection,ca=dv.select_matched(rp,cp,np.asarray([m.metadata['feature_mask'] for m in candidates]),
        ctx.y_validation,accuracies,reference.metadata['reference_validation_accuracy']-ctx.epsilon,
        caliper,pinned_first=True)
    models=rc.ModelSet(context=ctx);matches=[]
    for i,idx in enumerate(selection):
        source=reference[i] if idx is None else candidates[idx]
        model=copy.deepcopy(source,{id(ctx):ctx});models.append(model)
        accuracy=accuracies[i] if idx is None else ca[idx]
        matches.append(dict(slot=i,candidate_index=idx,reference_validation_accuracy=float(accuracies[i]),
            validation_accuracy=float(accuracy),accuracy_difference=float(accuracy-accuracies[i]),
            selection_reason='pinned_deployed_model' if i==0 else 'no_matched_candidate' if idx is None else 'matched_subset'))
    models.metadata=dict(diversity=True,model_matches=matches,candidate_pool_size=pool_size,
        masked_replacements=sum(i is not None for i in selection),
        unmatched_fallbacks=sum(i is None for i in selection[1:]),
        reference_model_hashes=[model_hash(m) for m in reference],**describe(models))
    assert models.metadata['model_hashes'][0]==models.metadata['reference_model_hashes'][0]
    return models


def save_set(run,models,factuals,seed,fold=0):
    path=run.path/f'seed_{seed}_fold_{fold}';path.mkdir(exist_ok=True)
    run.save_split(models.context)
    for i,m in enumerate(models):m.save(path/'selected_models'/f'model_{i:04d}')
    ctx=models.context
    rr.atomic_json(path/'selection_metadata.json',models.metadata)
    run.save_arrays(f'seed_{seed}_fold_{fold}/paired_tree_predictions.npz',candidate_ids=np.asarray(ctx.candidate_ids),
        candidates=np.asarray([x.numpy() for x in factuals]).reshape(-1,len(ctx.columns)),
        validation=np.stack([m.predict_probability(ctx.X_validation) for m in models],axis=1),
        test=np.stack([m.predict_probability(ctx.X_test) for m in models],axis=1))
