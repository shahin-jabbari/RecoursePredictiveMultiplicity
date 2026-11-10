"""Raw-data preparation for nested decision trees and held-out DT/RF panels."""
from collections import Counter
from pathlib import Path
import hashlib,shutil
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from io_tools import ROOT,setup_source,snapshot,read,write
from raw_cache import obtain

def generate(kind,root,ds,seed):
    key='tree-tradeoff'
    rel=Path('banks')/ds/f'seed_{seed}';b=root/rel
    cfg=read(snapshot(kind)/'protocol.json')
    if kind==key:
        setup_source(key)
        from common import rc,rt
        from native_prediction import native_saved_prediction
        rc.PROJECT_ROOT=ROOT
        X,y,act=rc.load_data_and_normalize(ds);ctx=rc.prepare_split(X,y,seed,ds,act)
        b.mkdir(parents=True);a={k:getattr(ctx,k) for k in ('X_train','y_train','X_validation','y_validation','X_test','y_test','train_ids','validation_ids','test_ids')}
        np.savez_compressed(b/'split.npz',**a,scaler_mean=ctx.scaler.mean_,scaler_scale=ctx.scaler.scale_)
        spec=cfg['primary_family'];entries=[];fps=set();selected=[];models=[];val=[]
        for i in range(spec['candidate_reserve_limit']):
            fitseed=seed+1000+i
            clf=DecisionTreeClassifier(max_depth=spec['max_depth'],min_samples_leaf=spec['min_samples_leaf'],max_features=spec['max_features'],random_state=fitseed).fit(a['X_train'],a['y_train'])
            model=rt.TreeModel.from_estimator(clf);v=clf.predict_proba(a['X_validation'])[:,list(clf.classes_).index(1)]
            np.testing.assert_allclose(v,model.predict_probability(a['X_validation']),rtol=0,atol=1e-12)
            t=model.trees[0];h=hashlib.sha256()
            for arr in (t.left,t.right,t.feature,t.threshold,t.probability):h.update(np.ascontiguousarray(arr).tobytes())
            fp=h.hexdigest();acc=float(((v>=.5)==a['y_validation']).mean())
            entries.append(dict(index=i,fit_seed=fitseed,fingerprint=fp,first_unique=fp not in fps,validation_accuracy=acc,nodes=len(t.left),leaves=int((t.left==-1).sum())))
            model.save(b/'candidates'/f'{i:04d}');models.append(model);fps.add(fp);val.append(v)
            if len(entries)>=spec['reference_accuracy_pool']:
                best=max(z['validation_accuracy'] for z in entries[:spec['reference_accuracy_pool']]);cutoff=best-spec['validation_accuracy_allowance']
                selected=[z['index'] for z in entries if z['first_unique'] and z['validation_accuracy']>=cutoff-1e-12][:24]
                if len(selected)==24:break
        if len(selected)!=24:raise RuntimeError('Too few admissible decision trees')
        test=np.stack([native_saved_prediction(models[i],a['X_test']) for i in selected]);valid=np.stack([val[i] for i in selected])
        for j,i in enumerate(selected):
            m=rt.TreeModel.load(b/'candidates'/f'{i:04d}');np.testing.assert_allclose(m.predict_probability(a['X_test']),test[j],rtol=0,atol=1e-12)
        np.savez_compressed(b/'native_prediction_audit.npz',test=test,validation=valid)
        ok=np.logical_and.reduce([2*(test[:m]<.5).sum(0)>=m for m in cfg['model_counts']])
        order=np.random.default_rng(seed).permutation(len(ok));eligible=order[ok[order]];chosen=eligible[:20]
        if len(chosen)!=20:raise RuntimeError('Too few eligible people')
        write(b/'bank.json',dict(dataset=ds,seed=seed,columns=ctx.columns,immutable_indices=sorted(set(range(len(ctx.columns)))-set(act)),selected_candidate_indices=selected,selected_test_positions=chosen,selected_source_ids=ctx.test_ids[chosen],eligible_random_order=eligible,eligible_count=len(eligible),random_order=order,best_validation_accuracy=best))
        write(b/'admission.json',dict(candidates=entries,selected=selected,best_validation_accuracy=best,minimum_admissible_accuracy=cutoff))
        return
    shutil.copytree(obtain(key,ds,seed)/rel,b)
    if kind=='tree-convergence':return
    meta=read(b/'bank.json')
    if kind=='tree-cost':
        setup_source(kind)
        from study import group_labels,GROUPS,case_key
        labels=group_labels(b,ds);order=meta['eligible_random_order'];chosen=list(order[:5]);prior=5
        if ds in GROUPS:
            for group in (GROUPS[ds]['first'],GROUPS[ds]['second']):
                for pos in order:
                    if sum(labels[z]==group for z in chosen)>=3:break
                    if labels[pos]==group and pos not in chosen:chosen.append(pos)
            prior=len(chosen)
        chosen=(chosen+[i for i in order if i not in chosen])[:20];available=Counter(labels[order])
        with np.load(b/'split.npz') as z:ids=z['test_ids'][chosen]
        meta.update(selected_test_positions=chosen,selected_source_ids=ids);write(b/'bank.json',meta)
        write(b/'cases.json',[dict(key=case_key(ds,seed,i),dataset=ds,seed=seed,person=i,position=pos,source_id=int(ids[i]),group=str(labels[pos]),eligible_group_weight=available[labels[pos]]/len(order),preserved=i<5,reused_cost=i<prior,pilot=seed in (401,406) and i<2) for i,pos in enumerate(chosen)])
        return
    # The final forest implementation dispatches both architectures with the frozen seed rule.
    setup_source('rf-generalization')
    from prepare_unseen import fit_candidate,fingerprint,common_eligibility,independent_prob,native_saved_prediction,rt
    write(root/'protocol.json',cfg)
    with np.load(b/'split.npz') as z:a=dict(z)
    cutoff=read(b/'admission.json')['minimum_admissible_accuracy']
    if kind=='rf-generalization':
        shutil.rmtree(b/'candidates');(b/'native_prediction_audit.npz').unlink()
        entries=[];fps=set();selected=[]
        for i in range(cfg['competing_candidate_limit']):
            m,e=fit_candidate(root,ds,seed,0,i,a,b/'candidates'/f'{i:04d}')
            e['first_unique']=e['fingerprint'] not in fps;fps.add(e['fingerprint']);entries.append(e)
            if i>=95:
                best=max(z['validation_accuracy'] for z in entries[:96]);cutoff=best-.05
                selected=[z['index'] for z in entries if z['first_unique'] and z['validation_accuracy']>=cutoff-1e-12][:24]
                if len(selected)==24:break
        if len(selected)!=24:raise RuntimeError('Insufficient admissible forests')
        write(b/'admission.json',dict(candidates=entries,selected=selected,best_validation_accuracy=best,minimum_admissible_accuracy=cutoff))
        meta.update(selected_candidate_indices=selected,best_validation_accuracy=best,family='random_forest',trees_per_model=10);write(b/'bank.json',meta)
    seen=[rt.TreeModel.load(b/'candidates'/f'{i:04d}') for i in meta['selected_candidate_indices']]
    pos=meta['selected_test_positions'];facts=a['X_test'][pos]
    write(b/'cohort_audit.json',dict(positions=pos,source_ids=a['test_ids'][pos],common_recourse_needed=common_eligibility(seen,facts),seen_factual_probabilities=independent_prob(seen,facts)))
    for stream,name in [(1,'refit'),(2,'bootstrap')]:
        folder=b/'panels'/name;fps={fingerprint(m) for m in seen};selected=[];entries=[];models=[]
        for i in range(cfg['unseen_candidate_limit']):
            m,e=fit_candidate(root,ds,seed,stream,i,a,folder/'candidates'/f'{i:04d}')
            fresh=e['fingerprint'] not in fps;fps.add(e['fingerprint']);e.update(distinct_from_seen_and_prior_candidates=fresh,accepted=bool(fresh and e['validation_accuracy']>=cutoff-1e-12));entries.append(e)
            if e['accepted']:selected.append(i);models.append(m)
            if len(selected)==50:break
        if len(selected)!=50:raise RuntimeError('Insufficient unseen models for '+name)
        write(folder/'admission.json',dict(panel=name,selected=selected,candidates=entries,minimum_admissible_accuracy=cutoff))
        native={k:np.stack([native_saved_prediction(m,a['X_'+k]) for m in models],axis=1) for k in ('test','validation')}
        np.testing.assert_allclose(native['test'],independent_prob(models,a['X_test']),rtol=0,atol=1e-12)
        np.savez_compressed(folder/'native_prediction_audit.npz',**native)
