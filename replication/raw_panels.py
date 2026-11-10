"""Derive unseen panels and disparity samples from the newly fitted competing bank."""
from pathlib import Path
from collections import Counter
import shutil,sys
import numpy as np
from io_tools import ROOT,setup_source,snapshot,read,write
from raw_cache import obtain

def cohort(ctx,z,cfg,seed):
    ok=np.logical_and.reduce([(z[:,:n]<0).sum(1)>=n/2 for n in cfg['cohort_model_sizes']])
    eligible=np.flatnonzero(ok)
    order=np.random.default_rng(seed+cfg['cohort_order_seed_offset']).permutation(eligible)
    if len(order)<cfg['instances']:raise RuntimeError('Insufficient eligible participants')
    return eligible,order,order[:cfg['instances']]

def generate(kind,root,ds,seed):
    if kind not in ('linear-generalization','linear-retraining','linear-cost','neural-cost','neural-generalization'):raise ValueError(kind)
    rel=Path('blocks')/ds/f'seed_{seed}';block=root/rel
    base=obtain('non-tree-tradeoff',ds,seed)
    if kind=='linear-retraining':
        src=next((ROOT/'experiments/linear_exact_unseen/retraining_transfer/phase_pilot/snapshots').glob('*mask_sampling_v11/source'))
        sys.path.insert(0,str(src));import panel_generator
        linear=obtain('linear-generalization',ds,seed)
        block.mkdir(parents=True);shutil.copyfile(linear/rel/'panel.npz',block/'panel.npz')
        panel_generator.FULL=base
        for arm in ('bootstrap_full_feature','bootstrap_matched_mask'):panel_generator.generate_panel(ds,seed,arm,root)
        return
    bank=block/'bank';shutil.copytree(base/rel/'bank',bank)
    if kind=='neural-cost':
        setup_source(kind)
        from common import labels,GROUPS
        cfg=read(snapshot(kind)/'protocol.json');selection=read(base/rel/'selection.json')
        positions=selection['test_positions'];ids=selection['instance_ids'];pre=read(bank/'preprocessing.json')
        with np.load(bank/'split_and_predictions.npz') as z:X=z['X_test'][positions].astype(np.float32).astype(float)
        labs=labels(X,pre,ds);counts=Counter(labs);chosen=[]
        if seed in (401,402,403):
            chosen=([int(np.flatnonzero(labs==GROUPS[ds][k])[0]) for k in ('first','second')] if ds in GROUPS and len(counts)==2 else [0,1])
        if ds in GROUPS:
            for k in ('first','second'):
                group=GROUPS[ds][k]
                for j,label in enumerate(labs):
                    if sum(labs[i]==group for i in chosen)>=2:break
                    if label==group and j not in chosen:chosen.append(j)
        chosen=(chosen+[j for j in range(len(X)) if j not in chosen])[:4]
        cases=[];free=list(map(int,pre['actionable_indices']));imm=sorted(set(range(X.shape[1]))-set(free))
        for j in chosen:
            x=X[j];iid=int(ids[j]);key=f'{ds}_{seed}_{iid}';lo=np.full(len(x),-np.inf);hi=np.full(len(x),np.inf);lo[imm]=hi[imm]=x[imm]
            d=block/'cases'/key;d.mkdir(parents=True)
            np.savez_compressed(d/'factual.npz',factual=x,lower=lo,upper=hi,immutable=imm)
            cases.append(dict(key=key,dataset=ds,seed=seed,instance_id=iid,source_position=j,group=str(labs[j]),group_weight=counts[labs[j]]/len(X),bank=str(rel/'bank'),free=free,maximum_budget=float(max(cfg['budgets'][ds]))))
        write(block/'cases.json',cases)
        return
    if kind.startswith('linear'):
        setup_source('linear-generalization')
        from pilot_engine import load_bank,panel
        cfg=read(snapshot('linear-generalization')/'protocol.json');models,ctx,A,b,meta,pre=load_bank(bank)
        X=ctx.X_test.astype(np.float32).astype(float);eligible,order,selected=cohort(ctx,X@A.T+b,cfg,seed)
        packed=dict(A=A,b=b)
        if kind=='linear-cost':
            sys.path.insert(0,str(snapshot(kind)/'source'))
            from design import groups,select
            groupspec=read(snapshot(kind)/'analysis_protocol.json')['groups']
            selected=select(order,groups(X,pre,groupspec[ds])) if ds in groupspec else order[:20]
        else:
            unseen,audit=panel(ctx,A,b,meta,cfg);write(block/'unseen_admission.json',audit)
            packed.update(unseen_A=unseen[:,:-1],unseen_b=unseen[:,-1])
    else:
        setup_source(kind)
        from models import load_bank,pack,logits
        import engine,torch
        torch.set_num_threads(1)
        cfg=read(snapshot(kind)/'protocol.json');models,ctx,meta,pre=load_bank(bank);seen=pack(models)
        X=ctx.X_test.astype(np.float32).astype(float);eligible,order,selected=cohort(ctx,logits(seen,X),cfg,seed)
        unseen=[];fps={engine.fingerprint(m) for m in models}
        for i in range(cfg['unseen_count']):
            m,rec=engine.train_one(cfg,ds,seed,i,models,ctx,meta,block/'unseen'/f'model_{i:03d}.npz')
            h=rec['functional_sha256']
            if h in fps:raise RuntimeError('Duplicate unseen neural model')
            fps.add(h);unseen.append(m)
        packed={'seen_'+k:v for k,v in seen.items()};packed.update({'unseen_'+k:v for k,v in pack(unseen).items()})
    write(block/'cohort.json',dict(eligible_count=len(eligible),eligible_indices=eligible,random_order=order,selected_indices=selected,selected_test_ids=ctx.test_ids[selected],future_20_prefix=order[:20],future_20_available=len(order)>=20))
    np.savez_compressed(block/'panel.npz',**packed,selected_X=X[selected],selected_indices=selected,test_ids=ctx.test_ids[selected],free=np.asarray(ctx.actionable_indices,int))
