from pathlib import Path
import json,zipfile
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from design import read,save,sha,utc,groups

def verify_trace(f,A,b,x0,free,B,cfg,ds):
    p=f['ORPM'];immutable=np.setdiff1d(np.arange(len(x0)),free)
    assert np.isfinite(p).all() and np.linalg.norm(p-x0)<=B+1e-8
    assert np.array_equal(p[immutable],x0[immutable])
    if not B:
        assert np.array_equal(p,x0);return
    points=f['orpm_points'];weights=f['orpm_weights'];T=cfg['rounds'].get(ds,cfg['rounds']['default'])
    assert len(points)==T and np.allclose(points.mean(0),p,atol=1e-10,rtol=0)
    assert np.all(np.linalg.norm(points-x0,axis=1)<=B+1e-8)
    assert np.array_equal(points[:,immutable],np.broadcast_to(x0[immutable],points[:,immutable].shape))
    losses=np.logaddexp(0.,-(points@A.T+b))
    logw=np.vstack([np.zeros(len(A)),np.cumsum(losses[:-1],axis=0)])*cfg['raw_outer_rate']
    logw-=logsumexp(logw,axis=1)[:,None]
    assert np.allclose(weights,np.exp(logw),atol=1e-9,rtol=1e-8)
    assert np.allclose(losses,f['orpm_losses'],atol=1e-9)

def verify(root):
    root=Path(root);cfg=read(root/'protocol.json');acfg=read(root/'analysis_protocol.json')
    source=Path(acfg['original_linear_result']);frozen=read(root/'freeze.json')['sha256']
    for name,h in frozen.items():assert sha(root/name)==h,'Frozen drift '+name
    preservation=read(root/'source_preservation.json')
    for name,h in preservation['original_input_sha256'].items():assert sha(name)==h,'Original drift '+name
    for name,h in preservation['copied_reports'].items():assert sha(root/name)==h,'Preservation copy drift '+name
    with open(root/'records.jsonl','rb') as allf,open(root/'preserved/orpm_records.jsonl','rb') as oldf:
        for line in oldf:assert allf.readline()==line,'Old record bytes changed'
        tail=allf.read()
    assert tail==(root/'new_records.jsonl').read_bytes()
    expected=set();panels={};plans=read(root/'cohort_plan.json');counts={};groupaudit=[]
    for plan in plans:
        ds,seed=plan['dataset'],plan['seed'];block=root/'blocks'/ds/f'seed_{seed}'
        old=source/'blocks'/ds/f'seed_{seed}'
        with np.load(block/'panel.npz') as f:p={k:f[k] for k in f.files}
        with np.load(old/'panel.npz') as f:
            for name in ['A','b','free']:assert np.array_equal(p[name],f[name])
            assert np.array_equal(p['selected_X'][:20],f['selected_X']) and np.array_equal(p['test_ids'][:20],f['test_ids'])
        panels[ds,seed]=p
        co=read(block/'cohort.json');orig=read(old/'cohort.json')
        assert co['random_order']==orig['random_order'] and co['selected_indices'][:20]==orig['selected_indices']
        # Independent filtered-order construction: union original prefix and
        # first min(20, eligible group size) entries per group, then saved order.
        if ds in acfg['groups']:
            with np.load(old/'bank/split_and_predictions.npz') as f:X=f['X_test'].astype(np.float32).astype(float)
            pre=read(old/'bank/preprocessing.json');labels=groups(X,pre,acfg['groups'][ds]);order=np.array(orig['random_order'])
            chosen=set(order[:20])
            for g in [acfg['groups'][ds]['first'],acfg['groups'][ds]['second']]:chosen.update(order[labels[order]==g][:20])
            selected=list(order[:20])+[int(i) for i in order[20:] if i in chosen]
            assert co['selected_indices']==selected
            for g in [acfg['groups'][ds]['first'],acfg['groups'][ds]['second']]:
                e=int(sum(labels[order]==g));s=int(sum(labels[selected]==g));assert s==min(20,e)
                groupaudit.append(dict(dataset=ds,seed=seed,group=g,eligible=e,selected=s,target=20))
        else:assert len(p['selected_X'])==20
        for person in range(len(p['selected_X'])):
            for n in acfg['model_counts']:
                case=block/'cases'/f'person_{person:02d}_n_{n}'
                if person<20:assert case.is_symlink() and case.resolve()==(old/'cases'/case.name).resolve()
                else:
                    assert not case.is_symlink()
                    for bi in range(len(cfg['budgets'][ds])):expected.add((ds,seed,person,n,bi))
    seen=set();traces=0
    for line in (root/'new_records.jsonl').read_text().splitlines():
        r=json.loads(line);ds,seed,person,n,bi=key=tuple(r[k] for k in ['dataset','seed','person','model_count','budget_index'])
        assert key not in seen and key in expected;seen.add(key)
        p=panels[ds,seed];x0=p['selected_X'][person];A=p['A'][:n];b=p['b'][:n];B=cfg['budgets'][ds][bi]
        assert r['budget']==B and r['test_id']==int(p['test_ids'][person]) and r['method']=='ORPM'
        with np.load(root/r['point_file']) as f:
            verify_trace(f,A,b,x0,p['free'],B,cfg,ds)
            assert np.array_equal(f['origin'],x0)
            assert abs(np.linalg.norm(f['ORPM']-x0)-r['cost'])<1e-9
            assert r['seen_joint_validity']==float(np.all(A@f['ORPM']+b>=1e-7*-1))
        traces+=int(B>0)
    assert seen==expected and len(seen)==14190 and traces==12291
    return dict(passed=True,verified_utc=utc(),original_records_byte_preserved=True,original_inputs_unchanged=True,original_cohort_prefixes_preserved=True,new_cohorts_independently_reconstructed=True,new_orpm_settings=len(seen),new_positive_orpm_traces=traces,new_people=633,combined_people=1833,numerical_cost_margin_singleton_audit='Performed separately in cost_report/verification.json before final completion',group_counts=groupaudit)

def verify_reports(root,droot):
    cost=root/'cost_report';checks={}
    for folder,expected in [(cost,30),(droot,44)]:
        v=read(folder/'verification.json');assert v['passed']
        assert len(list((folder/'figures').glob('*.png')))==expected
        assert len(list((folder/'figures').glob('*.pdf')))==expected
        for name in ['report.txt','report.html','figures_tables.zip']:assert (folder/name).stat().st_size>100
        with zipfile.ZipFile(folder/'figures_tables.zip') as z:
            assert z.testzip() is None
            assert len([n for n in z.namelist() if n.endswith('.png')])==expected
        inferential=list((folder/'tables').glob('*95ci.csv'))
        for p in inferential:
            frame=pd.read_csv(p);assert {'mean','ci_lower','ci_upper','n_seeds','df'}<=set(frame)
            ok=(frame.n_seeds>=2)&np.isfinite(frame['mean'])
            assert frame.loc[ok,['ci_lower','ci_upper']].notna().all().all()
            assert (frame.loc[ok,'ci_lower']<=frame.loc[ok,'mean']+1e-12).all()
            assert (frame.loc[ok,'mean']<=frame.loc[ok,'ci_upper']+1e-12).all()
        checks[str(folder)]=dict(passed=True,separate_panels=expected,ci_tables=len(inferential),archive_valid=True)
    assert read(cost/'verification.json')['singleton_solutions']==43992
    assert read(cost/'verification.json')['person_model_comparisons']==5499
    assert read(cost/'verification.json')['budget_settings']==40590
    # Independent calculation of one primary weighted summary per dataset.
    frame=pd.read_csv(cost/'budget_comparisons.csv');labels=pd.read_csv(cost/'group_labels.csv');counts=pd.read_csv(cost/'tables/group_count_audit.csv')
    computed=pd.read_csv(cost/'tables/success_seed_means.csv')
    for (ds,seed,n,B),z in frame.merge(labels,on=['dataset','seed','person'],validate='many_to_one').groupby(['dataset','seed','model_count','budget']):
        cc=counts[(counts.dataset==ds)&(counts.seed==seed)]
        if cc.empty:want=z.orpm_success.mean()
        else:
            want=sum(float(row.eligible_count/cc.eligible_count.sum())*z.loc[z.group==row.group,'orpm_success'].mean() for row in cc.itertuples())
        got=computed[(computed.dataset==ds)&(computed.seed==seed)&(computed.model_count==n)&(computed.budget==B)].orpm_success.item()
        assert abs(want-got)<1e-12
    checks['independent_population_weights']=True
    return checks
