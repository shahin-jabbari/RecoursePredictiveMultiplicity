from pathlib import Path
import json, hashlib, shutil
from datetime import datetime,timezone
import numpy as np

BASE=Path('/Users/sj987/Documents/ChatGPT/Code/Predictive_Multiplicity_Recourse/V7/experiments')
LINEAR=BASE/'linear_exact_unseen/phase_full/results/20261018T193056.984145Z_2353a4b874'
COST=BASE/'cost-of-multiplicity/results/20261021T120125Z'
DISPARITY=BASE/'disparity-of-multiplicity/results/20261021T121343Z'

def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(p)

def groups(X,pre,g):
    ix=pre['feature_names'].index(g['feature']);assert ix not in pre['actionable_indices']
    raw=X[:,ix]*pre['scale'][ix]+pre['mean'][ix]
    assert np.max(abs(raw-np.round(raw)))<1e-4
    raw=np.round(raw).astype(int)
    if g['kind']=='coded':
        assert set(np.unique(raw))<=set(g['codes']);first=raw==g['first_code']
    else:first=raw<g['threshold']
    return np.where(first,g['first'],g['second'])

def select(order,labels,target=20):
    selected=list(map(int,order[:20]));counts={g:int(np.sum(labels[selected]==g)) for g in np.unique(labels[order])}
    for i in order[20:]:
        g=labels[i]
        if counts[g]<target:selected.append(int(i));counts[g]+=1
    return np.asarray(selected,dtype=int)

def prepare(root):
    root=Path(root);cfg=read(root/'analysis_protocol.json');source=Path(cfg['original_linear_result'])
    hashes={}
    for initial in [COST,DISPARITY]:
        assert read(initial/'verification.json')['passed']
    # The completed audit already froze the old numerical outputs. Preserve that
    # manifest and independently recheck every input before any new recourse.
    prior=read(COST/'input_sha256.json')
    for name,digest in prior.items():
        assert sha(name)==digest, 'Prior input changed: '+name
        hashes[name]=digest
    shutil.copytree(COST,root/'preserved'/'cost_initial')
    shutil.copytree(DISPARITY,root/'preserved'/'disparity_initial')
    plans=[];inventory={};total_people=total_settings=0
    for ds in cfg['datasets']:
        for seed in cfg['seeds']:
            old=source/'blocks'/ds/f'seed_{seed}';block=root/'blocks'/ds/f'seed_{seed}'
            block.mkdir(parents=True);(block/'cases').mkdir()
            (block/'bank').symlink_to(old/'bank',target_is_directory=True)
            pre=read(old/'bank/preprocessing.json');cohort=read(old/'cohort.json')
            with np.load(old/'panel.npz') as f:p={k:f[k] for k in f.files}
            with np.load(old/'bank/split_and_predictions.npz') as f:X=f['X_test'].astype(np.float32).astype(float);ids=f['test_ids']
            A,b=p['A'],p['b'];logits=X@A.T+b
            eligible=np.flatnonzero(np.logical_and.reduce([(logits[:,:n]<0).sum(1)>=n/2 for n in cfg['model_counts']]))
            order=np.random.default_rng(seed+read(source/'protocol.json')['cohort_order_seed_offset']).permutation(eligible)
            assert cohort['random_order']==order.tolist() and cohort['selected_indices']==order[:20].tolist()
            labs=groups(X,pre,cfg['groups'][ds]) if ds in cfg['groups'] else None
            selected=select(order,labs) if labs is not None else order[:20]
            assert len(set(selected))==len(selected) and selected[:20].tolist()==cohort['selected_indices']
            detail=[]
            if labs is not None:
                for g in [cfg['groups'][ds]['first'],cfg['groups'][ds]['second']]:
                    ec=int(sum(labs[eligible]==g));sc=int(sum(labs[selected]==g))
                    assert sc==min(20,ec)
                    detail.append(dict(group=g,eligible_count=ec,old_count=int(sum(labs[selected[:20]]==g)),selected_count=sc,target=20,shortfall=max(0,20-sc)))
            cohort.update(selected_indices=selected.tolist(),selected_test_ids=ids[selected].tolist(),original_prefix=20,expansion_rule='append saved-order group entries until 20/group or all eligible',group_counts=detail)
            save(block/'cohort.json',cohort)
            p.update(selected_indices=selected,selected_X=X[selected],test_ids=ids[selected])
            np.savez_compressed(block/'panel.npz',**p)
            for person in range(20):
                for n in cfg['model_counts']:
                    name=f'person_{person:02d}_n_{n}';(block/'cases'/name).symlink_to(old/'cases'/name,target_is_directory=True)
            plan=dict(dataset=ds,seed=seed,old_people=20,new_people=len(selected)-20,total_people=len(selected),groups=detail)
            plans.append(plan);total_people+=len(selected);total_settings+=len(selected)*3*len(cfg['budgets'][ds])
            inventory[f'{ds}/{seed}']=dict(source_bank=str(old/'bank'),models=24,training=False,cohort_sha256=sha(block/'cohort.json'),panel_sha256=sha(block/'panel.npz'))
            print(utc(),'COHORT_FROZEN',ds,seed,'new',len(selected)-20,flush=True)
    assert total_people==1833 and total_settings==40590
    count=0
    with open(source/'records.jsonl') as inp,open(root/'records.jsonl','w') as dest,open(root/'preserved'/'orpm_records.jsonl','w') as original:
        for line in inp:
            r=json.loads(line)
            if r['method']=='ORPM':dest.write(line);original.write(line);count+=1
    assert count==26400
    save(root/'cohort_plan.json',plans);save(root/'model_inventory.json',inventory)
    save(root/'source_preservation.json',dict(original_input_sha256=hashes,original_orpm_records=26400,original_people=1200,source_records_sha256=sha(source/'records.jsonl'),copied_reports={str(p.relative_to(root)):sha(p) for p in (root/'preserved').rglob('*') if p.is_file()}))
    frozen={str(p.relative_to(root)):sha(p) for p in (root/'source').rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    for name in ['protocol.json','analysis_protocol.json','PROTOCOL.txt','PROTOCOL.html','cohort_plan.json','model_inventory.json','source_preservation.json']:
        frozen[name]=sha(root/name)
    for info in inventory.values():pass
    for ds in cfg['datasets']:
        for seed in cfg['seeds']:
            for name in ['cohort.json','panel.npz']:
                p=root/'blocks'/ds/f'seed_{seed}'/name;frozen[str(p.relative_to(root))]=sha(p)
    save(root/'freeze.json',dict(frozen_utc=utc(),no_new_recourse_yet=True,scope="production study; labeled implementation validation excluded",sha256=frozen))
    save(root/'preparation_verification.json',dict(passed=True,original_inputs_unchanged=True,all_cohorts_frozen=True,people=total_people,new_people=633,settings=total_settings,new_settings=14190,minimum_cost_solves=1899,positive_margin_solves=12291))
    save(root/'progress.json',dict(status='prepared',preserved_settings=26400,new_completed_settings=0,expected_new_settings=14190,completed_settings=26400,expected_settings=40590,training_required=False,updated_utc=utc()))
