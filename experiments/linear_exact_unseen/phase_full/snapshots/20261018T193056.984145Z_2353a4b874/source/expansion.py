"""Preserve the verified pilot while expanding only people, seeds and grid."""
from pathlib import Path
import shutil,json,time
import numpy as np
from common import *
from pilot_engine import load_bank,panel

KEYS=['dataset','seed','person','model_count','budget','method']
def record_key(row):return tuple(row[k] for k in KEYS)
def wanted(row,cfg):return row['dataset'] in cfg['datasets'] and row['seed'] in cfg['seeds'] and row['person']<cfg['instances'] and row['model_count'] in cfg['model_sizes'] and row['budget'] in cfg['budgets'][row['dataset']]
def pilot_rows(cfg):return [json.loads(s) for s in (Path(cfg['pilot_run'])/'records.jsonl').read_text().splitlines() if wanted(json.loads(s),cfg)]

def prepare(root,cfg):
    started=time.perf_counter();pilot=Path(cfg['pilot_run']);assert read(pilot/'verification.json')['passed']
    source_inventory=read(Path(cfg['source_run'])/'model_inventory.json');old_inventory=read(pilot/'model_inventory.json');out={};manifest={};copies={};original={}
    # Scientific artifacts are immutable inputs; presentation files are outside this audit.
    paths=[pilot/n for n in ['records.jsonl','verification.json','protocol.json','freeze.json','panels_frozen.json','model_inventory.json','input_manifest.json']]+[p for p in (pilot/'blocks').rglob('*') if p.is_file()]
    for p in paths:original[str(p)]=sha(p)
    for p,h in read(pilot/'input_manifest.json').items():assert sha(p)==h, 'Pilot input changed: '+p
    for p,h in read(pilot/'freeze.json')['source_sha256'].items():assert sha(pilot/p)==h,'Pilot frozen source changed: '+p
    def copy(old,new):
        new.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old,new);h=sha(old);assert sha(new)==h
        copies[str(new.relative_to(root))]=dict(source=str(old),sha256=h)
    for ds in cfg['datasets']:
        for seed in cfg['seeds']:
            key=f'{ds}/{seed}';bank=Path(source_inventory[key]['source_bank']);directory=root/'blocks'/ds/f'seed_{seed}';snapshot=directory/'bank';snapshot.mkdir(parents=True)
            names=['models.json','preprocessing.json','split_and_predictions.npz']+[f'model_{i:04d}.npz' for i in range(24)]
            for name in names:
                manifest[str(bank/name)]=sha(bank/name);shutil.copy2(bank/name,snapshot/name);assert sha(snapshot/name)==manifest[str(bank/name)]
            models,ctx,A,b,meta,prep=load_bank(snapshot);assert ctx.actionability_policy=='immutable'
            X=np.asarray(ctx.X_test,dtype=np.float32).astype(float);logits=X@A.T+b
            eligible=np.flatnonzero(np.logical_and.reduce([(logits[:,:n]<0).sum(1)>=n/2 for n in cfg['cohort_model_sizes']]))
            order=np.random.default_rng(seed+cfg['cohort_order_seed_offset']).permutation(eligible)
            assert len(order)>=cfg['instances'],f'{key}: only {len(order)} eligible people; need {cfg["instances"]}; no cohort relaxation permitted'
            selected=order[:cfg['instances']];cohort=dict(selection='optimization-only common majority rejection',eligible_count=len(eligible),eligible_indices=eligible.tolist(),random_order=order.tolist(),selected_indices=selected.tolist(),selected_test_ids=ctx.test_ids[selected].tolist(),future_20_prefix=order[:20].tolist(),future_20_available=bool(len(order)>=20))
            if key in old_inventory:
                olddir=pilot/'blocks'/ds/f'seed_{seed}';old=read(olddir/'cohort.json');saved=dict(np.load(olddir/'panel.npz',allow_pickle=False))
                assert old['random_order']==cohort['random_order'] and old['selected_indices'][:len(selected)]==cohort['selected_indices'][:len(old['selected_indices'])]
                assert np.array_equal(A,saved['A']) and np.array_equal(b,saved['b'])
                unseen=np.column_stack([saved['unseen_A'],saved['unseen_b']]);audit=read(olddir/'unseen_admission.json');copy(olddir/'unseen_admission.json',directory/'unseen_admission.json')
                for person in range(min(len(old['selected_indices']),cfg['instances'])):
                    for n in cfg['model_sizes']:
                        oldcase=olddir/'cases'/f'person_{person:02d}_n_{n}';case=directory/'cases'/oldcase.name
                        copy(oldcase/'minimum_cost.json',case/'minimum_cost.json')
            else:
                unseen,audit=panel(ctx,A,b,meta,cfg);save(directory/'unseen_admission.json',audit)
            save(directory/'cohort.json',cohort)
            np.savez_compressed(directory/'panel.npz',A=A,b=b,unseen_A=unseen[:,:-1],unseen_b=unseen[:,-1],selected_X=X[selected],selected_indices=selected,test_ids=ctx.test_ids[selected],free=np.asarray(ctx.actionable_indices,dtype=int))
            out[key]=dict(source_bank=str(bank),snapshot=str(snapshot.relative_to(root)),panel_sha256=sha(directory/'panel.npz'),cohort_sha256=sha(directory/'cohort.json'),unseen_admission_sha256=sha(directory/'unseen_admission.json'),models=24,unseen=50,pilot_panel_reused=key in old_inventory)
            print(utc(),'PANEL_FROZEN',key,'eligible',len(eligible),'attempts',audit['candidate_attempts'],'pilot_panel',key in old_inventory,flush=True)
            save(root/'progress.json',dict(status='preparing_panels',panels_completed=len(out),total_panels=cfg['expected_panels'],completed_records=0,expected_records=cfg['expected_records'],updated_utc=utc()))
    reused=[];copy_cases=set();skip=set()
    for oldrow in pilot_rows(cfg):
        ds=oldrow['dataset'];idx=cfg['budgets'][ds].index(oldrow['budget']);case=Path('blocks')/ds/f"seed_{oldrow['seed']}"/'cases'/f"person_{oldrow['person']:02d}_n_{oldrow['model_count']}"/f'budget_{idx}'
        if str(case) not in copy_cases:
            oldcase=pilot/Path(oldrow['point_file']).parent
            for name in ['points.npz','exact_margin.json']:copy(oldcase/name,root/case/name)
            copy_cases.add(str(case))
        row=dict(oldrow,point_file=str(case/'points.npz'),budget_index=idx);reused.append(row);skip.add((ds,oldrow['seed'],oldrow['person'],oldrow['model_count'],oldrow['budget']))
    assert len(reused)==cfg['preserved_pilot_records'],(len(reused),cfg['preserved_pilot_records'])
    save(root/'input_manifest.json',manifest);save(root/'model_inventory.json',out)
    save(root/'pilot_preservation.json',dict(pilot_run=str(pilot),original_sha256=original,copied_files=copies,preserved_records=len(reused),preserved_settings=len(skip),address_fields_changed=['point_file','budget_index']))
    save(root/'panels_frozen.json',dict(frozen_utc=utc(),count=len(out),inventory_sha256=sha(root/'model_inventory.json'),no_new_recourse_yet=True,pilot_panels_reused=sum(v['pilot_panel_reused'] for v in out.values()),preparation_seconds=time.perf_counter()-started))
    return reused,skip
