"""Compute the four published adaptations. One surrogate sequence per person and
one native base and RobX search per person and positive budget are shared."""
from pathlib import Path
import time
import pickle
import numpy as np
from io_tools import append, read, write, setup_source, snapshot, unpack

def run_study(run, ds, seed, cap, zero, native_run):
    setup_source('tree-literature')
    import torch
    from utils import recourse_trees as rt, recourse_core as rc
    from adv_component import run_adv, native_probabilities
    from surrogate_sequence import build_sequence, prefix_arrays
    from roar_raw_component import solve_raw_roar
    from roar_output import finalize_roar
    from first_tree_encoding import prepare_native_min_cost
    from native_base_capture import capture_native
    from native_base_output import analytic_factual_base
    from native_failure_output import finalize_native_base, robx_base_contract
    from robx_search_component import run_search
    from first_tree_pair import assemble_pair
    from comparison_rows import evaluate,assemble_baseline,base_row
    from adv_component import digest
    cfg = read(snapshot('tree-literature')/'design.json')
    unpack('tree-tradeoff', run, f'banks/{ds}/seed_{seed}/')
    bdir = run/'banks'/ds/f'seed_{seed}'
    meta = read(bdir/'bank.json')
    models = [rt.TreeModel.load(bdir/'candidates'/f'{i:04d}') for i in meta['selected_candidate_indices'][:24]]
    with np.load(bdir/'split.npz', allow_pickle=False) as split:
        train, test, ids = (split[k].copy() for k in ('X_train','X_test','test_ids'))
    binding=dict(people=[dict(source_test_position=pos,source_person_id=int(ids[pos]),
        immutable_indices=meta['immutable_indices'],LIME_seed=seed+i,RobX_probe_seed=seed+i,
        native_solver_seed=(seed+i)%2000000000) for i,pos in enumerate(meta['selected_test_positions'])])
    budgets = [0] if zero else cfg['budgets'][ds]
    native = {}
    for line in (native_run/'records.jsonl').open():
        import json
        row = json.loads(line)
        if row['method']=='ORPM' and row['dataset']==ds and row['seed']==seed:
            k = (row['person'],row['models'],row['budget'])
            if k in native:raise ValueError('Duplicate ORPM source')
            native[k] = row
    for person, identity in enumerate(binding['people'][:cap]):
        pos = identity['source_test_position']
        assert int(ids[pos]) == identity['source_person_id']
        x = torch.from_numpy(test[pos].copy())
        imm = identity['immutable_indices']
        lo = np.full(len(x),-np.inf); hi = np.full(len(x),np.inf)
        lo[imm] = hi[imm] = x.numpy()[imm]
        con = rc.Constraints(lo,hi,imm,'immutable')
        lime, probe, solver = (identity[k] for k in ('LIME_seed','RobX_probe_seed','native_solver_seed'))
        folder = run/'components'/f'person_{person}'
        folder.mkdir(parents=True)
        def record(name, function, *args):
            start = time.perf_counter()
            result = function(*args)
            seconds = time.perf_counter()-start
            with (folder/(name+'.pickle')).open('xb') as f:pickle.dump(result,f,protocol=4)
            append(run/'runtime.jsonl',dict(dataset=ds,seed=seed,person=person,component=name,seconds=seconds))
            return result
        seq = record('surrogate_sequence',build_sequence,models,x.numpy(),train,lime) if any(b>0 for b in budgets) else None
        for b in budgets:
            if b>0:
                oracle, _ = prepare_native_min_cost(models[0], x, b, con, solver)
                try:
                    if native_probabilities(models[0],x.numpy()[None,:])[0]>=.5:
                        raw=None; base=analytic_factual_base(oracle,solver)
                    else:
                        start=time.perf_counter()
                        oracle.problem.optimize()
                        wall=time.perf_counter()-start
                        append(run/'runtime.jsonl',dict(dataset=ds,seed=seed,person=person,component=f'native_solve_{b}',seconds=wall))
                        raw=record(f'native_raw_{b}',capture_native,oracle,solver,wall)
                        base=finalize_native_base(raw,oracle,solver)
                    with (folder/f'base_{b}.pickle').open('xb') as f:pickle.dump(base,f)
                    contract=robx_base_contract(base,oracle,solver,raw)
                    search=record(f'robx_{b}',run_search,models[0],x,b,train,contract,con,probe)
                    pair=record(f'pair_{b}',assemble_pair,models,x,b,train,con,probe,base,search,oracle,raw)
                finally:oracle.close()
            for m in cfg['models']:
                selected = models[:m]
                task=dict(dataset=ds,seed=seed,person=person,source_person_id=identity['source_person_id'],
                          source_test_position=pos,models=m,budget=b,prospective_pilot=False)
                saved=native[person,m,b]
                assert saved['source_person_id']==identity['source_person_id']
                observed=evaluate(np.asarray(saved['point'],dtype=np.float32),selected,x,b,con)
                assert observed['joint_validity']==saved['joint_validity'] and np.allclose(observed['probabilities'],saved['probabilities'][:m],rtol=0,atol=1e-12)
                append(run/'records.jsonl',base_row(task,'ORPM',observed,dict(source='fresh_native_predecessor'),
                    'fresh_native_predecessor',output_failure=False))
                def save(method,payload,base_output=None):
                    descriptor=dict(semantic_payload_sha256=digest(payload))
                    row=assemble_baseline(task,method,payload,descriptor,selected,x,con,base_output)
                    append(run/'records.jsonl',row)
                if b==0:
                    zero_payload=dict(schema='four-baseline-analytic-zero-v1',models=m,point=x.numpy())
                    for method in ['ADV-surrogate','ROAR-surrogate','RobX-constrained','OCEAN-style']:save(method,zero_payload)
                    continue
                adv=record(f'adv_{b}_{m}',run_adv,selected,x,b,con)
                save('ADV-surrogate',adv)
                a,d=prefix_arrays(seq,models,x.numpy(),train,lime,m)
                roar=record(f'roar_raw_{b}_{m}',solve_raw_roar,a,d,x.numpy(),b,con)
                ro=record(f'roar_{b}_{m}',finalize_roar,roar,a,d,selected,x.numpy(),b,con)
                save('ROAR-surrogate',dict(raw=roar,output=ro))
                for method in ['RobX-constrained','OCEAN-style']:save(method,pair,base)
