"""Fresh tree runs using the frozen solvers and the original model and cohort order."""
from pathlib import Path
import itertools
import time
import numpy as np
from io_tools import append, read, write, setup_source, snapshot, unpack

def inputs(key, run, ds, seed):
    unpack(key, run, f'banks/{ds}/seed_{seed}/')
    original = snapshot(key)
    cfg = read(original / 'protocol.json')
    cfg['absolute_deadline_epoch'] = None
    write(run / 'protocol.json', cfg)
    meta=read(run/'banks'/ds/f'seed_{seed}'/'bank.json')
    write(run/'cohorts.json',{ds:{str(seed):dict(positions=meta['selected_test_positions'],source_ids=meta['selected_source_ids'])}})
    return cfg

def tree_cells(key, run, ds, seed, cap, zero):
    source = setup_source(key)
    cfg = inputs(key, run, ds, seed)
    import common
    if key == 'tree-convergence':
        from fresh_tree_trajectory import trajectory
        meta=read(run/'banks'/ds/f'seed_{seed}'/'bank.json')
        plan=[dict(key=common.cell_key(ds,seed,i,m,b),dataset=ds,seed=seed,person=i,models=m,budget=b)
              for i in range(min(len(meta['selected_test_positions']), cap or len(meta['selected_test_positions'])))
              for b in cfg['positive_budgets'][ds] for m in cfg['model_counts']]
        write(run / 'task_plan.json', dict(cells=plan, expected_trajectories=len(plan)))
        write(run / 'reuse_manifest.json', dict(prefixes={}, complete={}, references={}))
        for i, c in enumerate(plan, 1):
            trajectory(run, ds, c['models'], c['budget'], i, seed, c['person'])
        return
    if key == 'rf-generalization':
        from forest_methods import run_method
    else:
        from methods import run_method
    cohorts = read(run / 'cohorts.json')[ds][str(seed)]
    people = len(cohorts['positions'])
    methods = cfg['methods']
    panels = {}
    if key == 'rf-generalization':
        import sys
        sys.path.insert(0,str(snapshot(key)/'verification_source'))
        from forest_audit import audit_cell
    if key.endswith('generalization'):
        from experiment import evaluation_metrics
        bankdir = run / 'banks' / ds / f'seed_{seed}'
        audit = read(bankdir / 'cohort_audit.json')
        for panel in cfg['panels']:
            folder = bankdir / 'panels' / panel
            panels[panel] = [common.rt.TreeModel.load(folder/'candidates'/f'{i:04d}')
                             for i in read(folder/'admission.json')['selected']]
    budgets = [0] if zero else cfg['budgets'][ds]
    for person, m, b in itertools.product(range(min(people, cap or people)), cfg['model_counts'], budgets):
        cell = dict(key=common.cell_key(ds, seed, person, m, b), dataset=ds, seed=seed, person=person,
                    models=m, budget=b, pilot=False)
        cell_rows = {}
        for method in methods:
            row = run_method(run, cell, method)
            cell_rows[method] = row
            append(run / 'records.jsonl', row)
            if panels:
                _, x, _, _ = common.bank(run, ds, m, seed, person)
                for panel, models in panels.items():
                    pr = common.rt.probabilities(models, row['point'])
                    fp = common.rt.probabilities(models, x)
                    append(run/'evaluation.jsonl', dict(dataset=ds, seed=seed, person=person, models=m,
                        budget=b, method=method, panel=panel, **evaluation_metrics(pr, row['joint_validity'], fp,
                        audit['common_recourse_needed'][person])))
        if key=='rf-generalization' and b>0:
            mm,x,_,_=common.bank(run,ds,m,seed,person)
            factual=dict(point=x.numpy(),probabilities=common.rt.probabilities(mm,x))
            audit_cell(run,cell,cell_rows,factual)
