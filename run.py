"""Portable entry point. Fresh runs require an explicit, new output directory."""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import defaultdict

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'replication'))
from io_tools import CATALOG, read, write, append, setup_source, snapshot, unpack

STUDIES = {
    'non-tree-convergence': 'Linear, neural and mixed model convergence',
    'tree-convergence': 'Decision-tree convergence, round trajectories and gap bounds',
    'non-tree-tradeoff': 'Linear, neural and mixed ORPM, ADV, ROAR, ElliCE, Reference',
    'tree-tradeoff': 'Decision-tree trade-off with native references',
    'tree-literature': 'Decision-tree ORPM, ADV-surrogate, ROAR-surrogate, RobX-constrained, OCEAN-style',
    'linear-cost': 'Linear cost and group disparity, expanded cohort',
    'neural-cost': 'Neural cost bounds and group disparity, bounded cohort',
    'tree-cost': 'Decision-tree cost bounds and group disparity, expanded cohort',
    'linear-generalization': 'Exact-linear benchmarks and same-structure unseen panel',
    'neural-generalization': 'Common-architecture neural benchmarks and unseen panel',
    'dt-generalization': 'Decision-tree unseen refit and bootstrap panels',
    'rf-generalization': 'Random-forest unseen refit and bootstrap panels',
    'linear-retraining': 'Transfer to fresh-bootstrap logistic panels, using newly trained refits',
}

def protocol(name):
    if name=='non-tree-convergence':return read(ROOT/'experiments/convergence_paper/config.json')
    if name=='linear-retraining':return read(ROOT/'experiments/linear_exact_unseen/retraining_transfer/protocol.json')
    if name=='non-tree-tradeoff':return read(snapshot(name)/'config.json')
    if name=='tree-literature':return read(snapshot(name)/'design.json')
    return read(snapshot(name)/'protocol.json')

def study_grid(name):
    cfg=protocol(name)
    seeds=cfg.get('seeds',cfg.get('pilot_seeds'))
    return list(itertools.product(cfg['datasets'],seeds))

def verify():
    entries=read(ROOT/'SHA256.json')
    for rel,expected in entries.items():
        h=hashlib.sha256()
        with (ROOT/rel).open('rb') as f:
            for block in iter(lambda:f.read(4<<20),b''):h.update(block)
        if h.hexdigest()!=expected:raise ValueError('Damaged file '+rel)
    print('Verified',len(entries),'packaged source, raw data and settings files.')

def non_tree(study, out, ds, seed, people, zero):
    import numpy as np
    cfg=protocol(study)
    if study=='non-tree-convergence':
        config=dict(cfg) if ds is None else dict(cfg,datasets=[ds],seeds=[seed],instances=min(people or cfg['instances'],cfg['instances']))
        write(out/'config.json',config)
        source=ROOT/'experiments/convergence_paper'
        import shutil
        from raw_cache import obtain
        for name in config['datasets']:
            for split_seed in config['seeds']:
                rel=Path('blocks')/name/f'seed_{split_seed}'
                shutil.copytree(obtain(study,name,split_seed)/rel,out/rel)
        shutil.copytree(source,out/'source',ignore=shutil.ignore_patterns('*.ipynb','*.json','__pycache__'))
        shutil.copytree(ROOT/'utils',out/'source/utils',ignore=shutil.ignore_patterns('__pycache__'))
        subprocess.run([sys.executable,str(out/'source/convergence_study.py'),'--root',str(out),'--config',str(out/'config.json')],check=True)
        return
    setup_source(study)
    unpack(study,out,f'blocks/{ds}/seed_{seed}/')
    folder=out/'blocks'/ds/f'seed_{seed}'
    if zero:cfg['budgets']={ds:[0]}
    cfg.update(datasets=[ds],seeds=[seed],instances=min(people or cfg['instances'],cfg['instances']))
    if study=='non-tree-tradeoff':
        from fresh_tradeoff import FreshTradeoff
        write(out/'config.json',cfg)
        job=FreshTradeoff(out,cfg)
        job.execute()
        write(out/'prediction_check.json',job.verify())
        return
    if people is not None:
        with np.load(folder/'panel.npz',allow_pickle=False) as z:
            panel={k:(z[k][:people] if k in ['selected_X','selected_indices','test_ids'] else z[k]) for k in z.files}
        np.savez_compressed(folder/'panel.npz',**panel)
    cfg['expected_records']=cfg['instances']*len(cfg['budgets'][ds])*len(cfg['model_sizes'])*len(cfg['methods'])
    write(out/'protocol.json',cfg)
    if study=='linear-generalization':
        import fresh_linear
        fresh_linear.run(out,cfg)
    else:
        import fresh_neural
        fresh_neural.run_settings(out,cfg)

def retraining(out,ds,seed,cap,zero):
    import numpy as np
    tmp=out/'linear'
    call('linear-generalization',tmp,ds,seed,cap,zero)
    cfg=protocol('linear-retraining')
    source=ROOT/'experiments/linear_exact_unseen/retraining_transfer'
    sys.path.insert(0,str(source))
    from runner import TOL
    panels={}
    for name in cfg['panels']:
        if name=='hessian_existing':
            path=tmp/'blocks'/ds/f'seed_{seed}'/'panel.npz'
            with np.load(path,allow_pickle=False) as z:A,b=z['unseen_A'],z['unseen_b']
            order=np.random.default_rng(81000000+seed+{'Synthetic':1,'COMPAS':2,'Give_Me_Some_Credit':3}[ds]).permutation(len(b))
        else:
            prefix=f'panels/{name}/{ds}/seed_{seed}.npz'
            unpack('linear-retraining',out,prefix)
            with np.load(out/prefix,allow_pickle=False) as z:A,b,order=z['A'],z['b'],z['order']
        panels[name]=A[order],b[order]
    for line in (tmp/'records.jsonl').open():
        r=json.loads(line)
        with np.load(tmp/r['point_file'],allow_pickle=False) as points:x=points[r['method']]
        for panel,(A,b) in panels.items():
            logits=A@x+b
            for k in cfg['unseen_panel_sizes']:
                z=logits[:k];ok=z>=-TOL;available=r['available']
                append(out/'evaluation.jsonl',dict(dataset=ds,seed=seed,person=r['person'],test_id=r['test_id'],
                    model_count=r['model_count'],budget=r['budget'],method=r['method'],panel=panel,unseen_panel_size=k,
                    available=available,mean_acceptance=float(ok.mean()) if available else 0.,
                    joint_validity=bool(available and ok.all()),strict_joint_validity=bool(available and (z>=0).all()),
                    worst_loss=float(np.logaddexp(0.,-z).max()),minimum_logit=float(z.min()),boundary_count=int((abs(z)<=TOL).sum())))


def call(study,out,ds,seed,people,zero,upstream=None):
    cmd=[sys.executable,str(ROOT/'run.py'),'bank',study,'--out',str(out),'--dataset',ds,'--seed',str(seed)]
    if people:cmd+=['--people',str(people)]
    if zero:cmd+=['--zero-only']
    if upstream:cmd+=['--orpm-from',str(upstream)]
    subprocess.run(cmd,check=True)

def bank(a):
    from tree import tree_cells
    name,out,ds,seed,cap,zero=a.study,a.out.resolve(),a.dataset,a.seed,a.people,a.zero_only
    if (ds,seed) not in study_grid(name):raise ValueError('Dataset or seed lies outside the frozen study scope')
    if cap is not None and cap<1:raise ValueError('People must be positive')
    if zero and name in ['tree-convergence','non-tree-convergence','tree-cost','linear-cost','neural-cost']:
        raise ValueError('This study needs positive budgets')
    out.mkdir(parents=True,exist_ok=False)
    write(out/'scope.json',dict(study=name,dataset=ds,seed=seed,people_limit=cap,zero_only=zero))
    if name in ['non-tree-convergence','non-tree-tradeoff','linear-generalization','neural-generalization']:
        non_tree(name,out,ds,seed,cap,zero)
    elif name=='tree-literature':
        from literature import run_study
        native=out/'native' if a.orpm_from is None else a.orpm_from.resolve()
        if a.orpm_from is None:call('tree-tradeoff',native,ds,seed,cap,zero)
        elif read(native/'scope.json')['study']!='tree-tradeoff' or not (native/'finished.json').is_file():raise ValueError('Expected a finished native decision-tree replication')
        run_study(out,ds,seed,cap,zero,native)
    elif name.endswith('-cost'):
        from cost import tree_cost,linear_cost,neural_cost
        {'tree-cost':tree_cost,'linear-cost':linear_cost,'neural-cost':neural_cost}[name](out,ds,seed,cap)
    elif name=='linear-retraining':retraining(out,ds,seed,cap,zero)
    else:tree_cells(name,out,ds,seed,cap,zero)
    write(out/'finished.json',dict(study=name,scope='requested dataset and seed',completed=True))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('list');sub.add_parser('verify')
    q=sub.add_parser('prepare',help='train model inputs only, without computing recourse');q.add_argument('study',choices=STUDIES);q.add_argument('--dataset',required=True);q.add_argument('--seed',type=int,required=True)
    for mode in ['run','bank']:
        q=sub.add_parser(mode,help='run the frozen grid' if mode=='run' else 'run one dataset and seed')
        q.add_argument('study',choices=STUDIES);q.add_argument('--out',type=Path,required=True)
        if mode=='bank':
            q.add_argument('--dataset',required=True);q.add_argument('--seed',type=int,required=True)
            q.add_argument('--people',type=int);q.add_argument('--zero-only',action='store_true')
        q.add_argument('--orpm-from',type=Path,help='matching tree-tradeoff output for a tree-literature run or bank')
    a=p.parse_args()
    if getattr(a,'orpm_from',None) is not None and a.study!='tree-literature':p.error('--orpm-from is for tree-literature only')
    if a.command=='list':
        for k,v in STUDIES.items():print(f'{k:23} {v}')
    elif a.command=='verify':verify()
    elif a.command=='prepare':
        if (a.dataset,a.seed) not in study_grid(a.study):raise ValueError('Outside the study design')
        from raw_cache import obtain
        name='tree-tradeoff' if a.study=='tree-literature' else a.study
        print(obtain(name,a.dataset,a.seed))
    elif a.command=='bank':bank(a)
    else:
        a.out.mkdir(parents=True,exist_ok=False)
        if a.study=='non-tree-convergence':
            non_tree(a.study,a.out,None,None,None,False)
        else:
            for ds,seed in study_grid(a.study):
                upstream=a.orpm_from.resolve()/ds/f'seed_{seed}' if a.orpm_from else None
                call(a.study,a.out/ds/f'seed_{seed}',ds,seed,None,False,upstream)
        if a.study=='tree-literature':
            setup_source('tree-literature')
            from comparison_reporting import frame,summaries,paired_summaries,write_csv
            data=frame(json.loads(line) for f in sorted(a.out.glob('*/seed_*/records.jsonl')) for line in f.open())
            summary,seeds=summaries(data)
            write_csv(a.out/'summary_intervals.csv',summary)
            write_csv(a.out/'seed_means.csv',seeds)
            write_csv(a.out/'paired_intervals.csv',paired_summaries(data))
        if a.study.endswith('-cost'):
            from summaries import summarize_cost,summarize_budget_disparity
            summarize_cost(a.out,a.study)
            summarize_budget_disparity(a.out)
        write(a.out/'finished.json',dict(study=a.study,banks=len(study_grid(a.study))))

if __name__=='__main__':
    for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[variable]='1'
    os.environ.update(RECOURSE_V7_PROJECT=str(ROOT),PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg')
    sys.dont_write_bytecode=True
    main()
