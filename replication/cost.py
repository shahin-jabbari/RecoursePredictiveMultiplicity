"""Matched singleton and common-action costs. Keep numerical bounds and group sampling weights."""
import numpy as np
from io_tools import read, write, append, snapshot, setup_source, unpack

def cost_row(run, ds, seed, person_id, group, weight, m, horizon, singles, joint, **extra):
    sl, su = np.asarray(singles,float).T
    jl, ju = joint
    if min(sl)<0 or np.any(sl>su+1e-6) or jl>ju+1e-6:
        raise ValueError('Inconsistent cost interval')
    append(run/'costs.jsonl',dict(dataset=ds,seed=seed,source_person_id=int(person_id),group=group,
        group_weight=weight,models=m,horizon=horizon,mean_single_lower=sl.mean(),mean_single_upper=su.mean(),
        hardest_single_lower=sl.max(),hardest_single_upper=su.max(),joint_lower=jl,joint_upper=ju,
        premium_mean_lower=max(0.,jl-su.mean()),premium_mean_upper=max(0.,ju-sl.mean()),
        premium_hardest_lower=max(0.,jl-su.max()),premium_hardest_upper=max(0.,ju-sl.max()),**extra))

def tree_cost(run, ds, seed, cap):
    from tree import inputs
    setup_source('tree-cost')
    cfg=inputs('tree-cost',run,ds,seed)
    from common import bank,cell_key
    from methods import run_method
    from study import singleton,witnesses_and_grid,source_by_case
    from cost_solver import solve_joint,verify_joint
    cases=[c for c in read(run/'banks'/ds/f'seed_{seed}'/'cases.json') if c['dataset']==ds and c['seed']==seed][:cap]
    write(run/'cohort_plan.json',cases)
    (run/'preserved').mkdir()
    for c in cases:
        person=c['person']
        mm,x,con,meta=bank(run,ds,24,seed,person)
        assert c['source_id']==meta['selected_source_ids'][person]
        singles=[singleton(model,x,con) for model in mm]
        write(run/'costs'/c['key']/'singletons.json',singles)
        tau=max(cfg['budgets'][ds])
        # Create fresh budget-grid witnesses for this exact person, including the enlarged group cohort.
        for m in cfg['model_counts']:
            for b in cfg['budgets'][ds]:
                cell=dict(key=cell_key(ds,seed,person,m,b),dataset=ds,seed=seed,person=person,models=m,budget=b,pilot=False)
                for method in ['ORPM','DirectMinimax','UniformTree','SingleTreeReference']:
                    row=run_method(run,cell,method)
                    append(run/'preserved'/'witness_records.jsonl',row)
                    if method=='ORPM':append(run/'records.jsonl',row)
        source_by_case.cache_clear()
        for m in cfg['model_counts']:
            known,grid,lower=witnesses_and_grid(run,c,m)
            out=solve_joint(run,c,m,singles,known,grid,lower)
            verify_joint(run,c,m,singles,out,known,grid,lower)
            write(run/'costs'/c['key']/f'm{m}'/'outcome.json',out)
            capped=[min(tau,item['cost']) if item['cost'] is not None else tau for item in singles[:m]]
            cost_row(run,ds,seed,c['source_id'],c['group'],c['eligible_group_weight'],m,tau,
                     [(v,v) for v in capped],(out['lower'],out['upper']),solver_status=out['status'])

    from analyze_study import person_rows
    metrics,transfer,solver = person_rows(run)
    metrics[metrics.cohort=='expanded'].rename(columns={'eligible_group_weight':'group_weight'}).to_csv(run/'person_metrics.csv',index=False)
    transfer.to_csv(run/'singleton_transfer.csv',index=False)
    solver.to_csv(run/'solver_status.csv',index=False)

def linear_cost(run,ds,seed,cap):
    import sys
    cfg=read(snapshot('linear-cost')/'analysis_protocol.json')
    src=setup_source('linear-cost')
    sys.path.insert(0,str(src/'frozen_linear'))
    from pilot_engine import load_bank,orpm,rc,torch
    from exact_linear import min_l2,environment,max_margin
    from design import groups
    from analyze_cost import singleton
    modelcfg=read(snapshot('linear-generalization')/'protocol.json')
    unpack('linear-cost',run,f'blocks/{ds}/seed_{seed}/')
    folder=run/'blocks'/ds/f'seed_{seed}'
    models,ctx,A,b,meta,pre=load_bank(folder/'bank')
    cohort=read(folder/'cohort.json')
    free=np.asarray(ctx.actionable_indices,dtype=int)
    labels=groups(ctx.X_test,pre,cfg['groups'][ds]) if ds in cfg['groups'] else np.repeat('all',len(ctx.X_test))
    eligible=np.asarray(cohort['eligible_indices'])
    tau=max(cfg['budgets'][ds])
    import pandas as pd
    metrics=[]
    with environment() as env:
        for pos in cohort['selected_indices'][:cap]:
            x=ctx.X_test[pos].astype(float)
            costs=[singleton(a,z,free)[0] for a,z in zip(A,A@x+b)]
            for m in cfg['model_counts']:
                _,info=min_l2(A[:m],b[:m],x,free,modelcfg['solver'],env)
                if info['available']:bounds=(min(tau,info['cost_lower']),min(tau,info['cost_upper']))
                elif info.get('independent_infeasibility'):bounds=(tau,tau)
                else:bounds=(0.,tau)
                group=labels[pos]
                cost_row(run,ds,seed,ctx.test_ids[pos],group,float((labels[eligible]==group).mean()),m,tau,
                    [(min(tau,v),min(tau,v)) for v in costs[:m]],bounds,solver_status=info['status'])
                modelset=rc.ModelSet(models[:m],context=ctx)
                cons=rc.constraints_for(modelset,torch.tensor(x,dtype=torch.float32))
                for budget in cfg['budgets'][ds]:
                    point,_,_=orpm(modelset,x,budget,cons,modelcfg,ds) if budget else (x,None,None)
                    margin,margin_info=max_margin(A[:m],b[:m],x,budget,free,modelcfg['solver'],env)
                    append(run/'budget.jsonl',dict(dataset=ds,seed=seed,source_person_id=int(ctx.test_ids[pos]),
                        group=group,models=m,budget=budget,orpm_point=point,orpm_joint=bool(np.min(A[:m]@point+b[:m])>=0),
                        max_margin_point=margin,max_margin=margin_info))

                    tolerance=cfg['logit_tolerance']
                    thresholds=[singleton(a,z,free,threshold=-tolerance)[0] for a,z in zip(A[:m],(A@x+b)[:m])]
                    separate=np.asarray(thresholds)<=budget+cfg['budget_tolerance']
                    mean,separately=float(separate.mean()),float(separate.all())
                    actual=float(np.min(A[:m]@point+b[:m])>=-tolerance)
                    lower=float(margin_info['margin_lower']>=-tolerance or actual)
                    upper=float(margin_info['margin_upper']>=-tolerance or lower)
                    values={'singleton_success':(mean,mean),'all_separate_success':(separately,separately),
                            'common_success':(lower,upper),'ORPM_success':(actual,actual),
                            'structural_multiplicity_penalty':(mean-upper,mean-lower),
                            'operational_multiplicity_penalty':(mean-actual,mean-actual),
                            'common_action_penalty':(separately-upper,separately-lower),'ORPM_shortfall':(lower-actual,upper-actual)}
                    for metric,(low,high) in values.items():
                        metrics.append(dict(dataset=ds,seed=seed,models=m,person=int(ctx.test_ids[pos]),budget=budget,
                            group=group,group_weight=float((labels[eligible]==group).mean()),metric=metric,lower=low,upper=high))
    pd.DataFrame(metrics).to_csv(run/'person_metrics.csv',index=False)

def neural_cost(run,ds,seed,cap):
    import sys
    setup_source('neural-cost')
    from common import load_net,indices,logits
    from solver import solve,environment
    cfg=read(snapshot('neural-cost')/'protocol.json')
    unpack('neural-cost',run,f'blocks/{ds}/seed_{seed}/')
    block=run/'blocks'/ds/f'seed_{seed}'
    cases=read(block/'cases.json')[:cap]
    write(run/'cohort_plan.json',cases)
    # Fresh same-person ORPM points are inputs to the bound audit. The solver checks them after its native call.
    sys.path.insert(0,str(snapshot('non-tree-tradeoff')/'source'))
    from fresh_tradeoff import FreshTradeoff,subset,rc,torch
    trade_cfg=read(snapshot('non-tree-tradeoff')/'config.json')
    trade_dir=run/'orpm_input';trade_dir.mkdir()
    unpack('non-tree-tradeoff',trade_dir,f'blocks/{ds}/seed_{seed}/')
    study=FreshTradeoff(trade_dir,trade_cfg)
    _,ctx,pool,_=study.block(ds,seed)
    with environment() as env:
        for c in cases:
            import shutil
            shutil.copytree(block/'cases'/c['key'],run/'cases'/c['key'])
            folder=run/'cases'/c['key']
            with np.load(folder/'factual.npz',allow_pickle=False) as f:x,lo,hi=(f[k].copy() for k in ('factual','lower','upper'))
            iid=c['instance_id']; source_index=np.flatnonzero(ctx.test_ids==iid)
            assert len(source_index)==1 and np.array_equal(ctx.X_test[source_index[0]],x)
            nets={i:load_net(run/c['bank']/f'model_{i:04d}.npz') for i in range(24,48)}
            points=[]; actions=[]
            for m in cfg['model_counts']:
                models=subset(pool,'neural',m);cons=rc.constraints_for(models,torch.tensor(x,dtype=torch.float32))
                for b in cfg['budgets'][ds]:
                    d=folder/'orpm'/f'm{m}_b{b:g}';d.mkdir(parents=True)
                    identity=dict(dataset=ds,seed=seed,instance_id=iid,family='neural',models=m)
                    point=study.orpm(d,models,x,b,cons,identity)[0] if b else x
                    points.append(point)
                    accepted=logits([nets[j] for j in indices(m)],point)>=cfg['success_logit_threshold']
                    actions.append(dict(models=m,budget=b,joint_validity=bool(accepted.all()),average_validity=float(accepted.mean()),cost=float(np.linalg.norm(point-x))))
                    append(run/'budget.jsonl',dict(**identity,budget=b,group=c['group'],point=point))
            write(folder/'saved_orpm.json',actions)
            tau=c['maximum_budget']
            def solve_set(name,ids):
                d=folder/'tasks'/name;d.mkdir(parents=True)
                point,info=solve([nets[j] for j in ids],x,tau,c['free'],lo,hi,env,cfg['maximum_optimization_seconds_per_solve'],d,points)
                write(d/'solution.json',dict(point=point,info=info))
                write(d/'result.json',dict(task=name,kind=name.split('_')[0],info=info))
                return info['capped_lower'],info['capped_upper']
            singles={i:solve_set(f'single_{i}',[i]) for i in nets}
            for m in cfg['model_counts']:
                js=indices(m)
                jlo,jhi=solve_set(f'joint_{m}',js)
                endpoints=[(singles[j][0],min(singles[j][1],jhi)) for j in js]
                cost_row(run,ds,seed,iid,c['group'],c['group_weight'],m,tau,endpoints,
                         (max(jlo,max(a for a,_ in endpoints)),jhi))
    import pandas as pd
    from analysis import pilot_rows
    _,curves,solvers=pilot_rows(run)
    pd.DataFrame(curves).to_csv(run/'person_metrics.csv',index=False)
    solvers.to_csv(run/'solver_status.csv',index=False)
