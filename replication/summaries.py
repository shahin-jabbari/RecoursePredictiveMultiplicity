"""Pointwise Student-t intervals across independent seed means."""
import json
import numpy as np
import pandas as pd
from scipy.stats import t
from io_tools import read, snapshot

def interval(values):
    v=np.asarray(values,float);n=len(v)
    mean=float(v.mean());half=float(t.ppf(.975,n-1)*v.std(ddof=1)/np.sqrt(n)) if n>1 else np.nan
    return mean,mean-half,mean+half

def summarize_cost(root,study):
    frame=pd.DataFrame([json.loads(line) for f in sorted(root.glob('*/seed_*/costs.jsonl')) for line in f.open()])
    design=read(snapshot('linear-cost')/'analysis_protocol.json')['groups']
    byseed=[]
    names=['mean_single','hardest_single','joint','premium_mean','premium_hardest']
    keys=['dataset','seed','models']
    for (ds,seed,m),bank in frame.groupby(keys):
        for metric in names:
            cols=[metric+'_lower',metric+'_upper']
            groups={}
            for label,group in bank.groupby('group'):
                if group.group_weight.nunique()!=1:raise ValueError('Inconsistent within-seed group weight')
                groups[label]=(group[cols].mean().to_numpy(),len(group),group.group_weight.iloc[0])
            def save(label,values,n):
                byseed.append(dict(dataset=ds,models=m,seed=seed,metric=metric,comparison=label,
                                   lower=values[0],upper=values[1],people=n))
            for label,(values,n,_) in groups.items():
                if label!='all':save(label,values,n)
            weights=np.array([g[2] for g in groups.values()])
            if not np.isclose(weights.sum(),1):raise ValueError('Incomplete group support in full run')
            save('eligible_population',sum(g[0]*g[2] for g in groups.values()),len(bank))
            if ds in design:
                first,second=design[ds]['first'],design[ds]['second']
                if first in groups and second in groups:
                    a,b=groups[first][0],groups[second][0]
                    save(first+' minus '+second,[a[0]-b[1],a[1]-b[0]],groups[first][1]+groups[second][1])
    seeds=pd.DataFrame(byseed);seeds.to_csv(root/'cost_seed_means.csv',index=False)
    rows=[]
    for key,g in seeds.groupby(['dataset','models','metric','comparison']):
        low=interval(g.lower);high=interval(g.upper)
        rows.append(dict(dataset=key[0],models=key[1],metric=key[2],comparison=key[3],people=g.people.sum(),
            seeds=g.seed.nunique(),lower=low[0],lower_ci_low=low[1],lower_ci_high=low[2],
            upper=high[0],upper_ci_low=high[1],upper_ci_high=high[2]))
    pd.DataFrame(rows).to_csv(root/'cost_and_disparity.csv',index=False)

def summarize_budget_disparity(root):
    inputs=[pd.read_csv(f) for f in sorted(root.glob('*/seed_*/person_metrics.csv'))]
    frame=pd.concat(inputs,ignore_index=True)
    design=read(snapshot('linear-cost')/'analysis_protocol.json')['groups']
    groups=frame.groupby(['dataset','models','budget','metric','seed','group'],as_index=False).agg(
        lower=('lower','mean'),upper=('upper','mean'),people=('lower','size'),weight=('group_weight','first'))
    seed_rows=[]
    for keys,g in groups.groupby(['dataset','models','budget','metric','seed']):
        ds,m,b,metric,seed=keys
        def add(label,lo,hi,n):
            seed_rows.append(dict(dataset=ds,models=m,budget=b,metric=metric,seed=seed,comparison=label,lower=lo,upper=hi,people=n))
        w=g.weight/g.weight.sum()
        add('eligible_population',sum(w*g.lower),sum(w*g.upper),g.people.sum())
        if ds not in design:continue
        first,second=design[ds]['first'],design[ds]['second']
        for row in g.itertuples():add(row.group,row.lower,row.upper,row.people)
        rows=g.set_index('group')
        if first in rows.index and second in rows.index:
            a,other=rows.loc[first],rows.loc[second]
            add(first+' minus '+second,a.lower-other.upper,a.upper-other.lower,a.people+other.people)
    seeds=pd.DataFrame(seed_rows)
    seeds.to_csv(root/'budget_seed_means.csv',index=False)
    rows=[]
    keys=['dataset','models','budget','metric','comparison']
    for identity,group in seeds.groupby(keys):
        low=interval(group.lower);high=interval(group.upper)
        rows.append(dict(**dict(zip(keys,identity)),people=group.people.sum(),seeds=len(group),
            lower=low[0],lower_ci_low=low[1],lower_ci_high=low[2],upper=high[0],upper_ci_low=high[1],upper_ci_high=high[2]))
    pd.DataFrame(rows).to_csv(root/'budget_and_disparity.csv',index=False)
