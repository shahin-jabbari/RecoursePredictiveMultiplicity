from pathlib import Path
import numpy as np
import pandas as pd
from analyze_cost import ci_table, draw

KEYS=['dataset','seed','person']

def weighted_aggregate(frame, labels, counts, keys, metrics, out, stem):
    merged=frame.merge(labels[KEYS+['group']],on=KEYS,validate='many_to_one')
    means=merged.groupby(keys+['seed','group'],dropna=False)[metrics].mean().reset_index()
    result=[]
    for identity, z in means.groupby(keys+['seed'],dropna=False):
        entry=dict(zip(keys+['seed'],identity))
        c=counts[(counts.dataset==entry['dataset']) & (counts.seed==entry['seed'])]
        if c.empty:
            assert len(z)==1
            entry.update(z.iloc[0][metrics].to_dict())
        else:
            w=c.set_index('group').eligible_count.astype(float)
            w=w/w.sum()
            groupmeans=z.set_index('group')[metrics].reindex(w.index)
            for metric in metrics:
                v=groupmeans[metric].to_numpy(float)
                entry[metric]=float(w.to_numpy()@v) if np.isfinite(v).all() else np.nan
        result.append(entry)
    seeds=pd.DataFrame(result)
    seeds.to_csv(out/'tables'/f'{stem}_seed_means.csv',index=False)
    ci=ci_table(seeds,keys,metrics)
    ci.to_csv(out/'tables'/f'{stem}_95ci.csv',index=False)
    return seeds,ci

def preserve_comparison(out,cfg):
    source=Path(cfg['preserved_cost_result'])
    for name in ['singleton_costs.csv','cost_comparisons.csv','budget_comparisons.csv','group_labels.csv']:
        old=pd.read_csv(source/name)
        combined=pd.read_csv(out/name)
        keys=KEYS+(['model'] if name.startswith('singleton') else ['model_count','budget_index'] if name.startswith('budget') else ['model_count'] if name.startswith('cost') else [])
        keep=combined[combined.person<20].sort_values(keys).reset_index(drop=True)
        old=old.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(keep,old,check_dtype=False,rtol=1e-9,atol=1e-9)
        old.to_csv(out/'tables'/('original_'+name),index=False)
    # Report original cohort unchanged; a paired phase contrast uses the SAME
    # eligible-group weights on both phases, dropping unmatched groups per seed.
    labels=pd.read_csv(source/'group_labels.csv')
    counts=pd.read_csv(out/'tables/group_count_audit.csv')
    for stem,filename in [('success','budget_comparisons.csv'),('cost','cost_comparisons.csv')]:
        data=pd.read_csv(source/filename)
        keys=['dataset','model_count']+(['budget'] if stem=='success' else [])
        ignore=keys+['seed','person','budget_index','orpm_returned_cost','tau','joint_cost_lower','joint_cost_upper','single_mean_cost','single_max_cost','rejected_model_count']
        metrics=[x for x in data if x not in ignore]
        weighted_aggregate(data,labels,counts,keys,metrics,out,'original_reweighted_'+stem)
        for suffix in ['seed_means','95ci']:
            p=source/'tables'/f'{stem}_{suffix}.csv'
            (out/'tables'/f'original_unweighted_{stem}_{suffix}.csv').write_bytes(p.read_bytes())

def bounded_difference(first,second,metrics):
    diff=first[metrics].copy()
    for m in metrics:
        other=m
        if m.endswith('_lower') and m[:-6]+'_upper' in metrics:other=m[:-6]+'_upper'
        if m.endswith('_upper') and m[:-6]+'_lower' in metrics:other=m[:-6]+'_lower'
        diff[m]=first[m]-second[other]
    return diff

def paired_effects(out,stem,keys,metrics):
    seeds=pd.read_csv(out/'tables'/f'{stem}_seed_means.csv')
    original=pd.read_csv(out/'tables'/f'original_reweighted_{stem}_seed_means.csv')
    idx=keys+['seed']
    a=seeds.set_index(idx).sort_index();b=original.set_index(idx).reindex(a.index)
    diff=bounded_difference(a,b,metrics).reset_index()
    diff.to_csv(out/'tables'/f'{stem}_phase_paired_seed_changes.csv',index=False)
    ci_table(diff,keys,metrics).to_csv(out/'tables'/f'{stem}_phase_paired_changes_95ci.csv',index=False)
    # Positive high-minus-low budgets and nested-count paired effects.
    if stem=='success':
        positives=seeds[seeds.budget>0]
        low=positives.groupby(['dataset','seed','model_count']).budget.transform('min')
        high=positives.groupby(['dataset','seed','model_count']).budget.transform('max')
        kk=['dataset','model_count','seed']
        x=positives[positives.budget==high].set_index(kk).sort_index()
        y=positives[positives.budget==low].set_index(kk).reindex(x.index)
        d=bounded_difference(x,y,metrics).reset_index()
        d.to_csv(out/'tables'/'budget_high_minus_low_seed_changes.csv',index=False)
        ci_table(d,['dataset','model_count'],metrics).to_csv(out/'tables'/'budget_high_minus_low_95ci.csv',index=False)
    kk=['dataset']+(['budget'] if stem=='success' else [])+['seed']
    for n in [12,24]:
        x=seeds[seeds.model_count==n].set_index(kk).sort_index()
        y=seeds[seeds.model_count==6].set_index(kk).reindex(x.index)
        d=bounded_difference(x,y,metrics).reset_index()
        d.to_csv(out/'tables'/f'{stem}_{n}_minus_6_seed_changes.csv',index=False)
        ci_table(d,kk[:-1],metrics).to_csv(out/'tables'/f'{stem}_{n}_minus_6_95ci.csv',index=False)

def more_plots(success,cost,out):
    import matplotlib.pyplot as plt
    for ds in sorted(success.dataset.unique()):
        for n in [6,12]:
            z=success[(success.dataset==ds)&(success.model_count==n)]
            fig,ax=plt.subplots(figsize=(7.8,5),layout='constrained')
            for m,l,c in [('single_success','Average singleton','#206095'),('separate_success','All separately','#2c8b57'),('joint_success_lower','Joint action lower bound','#a44a9e'),('orpm_success','ORPM','#d87418')]:draw(ax,z,'budget',m,l,c)
            ax.set(title=f'{ds} | {n} models',xlabel='L2 budget',ylabel='Success rate');ax.legend(fontsize=9);ax.grid(alpha=.18)
            fig.supxlabel('Eligible-group weights within seed; pointwise 95% t intervals across seeds.',fontsize=8)
            for e in ['png','pdf']:fig.savefig(out/'figures'/f'{ds}_success_{n}.{e}',dpi=150)
            plt.close(fig)
        z=cost[cost.dataset==ds]
        fig,ax=plt.subplots(figsize=(7.8,5),layout='constrained')
        for m,l,c in [('transfer_mean_other','Mean other-model acceptance','#206095'),('transfer_joint','All competing models accept','#a44a9e'),('transfer_joint_rejected','All accept, rejecting origins only','#d87418')]:draw(ax,z,'model_count',m,l,c)
        ax.set(title=f'{ds} | singleton-action transfer',xlabel='Competing models',ylabel='Acceptance rate',xticks=[6,12,24]);ax.legend(fontsize=9);ax.grid(alpha=.18)
        fig.supxlabel('Transfer within the saved competing bank; pointwise 95% t intervals.',fontsize=8)
        for e in ['png','pdf']:fig.savefig(out/'figures'/f'{ds}_singleton_transfer.{e}',dpi=150)
        plt.close(fig)

def group_plots(out,cfg):
    import matplotlib.pyplot as plt
    ci=pd.read_csv(out/'tables/success_group_95ci.csv')
    for ds,g in cfg['groups'].items():
        for n in [6,12,24]:
            for m,label in [('single_success','Average singleton success'),('joint_success_lower','Joint optimal success lower bound'),('orpm_success','ORPM success')]:
                fig,ax=plt.subplots(figsize=(7.8,5),layout='constrained')
                for group,color in [(g['first'],'#206095'),(g['second'],'#d87418')]:
                    z=ci[(ci.dataset==ds)&(ci.model_count==n)&(ci.group==group)]
                    draw(ax,z,'budget',m,group,color)
                ax.set(title=f'{ds} | {n} models | {label}',xlabel='L2 budget',ylabel='Success rate');ax.legend(fontsize=9);ax.grid(alpha=.18)
                fig.supxlabel('Same complete-pair seeds for both groups; pointwise 95% t intervals.',fontsize=8)
                for e in ['png','pdf']:fig.savefig(out/'figures'/f'{ds}_{n}_{m}_groups.{e}',dpi=150)
                plt.close(fig)
