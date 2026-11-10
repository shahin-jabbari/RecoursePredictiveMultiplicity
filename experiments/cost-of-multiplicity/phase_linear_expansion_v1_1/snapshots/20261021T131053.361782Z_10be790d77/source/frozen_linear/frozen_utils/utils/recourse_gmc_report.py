"""Paired seed-level reporting that keeps optimization intervals explicit."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import recourse_results as rr
from . import recourse_reporting as rp
from . import recourse_statistics as rs


def read(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def analyze(root):
    root=Path(root);outcomes=pd.DataFrame(read(root/'outcomes.jsonl'));curves=pd.DataFrame(read(root/'checkpoints.jsonl'))
    metadata=read(root/'model_diagnostics.jsonl');summaries=[]
    key=['family','seed','instance_id']
    assert not outcomes.duplicated(key+['diversity']).any()
    assert outcomes.groupby(key).size().eq(2).all()
    assert outcomes.feasible.all()
    for family,group in outcomes.groupby('family'):
        for label,prefix in [('returned_point',''),('uniform_response_mixture','mixed_')]:
            lo=prefix+'gap_lower_bound';hi=prefix+'gap_upper_bound'
            seeds=group.groupby(['seed','diversity'])[[lo,hi]].mean().unstack('diversity')
            record=dict(family=family,gap=label,n_seeds=len(seeds),paired_individuals=len(group)//2)
            for name,data in [('off_lower',seeds[(lo,False)]),('off_upper',seeds[(hi,False)]),
                    ('on_lower',seeds[(lo,True)]),('on_upper',seeds[(hi,True)]),
                    ('delta_lower',seeds[(lo,True)]-seeds[(hi,False)]),
                    ('delta_upper',seeds[(hi,True)]-seeds[(lo,False)])]:
                record.update({name+'_'+k:v for k,v in rs.mean_ci(data).items()})
            record['interpretation']='increase_resolved' if record['delta_lower_ci_low']>0 else \
                'decrease_resolved' if record['delta_upper_ci_high']<0 else 'not_resolved'
            summaries.append(record)
        for metric in ['valid','cost','dual_solve_uncertainty']:
            s=group.groupby(['seed','diversity'])[metric].mean().unstack('diversity')
            record=dict(family=family,metric=metric,n_seeds=len(s))
            for name,data in [('off',s[False]),('on',s[True]),('difference',s[True]-s[False])]:
                record.update({name+'_'+k:v for k,v in rs.mean_ci(data).items()})
            summaries.append(record)
        c=curves[curves.family.eq(family)]
        metrics=['gap_lower_bound','gap_upper_bound','mixed_gap_lower_bound','mixed_gap_upper_bound']
        seed_curves=c.groupby(['diversity','seed','round'])[metrics].mean().reset_index()
        rp.save_table(seed_curves,root,'Give_Me_Some_Credit',family+'_seed_gap_curves')
        rp.save_table(group.reset_index(drop=True),root,'Give_Me_Some_Credit',family+'_instance_results')
        fig,axes=plt.subplots(1,2,figsize=(12,4.7))
        for ax,prefix,title in zip(axes,['','mixed_'],['Returned candidate','Uniform response mixture']):
            for enabled,label in [(False,'Original model selection'),(True,'Matched feature subsets')]:
                part=seed_curves[seed_curves.diversity.eq(enabled)]
                mean=part.groupby('round')[metrics].mean()
                line,=ax.plot(mean.index,mean[prefix+'gap_upper_bound'],label=label+' — upper bound')
                ax.plot(mean.index,mean[prefix+'gap_lower_bound'],'--',color=line.get_color(),label=label+' — lower bound')
                ax.fill_between(mean.index.to_numpy(),mean[prefix+'gap_lower_bound'].to_numpy(),
                    mean[prefix+'gap_upper_bound'].to_numpy(),alpha=.12,color=line.get_color())
            ax.set(xlabel='ORPM round',ylabel='Gap (normalized logistic loss)' if family=='mixed_logistic' else 'Gap (squared probability loss)',
                   title=title,ylim=(0,None));ax.grid(alpha=.2)
        axes[1].legend(fontsize=7)
        fig.suptitle('Give Me Some Credit — '+('2 linear + 2 single-layer + 2 double-layer networks' if family=='mixed_logistic' else '5 decision trees'))
        fig.text(.5,.005,'Shading: optimization-bound intervals of the seed-averaged gap; statistical intervals are saved in the tables.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.035,1,.95));rp.save_figure(fig,root,'Give_Me_Some_Credit',family+'_duality_gap',False)
    summary=pd.DataFrame(summaries)
    rp.save_table(summary,root,'Give_Me_Some_Credit','paired_gap_summary')
    diagnostic_rows=[]
    for m in metadata:
        diagnostic_rows.append({k:m.get(k) for k in ['family','seed','diversity','validation_mean_accuracy',
            'validation_pairwise_disagreement','mean_predictors_used','masked_replacements','unmatched_fallbacks']})
    diag=pd.DataFrame(diagnostic_rows)
    rp.save_table(diag,root,'Give_Me_Some_Credit','paired_model_diagnostics')
    rr.atomic_json(root/'analysis.json',dict(summary=summaries,model_diagnostics=diagnostic_rows,
        paired_individuals=len(outcomes)//2,all_outputs_feasible=True))
    lines=['GMC: MIXED LOGISTIC LOSS AND DECISION-TREE SQUARED LOSS','',
        'Model count, deployed reference, held-out factuals, budget, and optimizer settings are held fixed within each diversity pair.',
        'The mixed model set is NONCONVEX in recourse coordinates. No affine-logit convex theorem is claimed.',
        'Returned-point and uniform-response-mixture gaps are reported separately. Tree feature vectors are not averaged.',
        'The neural lower bound comes from an exact ReLU encoding with tangent underestimators of logistic loss; local solutions supply upper bounds only.',
        'Tree bounds come from the original tree probability rules. All bounds are subject to solver numerical tolerances.',
        'A wide optimization interval means the actual gap is unresolved. Comparing upper bounds alone does not establish a change in the actual gap.',
        'Seed means are equally weighted. Statistical intervals use paired seed differences; individuals are not independent inference units.','']
    for r in summaries:
        if 'gap' not in r:continue
        lines.append(f"{r['family']} / {r['gap']} / {r['n_seeds']} seeds:")
        lines.append(f"  Original mean gap interval: [{r['off_lower_mean']:.6g}, {r['off_upper_mean']:.6g}]")
        lines.append(f"  Subset mean gap interval: [{r['on_lower_mean']:.6g}, {r['on_upper_mean']:.6g}]")
        lines.append(f"  Mean paired change interval: [{r['delta_lower_mean']:.6g}, {r['delta_upper_mean']:.6g}]")
        lines.append(f"  Outer statistical endpoints for that interval: [{r['delta_lower_ci_low']:.6g}, {r['delta_upper_ci_high']:.6g}]")
        lines.append(f"  Effect interpretation: {r['interpretation']}")
    (root/'report.txt').write_text('\n'.join(lines)+'\n')
    return summary
