"""Seed-level paired analysis for the time-budgeted diversity gap study."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import recourse_reporting as rp
from . import recourse_results as rr
from . import recourse_statistics as rs

LABELS={False:'All-feature models',True:'Accuracy-matched feature subsets'}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def paired(frame, metrics):
    rows=[]
    for ds,group in frame.groupby('dataset'):
        for metric in metrics:
            values=group.groupby(['seed','diversity'])[metric].mean().unstack('diversity').dropna()
            if values.empty:continue
            row=dict(dataset=ds,metric=metric,n_seeds=len(values))
            for label,vec in [('off',values[False]),('on',values[True]),('difference',values[True]-values[False])]:
                row.update({label+'_'+k:v for k,v in rs.mean_ci(vec).items()})
            rows.append(row)
    return pd.DataFrame(rows)


def analyze(root):
    root=Path(root)
    protocol=json.loads((root/'protocol.json').read_text())
    outcomes=pd.DataFrame(read_jsonl(root/'outcomes.jsonl'))
    curves=pd.DataFrame(read_jsonl(root/'checkpoints.jsonl'))
    metadata=read_jsonl(root/'model_diagnostics.jsonl')
    keys=['dataset','seed','instance_id']
    assert not outcomes.duplicated(keys+['diversity']).any()
    groups=outcomes.groupby(keys)
    assert groups.size().eq(2).all()
    assert groups.diversity.nunique().eq(2).all()
    assert groups.normalizer.nunique().eq(1).all()
    assert outcomes.feasible.all()
    assert (outcomes.gap_upper_bound<=outcomes.theory_upper_bound+1e-6).all()
    completed=outcomes[keys+['diversity']]
    curves=curves.merge(completed,on=keys+['diversity'],validate='many_to_one')
    outcomes['gap_at_most_001']=outcomes.gap_upper_bound.le(.01).astype(float)
    outcomes['gap_at_most_005']=outcomes.gap_upper_bound.le(.05).astype(float)
    outcomes['valid']=outcomes.valid.astype(float)
    metric_names=['gap_lower_bound','gap_upper_bound','raw_gap_lower_bound','raw_gap_upper_bound',
        'dual_solve_uncertainty','valid','cost','elapsed_seconds','gap_at_most_001','gap_at_most_005',
        'mean_oracle_error_upper_bound']
    summary=paired(outcomes,metric_names)
    completed_jobs=set(zip(outcomes.dataset,outcomes.seed))
    diag_keys=['validation_mean_accuracy','validation_pairwise_disagreement',
        'validation_probability_sd','mean_feature_group_jaccard_distance','distinct_feature_masks',
        'models_omitting_actionable_features','mean_predictors_used']
    diagnostics=[]
    for m in metadata:
        if (m['dataset'],m['seed']) not in completed_jobs:continue
        diagnostics.append(dict(dataset=m['dataset'],seed=m['seed'],diversity=m['diversity'],
            masked_replacements=m.get('masked_replacements',0),unmatched_fallbacks=m.get('unmatched_fallbacks',0),
            max_accuracy_match_error=max([abs(x['accuracy_difference']) for x in m.get('model_matches',[])],default=0.),
            **{k:m[k] for k in diag_keys}))
    diagnostics=pd.DataFrame(diagnostics)
    diagnostic_summary=paired(diagnostics,diag_keys+['masked_replacements','unmatched_fallbacks'])
    curve_metrics=['gap_lower_bound','gap_upper_bound','raw_gap_lower_bound','raw_gap_upper_bound']
    seed_curves=curves.groupby(['dataset','seed','diversity','round'])[curve_metrics].mean().reset_index()
    curve_stats=[]
    for (ds,enabled,t),g in seed_curves.groupby(['dataset','diversity','round']):
        for metric in curve_metrics:
            curve_stats.append(dict(dataset=ds,diversity=enabled,round=int(t),metric=metric,**rs.mean_ci(g[metric])))
    curve_stats=pd.DataFrame(curve_stats)
    verification=dict(completed_pairs=len(outcomes)//2,paired_cohorts=True,shared_normalizers=True,
        all_outputs_feasible=True,theory_bounds_respected=True,
        largest_gap_interval_width=float(outcomes.dual_solve_uncertainty.max()),
        largest_validation_match_error=float(diagnostics.max_accuracy_match_error.max()),
        instance_counts=outcomes.groupby(['dataset','seed','diversity']).size().reset_index(name='count').to_dict('records'))
    rr.atomic_json(root/'verification.json',verification)
    rr.atomic_json(root/'analysis.json',dict(protocol=protocol,verification=verification,
        summary=summary.to_dict('records'),diagnostics=diagnostic_summary.to_dict('records')))
    for ds in outcomes.dataset.unique():
        sub=summary[summary.dataset.eq(ds)].reset_index(drop=True)
        rp.save_table(sub,root,ds,'paired_final_gap_and_validity',latex=True)
        rp.save_table(outcomes[outcomes.dataset.eq(ds)].reset_index(drop=True),root,ds,'paired_instance_outcomes')
        rp.save_table(diagnostics[diagnostics.dataset.eq(ds)].reset_index(drop=True),root,ds,'model_diagnostics')
        rp.save_table(diagnostic_summary[diagnostic_summary.dataset.eq(ds)].reset_index(drop=True),root,ds,'paired_model_diagnostics')
        rp.save_table(seed_curves[seed_curves.dataset.eq(ds)].reset_index(drop=True),root,ds,'seed_gap_trajectories')
        rp.save_table(curve_stats[curve_stats.dataset.eq(ds)].reset_index(drop=True),root,ds,'gap_trajectory_summary')
        fig,axes=plt.subplots(1,2,figsize=(12,4.6))
        for ax,metric,label in zip(axes,['gap_upper_bound','raw_gap_upper_bound'],
                ['Duality-gap upper bound (common normalized loss)','Duality-gap upper bound (unnormalized log loss)']):
            for enabled in [False,True]:
                s=curve_stats[(curve_stats.dataset==ds)&(curve_stats.diversity==enabled)&(curve_stats.metric==metric)].sort_values('round')
                line,=ax.plot(s['round'],s['mean'],label=LABELS[enabled])
                ax.fill_between(s['round'].to_numpy(),s.ci_low.clip(lower=0).to_numpy(),s.ci_high.to_numpy(),
                    color=line.get_color(),alpha=.15)
            ax.set(xlabel='ORPM round',ylabel=label,ylim=(0,None));ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(rp.dataset_label(ds)+' — paired convex duality gap')
        fig.text(.5,.01,'Shading: pointwise 95% Student-t intervals across seed means.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.035,1,.95));rp.save_figure(fig,root,ds,'paired_duality_gap',False)
        # Paired differences make uncertainty in the effect directly visible.
        fig,ax=plt.subplots(figsize=(7,4.5))
        d=seed_curves[seed_curves.dataset.eq(ds)].pivot(index=['seed','round'],columns='diversity',values='gap_upper_bound')
        d['difference']=d[True]-d[False]
        rows=[dict(round=t,**rs.mean_ci(g['difference'])) for t,g in d.reset_index().groupby('round')]
        s=pd.DataFrame(rows)
        ax.plot(s['round'],s['mean'],color='#8046a5')
        ax.fill_between(s['round'].to_numpy(),s.ci_low.to_numpy(),s.ci_high.to_numpy(),color='#8046a5',alpha=.18)
        ax.axhline(0,color='black',lw=.8,ls='--');ax.grid(alpha=.2)
        ax.set(xlabel='ORPM round',ylabel='Gap change: feature subsets minus all features',
               title=rp.dataset_label(ds)+' — paired diversity effect')
        fig.tight_layout();rp.save_figure(fig,root,ds,'paired_gap_difference',False)
        rp.save_table(s,root,ds,'paired_gap_difference_curve')
    lines=['# Paired duality gap study with feature diversity','',
        f"Status: {protocol['status']}. Compute time: {protocol['elapsed_seconds']/60:.1f} minutes.",
        f"Six logistic models, {protocol['rounds']} ORPM rounds, up to {protocol['instances_per_seed']} individuals per seed; budgets: {protocol['datasets']}.",
        'Both conditions use identical splits, deployed model, held-out individuals, budgets, and optimizer settings. Feature-subset models are matched to each original model within two validation-accuracy percentage points; unmatched slots keep the original.',
        'Each paired individual uses one common loss bound for normalization. The experiment uses convex affine-logit cross-entropy loss, not squared probability loss or neural networks. Raw log-loss gaps are also saved.',
        'Means first average individuals within a seed and then weight seeds equally. Intervals below are paired pointwise 95% Student-t intervals across seeds, conditional on this dataset and repeated-holdout protocol. They are not population confidence guarantees or simultaneous confidence bands.',
        'Duality-gap optimization bounds and confidence intervals quantify different uncertainties. The recorded lower and upper gap bounds quantify optimization error; confidence intervals quantify variation across seed means.','',
        '|Dataset|Seeds|Paired individuals|All-feature gap|Feature-subset gap|Paired change [95% CI]|',
        '|---|---:|---:|---:|---:|---:|']
    for r in summary[summary.metric.eq('gap_upper_bound')].itertuples():
        n=len(outcomes[outcomes.dataset.eq(r.dataset)])//2
        lines.append(f'|{rp.dataset_label(r.dataset)}|{r.n_seeds}|{n}|{r.off_mean:.5f}|{r.on_mean:.5f}|{r.difference_mean:+.5f} [{r.difference_ci_low:+.5f}, {r.difference_ci_high:+.5f}]|')
    lines+=['','## Diversity actually achieved','',
        '|Dataset|Pairwise disagreement, off → on|Mean validation accuracy, off → on|Masked replacements out of 6|',
        '|---|---:|---:|---:|']
    for ds in outcomes.dataset.unique():
        d=diagnostic_summary[diagnostic_summary.dataset.eq(ds)].set_index('metric')
        p=d.loc['validation_pairwise_disagreement'];a=d.loc['validation_mean_accuracy'];m=d.loc['masked_replacements']
        lines.append(f'|{rp.dataset_label(ds)}|{p.off_mean*100:.2f}% → {p.on_mean*100:.2f}%|{a.off_mean*100:.2f}% → {a.on_mean*100:.2f}%|{m.on_mean:.2f}|')
    lines+=['','Interpret changes alongside measured disagreement. More feature subsets do not necessarily increase prediction disagreement, and a larger initial gap does not imply that ORPM fails to converge. The curves show how the fixed-round optimization error changes for the resulting model sets.',
        'Raw per-instance results, model parameters, feature masks, preprocessing splits, response trajectories, oracle diagnostics, and source snapshots are retained under this study directory. Dataset-specific figures and tables are in figures/ and table/.']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    return dict(root=str(root),summary=summary,diagnostics=diagnostic_summary)
