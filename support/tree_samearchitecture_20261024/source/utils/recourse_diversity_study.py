"""Paired diversity-off/on experiments and dataset-specific analysis exports."""
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import recourse_campaign as campaign
from . import recourse_core as rc
from . import recourse_reporting as rp
from . import recourse_results as rr
from . import recourse_statistics as rs

SECTIONS=('duality_gap','cost_validity')
LABELS={False:'Legacy feature selection',True:'Matched feature subsets'}


def _read_jsonl(path):
    if not Path(path).exists():
        return []
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def collect(index):
    diagnostics=[];outcomes=[];curves=[];cohorts={};identities={}
    for entry in index:
        enabled=entry['diversity'];section=entry['section']
        for directory in entry['runs']:
            run=Path(directory)
            manifest=json.loads((run/'manifest.json').read_text())
            cfg=manifest['configuration'];ds=cfg['dataset_name'];budget=cfg['recourse_budget']
            seed_meta={}
            for path in sorted(run.glob('seed_*_fold_*')):
                prep=json.loads((path/'preprocessing.json').read_text())
                data=json.loads((path/'models.json').read_text());meta=data['selection_metadata']
                n=len(data['models']);seed=prep['seed']
                identity=(section,ds,float(budget),n,seed)
                with np.load(path/'split_and_predictions.npz',allow_pickle=False) as saved:
                    ids=saved['candidate_ids'].tolist()
                    cohorts[(enabled,*identity)]=ids
                    identities[(enabled,*identity)]=dict(meta=meta,ids=ids,selection=cfg.get('candidate_selection','reference'),
                        test_ids=saved['test_ids'].tolist(),train_ids=saved['train_ids'].tolist(),
                        validation_ids=saved['validation_ids'].tolist())
                seed_meta[seed]=n
                record=dict(section=section,dataset=ds,budget=budget,n_models=n,seed=seed,
                    diversity=enabled,run=str(run),n_instances=len(ids),
                    candidate_selection=cfg.get('candidate_selection','reference'),
                    masked_replacements=meta.get('masked_replacements',0),
                    unmatched_fallbacks=meta.get('unmatched_fallbacks',0),
                    max_accuracy_match_error=max([abs(r['accuracy_difference']) for r in meta.get('model_matches',[])],default=0.))
                for key in ['validation_mean_accuracy','validation_pairwise_disagreement',
                    'validation_probability_sd','mean_feature_group_jaccard_distance',
                    'distinct_feature_masks','models_omitting_actionable_features','mean_predictors_used']:
                    record[key]=meta[key]
                diagnostics.append(record)
            records=_read_jsonl(run/'instances.jsonl')
            for r in records:
                if r.get('status') in {'solver_error','model_generation_failed'}:
                    raise RuntimeError(f'Failed experiment record: {run}: {r.get("error")}')
                if 'valid' not in r:continue
                outcomes.append({**{k:r.get(k) for k in ['dataset','seed','instance_id','method',
                    'budget','valid','max_loss','cost','elapsed_seconds','gap_lower_bound','gap_upper_bound']},
                    'n_models':seed_meta[r['seed']],'diversity':enabled,'section':section,
                    'gap_upper_bound':r.get('raw_gap_upper_bound',r.get('gap_upper_bound')),
                    'candidate_selection':cfg.get('candidate_selection','reference')})
            if section=='duality_gap':
                for r in _read_jsonl(run/'duality_checkpoints.jsonl'):
                    curves.append({**{k:r[k] for k in ['dataset','seed','instance_id','round',
                        'gap_lower_bound','gap_upper_bound']},'diversity':enabled,
                        'gap_lower_bound':r.get('raw_gap_lower_bound',r['gap_lower_bound']),
                        'gap_upper_bound':r.get('raw_gap_upper_bound',r['gap_upper_bound'])})
    checks=[]
    for key,off in identities.items():
        if key[0]:continue
        on=identities[(True,*key[1:])]
        fields=['test_ids','train_ids','validation_ids']
        if off['selection']=='reference':fields.append('ids')
        for item in fields:
            if off[item]!=on[item]:raise AssertionError(f'Unpaired {item}: {key}')
        if off['meta']['model_hashes']!=on['meta']['reference_model_hashes']:
            raise AssertionError(f'Accuracy matching did not use the actual control models: {key}')
        checks.append(dict(section=key[1],dataset=key[2],budget=key[3],n_models=key[4],seed=key[5],
                           reference_factual_ids=off['ids'],diverse_factual_ids=on['ids'],
                           common_factual_ids=sorted(set(off['ids']) & set(on['ids'])),
                           candidate_selection=off['selection'],same_splits=True,same_control_models=True))
    # Check cross-budget, cross-size cohorts and nesting independently in each condition.
    nesting=[]
    for enabled in [False,True]:
        for ds,seed in sorted({(k[2],k[5]) for k in identities if k[1]=='cost_validity'}):
            entries=[(k,v) for k,v in identities.items() if k[0]==enabled and k[1]=='cost_validity'
                     and k[2]==ds and k[5]==seed]
            reference_ids=entries[0][1]['ids'];by_size={};cohorts_by_size={}
            for key,value in entries:
                if value['selection']=='reference':
                    assert value['ids']==reference_ids,(key,'reference cohort changed with size/budget')
                if key[4] in cohorts_by_size:
                    assert value['ids']==cohorts_by_size[key[4]],(key,'cohort changed with budget')
                cohorts_by_size[key[4]]=value['ids']
                hashes=set(value['meta']['model_hashes'])
                if key[4] in by_size:assert hashes==by_size[key[4]],(key,'models changed with budget')
                by_size[key[4]]=hashes
            sizes=sorted(by_size)
            for a,b in zip(sizes,sizes[1:]):assert by_size[a].issubset(by_size[b]),(ds,seed,a,b)
            nesting.append(dict(diversity=enabled,dataset=ds,seed=seed,sizes=sizes,nested=True))
    outcome_columns=['dataset','seed','instance_id','method','budget','valid','max_loss','cost',
        'elapsed_seconds','gap_lower_bound','gap_upper_bound','n_models','diversity','section','candidate_selection']
    curve_columns=['dataset','seed','instance_id','round','gap_lower_bound','gap_upper_bound','diversity']
    return (pd.DataFrame(diagnostics),pd.DataFrame(outcomes,columns=outcome_columns),
            pd.DataFrame(curves,columns=curve_columns),dict(pairing=checks,nesting=nesting))


def paired_summary(outcomes):
    keys=['section','dataset','n_models','budget','method']
    if outcomes.empty:
        return pd.DataFrame(columns=keys+['metric','n_seeds','n_paired_individuals','candidate_selection',
            'off_mean','on_mean','difference_mean'])
    rows=[]
    for key,group in outcomes.groupby(keys):
        paired_ids=group.groupby(['seed','instance_id','diversity']).size().unstack('diversity')
        common=paired_ids.reindex(columns=[False,True]).dropna().index
        paired=group.set_index(['seed','instance_id']).loc[common].reset_index()
        for metric,scale in [('valid',100),('max_loss',1),('cost',1),('gap_upper_bound',1)]:
            if group[metric].isna().all():continue
            seeds=paired.groupby(['diversity','seed'])[metric].mean().unstack(0).reindex(columns=[False,True]).dropna()
            row=dict(zip(keys,key),metric=metric,n_seeds=len(seeds),n_paired_individuals=len(common),
                candidate_selection=group['candidate_selection'].iloc[0] if 'candidate_selection' in group else 'reference')
            for label,values in [('off',seeds[False]*scale),('on',seeds[True]*scale),
                                 ('difference',(seeds[True]-seeds[False])*scale)]:
                s=rs.mean_ci(values)
                row.update({label+'_'+k:s[k] for k in ['mean','sd','se','ci_low','ci_high','ci_status']})
            rows.append(row)
    return pd.DataFrame(rows)


def report(index,root,show=False):
    root=Path(root)
    diagnostics,outcomes,curves,checks=collect(index)
    summary=paired_summary(outcomes)
    rr.atomic_json(root/'pairing_verification.json',checks)
    rr.atomic_json(root/'analysis_records.json',dict(diagnostics=diagnostics.to_dict('records'),
        summary=summary.to_dict('records'),outcomes=outcomes.to_dict('records')))
    for ds in diagnostics.dataset.unique():
        d=diagnostics[diagnostics.dataset.eq(ds)]
        s=summary[summary.dataset.eq(ds)]
        o=outcomes[outcomes.dataset.eq(ds)]
        rp.save_table(d,root,ds,'diversity_model_diagnostics')
        rp.save_table(s,root,ds,'diversity_paired_results',latex=True)
        rp.save_table(o,root,ds,'diversity_instance_results')
        valid=s[(s.section=='cost_validity')&(s.method=='ORPM')&(s.metric=='valid')]
        if not valid.empty:
            sizes=sorted(valid.n_models.unique())
            fig,axes=plt.subplots(1,len(sizes),figsize=(5*len(sizes),4.3),squeeze=False,sharey=True)
            for ax,n in zip(axes[0],sizes):
                sub=valid[valid.n_models.eq(n)].sort_values('budget')
                for label,enabled in [('off',False),('on',True)]:
                    artist,=ax.plot(sub.budget,sub[label+'_mean'],'o-',label=LABELS[enabled])
                    # Truncate only displayed uncertainty, never the saved raw limits.
                    ax.fill_between(sub.budget.to_numpy(),sub[label+'_ci_low'].clip(0,100).to_numpy(),
                        sub[label+'_ci_high'].clip(0,100).to_numpy(),color=artist.get_color(),alpha=.1)
                ax.set(xlabel='Recourse budget (standardized L2 distance)',title=f'{n} competing models',ylim=(-2,102))
                ax.grid(alpha=.2)
            axes[0,0].set_ylabel('Joint validity (%)');axes[0,-1].legend(fontsize=8)
            fig.suptitle(rp.dataset_label(ds)+' — ORPM: matched individuals and model counts')
            fig.tight_layout();rp.save_figure(fig,root,ds,'diversity_budget_joint_validity',show)
            budgets=sorted(valid.budget.unique())
            fig,axes=plt.subplots(1,len(budgets),figsize=(5*len(budgets),4.3),squeeze=False,sharey=True)
            for ax,b in zip(axes[0],budgets):
                sub=valid[valid.budget.eq(b)].sort_values('n_models')
                for label,enabled in [('off',False),('on',True)]:
                    ax.plot(sub.n_models,sub[label+'_mean'],'o-',label=LABELS[enabled])
                ax.set(xlabel='Number of competing models',title=f'Budget {b:g}',ylim=(-2,102),xticks=sizes)
                ax.grid(alpha=.2)
            axes[0,0].set_ylabel('Joint validity (%)');axes[0,-1].legend(fontsize=8)
            fig.suptitle(rp.dataset_label(ds)+' — ORPM: set size at fixed budget')
            fig.tight_layout();rp.save_figure(fig,root,ds,'diversity_size_joint_validity',show)
            # Show the diversity actually attained alongside accuracy matching.
            unique=d[d.section.eq('cost_validity')].drop_duplicates(['diversity','seed','n_models'])
            fig,axes=plt.subplots(1,2,figsize=(11,4.3))
            for enabled,part in unique.groupby('diversity'):
                means=part.groupby('n_models')[['validation_pairwise_disagreement','validation_mean_accuracy']].mean()*100
                for ax,metric in zip(axes,['validation_pairwise_disagreement','validation_mean_accuracy']):
                    ax.plot(means.index,means[metric],'o-',label=LABELS[enabled]);ax.set_xticks(sizes);ax.grid(alpha=.2)
            axes[0].set(xlabel='Number of competing models',ylabel='Validation pairwise disagreement (%)')
            axes[1].set(xlabel='Number of competing models',ylabel='Mean validation accuracy (%)');axes[1].legend(fontsize=8)
            fig.suptitle(rp.dataset_label(ds)+' — measured model diversity and accuracy')
            fig.tight_layout();rp.save_figure(fig,root,ds,'diversity_disagreement_accuracy',show)
        c=curves[curves.dataset.eq(ds)]
        if not c.empty:
            common=c.groupby(['seed','instance_id','round','diversity']).size().unstack('diversity').reindex(columns=[False,True]).dropna().index
            c=c.set_index(['seed','instance_id','round']).loc[common].reset_index()
        if not c.empty:
            seed=c.groupby(['diversity','seed','round'])[['gap_lower_bound','gap_upper_bound']].mean().reset_index()
            rp.save_table(seed,root,ds,'diversity_duality_seed_curves')
            fig,ax=plt.subplots(figsize=(7,4.6))
            for enabled,part in seed.groupby('diversity'):
                rows=[dict(round=r,**rs.mean_ci(g.gap_upper_bound)) for r,g in part.groupby('round')]
                stats=pd.DataFrame(rows).sort_values('round')
                artist,=ax.plot(stats['round'],stats['mean'],'o-',label=LABELS[enabled])
                ax.fill_between(stats['round'].to_numpy(),stats.ci_low.clip(lower=0).to_numpy(),
                                stats.ci_high.to_numpy(),color=artist.get_color(),alpha=.12)
            ax.set(xlabel='ORPM round',ylabel='Duality-gap upper bound (raw cross entropy)',
                title=rp.dataset_label(ds)+' — convex logistic experiment')
            ax.grid(alpha=.2);ax.legend();fig.tight_layout()
            rp.save_figure(fig,root,ds,'diversity_duality_gap',show)
    lines=['# V7 paired experiment','',
        'Both conditions use identical seeds, splits, budgets, model counts, and optimization settings. Reference cohorts are fixed; majority cohorts may differ. Paired differences use only shared evaluated individuals, with paired counts reported.',
        'Accuracy matching uses validation data only and a two-percentage-point default per-model caliper. Unmatched slots retain the original model.',
        'Feature-subset generation encourages disagreement; it does not guarantee that ensemble disagreement increases.',
        'Quick uses two seeds and three factuals per seed; differences are exploratory, not evidence of statistical significance.',
        'Gap bounds are optimization bounds. Confidence intervals across seeds measure a different uncertainty.',
        'Gap comparisons use raw affine-logit BCE to avoid set-dependent normalization effects; trade-offs use squared probability loss. Different loss units are not directly comparable.','',
        '## ORPM joint validity (percent)','',
        '|Dataset|Models|Budget|Diversity off|Diversity on|Change (pp)|',
        '|---|---:|---:|---:|---:|---:|']
    for r in summary[(summary.section=='cost_validity')&(summary.method=='ORPM')&(summary.metric=='valid')].itertuples():
        lines.append(f'|{r.dataset}|{r.n_models}|{r.budget:g}|{r.off_mean:.2f}|{r.on_mean:.2f}|{r.difference_mean:+.2f}|')
    lines+=['','## Final certified duality-gap upper bound','',
            '|Dataset|Diversity off|Diversity on|','|---|---:|---:|']
    for r in summary[(summary.section=='duality_gap')&(summary.metric=='gap_upper_bound')].itertuples():
        lines.append(f'|{r.dataset}|{r.off_mean:.6g}|{r.on_mean:.6g}|')
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    return dict(directory=str(root),summary=summary,diagnostics=diagnostics,pairing=checks)


def run_paired(mode='quick',seed=42,resume=True,show=False):
    from . import recourse_runtime as runtime
    if rr.NOTEBOOK_VERSION!='V7':
        raise RuntimeError('Initialize the V7 runtime first.')
    root=Path(rr.RESULTS_ROOT)/'V7'/campaign.mode_name(mode)/campaign.CANDIDATE_SELECTION/'diversity_comparison'/rr._stamp()
    root.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();index=[];original=campaign.DIVERSITY_ENABLED
    try:
        for enabled in [False,True]:
            campaign.DIVERSITY_ENABLED=enabled
            for section in SECTIONS:
                print(f'PAIRED STUDY: diversity={enabled}; {section}',flush=True)
                result=runtime.run_section(section,mode,seed,resume,show)
                index.append(dict(diversity=enabled,section=section,
                    campaign_directory=result.attrs['campaign_directory'],
                    runs=result['Run Directory'].unique().tolist()))
                rr.atomic_json(root/'study_index.json',dict(mode=campaign.mode_name(mode),seed=seed,
                    candidate_selection=campaign.CANDIDATE_SELECTION,diversity_options=dict(campaign.DIVERSITY_OPTIONS),status='running',runs=index))
        result=report(index,root,show)
        rr.atomic_json(root/'study_index.json',dict(mode=campaign.mode_name(mode),seed=seed,
            candidate_selection=campaign.CANDIDATE_SELECTION,diversity_options=dict(campaign.DIVERSITY_OPTIONS),status='completed',runs=index,
            elapsed_seconds=time.perf_counter()-started))
        print('Paired diversity report:',root,flush=True)
        return result
    except Exception as exc:
        rr.atomic_json(root/'study_index.json',dict(mode=campaign.mode_name(mode),seed=seed,
            status='failed',error=str(exc),runs=index,elapsed_seconds=time.perf_counter()-started))
        raise
    finally:campaign.DIVERSITY_ENABLED=original
