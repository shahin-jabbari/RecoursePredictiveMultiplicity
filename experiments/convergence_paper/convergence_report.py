"""Per-individual convergence figures; no convergence decision from a mean gap."""
from pathlib import Path
import json, zipfile
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from convergence_study import load, csv, crossing_summary, classify

LABELS={'linear':'Logistic only','neural':'Neural only','mixed':'Logistic + neural'}
KEY=['dataset','seed','instance_id','family','models','budget']
CONDITION=['dataset','family','models','budget','gap_kind','threshold']

def summarize_seed_fractions(frame,groups,columns):
    out=[]
    for values,g in frame.groupby(groups,dropna=False):
        values=values if isinstance(values,tuple) else (values,)
        for col in columns:
            a=g[col].to_numpy(float);n=len(a);sd=float(a.std(ddof=1)) if n>1 else np.nan
            half=float(student_t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else np.nan
            base=dict(zip(groups,values));base['summary_metric' if 'metric' in base else 'metric']=col
            out.append(dict(base,mean=float(a.mean()),seeds=n,
                ci_halfwidth=half,ci_low=float(a.mean()-half),ci_high=float(a.mean()+half),
                interval_available=bool(n>1 and sd>0)))
    return pd.DataFrame(out)

def analyze(root,plots=True):
    root=Path(root);c=load(root/'protocol.json')['configuration']
    paths=sorted((root/'blocks').glob('*/seed_*/person_*/*/models_*/budget_*/gaps.csv'))
    if not paths:raise RuntimeError('No completed gap evaluations')
    df=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    if df.duplicated(KEY+['gap_kind','round']).any():raise RuntimeError('Duplicate gap records')
    from utils.recourse_bound_checks import DIAGNOSTIC_VERSION
    nonconvex=df.family.ne('linear')
    if nonconvex.any() and ('diagnostic_version' not in df or not df.loc[nonconvex,'diagnostic_version'].eq(DIAGNOSTIC_VERSION).all()):
        raise RuntimeError('Legacy nonconvex diagnostics remain; finish numerical-bound replay before reporting')
    expected=load(root/'protocol.json')['expected_trajectories']
    if len(df)!=expected*2*len(c['checkpoints']):raise RuntimeError('Incomplete study; do not label final')
    if not np.isfinite(df[['gap_lower','gap_upper','dual_lower','dual_upper']]).all().all():raise RuntimeError('Non-finite bounds')
    if (df.gap_lower>df.gap_upper+1e-6).any():raise RuntimeError('Reversed gap bounds')
    csv(df,root/'checkpoints.csv');cross=[];states=[]
    for vals,g in df.groupby(KEY+['gap_kind']):
        base=dict(zip(KEY+['gap_kind'],vals))
        for threshold in c['raw_gap_thresholds']:
            crossing=crossing_summary(g,threshold,c['confirmations']);cross.append(dict(base,**crossing))
            first=crossing['first_confirmed_round']
            for r in g.itertuples():
                status=classify(r.gap_lower,r.gap_upper,threshold)
                states.append(dict(base,threshold=threshold,round=r.round,certified_below=float(status=='certified_below'),
                    definitely_above=float(status=='definitely_above'),unresolved=float(status=='unresolved'),
                    possibly_below=float(status!='definitely_above'),confirmed_by_round=float(first is not None and first<=r.round)))
    crossings=pd.DataFrame(cross);state=pd.DataFrame(states)
    csv(crossings,root/'instance_crossings.csv')
    metrics=['certified_below','definitely_above','unresolved','possibly_below','confirmed_by_round']
    seed=state.groupby(CONDITION+['round','seed'])[metrics].mean().reset_index()
    csv(seed,root/'seed_fractions.csv');fractions=summarize_seed_fractions(seed,CONDITION+['round'],metrics)
    csv(fractions,root/'convergence_fractions.csv')
    # Effects are paired across the SAME individuals and then summarized by seed.
    final=state[state['round'].eq(c['rounds'])];contrasts=[]
    def compare(a,b,match,name):
        paired=a.merge(b,on=match,suffixes=('_a','_b'),validate='one_to_one')
        for metric in ['certified_below','possibly_below','unresolved']:
            paired['difference']=paired[metric+'_a']-paired[metric+'_b']
            grp=[x for x in match if x!='instance_id']
            v=paired.groupby(grp).difference.mean().reset_index()
            v['contrast']=name;v['metric']=metric;contrasts.append(v)
    match=['dataset','seed','instance_id','family','budget','gap_kind','threshold']
    compare(final[final.models.eq(max(c['model_sizes']))],final[final.models.eq(min(c['model_sizes']))],match,'largest minus smallest model set')
    for ds in c['datasets']:
        g=final[final.dataset.eq(ds)];match=['dataset','seed','instance_id','family','models','gap_kind','threshold']
        compare(g[g.budget.eq(max(c['budgets'][ds]))],g[g.budget.eq(min(c['budgets'][ds]))],match,'largest minus smallest budget')
    for family in ['neural','mixed']:
        match=['dataset','seed','instance_id','models','budget','gap_kind','threshold']
        compare(final[final.family.eq(family)],final[final.family.eq('linear')],match,f'{family} minus linear')
    effects=pd.concat(contrasts,ignore_index=True);csv(effects,root/'paired_seed_effects.csv')
    group=[x for x in effects.columns if x not in ['seed','difference']]
    effect_summary=summarize_seed_fractions(effects,group,['difference']);csv(effect_summary,root/'paired_effect_summary.csv')
    diagnostics=[]
    for p in (root/'blocks').glob('*/seed_*/set_diagnostics.json'):diagnostics.extend(load(p))
    dsdiag=pd.DataFrame(diagnostics);csv(dsdiag.drop(columns=['model_hashes'],errors='ignore'),root/'model_set_diagnostics.csv')
    text=['Per-instance ORPM convergence study',json.dumps(c,indent=2),
        'The cohort is held out. Recourse is evaluated on the fixed competing models, not on unseen models.',
        'All families use raw logistic loss. Logistic sets are convex in the recourse vector; neural/mixed sets generally are not.',
        'Averaged-point gaps evaluate the returned full arithmetic average. Randomized-response gaps evaluate a uniform random response.',
        'Saved regret decomposition: randomized gap = adversary regret + recourse regret; only the latter depends on the unknown dual minimum. Averaging difference is point primal minus randomized primal and can have either sign.',
        'Both gaps use the full mean adversary weights. The two primal terms differ. A neural averaged-point gap need not vanish even at a mixed equilibrium.',
        'Bounds enclose the gap over the encoded continuous domain, subject to numerical tolerances. Wide bounds mean unresolved, not nonconverged.',
        'Neural/mixed bounds use witness_checked_v2: tighter solver tolerances, tiny-tail tangent omission, and revocable independent bound sources. Contradicted numerical sources and all their transported uses are discarded; they are never clipped to a feasible objective.',
        'Neural/mixed dual evaluation is retrospective and can reuse every saved feasible response, including responses from later rounds. This does not alter any model, ORPM iterate, primal loss, or adversary weight. Checkpoint crossings are post-hoc estimates, not implementable online stopping times.',
        'A case is certified below a threshold only if the gap upper bound is below it. It is definitely above only if the lower bound exceeds it.',
        'Confirmed crossing requires two consecutive measured checkpoints. Sustained means all remaining measured checkpoints through the horizon; no between-checkpoint claim.',
        'First crossings are checkpoint-resolved, not exact hitting times. No imputation of failures/unresolved cases as T_max, and no average over successful cases only.',
        'The primary figures display per-person curves/status and fractions, not a mean gap used as a stopping rule.',
        'Confidence intervals are pointwise Student-t intervals over seed-level fractions; they are not global-solver intervals and are not simultaneous.',
        'Zero observed variation does not establish zero population uncertainty; unavailable/degenerate intervals are explicitly flagged.',
        'All counts are nested within family and share component pools across families. The same people satisfy majority rejection (ties included) for all nine sets.',
        'Accuracy matching is to a full-feature control within architecture, not perfect matching across architectures. See validation/disagreement diagnostics.',
        'The fixed raw outer rate is calibrated from earlier, disjoint-seed pilots. Training, algorithm, and global evaluation runtimes are distinct.',
        'Convex theory curve uses the ACTUAL fixed-rate bound log(n)/(alpha*t) + alpha*B^2/8 + mean certified oracle error.',
        'That worst-case bound can be loose. Do not substitute a time-retuned square-root rate for this fixed-rate trajectory.',
        'Squared-loss validity experiments are not interchangeable with these logistic-loss convergence experiments.',
        'Synthetic uses 2,000 observations for every family to hold the data and test cohort fixed.',
        'Continuous feature constraints are the V7 relaxation; convergence does not establish practical actionability or valid recourse.',
        f'Completed trajectories: {expected}; saved gap rows: {len(df)}.',
        'Final fractions at primary threshold (raw gap):']
    selected=fractions[fractions['round'].eq(c['rounds'])&fractions.threshold.eq(c['primary_raw_gap_threshold'])&fractions.metric.isin(['certified_below','definitely_above','unresolved'])]
    text.append(selected.to_string(index=False))
    for ds in c['datasets']:
        table=root/'table'/ds;table.mkdir(parents=True,exist_ok=True)
        csv(df[df.dataset.eq(ds)],table/'per_instance_gap_checkpoints.csv')
        csv(crossings[crossings.dataset.eq(ds)],table/'per_instance_threshold_crossings.csv')
        csv(fractions[fractions.dataset.eq(ds)],table/'convergence_fractions_seed_intervals.csv')
        csv(effect_summary[effect_summary.dataset.eq(ds)],table/'paired_budget_size_family_effects.csv')
        csv(dsdiag[dsdiag.dataset.eq(ds)].drop(columns=['model_hashes'],errors='ignore'),table/'validation_accuracy_disagreement.csv')
        if plots:plot_dataset(root,ds,df[df.dataset.eq(ds)],fractions[fractions.dataset.eq(ds)],c)
    (root/'report.txt').write_text('\n\n'.join(text)+'\n')
    with zipfile.ZipFile(root/'figures_tables.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for base in ['figures','table']:
            for p in (root/base).rglob('*'):
                if p.is_file():archive.write(p,p.relative_to(root))
        for name in ['report.txt','protocol.json','paired_effect_summary.csv']:archive.write(root/name,name)

def plot_dataset(root,ds,df,fractions,c):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    figdir=root/'figures'/ds;figdir.mkdir(parents=True,exist_ok=True)
    colors={6:'#0072B2',12:'#E69F00',24:'#009E73'};budgets=c['budgets'][ds]
    def save(fig,name):
        fig.tight_layout()
        for ext in ['pdf','png']:fig.savefig(figdir/f'{name}.{ext}',dpi=150,bbox_inches='tight')
        plt.close(fig)
    for kind in ['averaged_point','randomized_response']:
        d=df[df.gap_kind.eq(kind)];title='Averaged recommendation' if kind=='averaged_point' else 'Uniform random response'
        for threshold in c['raw_gap_thresholds']:
            fig,axes=plt.subplots(3,len(budgets),figsize=(11.5,8.3),squeeze=False,sharex=True,sharey=True)
            for fi,family in enumerate(c['families']):
                for bi,budget in enumerate(budgets):
                    ax=axes[fi,bi]
                    for n in c['model_sizes']:
                        g=fractions[fractions.gap_kind.eq(kind)&fractions.threshold.eq(threshold)&fractions.family.eq(family)&fractions.budget.eq(budget)&fractions.models.eq(n)]
                        a=g[g.metric.eq('certified_below')].sort_values('round');b=g[g.metric.eq('possibly_below')].sort_values('round')
                        ax.plot(a['round'],a['mean'],color=colors[n],label=f'{n} models')
                        ax.fill_between(a['round'],a['mean'],b['mean'],color=colors[n],alpha=.08)
                        # The dotted line is an uncertainty ceiling, NOT a confidence interval.
                        ax.plot(b['round'],b['mean'],color=colors[n],ls=':',lw=.8)
                    ax.set(xscale='log',ylim=(-.02,1.02),title=f'{LABELS[family]}, budget {budget:g}')
                    if fi==2:ax.set_xlabel('ORPM rounds')
                    if bi==0:ax.set_ylabel('Fraction of held-out individuals')
                    ax.grid(alpha=.18)
            axes[0,0].legend(fontsize=8);fig.suptitle(f'{ds}: {title}, raw gap ≤ {threshold:g}\nSolid: certified; dotted: includes unresolved cases',fontsize=12)
            save(fig,f'{kind}_converged_fraction_threshold_{threshold:g}')
        # Per-person curves at the largest size; no mean gap hides difficult people.
        fig,axes=plt.subplots(3,len(budgets),figsize=(11.5,8.3),squeeze=False,sharex=True)
        for fi,family in enumerate(c['families']):
            for bi,budget in enumerate(budgets):
                ax=axes[fi,bi];g=d[d.family.eq(family)&d.budget.eq(budget)&d.models.eq(max(c['model_sizes']))]
                for _,case in g.groupby(['seed','instance_id']):
                    case=case.sort_values('round');lo=case.gap_lower.clip(lower=1e-7);hi=case.gap_upper.clip(lower=1e-7)
                    ax.plot(case['round'],hi,color='#0072B2',alpha=.14,lw=.6)
                    ax.plot(case['round'],lo,color='#D55E00',alpha=.1,lw=.5)
                ax.axhline(c['primary_raw_gap_threshold'],color='black',ls='--',lw=.8)
                ax.set(xscale='log',yscale='log',title=f'{LABELS[family]}, budget {budget:g}')
                if fi==2:ax.set_xlabel('ORPM rounds')
                if bi==0:ax.set_ylabel('Individual raw duality gap bounds')
                ax.grid(alpha=.15)
        fig.suptitle(f'{ds}: {title}, {max(c["model_sizes"])} models\nEach line is one case; blue upper / orange lower; values below 1e-7 clipped',fontsize=12)
        save(fig,f'{kind}_individual_gap_curves_models_{max(c["model_sizes"])}')
        # Heatmaps include every size and budget, with uncertainty explicitly gray.
        for family in c['families']:
            fig,axes=plt.subplots(len(budgets),len(c['model_sizes']),figsize=(10.8,8.3),squeeze=False)
            for bi,budget in enumerate(budgets):
                for ni,n in enumerate(c['model_sizes']):
                    ax=axes[bi,ni];g=d[d.family.eq(family)&d.budget.eq(budget)&d.models.eq(n)].copy()
                    threshold=c['primary_raw_gap_threshold'];g['status']=[0 if hi<=threshold else 2 if lo>threshold else 1 for lo,hi in zip(g.gap_lower,g.gap_upper)]
                    # Fixed seed/person order within dataset across all compared panels.
                    matrix=g.pivot(index=['seed','instance_id'],columns='round',values='status').sort_index()
                    ax.imshow(matrix.to_numpy(),aspect='auto',interpolation='nearest',cmap=ListedColormap(['#009E73','#CCCCCC','#D55E00']),norm=BoundaryNorm([-.5,.5,1.5,2.5],3))
                    ax.set(title=f'{n} models, budget {budget:g}',xticks=range(len(matrix.columns)),xticklabels=matrix.columns)
                    ax.tick_params(axis='x',labelsize=7)
                    if bi==len(budgets)-1:ax.set_xlabel('ORPM rounds (checkpoint spacing)')
                    if ni==0:ax.set_ylabel('Held-out individual (seed/person order)')
            fig.suptitle(f'{ds}: {LABELS[family]}, {title}, gap ≤ {threshold:g}\nGreen: certified; gray: unresolved; orange: definitely above',fontsize=12)
            save(fig,f'{kind}_per_instance_status_{family}')
