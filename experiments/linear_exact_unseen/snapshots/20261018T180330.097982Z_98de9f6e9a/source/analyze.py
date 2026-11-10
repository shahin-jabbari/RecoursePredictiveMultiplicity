from pathlib import Path
import json,html,zipfile
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import read,save,utc

METRICS=['seen_joint_validity','unseen_joint_validity','seen_strict_joint_validity','unseen_strict_joint_validity','seen_mean_validity','unseen_mean_validity','seen_minimum_logit','unseen_minimum_logit','seen_worst_loss','unseen_worst_loss','cost','seconds','available','exact_loss_gap','cost_excess_over_minimum','minimum_cost_lower','minimum_cost_upper','transfer_change']
LABELS={'seen_joint_validity':'Optimization-panel joint validity','unseen_joint_validity':'Unseen-panel joint validity (50 models)','seen_mean_validity':'Optimization-panel mean acceptance','unseen_mean_validity':'Unseen-panel mean acceptance','exact_loss_gap':'Worst logistic loss gap vs exact optimum','transfer_change':'Unseen minus optimization joint validity'}
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#777777','#222222','#E69F00']

def ci(v,bounded=False):
    v=np.asarray(v,dtype=float);v=v[np.isfinite(v)];n=len(v);m=float(np.mean(v)) if n else np.nan;h=float(t.ppf(.975,n-1)*np.std(v,ddof=1)/np.sqrt(n)) if n>1 else np.nan
    lo,hi=m-h,m+h
    return dict(mean=m,ci_low=lo,ci_high=hi,plot_low=max(0,lo) if bounded else lo,plot_high=min(1,hi) if bounded else hi,seeds=n)

def summarize(df,group,metrics):
    seed=df.groupby(group+['seed'],dropna=False)[metrics].mean().reset_index();out=[]
    for key,g in seed.groupby(group,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        base=dict(zip(group,key))
        for metric in metrics:
            bounded=metric.endswith('validity') or metric=='available'
            out.append(dict(base,metric=metric,**ci(g[metric],bounded)))
    return pd.DataFrame(out),seed

def formatted(df):
    out=df.copy();out['estimate [95% CI]']=out.apply(lambda r:'not estimable' if not np.isfinite(r['mean']) else f"{r['mean']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]",axis=1)
    return out.drop(columns=['mean','ci_low','ci_high','plot_low','plot_high'])

def analyze(root):
    root=Path(root);cfg=read(root/'protocol.json');verification=read(root/'verification.json');figs=root/'figures';tables=root/'tables';figs.mkdir(exist_ok=True);tables.mkdir(exist_ok=True)
    rows=[json.loads(s) for s in (root/'records.jsonl').read_text().splitlines()];df=pd.DataFrame([{k:v for k,v in r.items() if k!='info'} for r in rows]);df['available']=df['available'].astype(float);df['transfer_change']=df.unseen_joint_validity-df.seen_joint_validity
    df.to_csv(root/'evaluation_records.csv',index=False)
    group=['dataset','model_count','budget','method'];cell,seeds=summarize(df,group,METRICS)
    cell.to_csv(tables/'all_cells_95ci.csv',index=False);formatted(cell).to_csv(tables/'all_cells_formatted_95ci.csv',index=False);seeds.to_csv(tables/'seed_means.csv',index=False)
    high=df[df.apply(lambda r:r.budget==max(cfg['budgets'][r.dataset]),axis=1)].copy()
    high24=high[high.model_count==24]
    primary,primary_seeds=summarize(high24,['dataset','method'],METRICS);primary.to_csv(tables/'primary_largest_budget_n24_95ci.csv',index=False);formatted(primary).to_csv(tables/'primary_largest_budget_n24_formatted_95ci.csv',index=False)
    paired=[]
    for ds in cfg['datasets']:
        subset=primary_seeds[primary_seeds.dataset==ds]
        ref=subset[subset.method=='ORPM'].set_index('seed')
        for m in cfg['methods']:
            if m=='ORPM':continue
            comp=subset[subset.method==m].set_index('seed')
            for metric in ['seen_joint_validity','unseen_joint_validity','seen_mean_validity','unseen_mean_validity','seen_worst_loss']:
                paired.append(dict(dataset=ds,contrast='ORPM minus '+m,metric=metric,**ci(ref[metric]-comp[metric])))
    paired=pd.DataFrame(paired);paired.to_csv(tables/'paired_method_differences_95ci.csv',index=False)
    effects=[]
    for ds in cfg['datasets']:
        for method in cfg['methods']:
            sub=df[(df.dataset==ds)&(df.method==method)]
            for metric in ['seen_joint_validity','unseen_joint_validity','seen_mean_validity','unseen_mean_validity','seen_worst_loss']:
                a=sub[(sub.model_count==24)&(sub.budget==max(cfg['budgets'][ds]))].groupby('seed')[metric].mean()
                b=sub[(sub.model_count==24)&(sub.budget==0)].groupby('seed')[metric].mean()
                effects.append(dict(dataset=ds,method=method,contrast='largest budget minus zero at n24',metric=metric,**ci(a-b)))
                b=sub[(sub.model_count==6)&(sub.budget==max(cfg['budgets'][ds]))].groupby('seed')[metric].mean()
                effects.append(dict(dataset=ds,method=method,contrast='24 minus 6 models at largest budget',metric=metric,**ci(a-b)))
    pd.DataFrame(effects).to_csv(tables/'paired_budget_and_model_count_effects_95ci.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','font.family':'DejaVu Sans','pdf.fonttype':42})
    plotted=[]
    for ds in cfg['datasets']:
        short='GMC' if ds=='Give_Me_Some_Credit' else ds
        for n in cfg['model_sizes']:
            for metric in ['seen_joint_validity','unseen_joint_validity','seen_mean_validity','unseen_mean_validity','exact_loss_gap']:
                part=cell[(cell.dataset==ds)&(cell.model_count==n)&(cell.metric==metric)]
                fig,ax=plt.subplots(figsize=(7.6,4.9))
                for k,m in enumerate(cfg['methods']):
                    q=part[part.method==m].sort_values('budget');xx=q.budget.to_numpy();yy=q['mean'].to_numpy();lo=q.plot_low.to_numpy();hi=q.plot_high.to_numpy()
                    ax.plot(xx,yy,marker='o',ms=3,color=COLORS[k],label=m,lw=1.6,ls='--' if m.startswith('Exact') else '-')
                    ax.fill_between(xx,lo,hi,color=COLORS[k],alpha=.10)
                ax.set(xlabel='L2 budget (standardized feature units)',ylabel=LABELS[metric],title=f'{short} · {n} optimization models · pilot')
                if metric.endswith('validity'):ax.set_ylim(-.03,1.03)
                if metric=='exact_loss_gap':ax.set_yscale('symlog',linthresh=.01)
                ax.grid(alpha=.2);ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,-.17),fontsize=8,frameon=False)
                fig.text(.01,.01,'Pointwise 95% t intervals across 3 seeds; 5 people/seed. Rate bands clipped to [0,1].',fontsize=8)
                fig.subplots_adjust(bottom=.26);name=f'{short}_n{n}_{metric}_budget';fig.savefig(figs/(name+'.png'),dpi=150);fig.savefig(figs/(name+'.pdf'));plt.close(fig);plotted.append(name)
        for metric in ['seen_joint_validity','unseen_joint_validity','seen_mean_validity','unseen_mean_validity','exact_loss_gap']:
            part=cell[(cell.dataset==ds)&(cell.budget==max(cfg['budgets'][ds]))&(cell.metric==metric)];fig,ax=plt.subplots(figsize=(7.6,4.9))
            for k,m in enumerate(cfg['methods']):
                q=part[part.method==m].sort_values('model_count');xx=q.model_count.to_numpy();yy=q['mean'].to_numpy();ax.plot(xx,yy,marker='o',ms=3,color=COLORS[k],label=m,lw=1.6,ls='--' if m.startswith('Exact') else '-');ax.fill_between(xx,q.plot_low.to_numpy(),q.plot_high.to_numpy(),color=COLORS[k],alpha=.10)
            ax.set(xlabel='Number of optimization models (nested prefixes)',ylabel=LABELS[metric],title=f'{short} · budget {max(cfg["budgets"][ds]):g} · pilot');ax.set_xticks(cfg['model_sizes'])
            if metric.endswith('validity'):ax.set_ylim(-.03,1.03)
            if metric=='exact_loss_gap':ax.set_yscale('symlog',linthresh=.01)
            ax.grid(alpha=.2);ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,-.17),fontsize=8,frameon=False);fig.text(.01,.01,'Pointwise 95% t intervals across 3 seeds; 5 people/seed. Rate bands clipped to [0,1].',fontsize=8);fig.subplots_adjust(bottom=.26);name=f'{short}_{metric}_model_count';fig.savefig(figs/(name+'.png'),dpi=150);fig.savefig(figs/(name+'.pdf'));plt.close(fig);plotted.append(name)
    def value(ds,m,metric,percent=False):
        r=primary[(primary.dataset==ds)&(primary.method==m)&(primary.metric==metric)].iloc[0];scale=100 if percent else 1
        return f"{scale*r['mean']:.3f} [{scale*r.ci_low:.3f}, {scale*r.ci_high:.3f}]"+('%' if percent else '')
    ready=verification['passed'] and verification['future_20_available_all_pilot_blocks']
    lines=['V7 EXACT LINEAR COMPARISON AND UNSEEN-MODEL PILOT',f'Generated: {utc()}',f'Run: {root}','',f'READINESS: {"READY to propose the full study; not launched" if ready else "NOT READY: resolve failed verification or cohort gates before a full study"}.',f'Completed {len(rows):,} method records; {cfg["expected_settings"]} settings; 45 people across 9 dataset/seed blocks.',f'Independent verification passed: {verification["passed"]}. All pilot blocks support a 20-person prefix: {verification["future_20_available_all_pilot_blocks"]}.','', 'DESIGN AND INTERPRETATION','Three seeds (401–403), five people/seed/dataset; nested 6/12/24 optimization models.','Seven methods; 50 unseen full-feature logistic hypotheses fixed per seed. All nine panels frozen before recourse.','Exact comparators: max minimum logit (convex SOCP) and minimum L2 joint-valid cost (convex QP).','The old MILP wrapper has an incompatible L1 objective and positive margin; it is not used as a like-for-like comparator.','This pilot does not test discrete interventions, neural transfer, new training datasets or external populations.','Unseen hypotheses share training data/center and have no feature masks; optimization banks include masks.','Thus seen/unseen differences combine transfer with a change in panel composition and size.','', 'CONFIDENCE INTERVALS','Every estimate below uses a pointwise 95% Student-t interval across 3 equally weighted seed means (df=2).','Paired contrasts are differenced within seed. Each seed mean uses the same five people.','Raw t limits may extend outside [0,100%]; plot rate bands are clipped. Raw CSV limits remain available.','Three seeds give imprecise, unstable intervals: these are pilot descriptions, not confirmatory claims.','A zero-width interval at a boundary means all three seed means coincided, not population certainty.','Design counts and deterministic verification facts are exact, not sampling estimates.','', 'PRIMARY SNAPSHOT: 24 OPTIMIZATION MODELS AT THE LARGEST PILOT BUDGET','Numbers are mean [95% CI]. Seen = joint validity on the optimization set. Unseen = all 50 models valid.']
    for ds in cfg['datasets']:
        lines+=['',f'{ds} — budget {max(cfg["budgets"][ds]):g}']
        for m in cfg['methods']:lines.append(f'{m:15s} seen {value(ds,m,"seen_joint_validity",True)}; unseen {value(ds,m,"unseen_joint_validity",True)}; unseen mean acceptance {value(ds,m,"unseen_mean_validity",True)}')
        lines+=['ORPM worst-loss excess above exact max-margin: '+value(ds,'ORPM','exact_loss_gap'),'ORPM unseen minus seen joint validity: '+value(ds,'ORPM','transfer_change',True),'ORPM minus each comparator (joint-validity percentage points; paired 95% CI):']
        for m in cfg['methods']:
            if m=='ORPM':continue
            q=paired[(paired.dataset==ds)&(paired.contrast=='ORPM minus '+m)&(paired.metric=='unseen_joint_validity')].iloc[0]
            lines.append(f'  {m}: {100*q["mean"]:.3f} [{100*q.ci_low:.3f}, {100*q.ci_high:.3f}] pp on unseen panel')
    lines+=['','NUMERICAL AND OPTIMIZATION AUDIT',json.dumps(verification['counts'],indent=2),'All exact bounds are independently evaluated at frozen numerical tolerances; not interval-arithmetic proofs.','Strict zero-threshold validity is included alongside tolerance-aware validity in every cell table.','Minimum-cost solutions often lie on a decision boundary: use the strict-sensitivity columns when interpreting them.','Unavailable ExactMinL2 outputs are included as zero validity; factual placeholders are not successful outputs.','Cost and loss of unavailable placeholders are raw diagnostics, not successful recourse costs.','Conditional cost excess is only defined when a returned point is jointly valid and a minimum-cost solution exists.','ExactMinL2 timings include the cached unbudgeted solve for each positive budget; do not sum repeated cached timings.','Failures: '+json.dumps(verification['failures']),'','BUDGET, MODEL COUNT AND GENERALIZATION','See paired_budget_and_model_count_effects_95ci.csv for largest-minus-zero budget effects at n24','and paired 24-minus-6 model effects at the largest budget, with 95% CIs.','The nested model-count design preserves individuals and the reference model. More optimization constraints','can reduce attainable seen validity; unseen validity need not be monotone because the returned intervention changes.','ExactMaxMargin identifies the attainable worst-loss frontier; a method gap can reflect finite optimization.','ExactMinL2 answers a distinct cost question, and may transfer less well because its seen margin is zero.','Do not interpret all 50 unseen hypotheses as independent replications; seed is the uncertainty unit.','', 'FULL-STUDY DECISION','A passing pilot establishes pipeline readiness within the tested scope. It does not establish strong unseen-model generalization.','Proposed expansion: six datasets, ten seeds, twenty people, all original budgets, same frozen algorithms.','Keep five-person prefixes, the pilot unseen panels, and prior pilot results when extending.','Untested Polish, German and ACSIncome blocks still require admission/cohort checks without relaxing the rules.','The proposed full study is saved but has NOT been launched.','', 'OUTPUTS',f'{len(plotted)} separate figure panels, each saved as PDF and PNG. No combined subfigure files.','tables/all_cells_95ci.csv: full numerical estimates, untruncated 95% bounds, seed counts.','tables/primary_largest_budget_n24_95ci.csv: primary snapshot; paired method and design-effect tables accompany it.','tables/seed_means.csv: seed-level audit data (not inferential estimates).','evaluation_records.csv and records.jsonl: per-intervention raw data (not aggregate estimates).','figures_tables.zip: all individual figures and tables plus report/protocol/verification.']
    report='\n'.join(lines)+'\n';(root/'report.txt').write_text(report)
    body='<html><head><meta charset="utf-8"><title>V7 linear pilot report</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;color:#16202b}pre{white-space:pre-wrap;line-height:1.5;font:14px ui-monospace}img{width:100%;max-width:850px}figure{margin:40px 0;border-top:1px solid #ddd}a{color:#0072b2}</style></head><body><h1>V7 exact linear and unseen-model pilot</h1><pre>'+html.escape(report)+'</pre>'
    for name in plotted:body+=f'<figure><img src="figures/{name}.png" alt="{name}"><figcaption>{html.escape(name)} · <a href="figures/{name}.pdf">Separate PDF</a></figcaption></figure>'
    (root/'report.html').write_text(body+'</body></html>')
    save(root/'analysis.json',dict(completed_utc=utc(),figure_panels=len(plotted),each_saved_separately=True,all_inferential_tables_have_95ci=True,ready_for_full_study=ready,full_study_launched=False))
    with zipfile.ZipFile(root/'figures_tables.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(list(figs.glob('*'))+list(tables.glob('*'))+[root/'report.txt',root/'report.html',root/'protocol.json',root/'PROTOCOL.txt',root/'verification.json',root/'analysis.json']):z.write(path,path.relative_to(root))
    with zipfile.ZipFile(root/'figures_tables.zip') as z:assert z.testzip() is None

if __name__=='__main__':
    import sys
    analyze(Path(sys.argv[1]))
