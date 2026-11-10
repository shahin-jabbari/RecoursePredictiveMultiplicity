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
    df=pd.DataFrame([{k:v for k,v in json.loads(s).items() if k!='info'} for s in open(root/'records.jsonl')]);df['available']=df['available'].astype(float);df['transfer_change']=df.unseen_joint_validity-df.seen_joint_validity
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
                ax.set(xlabel='L2 budget (standardized feature units)',ylabel=LABELS[metric],title=f'{short} · {n} optimization models · full study')
                if metric.endswith('validity'):
                    ax.set_ylim(-.03,1.03);ax.set_yticks(np.linspace(0,1,6));ax.set_yticklabels([f'{v:.1f}' for v in np.linspace(0,1,6)]);ax.tick_params(labelleft=True)
                if metric=='exact_loss_gap':ax.set_yscale('symlog',linthresh=.01)
                ax.grid(alpha=.2);ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,-.17),fontsize=8,frameon=False)
                fig.text(.01,.01,'Pointwise 95% t CIs: 10 seeds, 20 people/seed. Rate bands clipped to [0,1].',fontsize=8)
                fig.subplots_adjust(bottom=.26);name=f'{short}_n{n}_{metric}_budget';fig.savefig(figs/(name+'.png'),dpi=150);fig.savefig(figs/(name+'.pdf'));plt.close(fig);plotted.append(name)
        for metric in ['seen_joint_validity','unseen_joint_validity','seen_mean_validity','unseen_mean_validity','exact_loss_gap']:
            part=cell[(cell.dataset==ds)&(cell.budget==max(cfg['budgets'][ds]))&(cell.metric==metric)];fig,ax=plt.subplots(figsize=(7.6,4.9))
            for k,m in enumerate(cfg['methods']):
                q=part[part.method==m].sort_values('model_count');xx=q.model_count.to_numpy();yy=q['mean'].to_numpy();ax.plot(xx,yy,marker='o',ms=3,color=COLORS[k],label=m,lw=1.6,ls='--' if m.startswith('Exact') else '-');ax.fill_between(xx,q.plot_low.to_numpy(),q.plot_high.to_numpy(),color=COLORS[k],alpha=.10)
            ax.set(xlabel='Number of optimization models (nested prefixes)',ylabel=LABELS[metric],title=f'{short} · budget {max(cfg["budgets"][ds]):g} · full study');ax.set_xticks(cfg['model_sizes'])
            if metric.endswith('validity'):
                ax.set_ylim(-.03,1.03);ax.set_yticks(np.linspace(0,1,6));ax.set_yticklabels([f'{v:.1f}' for v in np.linspace(0,1,6)]);ax.tick_params(labelleft=True)
            if metric=='exact_loss_gap':ax.set_yscale('symlog',linthresh=.01)
            ax.grid(alpha=.2);ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,-.17),fontsize=8,frameon=False);fig.text(.01,.01,'Pointwise 95% t CIs: 10 seeds, 20 people/seed. Rate bands clipped to [0,1].',fontsize=8);fig.subplots_adjust(bottom=.26);name=f'{short}_{metric}_model_count';fig.savefig(figs/(name+'.png'),dpi=150);fig.savefig(figs/(name+'.pdf'));plt.close(fig);plotted.append(name)
    # The pilot is nested in this expansion; show both levels, not an independent phase test.
    pilot_table=pd.read_csv(Path(cfg['pilot_run'])/'tables/all_cells_95ci.csv')
    common=cell.merge(pilot_table,on=['dataset','model_count','budget','method','metric'],suffixes=('_full','_pilot'))
    common.to_csv(tables/'pilot_and_full_common_cells_95ci.csv',index=False)
    def value(ds,m,metric,percent=False):
        r=primary[(primary.dataset==ds)&(primary.method==m)&(primary.metric==metric)].iloc[0];scale=100 if percent else 1
        return f"{scale*r['mean']:.4f} [{scale*r.ci_low:.4f}, {scale*r.ci_high:.4f}]"+('%' if percent else '')
    lines=['V7 EXACT LINEAR AND UNSEEN-MODEL FULL STUDY',f'Generated: {utc()}',f'Run: {root}',f'Independent verification passed: {verification["passed"]}',f'{len(df):,} method records; {cfg["expected_settings"]:,} settings; 1,200 selected people across 60 dataset/seed blocks.',f'Preserved pilot records: {verification["preserved_pilot_records"]:,}; new records: {verification["new_records"]:,}.','', 'DESIGN','Six datasets, seeds 401–410, twenty people per seed/dataset. Nested 6/12/24 competing models.','Seven methods and 7–8 budgets per dataset, including zero. All methods and numerical settings remain frozen.','50 fixed unseen full-feature logistic hypotheses per seed, admitted from training/validation data before recourse.','The first five pilot people, all nine original unseen panels and all 3,780 pilot evaluations are preserved.','Same people and unseen panels across budgets, competing-set sizes and methods.','', 'CONFIDENCE INTERVALS','All estimates use pointwise 95% Student-t CIs over ten equally weighted seed means (df=9).','Paired contrasts are differenced within seed first. The twenty people and fifty models are not independent seeds.','Raw t limits are retained in tables; rate plots clip to [0,1]. Zero-width intervals at a boundary do not imply certainty.','Missing conditional estimates have fewer contributing seeds, reported explicitly. Exact design counts need no sampling CI.','No multiplicity-adjusted superiority claims. Seeds share the underlying finite dataset; inference is conditional on this protocol.','', 'PRIMARY SNAPSHOT: 24 COMPETING MODELS AT THE LARGEST BUDGET','Numbers are mean [95% CI]. Joint validity means every model in that panel accepts the same intervention.']
    for ds in cfg['datasets']:
        lines+=['',ds+' — budget '+str(max(cfg['budgets'][ds]))]
        for m in cfg['methods']:lines.append(f'{m:15s} seen joint {value(ds,m,"seen_joint_validity",True)}; unseen joint {value(ds,m,"unseen_joint_validity",True)}; unseen mean {value(ds,m,"unseen_mean_validity",True)}')
        lines+=['ORPM exact worst-loss gap: '+value(ds,'ORPM','exact_loss_gap'),'ORPM unseen-minus-seen joint validity: '+value(ds,'ORPM','transfer_change',True),'Paired unseen joint-validity differences (ORPM minus comparator, percentage points):']
        for m in cfg['methods']:
            if m=='ORPM':continue
            q=paired[(paired.dataset==ds)&(paired.contrast=='ORPM minus '+m)&(paired.metric=='unseen_joint_validity')].iloc[0];lines.append(f'  {m}: {100*q["mean"]:.4f} [{100*q.ci_low:.4f}, {100*q.ci_high:.4f}]')
    lines+=['','BUDGET AND COMPETING-SET SIZE','Separate panels plot every budget at each model count and every model count at the largest budget.','The paired design-effect table reports largest-minus-zero budget and 24-minus-6 model contrasts, each with a 95% CI.','More competing constraints can reduce attainable seen joint validity. Unseen validity need not vary monotonically.','Raw budget magnitudes are dataset-specific standardized L2 distances, not directly comparable monetary costs.','', 'EXACT OPTIMIZATION AND COST','ExactMaxMargin is a convex SOCP maximizing the minimum seen logit, equivalent to minimizing the worst logistic loss.','ExactMinL2 is a convex QP for minimum L2 joint-valid recourse. These do not require binary variables.','The legacy MILP wrapper uses a different L1 objective and positive margin and is not an equivalent comparator.','Every exact bound and infeasibility result is independently checked. Solver numerical tolerances are frozen.','ExactMinL2 no-output cases contribute zero primary validity; saved factual placeholders are not successful interventions.','Raw placeholder cost/loss is diagnostic only. Exact-loss gaps are defined only for available outputs.','Conditional cost excess is restricted to seen-joint-valid outputs with a finite minimum cost.','Cached ExactMinL2 solve time is repeated across budgets; repeated timings must not be summed as compute totals.','Strict zero-threshold validity and boundary-sensitive counts accompany tolerance-aware outcomes.','', 'UNSEEN-MODEL SCOPE','The panel contains fresh full-feature logistic parameter hypotheses sharing training data and a fitted center.','Sampling uses a training-Hessian ellipsoid, which is related to ElliCE’s uncertainty family; this is not distribution-neutral.','Optimization banks include feature masks, so seen/unseen changes also reflect different panel composition and size.','The experiment does not establish generalization to new training datasets, unseen feature-mask distributions, neural families or external populations.','A small exact seen-objective gap with poor unseen validity indicates an objective/panel mismatch, not necessarily insufficient optimization.','', 'PILOT COMPARISON','pilot_and_full_common_cells_95ci.csv shows pilot and full-study estimates at identical budgets/counts, with CIs for each phase.','The pilot is included in the full sample, so these phases are dependent; the table does not claim an independent replication test.','Pilot interventions, timings, scientific metrics, traces and exact certificates were copied unchanged and rechecked.','', 'INDEPENDENT VERIFICATION',json.dumps(verification,indent=2),'', 'OUTPUTS',f'{len(plotted)} separate panels, each as PNG and PDF; no composite subfigure files.','tables/all_cells_95ci.csv and formatted counterpart: estimates with raw 95% CI limits.','tables/paired_method_differences_95ci.csv and paired_budget_and_model_count_effects_95ci.csv: paired contrasts.','tables/seed_means.csv and evaluation_records.csv: raw audit data, not confidence-interval estimates.','report.html includes all separate panels. figures_tables.zip includes the report, protocol, verification and tables.']
    report='\n'.join(lines)+'\n';(root/'report.txt').write_text(report)
    page='<html><head><meta charset="utf-8"><title>V7 full linear study</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;color:#16202b}pre{white-space:pre-wrap;line-height:1.5;font:14px ui-monospace}img{width:100%;max-width:850px}figure{margin:40px 0;border-top:1px solid #ddd}</style></head><body><h1>V7 exact linear and unseen-model full study</h1><pre>'+html.escape(report)+'</pre>'
    for name in plotted:page+=f'<figure><img src="figures/{name}.png" alt="{name}"><figcaption>{html.escape(name)} · <a href="figures/{name}.pdf">Separate PDF</a></figcaption></figure>'
    (root/'report.html').write_text(page+'</body></html>')
    save(root/'analysis.json',dict(completed_utc=utc(),figure_panels=len(plotted),each_saved_separately=True,all_inferential_tables_have_95ci=True,verification_passed=verification['passed'],full_study_complete=verification['passed']))
    with zipfile.ZipFile(root/'figures_tables.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(list(figs.glob('*'))+list(tables.glob('*'))+[root/n for n in ['report.txt','report.html','protocol.json','PROTOCOL.txt','verification.json','analysis.json']]):z.write(p,p.relative_to(root))
    with zipfile.ZipFile(root/'figures_tables.zip') as z:assert z.testzip() is None

if __name__=='__main__':
    import sys
    analyze(Path(sys.argv[1]))
