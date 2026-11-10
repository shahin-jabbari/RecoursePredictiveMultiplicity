from experiment import *
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import html,zipfile

METRICS=['seen_joint','unseen_mean','unseen_joint','unseen_worst_probability','factual_unseen_mean','factual_unseen_joint','mean_acceptance_gain','joint_acceptance_gain','seen_to_unseen_joint_drop','unseen_joint_5','unseen_joint_10','unseen_joint_25','unseen_joint_50']
KEYS=['dataset','panel','models','budget','method']
COLORS=['#126E82','#D97706','#7750A4','#3A7D44']

def interval(values):
 a=np.asarray(values,float);n=len(a);mean=float(a.mean()) if n else None
 se=float(a.std(ddof=1)/np.sqrt(n)) if n>1 else None
 half=float(t.ppf(.975,n-1)*se) if n>1 else None
 return dict(mean=mean,ci_low=mean-half if half is not None else None,ci_high=mean+half if half is not None else None,se=se,seeds=n,df=n-1 if n else None)

def summarize(df,conditional=None):
 if conditional=='seen_valid':df=df[df.seen_joint==1]
 if conditional=='recourse_needed':df=df[df.recourse_needed]
 seedmeans=df.groupby(KEYS+['seed'],dropna=False)[METRICS].mean().reset_index();out=[]
 for key,g in seedmeans.groupby(KEYS):
  for metric in METRICS:out.append(dict(zip(KEYS,key),metric=metric,subset=conditional or 'all_saved_people',**interval(g[metric])))
 return seedmeans,pd.DataFrame(out)

def paired(seedmeans):
 out=[]
 def emit(base,label,left,right,metrics):
  z=left.merge(right,on='seed',suffixes=('_a','_b'))
  for metric in metrics:out.append(dict(base,contrast=label,metric=metric,**interval(z[metric+'_a']-z[metric+'_b'])))
 metrics=['seen_joint','unseen_mean','unseen_joint','seen_to_unseen_joint_drop','joint_acceptance_gain']
 for (ds,panel,m,method),g in seedmeans.groupby(['dataset','panel','models','method']):
  bs=sorted(b for b in g.budget.unique() if b>0)
  emit(dict(dataset=ds,panel=panel,models=m,method=method,budget_low=bs[0],budget_high=bs[-1]),'highest_minus_lowest_positive_budget',g[g.budget==bs[-1]],g[g.budget==bs[0]],metrics)
 for (ds,panel,b,method),g in seedmeans.groupby(['dataset','panel','budget','method']):
  ms=sorted(g.models.unique())
  for low,high in [(6,12),(12,24),(6,24)]:
   if low in ms and high in ms:emit(dict(dataset=ds,panel=panel,budget=b,method=method,models_low=low,models_high=high),'larger_minus_smaller_set',g[g.models==high],g[g.models==low],metrics)
 for (ds,panel,b,m),g in seedmeans.groupby(['dataset','panel','budget','models']):
  for method in g.method.unique():
   if method!='ORPM':emit(dict(dataset=ds,panel=panel,budget=b,models=m,comparator=method),'ORPM_minus_comparator',g[g.method=='ORPM'],g[g.method==method],metrics)
 return pd.DataFrame(out)

def analyze(root):
 p=read(root/'protocol.json');df=pd.DataFrame(rows(root/'evaluation.jsonl'));table=root/'tables';figdir=root/'figures';table.mkdir();figdir.mkdir()
 seeds,summary=summarize(df);seeds.to_csv(table/'seed_means.csv',index=False);summary.to_csv(table/'summary_95ci.csv',index=False)
 for subset in ['seen_valid','recourse_needed']:
  ss,ci=summarize(df,subset);ss.to_csv(table/f'{subset}_seed_means.csv',index=False);ci.to_csv(table/f'{subset}_95ci.csv',index=False)
 df.groupby(KEYS+['seed']).agg(people=('person','size'),seen_valid_people=('seen_joint','sum'),recourse_needed_people=('recourse_needed','sum')).reset_index().to_csv(table/'denominators.csv',index=False)
 paired(seeds).to_csv(table/'paired_effects_95ci.csv',index=False)
 source=pd.DataFrame(rows(root/'records.jsonl'))
 source.groupby(['dataset','models','budget','method','seed'])[[c for c in ['timeout_fraction','native_certified_fraction','rejected_certificate_fraction','rejected_incumbent_fraction','cost'] if c in source]].mean().reset_index().to_csv(table/'solver_and_cost_seed_means.csv',index=False)
 diagnostics=[]
 for ds in p['datasets']:
  for seed in p['seeds']:
   b=root/'banks'/ds/f'seed_{seed}';test=np.load(b/'split.npz')['X_test'];mm,_,_,_=bank(root,ds,24,seed,0)
   seen=independent_prob(mm,test)>=.5
   for panel in PANELS:
    pdirectory=b/'panels'/panel;ad=read(pdirectory/'admission.json');pr=np.load(pdirectory/'native_prediction_audit.npz')['test']>=.5
    # Pairwise disagreement of all unseen model pairs and with seen models on the fixed held-out test split.
    n=pr.shape[1];counts=pr.sum(axis=1);within=float(np.mean(2*counts*(n-counts)/(n*(n-1))))
    cross=float(np.mean(pr.mean(1)*(1-seen.mean(1))+(1-pr.mean(1))*seen.mean(1)))
    diagnostics.append(dict(dataset=ds,seed=seed,panel=panel,candidates=len(ad['candidates']),accepted=50,cutoff=ad['minimum_admissible_accuracy'],validation_mean=np.mean([ad['candidates'][i]['validation_accuracy'] for i in ad['selected']]),unseen_pair_disagreement=within,seen_unseen_disagreement=cross))
 pd.DataFrame(diagnostics).to_csv(table/'panel_diagnostics.csv',index=False)
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 names=[]
 def draw(ax,g,metric,title):
  for color,method in zip(COLORS,p['methods']):
   q=g[(g.method==method)&(g.metric==metric)].sort_values('budget');x=q.budget.to_numpy();y=q['mean'].to_numpy();lo=q.ci_low.to_numpy();hi=q.ci_high.to_numpy()
   ax.plot(x,y,'o-',label=method,color=color,ms=3,lw=1.4);ax.fill_between(x,np.maximum(0,lo),np.minimum(1,hi),color=color,alpha=.12)
  ax.set(title=title,xlabel='Standardized L2 budget',ylim=(-.025,1.025));ax.grid(alpha=.15)
 for ds in p['datasets']:
  for panel in PANELS:
   for m in p['model_counts']:
    g=summary[(summary.dataset==ds)&(summary.panel==panel)&(summary.models==m)]
    fig,axes=plt.subplots(1,3,figsize=(10.8,3.2),sharey=True)
    for ax,metric,title in zip(axes,['seen_joint','unseen_mean','unseen_joint'],['Seen joint validity','Mean unseen acceptance','All-50 unseen joint validity']):draw(ax,g,metric,title)
    axes[0].set_ylabel('Fraction');handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,-.01))
    fig.suptitle(f"{LABELS.get(ds,ds)} | {m} {'forests' if p['arm']=='random_forest_pilot' else 'trees'} | {panel} panel"+(' | TWO-SEED PILOT' if p['arm']=='random_forest_pilot' else ''))
    fig.tight_layout(rect=[0,.09,1,.94]);name=f'{ds}_{panel}_m{m}_budget';names.append(name)
    for ext in ['png','pdf']:fig.savefig(figdir/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
   for m in [24]:
    b=max(p['budgets'][ds]);g=summary[(summary.dataset==ds)&(summary.panel==panel)&(summary.models==m)&(summary.budget==b)]
    fig,ax=plt.subplots(figsize=(5.8,3.8))
    for color,method in zip(COLORS,p['methods']):
     q=[g[(g.method==method)&(g.metric==f'unseen_joint_{k}')].iloc[0] for k in [5,10,25,50]]
     ax.plot([5,10,25,50],[r['mean'] for r in q],'o-',color=color,label=method,ms=4)
     ax.fill_between([5,10,25,50],[max(0,r.ci_low) for r in q],[min(1,r.ci_high) for r in q],color=color,alpha=.12)
    ax.set(xlabel='Number of unseen models (fixed nested prefixes)',ylabel='Joint validity',ylim=(-.025,1.025),title=f'{LABELS.get(ds,ds)} | {panel} | m=24, B={b:g}')
    ax.legend(fontsize=8,frameon=False);ax.grid(alpha=.15);fig.tight_layout();name=f'{ds}_{panel}_panel_size';names.append(name)
    for ext in ['png','pdf']:fig.savefig(figdir/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
   stage(root,'report_figures',dataset=ds,figures=len(names))
 def formatted(g,metric):
  r=g[g.metric==metric].iloc[0];return f"{100*r['mean']:.1f}% [{100*r.ci_low:.1f}, {100*r.ci_high:.1f}]"
 lines=[f"SAME-ARCHITECTURE {'RANDOM-FOREST PILOT' if p['arm']=='random_forest_pilot' else 'DECISION-TREE UNSEEN EVALUATION'}",'',
  'Methods and budgets are reported separately; all original people/actions are preserved. Unseen panels are new-seed refits and outer bootstrap refits, each with 50 distinct validation-admissible models. Same architecture means fixed hyperparameters, not identical learned splits.',
  'Each forest is one probability-averaging predictor with 10 depth-four constituent trees. The decision-tree reference preserves its original best-probability/least-cost-tie rule; the forest reference is a separate minimum-cost threshold solve. Native baseline actions are never replaced by ORPM actions.',
  'Intervals are pointwise 95% Student t intervals on person means within seed, with equal seed weights. Two-seed pilot intervals have df=1 and are not reliable full-study efficacy estimates. Ten-seed decision-tree intervals have df=9. Plots clip intervals to [0,1] for display; tables retain raw endpoints. Numerical solver uncertainty is distinct from sampling uncertainty.',
  'The fixed cohort was selected using the old decision-tree bank. Factual acceptance/gains and the fixed common-recourse-needed subset are reported separately; no forest-specific reselection was done. Conditional seen-valid and recourse-needed tables include supporting seed/person denominators.',
  'The same unseen panels score every method and setting; do not treat models or repeated methods as independent replicates. No averaging across budgets, set sizes or datasets. Conclusions are conditional on saved splits, architecture, admission and fitting/bootstrapping procedure. All-50 joint validity is stricter than mean unseen acceptance. Prefix panel-size curves use a fixed saved order.',
  'SingleForestReference with timeout/no accepted native incumbent returns the factual as an explicitly failed method outcome, never as proof of infeasibility. Forest full study remains UNAUTHORIZED and is not launched.','', 'ORPM at m=24 and the largest tested budget (mean [pointwise 95% CI]):']
 for ds in p['datasets']:
  for panel in PANELS:
   g=summary[(summary.dataset==ds)&(summary.panel==panel)&(summary.models==24)&(summary.budget==max(p['budgets'][ds]))&(summary.method=='ORPM')]
   lines.append(f"{LABELS.get(ds,ds)}; {panel}; B={max(p['budgets'][ds]):g}: seen {formatted(g,'seen_joint')}; unseen mean {formatted(g,'unseen_mean')}; all-50 joint {formatted(g,'unseen_joint')}")
 if (root/'full_study_estimate.json').exists():lines.extend(['','FULL FOREST PROJECTION',json.dumps(read(root/'full_study_estimate.json'),indent=2)])
 text='\n\n'.join(lines);(root/'report.txt').write_text(text+'\n')
 gallery=''.join(f'<figure><img src="figures/{n}.png" alt="{n}"><figcaption><a href="figures/{n}.pdf">PDF: {n}</a></figcaption></figure>' for n in names)
 (root/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Same-architecture unseen evaluation</title><style>body{font:16px system-ui;max-width:1150px;margin:35px auto;padding:20px;color:#183038}pre{white-space:pre-wrap}img{width:100%}figure{margin:40px 0}</style><h1>Same-architecture unseen evaluation</h1><pre>'+html.escape(text)+'</pre>'+gallery)
 with zipfile.ZipFile(root/'figures_tables.zip','w',zipfile.ZIP_DEFLATED) as z:
  for folder in [figdir,table]:
   for file in sorted(folder.iterdir()):z.write(file,str(file.relative_to(root)))
  for name in ['report.txt','report.html','protocol.json','verification.json']:
   z.write(root/name,name)
  if (root/'full_study_estimate.json').exists():z.write(root/'full_study_estimate.json','full_study_estimate.json')
 with zipfile.ZipFile(root/'figures_tables.zip') as z:assert z.testzip() is None
 expected=48 if p['arm']=='decision_tree' else 36
 assert len(names)==expected and len(list(figdir.glob('*.png')))==expected and len(list(figdir.glob('*.pdf')))==expected
 write(root/'analysis.json',dict(completed=True,utc=now(),figures=expected,png=expected,pdf=expected,summary_rows=len(summary),seed_count=len(p['seeds']),intervals='pointwise Student t; df=n_seed-1; conditional subsets report support',full_forest_authorized=False))
