from common import *
import csv,html,zipfile,collections
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FuncFormatter

METRICS=['gap_lower','gap_upper','ever_attained','current_attained','joint_validity','cost','point_gap_upper']
LABELS={'Synthetic':'Synthetic','COMPAS':'COMPAS','Polish':'Polish','German':'German','Give_Me_Some_Credit':'GMC','Folktables':'ACSIncome'}
def ci(values,clip=None):
 a=np.asarray(values,float);n=len(a);mu=np.mean(a,axis=0)
 if n>1:
  half=student_t.ppf(.975,n-1)*np.std(a,axis=0,ddof=1)/np.sqrt(n);lo=mu-half;hi=mu+half
 else:lo=np.full_like(mu,np.nan);hi=np.full_like(mu,np.nan)
 if clip is not None:lo=np.maximum(clip[0],lo);hi=np.minimum(clip[1],hi)
 return mu,lo,hi,n
def write_csv(path,rs):
 assert rs
 with path.open('w',newline='') as f:
  keys=list(dict.fromkeys(k for row in rs for k in row));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rs)
def stats_record(values,**fields):
 mu,lo,hi,n=ci(values)
 return dict(**fields,mean=float(mu),ci95_low=float(lo),ci95_high=float(hi),seeds=n,df=n-1)
def label_gap_axis(ax):
 lower,upper=ax.get_ylim();lower=max(0.,lower);ax.set_ylim(lower,upper)
 ticks=[v for v in [0,.001,.002,.005,.01,.02,.05,.1,.2,.5,1] if lower<=v<=upper]
 ax.yaxis.set_major_locator(FixedLocator(ticks));ax.yaxis.set_major_formatter(FuncFormatter(lambda value,_:f'{value:g}'));ax.minorticks_off()

def analyze(root):
 root=Path(root);p=read(root/'protocol.json');plan=read(root/'task_plan.json');assert read(root/'verification.json')['passed'];deadline_check(root)
 figures=root/'figures';tables=root/'tables';figures.mkdir(exist_ok=True);tables.mkdir(exist_ok=True)
 grouped={};solver_grouped={};status=collections.Counter();rejected=incumbents=0
 for cell in plan['cells']:
  d=root/'trajectories'/cell['key'];rs=rows(d/'rounds.jsonl');ds=cell['dataset'];m=cell['models'];b=cell['budget'];seed=cell['seed']
  values=np.array([[r['mixture_gap'][0],r['mixture_gap'][1],int(r['first_certified_round'] is not None),int(r['mixture_gap'][1]<=.005),int(r['joint_valid']),r['best_cost'],r['point_gap'][1]] for r in rs])
  grouped.setdefault((ds,m,b),{}).setdefault(seed,[]).append(values)
  infos=[r['oracle'] for r in rs]+[r['oracle'] for r in rows(d/'diagnostics.jsonl')]+[read(d/'reference.json')['oracle']]
  for info in infos:status[str(info['solver_status'])]+=1;rejected+=int(info.get('certificate_rejected',False));incumbents+=int(info.get('incumbent_rejected',False))
  solver_grouped.setdefault((ds,m,b),{}).setdefault(seed,[]).append([sum(z['solver_status']==9 for z in infos)/len(infos),sum(z['global_certificate'] for z in infos)/len(infos),sum(z.get('certificate_rejected',False) for z in infos)/len(infos)])
 seeds={k:{seed:np.mean(a,axis=0) for seed,a in v.items()} for k,v in grouped.items()}
 curve_rows=[];cell_rows=[];seed_rows=[];solver_rows=[]
 for (ds,m,b),v in seeds.items():
  ordered=sorted(v);arr=np.array([v[s] for s in ordered])
  for j,metric in enumerate(METRICS):
   clip=(0,1) if metric in ['ever_attained','current_attained','joint_validity'] else (0,np.inf)
   mu,lo,hi,n=ci(arr[:,:,j],clip)
   for t in range(200):
    row=dict(dataset=ds,models=m,budget=b,round=t+1,metric=metric,mean=mu[t],ci95_low=lo[t],ci95_high=hi[t],seeds=n,df=n-1)
    curve_rows.append(row)
    if t+1 in [50,100,200]:cell_rows.append(row)
   for seed in ordered:
    for t in [50,100,200]:seed_rows.append(dict(dataset=ds,models=m,budget=b,seed=seed,round=t,metric=metric,seed_mean=v[seed][t-1,j],people=len(grouped[(ds,m,b)][seed])))
  for j,metric in enumerate(['timeout_fraction','native_certified_fraction','rejected_certificate_fraction']):
   solver_rows.append(stats_record([np.mean(solver_grouped[(ds,m,b)][s],axis=0)[j] for s in ordered],dataset=ds,models=m,budget=b,metric=metric))
 paired=[]
 for ds in p['datasets']:
  budgets=p['positive_budgets'][ds]
  for m in p['model_counts']:
   for b in budgets:
    v=seeds[(ds,m,b)]
    for j,metric in enumerate(METRICS):paired.append(stats_record([a[199,j]-a[99,j] for a in v.values()],contrast='round_200_minus_100',dataset=ds,models=m,budget=b,metric=metric))
   if len(budgets)>1:
    a,c=seeds[(ds,m,budgets[-1])],seeds[(ds,m,budgets[0])];ss=sorted(set(a)&set(c))
    for j,metric in enumerate(METRICS):paired.append(stats_record([a[s][199,j]-c[s][199,j] for s in ss],contrast='high_minus_low_budget_at_200',dataset=ds,models=m,low_budget=budgets[0],high_budget=budgets[-1],metric=metric))
  for b in budgets:
   for low,high in zip(p['model_counts'][:-1],p['model_counts'][1:]):
    a,c=seeds[(ds,high,b)],seeds[(ds,low,b)];ss=sorted(set(a)&set(c))
    for j,metric in enumerate(METRICS):paired.append(stats_record([a[s][199,j]-c[s][199,j] for s in ss],contrast='larger_minus_smaller_set_at_200',dataset=ds,budget=b,low_models=low,high_models=high,metric=metric))
 write_csv(tables/'curves_95ci.csv',curve_rows);write_csv(tables/'cells_50_100_200_95ci.csv',cell_rows);write_csv(tables/'seed_means.csv',seed_rows);write_csv(tables/'paired_effects_95ci.csv',paired);write_csv(tables/'solver_outcomes_95ci.csv',solver_rows)
 colors=['#0072B2','#D55E00','#009E73'];rounds=np.arange(1,201)
 def draw(ax,ds,m,metric):
  for b,color in zip(p['positive_budgets'][ds],colors):
   j=METRICS.index(metric);vals=np.array([v[:,j] for v in seeds[(ds,m,b)].values()]);mu,lo,hi,n=ci(vals,(0,1) if metric=='ever_attained' else (0,np.inf))
   ax.plot(rounds,mu,color=color,label=f'L2 budget {b:g}');ax.fill_between(rounds,lo,hi,color=color,alpha=.15)
  ax.set_xlabel('ORPM rounds');ax.grid(alpha=.2);ax.legend(fontsize=8)
  if metric=='gap_upper':
   ax.axhline(.005,c='black',ls=':',label='target 0.005');ax.set_yscale('symlog',linthresh=.001);ax.set_ylabel('Mean certified upper mixture-gap bound');label_gap_axis(ax)
  else:ax.set_ylim(-.02,1.02);ax.set_ylabel('Fraction attaining target by round')
 for ds in p['datasets']:
  for m in p['model_counts']:
   deadline_check(root);fig,axes=plt.subplots(1,2,figsize=(10,3.7))
   for ax,metric in zip(axes,['gap_upper','ever_attained']):draw(ax,ds,m,metric)
   fig.suptitle(f'{LABELS[ds]} | {m} trees | pointwise 95% seed intervals');fig.tight_layout()
   for ext in ['png','pdf']:fig.savefig(figures/f'{ds}_m{m}.{ext}',dpi=180)
   plt.close(fig)
   for metric in ['gap_upper','ever_attained']:
    fig,ax=plt.subplots(figsize=(5.4,3.8));draw(ax,ds,m,metric);ax.set_title(f'{LABELS[ds]}, {m} trees');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(figures/f'{ds}_m{m}_{metric}.{ext}',dpi=180)
    plt.close(fig)
 description=f"Tree-convergence main study: {len(p['datasets'])} datasets, {len(p['seeds'])} seeds, up to {p['people_per_dataset_seed']} held-out people per dataset/seed; {plan['people']} actual people, {plan['expected_trajectories']} trajectories and {plan['expected_rounds']} ORPM rounds. Nested 6/12/24 depth-four decision trees, the same dataset-specific positive L2 budgets, 200 rounds, eta=sqrt(log(m)/200), and loss (1-p)^2. Each trajectory starts at the factual with uniform model weights; actions are not transferred across budgets or set sizes. Best observed worst-loss action is returned, with a cost tie-break.\n\n"
 description+="The primary outcome is first certified attainment of upper empirical-action-mixture gap <=0.005; current attainment is separately tabulated. Deterministic validity (all competing models accept) and the deterministic-action gap remain separate outcomes. Budget/count/dataset results are never pooled. Individuals are averaged within seed and seeds weighted equally. All plotted bands and inferential tables use pointwise 95% Student-t intervals over seed means (df=9 with ten seeds), descriptive and not simultaneous. Lower/upper numerical optimization bounds are stored separately and are not sampling confidence intervals.\n\n"
 description+="Model admission uses a fixed best-of-first-96 validation reference with allowance 0.05; an ordered reserve of at most 1536 candidates supplies 24 distinct eligible trees without relaxing the cutoff. All 60 banks and paired cohorts froze before recourse. Oracle/checkpoint/reference optimization caps remain 1/2/30 seconds and one thread. Timeouts and numerical-certificate rejections are outcomes; verified bounds and feasible fallback actions are independently audited. The calibration is separate (seed 390) and is not included in the main estimates.\n\n"
 description+='This expansion preserves completed trajectories and continues saved 100-round prefixes with independently evaluated checkpoints. Numerical bounds and sampling uncertainty are distinct.\n\n'
 description+=f"Native solver statuses: {dict(status)}. Rejected numerical certificates: {rejected}; rejected budget-infeasible incumbents: {incumbents}. Detailed frequencies with seed-level intervals are in solver_outcomes_95ci.csv. Historical plus newly recorded trajectory work (not expansion elapsed time): {sum(z['total_seconds'] for z in rows(root/'summaries.jsonl'))/3600:.2f} hours, excluding preparation and final verification/reporting.\n\n"
 def formatted(row):return f"{row['mean']:.4f} [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]"
 description+='Cell results at round 200 (mean [pointwise 95% CI]):\n'
 for ds in p['datasets']:
  for m in p['model_counts']:
   for b in p['positive_budgets'][ds]:
    subset={r['metric']:r for r in cell_rows if r['dataset']==ds and r['models']==m and r['budget']==b and r['round']==200}
    description+=f"{LABELS[ds]}, {m} trees, B={b:g}: upper mixture gap {formatted(subset['gap_upper'])}; ever attained {formatted(subset['ever_attained'])}; current attained {formatted(subset['current_attained'])}; joint validity {formatted(subset['joint_validity'])}.\n"
 description+='\nPaired effects (same people/seeds; mean difference [95% CI]):\n'
 for row in paired:
  if row['metric'] not in ['joint_validity','gap_upper','ever_attained']:continue
  desc=', '.join(f'{k}={v}' for k,v in row.items() if k not in ['mean','ci95_low','ci95_high','df','seeds'])
  description+=desc+': '+formatted(row)+'\n'
 (root/'report.txt').write_text(description)
 document='<!doctype html><html><meta charset="utf-8"><title>Tree convergence: 10 seeds × 20 people</title><style>body{font:16px system-ui;max-width:1150px;margin:35px auto;padding:20px}pre{white-space:pre-wrap;line-height:1.5}img{max-width:100%}</style><h1>Tree convergence: 10 seeds × 20 people</h1><pre>'+html.escape(description)+'</pre>'
 for ds in p['datasets']:
  for m in p['model_counts']:document+=f'<h2>{LABELS[ds]}: {m} trees</h2><img src="figures/{ds}_m{m}.png">'
 (root/'report.html').write_text(document+'</html>')
 count=len(p['datasets'])*len(p['model_counts']);write(root/'analysis.json',dict(completed=True,utc=now(),trajectories=plan['expected_trajectories'],people=plan['people'],seeds=len(p['seeds']),sampling_CIs=True,confidence=.95,df=len(p['seeds'])-1,separate_panels=count*2,combined_figures=count,native_solver_statuses=dict(status),rejected_certificates=rejected,rejected_infeasible_incumbents=incumbents,pointwise_not_simultaneous=True,no_budget_count_dataset_pooling=True))
 assert len(list(figures.glob('*.png')))==count*3 and len(list(figures.glob('*.pdf')))==count*3
 with zipfile.ZipFile(root/'figures_tables.zip','w',zipfile.ZIP_DEFLATED) as z:
  for folder in [figures,tables]:
   for f in folder.iterdir():z.write(f,f.relative_to(root))
  for name in ['report.txt','report.html','analysis.json','verification.json','protocol.json','task_plan.json','model_inventory.json']:z.write(root/name,name)
 with zipfile.ZipFile(root/'figures_tables.zip') as z:assert z.testzip() is None
 deadline_check(root)
