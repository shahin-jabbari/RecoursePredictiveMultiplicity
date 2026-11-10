import csv, html, json, zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *

NAMES={'Give_Me_Some_Credit':'GMC','Folktables':'ACSIncome'}
def timing_estimate(root,completed=None):
 root=Path(root);cfg=read(root/'protocol.json');rows=[json.loads(s) for s in (root/'timing.jsonl').read_text().splitlines()] if (root/'timing.jsonl').exists() else []
 demand=read(root/'source_compute_demand.json');planned=read(root/'cohort_plan.json');by=defaultdict(list)
 for r in rows:by[(r['dataset'],r['kind'])].append(r)
 full=remaining=0.;details=[]
 for ds in DATASETS:
  for kind in ['singleton_shallow','singleton_deep','joint_6','joint_12','joint_24']:
   got=by[(ds,kind)];non=[r['seconds'] for r in got if r['optimization_seconds']>0]
   allsimilar=[r['seconds'] for r in rows if r['kind']==kind and r['optimization_seconds']>0]
   rate=float(np.mean(non)) if non else float(np.mean(allsimilar)) if allsimilar else 10.5
   future=sum(d['nontrivial'] for d in demand if d['dataset']==ds and d['kind']==kind)
   full+=rate*future
   total=sum(c['dataset']==ds for c in planned)*(12 if kind.startswith('singleton') else 1)
   unknown=max(0,total-len(got));fraction=(sum(r['optimization_seconds']>0 for r in got)/len(got)) if got else .75 if kind.startswith('singleton') else 1.
   remaining+=unknown*fraction*rate
   details.append(dict(dataset=ds,kind=kind,observed_tasks=len(got),observed_optimizations=len(non),mean_nontrivial_wall_seconds=rate,full_nontrivial_tasks=future,projected_full_seconds=rate*future))
 return dict(updated_utc=utc(),new_completed_tasks=len(rows),expected_tasks=cfg['expected_tasks'],pilot_remaining_hours=(remaining+300)/3600,
  pilot_remaining_planning_hours=[max(0,(.7*remaining+180)/3600),(1.6*remaining+900)/3600],full_saved_cohort_hours=(full*1.15+600)/3600,
  full_saved_cohort_planning_hours=[(full*.7+300)/3600,(full*1.7+1800)/3600],full_optimization_cap_hours=sum(d['nontrivial'] for d in demand)*10/3600,
  same_ten_second_limits=True,full_expansion_authorized=False,note='Timing extrapolation, not a statistical confidence interval. Selected people, hard instances, encoding and I/O can change runtimes. A 60-second study needs separate calibration.',details=details)

def ci(v):
 a=np.asarray(v,float);n=len(a);mean=float(np.mean(a));half=float(t.ppf(.975,n-1)*np.std(a,ddof=1)/np.sqrt(n)) if n>1 else None
 return mean,None if half is None else mean-half,None if half is None else mean+half,n
def summarize(frame,keys):
 rows=[]
 for key,g in frame.groupby(keys,dropna=False,sort=False):
  if not isinstance(key,tuple):key=(key,)
  r=dict(zip(keys,key));l=ci(g['lower']);u=ci(g['upper'])
  r.update(estimate_lower=l[0],estimate_upper=u[0],lower_endpoint_ci95_low=l[1],lower_endpoint_ci95_high=l[2],upper_endpoint_ci95_low=u[1],upper_endpoint_ci95_high=u[2],seeds=l[3],df=l[3]-1)
  rows.append(r)
 return pd.DataFrame(rows)
def seed_means(rows,keys,weighted=False):
 frame=pd.DataFrame(rows);out=[]
 for key,g in frame.groupby(keys+['seed'],sort=False):
  r=dict(zip(keys+['seed'],key if isinstance(key,tuple) else (key,)))
  if weighted:
   grouped=g.groupby('group',sort=False).agg(lower=('lower','mean'),upper=('upper','mean'),weight=('group_weight','first'))
   w=grouped['weight']/grouped['weight'].sum();lo=float(sum(w*grouped['lower']));hi=float(sum(w*grouped['upper']))
  else:lo=float(g['lower'].mean());hi=float(g['upper'].mean())
  r.update(lower=lo,upper=hi,people=len(g));out.append(r)
 return pd.DataFrame(out)
def row_interval(c,n,metric,lo,hi,**kw):
 assert lo<=hi+1e-7,(metric,lo,hi)
 return dict(dataset=c['dataset'],seed=c['seed'],instance_id=c['instance_id'],group=c['group'],group_weight=c['group_weight'],models=n,metric=metric,lower=float(lo),upper=float(max(lo,hi)),**kw)
def success(info,B):
 lo=float(info['available'] and info['cost_upper']<=B+1e-8)
 hi=float(not info['infeasible_within_horizon'] and (info['cost_lower'] is None or info['cost_lower']<=B+1e-8))
 assert lo<=hi
 return lo,hi
def pilot_rows(root):
 costs=[];curves=[];outcomes=[]
 for c in read(root/'cohort_plan.json'):
  directory=root/'cases'/c['key'];task={}
  for p in (directory/'tasks').glob('*/result.json'):
   r=read(p);task[r['task']]=r['info'];outcomes.append(dict(dataset=c['dataset'],seed=c['seed'],kind=r['kind'],timeout=float(r['info']['solver_status']==9),certified=float(r['info']['certified']),native_incumbent=float(r['info']['native_incumbent']),width=r['info']['capped_upper']-r['info']['capped_lower']))
  original=read(directory/'saved_orpm.json')
  for n in [6,12,24]:
   single=[task[f'single_{j}'] for j in indices(n)];joint=task[f'joint_{n}'];L=np.array([s['capped_lower'] for s in single]);U=np.array([s['capped_upper'] for s in single])
   jl=max(joint['capped_lower'],float(L.max()));ju=joint['capped_upper'];U=np.minimum(U,ju)
   assert np.all(L<=U+1e-5) and jl<=ju+1e-5
   vals={'singleton_capped_cost':(L.mean(),U.mean()),'hardest_singleton_capped_cost':(L.max(),U.max()),'common_capped_cost':(jl,ju),
    'common_minus_mean_singleton':(max(0,jl-U.mean()),max(0,ju-L.mean())),
    'common_minus_hardest_singleton':(max(0,jl-U.max()),max(0,ju-L.max()))}
   for metric,(lo,hi) in vals.items():costs.append(row_interval(c,n,metric,lo,hi))
   for r in [r for r in original if r['models']==n]:
    B=r['budget'];s=[success(v,B) for v in single];sl=float(np.mean([v[0] for v in s]));su=float(np.mean([v[1] for v in s]));el=min(v[0] for v in s);eu=min(v[1] for v in s)
    cl,cu=success(joint,B);cl=max(cl,float(r['joint_validity']));cu=min(cu,eu)
    # A verified common witness also proves each separate-model feasibility.
    el=max(el,cl);sl=max(sl,cl);su=max(su,sl);eu=max(eu,el);o=float(r['joint_validity'])
    for metric,(lo,hi) in {'singleton_success':(sl,su),'all_separate_success':(el,eu),'common_success':(cl,cu),'ORPM_success':(o,o),
     'harder_model_penalty':(max(0,sl-eu),max(0,su-el)),'common_action_penalty':(max(0,el-cu),max(0,eu-cl)),
     'ORPM_shortfall':(max(0,cl-o),max(0,cu-o))}.items():curves.append(row_interval(c,n,metric,lo,hi,budget=B))
 return costs,curves,pd.DataFrame(outcomes)
def gaps(seedframe,keys):
 out=[]
 for ds,gp in GROUPS.items():
  g=seedframe[seedframe.dataset==ds];a=g[g.group==gp['first']];b=g[g.group==gp['second']]
  p=a.merge(b,on=keys+['seed'],suffixes=('_a','_b'))
  for _,r in p.iterrows():
   out.append({**{k:r[k] for k in keys+['seed']},'comparison':gp['first']+' minus '+gp['second'],'lower':r.lower_a-r.upper_b,'upper':r.upper_a-r.lower_b,'people_first':r.people_a,'people_second':r.people_b})
 return pd.DataFrame(out)
def paired_change(seedframe,column,low,high,keys,label):
 a=seedframe[seedframe[column]==high];b=seedframe[seedframe[column]==low]
 p=a.merge(b,on=keys+['seed'],suffixes=('_a','_b'));rows=[]
 for _,r in p.iterrows():rows.append({**{k:r[k] for k in keys+['seed']},'comparison':label,'lower':r.lower_a-r.upper_b,'upper':r.upper_a-r.lower_b})
 return pd.DataFrame(rows)
def plot(table,ds,metrics,xcol,title,ylabel,path,filter_values=None):
 fig,ax=plt.subplots(figsize=(7.5,5.2));part=table[table.dataset==ds]
 if filter_values:
  for key,value in filter_values.items():part=part[part[key]==value]
 colors=plt.get_cmap('tab10').colors
 for i,metric in enumerate(metrics):
  z=part[part.metric==metric].sort_values(xcol)
  if z.empty:continue
  x=z[xcol].to_numpy(float);lo=z.estimate_lower.to_numpy(float);hi=z.estimate_upper.to_numpy(float);mid=(lo+hi)/2
  ax.fill_between(x,lo,hi,color=colors[i],alpha=.15)
  ax.plot(x,mid,'o-',color=colors[i],label=metric.replace('_',' '))
  # These bars enclose the pointwise 95% intervals of both bound endpoints.
  low=z.lower_endpoint_ci95_low.to_numpy(float);high=z.upper_endpoint_ci95_high.to_numpy(float)
  if np.isfinite(low).all():ax.errorbar(x,mid,yerr=np.maximum(0,np.array([mid-low,high-mid])),fmt='none',ecolor=colors[i],capsize=3,alpha=.7)
 ax.axhline(0,color='#999',lw=.6);ax.set(title=f'{NAMES.get(ds,ds)} — {title}',xlabel='L2 budget' if xcol=='budget' else 'Competing models',ylabel=ylabel)
 ax.legend(fontsize=8,loc='best');ax.grid(alpha=.2);fig.text(.01,.01,'Shading: numerical identification range. Bars: pointwise 95% CIs for bound endpoints. Pilot: 3 seeds.',fontsize=8)
 fig.tight_layout(rect=(0,.04,1,1));path.parent.mkdir(parents=True,exist_ok=True)
 for ext in ['png','pdf']:fig.savefig(path.with_suffix('.'+ext),dpi=180)
 plt.close(fig)
def table_html(df):return df.to_html(index=False,float_format=lambda x:f'{x:.4g}',border=0)
def report(dest,title,paragraphs,tables,images,extras):
 dest.mkdir(parents=True,exist_ok=True)
 text=title+'\n\n'+'\n\n'.join(paragraphs)+'\n\n'
 body='<h1>'+html.escape(title)+'</h1>'+''.join('<p>'+html.escape(p)+'</p>' for p in paragraphs)
 for name,df in tables:
  df.to_csv(dest/'tables'/name,index=False);text+=name+'\n'+df.to_string(index=False)+'\n\n';body+='<h2>'+html.escape(name)+'</h2>'+table_html(df)
 for image in images:body+='<figure><img src="figures/'+image.name+'" style="max-width:100%"></figure>'
 (dest/'report.txt').write_text(text);(dest/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>'+html.escape(title)+'</title><style>body{font:16px/1.5 system-ui;max-width:1200px;margin:3em auto;padding:0 2em}table{display:block;overflow:auto;font-size:12px;border-collapse:collapse}td,th{padding:5px;border-bottom:1px solid #ddd}h2{margin-top:2em}</style>'+body)
 with zipfile.ZipFile(dest/'figures_tables.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
  for p in list((dest/'tables').glob('*'))+list((dest/'figures').glob('*'))+[dest/'report.txt',dest/'report.html']:z.write(p,str(p.relative_to(dest)))
 return dict(panels=len(images),png=len(list((dest/'figures').glob('*.png'))),pdf=len(list((dest/'figures').glob('*.pdf'))),tables=len(tables),archive_sha256=sha(dest/'figures_tables.zip'))
def analyze(root):
 root=Path(root);cfg=read(root/'protocol.json');costdest=root/'cost_report';disdest=Path(cfg['disparity_result'])
 for dest in [costdest,disdest]:(dest/'tables').mkdir(parents=True,exist_ok=True);(dest/'figures').mkdir(exist_ok=True)
 costs,curves,outcomes=pilot_rows(root)
 cs=seed_means(costs,['dataset','models','metric'],True);ss=seed_means(curves,['dataset','models','budget','metric'],True)
 cc=summarize(cs,['dataset','models','metric']);sc=summarize(ss,['dataset','models','budget','metric'])
 cg=seed_means(costs,['dataset','models','metric','group']);sg=seed_means(curves,['dataset','models','budget','metric','group'])
 cgap=gaps(cg,['dataset','models','metric']);sgap=gaps(sg,['dataset','models','budget','metric'])
 cgs=summarize(cgap,['dataset','models','metric','comparison']);sgs=summarize(sgap,['dataset','models','budget','metric','comparison'])
 source=pd.read_csv(root/'saved_orpm_results.csv');source['joint_validity']=source.joint_validity.astype(float)
 reused=[]
 for _,r in source.iterrows():reused.append(dict(dataset=r.dataset,seed=r.seed,instance_id=r.instance_id,models=r.models,budget=r.budget,group=r.group,group_weight=r.group_weight,metric='saved_ORPM_success',lower=r.joint_validity,upper=r.joint_validity))
 rs=seed_means(reused,['dataset','models','budget','metric']);rc=summarize(rs,['dataset','models','budget','metric'])
 rg=seed_means(reused,['dataset','models','budget','metric','group']);rgg=gaps(rg,['dataset','models','budget','metric']);rgc=summarize(rgg,['dataset','models','budget','metric','comparison'])
 bench=[]
 for metric in ['timeout','certified','native_incumbent','width']:
  for (ds,seed,kind),g in outcomes.groupby(['dataset','seed','kind']):
   val=float(g[metric].mean());bench.append(dict(dataset=ds,seed=seed,kind=kind,metric=metric,lower=val,upper=val))
 bc=summarize(pd.DataFrame(bench),['dataset','kind','metric'])
 nc=paired_change(cs,'models',6,24,['dataset','metric'],'24 minus 6 models')
 ns=paired_change(ss,'models',6,24,['dataset','budget','metric'],'24 minus 6 models')
 budget_parts=[]
 for ds in DATASETS:
  bs=[x for x in cfg['budgets'][ds] if x>0]
  budget_parts.append(paired_change(ss[ss.dataset==ds],'budget',min(bs),max(bs),['dataset','models','metric'],'largest minus smallest positive budget'))
 budget_change=pd.concat(budget_parts,ignore_index=True)
 eta=timing_estimate(root);save(root/'runtime_estimate.json',eta)
 cimages=[];dimages=[]
 for ds in DATASETS:
  for n in [6,12,24]:
   p=costdest/'figures'/f'{ds}_n{n}_success';plot(sc,ds,['singleton_success','all_separate_success','common_success','ORPM_success'],'budget',f'pilot success, {n} models','Fraction of people / mean over models',p,{'models':n});cimages.append(p.with_suffix('.png'))
  p=costdest/'figures'/f'{ds}_cost';plot(cc,ds,['singleton_capped_cost','hardest_singleton_capped_cost','common_capped_cost'],'models','capped minimum L2','Standardized L2',p);cimages.append(p.with_suffix('.png'))
  p=costdest/'figures'/f'{ds}_premium';plot(cc,ds,['common_minus_mean_singleton','common_minus_hardest_singleton'],'models','multiplicity cost premium','Additional standardized L2',p);cimages.append(p.with_suffix('.png'))
  if ds in GROUPS:
   p=disdest/'figures'/f'{ds}_cost_gap';plot(cgs,ds,['singleton_capped_cost','common_capped_cost','common_minus_mean_singleton'],'models','first-minus-second group cost gap','Difference in standardized L2',p);dimages.append(p.with_suffix('.png'))
   for n in [6,12,24]:
    p=disdest/'figures'/f'{ds}_n{n}_success_gap';plot(sgs,ds,['singleton_success','common_success','ORPM_success'],'budget',f'group success gap, {n} models','First-minus-second group difference',p,{'models':n});dimages.append(p.with_suffix('.png'))
 counts=pd.DataFrame(read(root/'group_counts.json'));counts.to_csv(disdest/'tables/group_counts_descriptive.csv',index=False)
 common=[
  'Completed timed pilot: 36 saved people, three seeds, six datasets, 864 singleton cost tasks and 108 common-action tasks. Models and ORPM actions are reused from the trade-off study; no training. These neural sets mix 20-ReLU and 50/100-ReLU architectures in equal proportions.',
  'The figures concern the new small pilot. Separate tables summarize all 26,400 saved neural ORPM settings across ten seeds and twenty people per seed. Pilot and source cohorts overlap and are not independent replications.',
  'Minimum costs are capped at the dataset maximum budget. Numerical lower/upper bounds are separate from pointwise 95% Student-t sampling intervals on each endpoint (pilot df=2; reused source df=9). The endpoint envelope is descriptive, not a simultaneous band. Shading shows optimization uncertainty; bars show endpoint sampling intervals. Exact values and intervals are in the tables.',
  'Each nontrivial optimization received ten seconds and one thread. Timeouts and missing incumbents remain outcomes. No incumbent does not mean infeasible. Saved ORPM witnesses only improve upper bounds after solving, without warm-starting MIQCP. Native solver outcomes are reported separately. An unresolved range cannot establish either a cost premium or its absence.',
  'The pilot is for timing and implementation validation; subgroup samples are too small for firm disparity conclusions. All statements are conditional on the saved benchmark splits, fitted models, selected people and immutable-feature action regions. Standardized L2 is not money; no causal fairness or arbitrary-network generalization claim.',
  f'Timing-based full saved-cohort estimate at these same ten-second limits: {eta["full_saved_cohort_hours"]:.1f} hours central; planning range {eta["full_saved_cohort_planning_hours"][0]:.1f}–{eta["full_saved_cohort_planning_hours"][1]:.1f} hours. This is extrapolation, not a confidence interval. Full expansion has not been launched.']
 costtables=[('pilot_cost_95ci.csv',cc),('pilot_success_95ci.csv',sc),('saved_ORPM_success_95ci.csv',rc),('solver_outcomes_95ci.csv',bc),
  ('paired_model_count_cost_effects_95ci.csv',summarize(nc,['dataset','metric','comparison'])),
  ('paired_model_count_success_effects_95ci.csv',summarize(ns,['dataset','budget','metric','comparison'])),
  ('paired_budget_effects_95ci.csv',summarize(budget_change,['dataset','models','metric','comparison']))]
 distables=[('pilot_group_cost_95ci.csv',summarize(cg,['dataset','models','metric','group'])),('pilot_group_success_95ci.csv',summarize(sg,['dataset','models','budget','metric','group'])),('pilot_group_cost_gaps_95ci.csv',cgs),('pilot_group_success_gaps_95ci.csv',sgs),('saved_ORPM_group_gaps_95ci.csv',rgc)]
 cost_findings=[];disparity_findings=[]
 for _,r in cc[(cc.models==24)&(cc.metric=='common_minus_mean_singleton')].iterrows():
  cost_findings.append(f'{NAMES.get(r.dataset,r.dataset)}, 24 models: common-action capped cost premium over the average singleton lies in [{r.estimate_lower:.3g}, {r.estimate_upper:.3g}] standardized L2; the envelope of endpoint 95% sampling intervals is [{r.lower_endpoint_ci95_low:.3g}, {r.upper_endpoint_ci95_high:.3g}].')
 for _,r in cgs[(cgs.models==24)&(cgs.metric=='common_minus_mean_singleton')].iterrows():
  disparity_findings.append(f'{NAMES.get(r.dataset,r.dataset)}, 24 models: first-minus-second group gap in added multiplicity burden lies in [{r.estimate_lower:.3g}, {r.estimate_upper:.3g}] L2; endpoint 95% interval envelope [{r.lower_endpoint_ci95_low:.3g}, {r.upper_endpoint_ci95_high:.3g}], from {int(r.seeds)} paired seeds.')
 c=report(costdest,'Neural cost of multiplicity: timed reuse pilot',common+['Singleton success permits a different action for each model. All-separate success requires every model to have its own feasible action; common success requires one action for all models. Their differences separate the harder-model effect, common-action burden and ORPM shortfall.']+cost_findings,costtables,cimages,[])
 d=report(disdest,'Neural disparity of multiplicity: timed reuse pilot',common+['Group gaps are first minus second: recorded female minus male for COMPAS/ACSIncome; under 40 minus 40 and older for German/GMC. Missing original groups are omitted from paired comparisons, never filled with zero. Pilot overall means use source-cohort group proportions, while group means are conditional.']+disparity_findings,distables,dimages,[])
 save(root/'analysis.json',dict(passed=True,cost=c,disparity=d,completed_utc=utc(),confidence_intervals='pointwise descriptive t intervals for each endpoint',pilot_seed_df=2,source_seed_df=9))
