#!/usr/bin/env python3
import argparse,csv,datetime as dt,hashlib,html,importlib.util,json,os,sys,time,zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t

ROOT0=Path('/Users/sj987/Documents/ChatGPT/Code/Predictive_Multiplicity_Recourse/V7/experiments/linear_exact_unseen/retraining_transfer/phase_pilot/results/20261019T115745.955161Z_0c2637db86')
s=importlib.util.spec_from_file_location('frozen_transfer_runner',ROOT0/'source/runner.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
METRICS=m.METRICS;DATASETS=m.DATASETS;SEEDS=m.SEEDS;METHODS=m.METHODS;MODEL_COUNTS=m.MODEL_COUNTS;UNSEEN_COUNTS=m.UNSEEN_COUNTS;PANEL_TYPES=m.PANEL_TYPES;BUDGETS=m.BUDGETS
SOURCE=m.FULL
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha_file(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def write_json(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def log(*a):print(utc(),*a,flush=True)
def recovery(root):
 root=Path(root);tables=root/'tables';tables.mkdir(exist_ok=True)
 progress=json.loads((root/'progress.json').read_text())
 if progress.get('status')!='failed' or progress.get('completed_records')!=1080000:
  raise RuntimeError('partial run is not at the expected recoverable checkpoint')
 # Validate the existing partial file: 90,000 source records x 12 panel evaluations.
 evalfile=root/'evaluation_records.csv'
 with evalfile.open(newline='') as f: existing=sum(1 for _ in csv.DictReader(f))
 if existing!=1080000:raise RuntimeError(f'partial CSV has {existing} rows, expected 1,080,000')
 source_rows=[]
 with (SOURCE/'records.jsonl').open() as f:
  for line in f:
   r=json.loads(line)
   if r['dataset'] in DATASETS:source_rows.append(r)
 if len(source_rows)!=92400:raise RuntimeError(f'expected 92,400 source rows, got {len(source_rows)}')
 panels={}
 for ds in DATASETS:
  for seed in SEEDS:
   path=SOURCE/'blocks'/ds/f'seed_{seed}'/'panel.npz'
   with np.load(path,allow_pickle=False) as z:ha,hb=z['unseen_A'].copy(),z['unseen_b'].copy()
   rng=np.random.default_rng(81000000+seed+{'Synthetic':1,'COMPAS':2,'Give_Me_Some_Credit':3}[ds])
   panels[(ds,seed,'hessian_existing')]=(ha,hb,rng.permutation(len(hb)))
   for arm in ['bootstrap_full_feature','bootstrap_matched_mask']:
    path=root/'panels'/arm/ds/f'seed_{seed}.npz'
    with np.load(path,allow_pickle=False) as z:panels[(ds,seed,arm)]=(z['A'].copy(),z['b'].copy(),z['order'].copy())
 # Resume safely if an earlier recovery attempt appended rows before it stopped.
 base_rows=1080000
 if existing<base_rows or existing>1108800 or (existing-base_rows)%12:
  raise RuntimeError(f'partial CSV has non-resumable row count {existing}')
 already_recovered=(existing-base_rows)//12
 cache={}
 def load_points(rel):
  if rel not in cache:
   with np.load(SOURCE/rel,allow_pickle=False) as z:cache[rel]={k:z[k].copy() for k in z.files if k in METHODS}
   if len(cache)>256:cache.pop(next(iter(cache)))
  return cache[rel]
 fields=['dataset','seed','person','test_id','competing_model_count','budget','budget_index','method','available','panel_type','unseen_panel_size']+METRICS+['boundary_count']
 tol=1e-7;written=0
 with evalfile.open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields)
  for r in source_rows[90000+already_recovered:]:
   ds=r['dataset'];seed=int(r['seed']);method=r['method'];available=bool(r['available'])
   x=np.asarray(load_points(r['point_file'])[method],float)
   for arm in PANEL_TYPES:
    A,b,order=panels[(ds,seed,arm)];zall=np.asarray(A[order],float)@x+np.asarray(b[order],float)
    for k in UNSEEN_COUNTS:
     z=zall[:k];valid=z>=-tol;strict=z>=0
     row={'dataset':ds,'seed':seed,'person':int(r['person']),'test_id':int(r['test_id']),
      'competing_model_count':int(r['model_count']),'budget':float(r['budget']),'budget_index':int(r['budget_index']),
      'method':method,'available':available,'panel_type':arm,'unseen_panel_size':k,
      'joint_validity':float(bool(available and valid.all())),
      'strict_joint_validity':float(bool(available and strict.all())),
      'mean_acceptance':float(valid.mean()) if available else 0.,
      'minimum_logit':float(z.min()),'worst_loss':float(np.logaddexp(0.,-z).max()),
      'boundary_count':int(np.sum(np.abs(z)<=tol))}
     w.writerow(row);written+=1
 remaining=28800-already_recovered*12
 if written!=remaining:raise RuntimeError(f'expected {remaining} remaining rows, appended {written}')
 log('RECOVERED_REMAINDER_ROWS',already_recovered*12+written)
 # Aggregate each person's score to a seed mean, then infer across the ten outer seeds.
 sums=defaultdict(float);counts=defaultdict(int)
 with evalfile.open(newline='') as f:
  for r in csv.DictReader(f):
   prefix=(r['dataset'],r['panel_type'],int(r['unseen_panel_size']),int(r['competing_model_count']),float(r['budget']),r['method'],int(r['seed']))
   for metric in METRICS:
    key=prefix+(metric,);sums[key]+=float(r[metric]);counts[key]+=1
 seedmeans={}
 for key,val in sums.items():
  n=counts[key]
  if n!=20:raise RuntimeError(f'expected 20 people in seed cell {key}, found {n}')
  seedmeans[key]=val/n
 with (tables/'seed_means.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['dataset','panel_type','unseen_panel_size','competing_model_count','budget','method','seed','metric','mean_person_value','people'])
  for key,val in sorted(seedmeans.items(),key=lambda x:str(x[0])):w.writerow(list(key[:-1])+[key[-1],val,20])
 groups=defaultdict(list)
 for key,val in seedmeans.items():groups[key[:-2]+(key[-1],)].append((key[-2],val))
 summary={}
 with (tables/'all_cells_95ci.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['dataset','panel_type','unseen_panel_size','competing_model_count','budget','method','metric','mean','ci_low','ci_high','seeds'])
  for key,values in sorted(groups.items(),key=lambda x:str(x[0])):
   values=sorted(values);a=np.asarray([v for seed,v in values]);mean=float(a.mean());half=float(student_t.ppf(.975,len(a)-1))*float(a.std(ddof=1))/np.sqrt(len(a))
   cell=key[:-1];metric=key[-1];w.writerow(list(cell)+[metric,mean,mean-half,mean+half,len(a)])
   summary[cell+(metric,)]=(mean,mean-half,mean+half,len(a))
 effects=[]
 def effect(ds,arm,k,n,budget,method,metric,label,left,right):
  vals=[]
  for seed in SEEDS:
   a=seedmeans.get((ds,arm,k,n,left,method,seed,metric));b=seedmeans.get((ds,arm,k,n,right,method,seed,metric))
   if a is not None and b is not None:vals.append(b-a)
  if len(vals)<2:return
  a=np.asarray(vals);mean=float(a.mean());half=float(student_t.ppf(.975,len(a)-1))*float(a.std(ddof=1))/np.sqrt(len(a))
  effects.append(dict(dataset=ds,panel_type=arm,unseen_panel_size=k,competing_model_count=n,budget=budget,method=method,metric=metric,contrast=label,mean=mean,ci_low=mean-half,ci_high=mean+half,seeds=len(a)))
 for ds in DATASETS:
  b0=float(BUDGETS[ds][0]);bmax=float(BUDGETS[ds][-1])
  for arm in PANEL_TYPES:
   for k in UNSEEN_COUNTS:
    for n in MODEL_COUNTS:
     for method in METHODS:
      for metric in METRICS:
       effect(ds,arm,k,n,bmax,method,metric,'largest budget minus zero',b0,bmax)
       effect(ds,arm,k,n,bmax,method,metric,'24 competing models minus 6',6,24)
       effect(ds,arm,50,n,bmax,method,metric,'50 unseen models minus 5',5,50)
      for newarm in ['bootstrap_full_feature','bootstrap_matched_mask']:
       for metric in METRICS:
        vals=[]
        for seed in SEEDS:
         a=seedmeans.get((ds,newarm,50,n,bmax,method,seed,metric));b=seedmeans.get((ds,'hessian_existing',50,n,bmax,method,seed,metric))
         if a is not None and b is not None:vals.append(a-b)
        if len(vals)>1:
         x=np.asarray(vals);mean=float(x.mean());half=float(student_t.ppf(.975,len(x)-1))*float(x.std(ddof=1))/np.sqrt(len(x))
         effects.append(dict(dataset=ds,panel_type=newarm,unseen_panel_size=50,competing_model_count=n,budget=bmax,method=method,metric=metric,contrast='bootstrap minus existing Hessian panel at largest budget',mean=mean,ci_low=mean-half,ci_high=mean+half,seeds=len(x)))
 with (tables/'paired_effects_95ci.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(effects[0]));w.writeheader();w.writerows(effects)
 # Separate figures; each line has pointwise 95% CIs over ten seed means.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 figdir=root/'figures';figdir.mkdir(exist_ok=True);made=0
 for ds in DATASETS:
  for arm in PANEL_TYPES:
   for method in METHODS:
    for metric in ['joint_validity','mean_acceptance']:
     fig,ax=plt.subplots(figsize=(6.6,4.5))
     for n,style in zip(MODEL_COUNTS,['-','--',':']):
      for k,marker in zip(UNSEEN_COUNTS,['o','s','^','D']):
       xs=BUDGETS[ds];ys=[];lo=[];hi=[]
       for B in xs:
        v=summary.get((ds,arm,k,n,float(B),method,metric))
        if v is None:ys.append(np.nan);lo.append(np.nan);hi.append(np.nan)
        else:ys.append(v[0]);lo.append(v[1]);hi.append(v[2])
       y=np.asarray(ys);lc=np.clip(lo,0,1);hc=np.clip(hi,0,1)
       ax.plot(xs,y,linestyle=style,marker=marker,label=f'{n} competing, {k} unseen')
       ax.fill_between(xs,lc,hc,alpha=.06)
     ax.set_ylim(0,1);ax.set_xlabel('L2 budget (dataset standardized units)');ax.set_ylabel(metric.replace('_',' '))
     ax.set_title(f'{ds} | {method} | {arm}');ax.legend(fontsize=6,ncol=2);ax.grid(alpha=.2);fig.tight_layout()
     stem=f'{ds}__{method}__{arm}__{metric}'
     fig.savefig(figdir/(stem+'.png'),dpi=150);fig.savefig(figdir/(stem+'.pdf'));plt.close(fig);made+=1
 # Compact report documents interpretation and maximum-budget estimates.
 lines=['V7 RETRAINED-LOGISTIC TRANSFER PILOT (RECOVERED FINALIZATION)',f'Generated: {utc()}',f'Run: {root}',
  'Completed 1,108,800 person/setting/panel-size evaluations across 92,400 saved recourse records.',
  'Recovery note: the initial process stopped during finalization because the tables directory was not created before writing seed_means.csv. All frozen panels and the partial evaluation CSV were preserved; 28,800 rows were appended from the last checkpoint before analysis.',
  'Three panels: existing Hessian reference; 50 full-feature logistic fits per seed from ordinary bootstrap training resamples; 50 fits with masks sampled from the empirical 24-model bank distribution and bootstrap refits.',
  'All recourse points were reused. No optimization rerun, validation-based panel rejection, or outcome-based tuning.',
  'All inferential CIs are pointwise 95% Student-t over ten equally weighted seed means (df=9), with paired contrasts within seed. The models and people are nested observations, not independent seeds.',
  'This extends the completed study and reuses its split/cohort; it is not an independent replication. Bootstrap panels estimate training-sample sensitivity conditional on the saved data/splits.',
  'Primary outcome is all-K joint validity; mean acceptance is reported separately. Mixed-integer action constraints were not evaluated in this pilot.',
  'MAXIMUM BUDGET, 24 COMPETING MODELS, 50 UNSEEN MODELS: mean [95% CI]']
 for ds in DATASETS:
  lines.append(f'\n{ds} — {BUDGETS[ds][-1]}')
  for arm in PANEL_TYPES:
   lines.append(arm)
   for method in METHODS:
    a=summary[(ds,arm,50,24,float(BUDGETS[ds][-1]),method,'joint_validity')]
    b=summary[(ds,arm,50,24,float(BUDGETS[ds][-1]),method,'mean_acceptance')]
    lines.append(f'{method:16s} all-50 joint {100*a[0]:.1f}% [{100*a[1]:.1f}, {100*a[2]:.1f}]%; mean acceptance {100*b[0]:.1f}% [{100*b[1]:.1f}, {100*b[2]:.1f}]%')
 report='\n'.join(lines)+'\n';(root/'report.txt').write_text(report)
 cards=[]
 for image in sorted(figdir.glob('*.png')):
  cards.append(f'<figure><a href=\"figures/{html.escape(image.stem)}.pdf\"><img loading=\"lazy\" src=\"figures/{html.escape(image.name)}\" width=\"520\"></a><figcaption>{html.escape(image.stem)}</figcaption></figure>')
 (root/'report.html').write_text('<!doctype html><meta charset=\"utf-8\"><title>Retrained model transfer pilot</title><h1>V7 retrained logistic transfer pilot</h1><pre>'+html.escape(report)+'</pre><h2>Tables</h2><ul><li><a href=\"tables/all_cells_95ci.csv\">Cell estimates and 95% CIs</a></li><li><a href=\"tables/paired_effects_95ci.csv\">Paired effects and 95% CIs</a></li><li><a href=\"tables/seed_means.csv\">Seed means</a></li></ul><h2>Separate figure panels</h2>'+''.join(cards))
 # Recheck all source hashes and panel counts, then package output.
 source_manifest=json.loads((root/'input_manifest.json').read_text())['source']
 changed=[name for name,digest in source_manifest.items() if sha_file(SOURCE/name)!=digest]
 if changed:raise RuntimeError(f'source preservation hash mismatch: {changed[:3]}')
 inventory=json.loads((root/'panel_inventory.json').read_text())
 if len(inventory)!=60:raise RuntimeError(f'expected 60 frozen new panels, found {len(inventory)}')
 for item in inventory:
  f=root/item['path']; 
  if sha_file(f)!=item['sha256']:raise RuntimeError(f'panel hash mismatch {f}')
  with np.load(f,allow_pickle=False) as z:
   new_hashes={m.theta_hash(a,b) for a,b in zip(z['A'],z['b'])}
   if z['A'].shape[0]!=50 or len(new_hashes)!=50:raise RuntimeError(f'invalid/duplicate panel {f}')
  bank=SOURCE/'blocks'/item['dataset']/f"seed_{item['seed']}"/'bank'
  _,_,oa,ob,_,_=m.load_bank(bank)
  old_hashes={m.theta_hash(a,b) for a,b in zip(oa,ob)}
  if new_hashes & old_hashes:raise RuntimeError(f'panel overlaps optimization bank: {f}')
 if made!=126:raise RuntimeError(f'expected 126 separate figures, generated {made}')
 with (root/'evaluation_records.csv').open(newline='') as f:total=sum(1 for _ in csv.DictReader(f))
 if total!=1108800:raise RuntimeError(f'final record count {total}, expected 1,108,800')
 verification={'passed':True,'verified_utc':utc(),'expected_evaluation_rows':1108800,'actual_evaluation_rows':total,
  'source_preservation':True,'source_hashes_checked':len(source_manifest),'panels':len(inventory),'new_models':3000,
  'models_per_panel':50,'panel_sha256_verified':True,'unique_parameters_within_new_panels':True,
  'figure_panels':made,'each_separate_png_pdf':True,'failures':[],
  'recovered_finalization_error':'FileNotFoundError: missing tables directory; no model fit or saved-point evaluation was lost.',
  'confidence_intervals':'95% Student-t across ten seed means, df=9'}
 write_json(root/'verification.json',verification)
 write_json(root/'analysis.json',{'completed_utc':utc(),'figure_panels':made,'each_saved_separately':True,
  'all_inferential_tables_have_95ci':True,'verification_passed':True,'study_complete':True})
 write_json(root/'recovery.json',{'recovered_utc':utc(),'source_failure':json.loads((root/'failure.json').read_text()),
  'recovered_rows':written,'existing_rows_preserved':existing,'complete_rows':total,
  'no_model_fits_repeated':True,'no_saved_interventions_recomputed':True,'recovery_script_sha256':sha_file(Path(__file__))})
 with zipfile.ZipFile(root/'figures_tables.zip','w',zipfile.ZIP_DEFLATED) as z:
  for f in root.rglob('*'):
   if f.is_file() and f.name not in ['run.log','worker.lock','progress.json','figures_tables.zip'] and 'plot_cache' not in str(f):
    z.write(f,str(f.relative_to(root)))
 write_json(root/'progress.json',{'status':'complete_after_recovery','expected_records':total,'completed_records':total,
  'source_records':92400,'source_records_completed':92400,'panels_completed':60,'updated_utc':utc(),
  'verification_passed':True,'recovered_finalization_error':True})
 with (root/'run.log').open('a') as f:f.write(f'{utc()} PILOT_COMPLETED_AFTER_RECOVERY all rows verified; report and archive saved\n')
 print('RECOVERY_COMPLETE',root,'ROWS',total,'FIGURES',made)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();recovery(a.root)

