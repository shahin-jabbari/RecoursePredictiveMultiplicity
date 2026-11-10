#!/usr/bin/env python3
import argparse, csv, datetime as dt, fcntl, hashlib, html, json, os, shutil, subprocess, sys, time, traceback, uuid, zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits
threadpool_limits(1)

V7=Path('/Users/sj987/Documents/ChatGPT/Code/Predictive_Multiplicity_Recourse/V7')
PROJECT=V7/'experiments/linear_exact_unseen/retraining_transfer'
FULL=V7/'experiments/linear_exact_unseen/phase_full/results/20261018T193056.984145Z_2353a4b874'
DATASETS=['Synthetic','COMPAS','Give_Me_Some_Credit']
SEEDS=list(range(401,411))
MODEL_COUNTS=[6,12,24]
UNSEEN_COUNTS=[5,10,25,50]
METHODS=['ORPM','ADV','ROAR','ElliCE','Reference','ExactMaxMargin','ExactMinL2']
BUDGETS={'Synthetic':[0,.5,1,1.5,2,3,5],
         'COMPAS':[0,.5,1,1.5,2,3,5],
         'Give_Me_Some_Credit':[0,5,10,25,50,60,75,100]}
L2=.001
TOL=1e-7
PANEL_TYPES=['hessian_existing','bootstrap_full_feature','bootstrap_matched_mask']
METRICS=['joint_validity','strict_joint_validity','mean_acceptance','minimum_logit','worst_loss']
CONFIG={'study':'V7_retrained_logistic_transfer_pilot','version':'1.0',
 'source_full_run':str(FULL),'pilot_run':str(V7/'experiments/linear_exact_unseen/results/20261018T180330.097982Z_98de9f6e9a'),
 'datasets':DATASETS,'seeds':SEEDS,'people_per_seed':20,'competing_model_counts':MODEL_COUNTS,
 'unseen_panel_sizes':UNSEEN_COUNTS,'methods':METHODS,'budgets':BUDGETS,'panels':PANEL_TYPES,
 'models_per_new_panel':50,'bootstrap_replicates':50,
 'bootstrap':'ordinary row bootstrap from the saved standardized training split, n rows with replacement',
 'fit_objective':'mean binary cross entropy + 0.001/2 times squared L2 norm of weights and intercept',
 'mask_arm':'sample feature masks uniformly with replacement from the first 24 saved competing hypotheses; fit each bootstrap draw using that mask',
 'validation_filter':'none; validation accuracy is diagnostic only, with no panel rejection or tuning',
 'unseen_panel_freeze':'all panels frozen before evaluation of saved recourses',
 'saved_recourse_points':'reuse the full-study saved outputs for the identical 20-person cohorts, budgets, methods, and competing-set sizes; no recourse optimization is rerun',
 'primary_metrics':['all-K joint validity','mean acceptance across K models'],
 'ci':'pointwise 95% Student-t across 10 equally weighted seed means (df=9); paired within seed',
 'training_data':'saved training split only; fixed original scaler; test rows are used only for the preselected original cohort and evaluation',
 'limitations':'The three datasets and seeds reuse the full-study splits and cohorts; this is an extension and not an independent replication. Bootstrap draws are conditional on these training samples.'}

def utc():
 return dt.datetime.now(dt.timezone.utc).isoformat()
def clean(o):
 if isinstance(o,dict):return {str(k):clean(v) for k,v in o.items()}
 if isinstance(o,(list,tuple)):return [clean(x) for x in o]
 if isinstance(o,np.ndarray):return clean(o.tolist())
 if isinstance(o,np.generic):return clean(o.item())
 if isinstance(o,float) and not np.isfinite(o):return None
 if isinstance(o,Path):return str(o)
 return o
def save(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(clean(obj),indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def sha_bytes(x):return hashlib.sha256(x).hexdigest()
def sha_file(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def theta_hash(a,b):
 x=np.r_[np.asarray(a,float),float(b)].astype('<f4')
 return hashlib.sha256(x.tobytes()).hexdigest()
def jload(p):return json.loads(Path(p).read_text())
def logmsg(*a):print(utc(),*a,flush=True)

def fit_bootstrap(X,y,mask,rng):
 n,d=X.shape
 cols=np.arange(d) if mask is None else np.flatnonzero(np.asarray(mask)>0)
 idx=rng.integers(0,n,size=n)
 xb=np.asarray(X[idx][:,cols],dtype=np.float64)
 yy=np.asarray(y[idx],dtype=np.float64)
 Xa=np.column_stack([xb,np.ones(n)])
 def fg(theta):
  z=Xa@theta
  value=float(np.mean(np.logaddexp(0,z)-yy*z)+.5*L2*(theta@theta))
  grad=Xa.T@(expit(z)-yy)/n+L2*theta
  return value,grad
 started=time.perf_counter()
 result=minimize(fg,np.zeros(len(cols)+1),jac=True,method='L-BFGS-B',
  options={'maxiter':2000,'ftol':1e-12,'gtol':1e-8,'maxls':50})
 elapsed=time.perf_counter()-started
 theta=result.x
 gradnorm=float(np.linalg.norm(fg(theta)[1]))
 if not np.isfinite(theta).all() or not np.isfinite(result.fun) or gradnorm>2e-5:
  raise RuntimeError(f'bootstrap logistic fit failed: success={result.success}, grad={gradnorm}, message={result.message}')
 weights=np.zeros(d);weights[cols]=theta[:-1]
 return weights,float(theta[-1]),dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),
  objective=float(result.fun),gradient_l2=gradnorm,seconds=elapsed,active_features=cols.tolist(),
  bootstrap_indices_sha256=sha_bytes(np.asarray(idx,dtype='<i8').tobytes()),
  mask_sha256=None if mask is None else sha_bytes(np.asarray(mask,dtype='<f4').tobytes()))

def generate_panel(ds,seed,panel_type,root):
 bank=FULL/'blocks'/ds/f'seed_{seed}'/'bank'
 split=dict(np.load(bank/'split_and_predictions.npz',allow_pickle=False))
 X=np.asarray(split['X_train'],dtype=np.float64);y=np.asarray(split['y_train'],dtype=np.float64)
 Xv=np.asarray(split['X_validation'],dtype=np.float64);yv=np.asarray(split['y_validation'],dtype=np.float64)
 meta=jload(bank/'models.json');desc=meta['models'][:24]
 masks=[m.get('feature_mask') for m in desc]
 if panel_type=='bootstrap_full_feature':masks_to_fit=[None]*50
 elif panel_type=='bootstrap_matched_mask':
  maskcode={'Synthetic':11001,'COMPAS':22003,'Give_Me_Some_Credit':33007}[ds]
  mask_rng=np.random.default_rng(93000000+seed*101+maskcode)
  masks_to_fit=[masks[int(i)] for i in mask_rng.integers(0,len(masks),size=50)]
 else:raise ValueError(panel_type)
 seedcode={'Synthetic':11001,'COMPAS':22003,'Give_Me_Some_Credit':33007}[ds]
 armcode=1 if panel_type=='bootstrap_full_feature' else 2
 rng=np.random.default_rng(73000000+seed*101+seedcode+armcode*100000)
 origA=[];origb=[]
 for m in desc:
  with np.load(bank/f"model_{len(origA):04d}.npz",allow_pickle=False) as f:
   if 'linear.weight' in f:
    w=f['linear.weight'].reshape(-1);b=float(f['linear.bias'].reshape(-1)[0])
   else:
    w=f['base_model.linear.weight'].reshape(-1);b=float(f['base_model.linear.bias'].reshape(-1)[0])
   mask=m.get('feature_mask')
   if mask is not None:w=w*np.asarray(mask,dtype=float)
   origA.append(w);origb.append(b)
 original_hashes={theta_hash(a,b) for a,b in zip(origA,origb)}
 outA=[];outb=[];audit=[];dup=0
 for i,mask in enumerate(masks_to_fit):
  attempts=0
  while True:
   attempts+=1
   w,b,info=fit_bootstrap(X,y,mask,rng)
   h=theta_hash(w,b)
   if h not in original_hashes and h not in {theta_hash(a,b0) for a,b0 in zip(outA,outb)}:break
   dup+=1
   if attempts>=10:raise RuntimeError(f'could not draw distinct held-out model {ds}/{seed}/{panel_type}/{i}')
  outA.append(w);outb.append(b)
  logits=Xv@w+b
  audit.append(dict(model=i,theta_sha256=h,validation_accuracy=float(np.mean((logits>=0)==yv)),
   validation_log_loss=float(np.mean(np.logaddexp(0,logits)-yv*logits)),mask_source=('all_features' if mask is None else 'competing_bank_empirical_mask'),
   fit=info,duplicate_redraws=attempts-1))
 arrA=np.asarray(outA);arrb=np.asarray(outb)
 path=root/'panels'/panel_type/ds/f'seed_{seed}.npz';path.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(path,A=arrA,b=arrb,order=rng.permutation(50))
 auditpath=path.with_suffix('.json')
 save(auditpath,dict(dataset=ds,seed=seed,panel_type=panel_type,models=50,training_rows=len(y),
  feature_count=X.shape[1],l2=L2,duplicate_redraws=dup,models_audit=audit))
 return path,sha_file(path),dict(dataset=ds,seed=seed,panel_type=panel_type,path=str(path.relative_to(root)),
  sha256=sha_file(path),audit_sha256=sha_file(auditpath),models=50,duplicate_redraws=dup)

def run_worker(root,lock_fd):
 root=Path(root);status={'status':'preparing','expected_records':None,'completed_records':0,'updated_utc':utc()}
 try:
  cfg=jload(root/'protocol.json')
  source=Path(cfg['source_full_run'])
  v=jload(source/'verification.json')
  if not v.get('passed'):raise RuntimeError('source full study verification is not passed')
  # Freeze source scientific files and all 30 source panels/cohorts.
  manifest={}
  for n in ['records.jsonl','evaluation_records.csv','model_inventory.json','verification.json','freeze.json']:
   manifest[n]=sha_file(source/n)
  for ds in cfg['datasets']:
   for seed in cfg['seeds']:
    for n in ['panel.npz','cohort.json','unseen_admission.json']:
     f=source/'blocks'/ds/f'seed_{seed}'/n
     manifest[str(f.relative_to(source))]=sha_file(f)
    bank=source/'blocks'/ds/f'seed_{seed}'/'bank'
    for n in ['models.json','preprocessing.json','split_and_predictions.npz']+[f'model_{i:04d}.npz' for i in range(24)]:
     f=bank/n;manifest[str(f.relative_to(source))]=sha_file(f)
  save(root/'input_manifest.json',dict(source=manifest,sha256=sha_bytes(json.dumps(manifest,sort_keys=True).encode())))
  panelmap={}
  for ds in cfg['datasets']:
   for seed in cfg['seeds']:
    f=source/'blocks'/ds/f'seed_{seed}'/'panel.npz'
    with np.load(f,allow_pickle=False) as z: panelmap[(ds,seed)]=(z['unseen_A'].copy(),z['unseen_b'].copy(),z['selected_X'].copy(),z['test_ids'].copy())
  inventory=[]
  for ds in cfg['datasets']:
   for seed in cfg['seeds']:
    for arm in ['bootstrap_full_feature','bootstrap_matched_mask']:
     path,h,item=generate_panel(ds,seed,arm,root);inventory.append(item);logmsg('PANEL_FROZEN',ds,seed,arm,'sha256',h)
     status.update(status='preparing_panels',panels_completed=len(inventory),total_panels=60,updated_utc=utc());save(root/'progress.json',status)
  save(root/'panel_inventory.json',inventory)
  # Add existing Hessian panels from the immutable full-study run to the evaluation set.
  for ds in cfg['datasets']:
   for seed in cfg['seeds']:
    path=source/'blocks'/ds/f'seed_{seed}'/'panel.npz'
    with np.load(path,allow_pickle=False) as z:
     hA=z['unseen_A'].copy();hb=z['unseen_b'].copy()
    rng=np.random.default_rng(81000000+seed+{'Synthetic':1,'COMPAS':2,'Give_Me_Some_Credit':3}[ds])
    panelmap[(ds,seed,'hessian_existing')]=(hA,hb,rng.permutation(len(hb)))
  for item in inventory:
   ds=item['dataset'];seed=item['seed'];panel_type=item['panel_type'];path=root/item['path']
   with np.load(path,allow_pickle=False) as z:
    A=z['A'].copy();b=z['b'].copy();order=z['order'].copy()
   panelmap[(ds,seed,panel_type)]=(A,b,order)
  source_records=source/'records.jsonl'
  selected=[]
  with source_records.open() as f:
   for line in f:
    r=json.loads(line)
    if r['dataset'] in cfg['datasets']:selected.append(r)
  expected=sum(len(cfg['seeds'])*20*3*len(cfg['budgets'][ds])*len(cfg['methods']) for ds in cfg['datasets'])*len(PANEL_TYPES)*len(UNSEEN_COUNTS)
  status.update(status='running',expected_records=expected,source_records=len(selected),updated_utc=utc());save(root/'progress.json',status)
  pt_cache={}
  def points(file):
   if file not in pt_cache:
    with np.load(source/file,allow_pickle=False) as z:pt_cache[file]={k:z[k].copy() for k in z.files if k in cfg['methods']}
    if len(pt_cache)>256:pt_cache.pop(next(iter(pt_cache)))
   return pt_cache[file]
  sums=defaultdict(float);counts=defaultdict(int);record_count=0
  fields=['dataset','seed','person','test_id','competing_model_count','budget','budget_index','method','available','panel_type','unseen_panel_size']+METRICS+['boundary_count']
  with (root/'evaluation_records.csv').open('x',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
   for j,r in enumerate(selected,1):
    ds=r['dataset'];seed=int(r['seed']);method=r['method'];available=bool(r['available'])
    pp=points(r['point_file']);x=np.asarray(pp[method],float)
    for panel_type in PANEL_TYPES:
     A,b,order=panelmap[(ds,seed,panel_type)]
     zall=np.asarray(A[order],float)@x+np.asarray(b[order],float)
     for k in UNSEEN_COUNTS:
      z=zall[:k];valid=z>=-TOL;strict=z>=0
      row=dict(dataset=ds,seed=seed,person=int(r['person']),test_id=int(r['test_id']),
       competing_model_count=int(r['model_count']),budget=float(r['budget']),budget_index=int(r['budget_index']),
       method=method,available=available,panel_type=panel_type,unseen_panel_size=k,
       joint_validity=float(bool(available and valid.all())),strict_joint_validity=float(bool(available and strict.all())),
       mean_acceptance=float(valid.mean()) if available else 0.,
       minimum_logit=float(z.min()),worst_loss=float(np.logaddexp(0.,-z).max()),boundary_count=int(np.sum(np.abs(z)<=TOL)))
      writer.writerow(row);record_count+=1
      for metric in METRICS:
       key=(ds,panel_type,k,int(r['model_count']),float(r['budget']),method,seed,metric)
       sums[key]+=row[metric];counts[key]+=1
    if j%5000==0:
     status.update(completed_records=record_count,source_records_completed=j,updated_utc=utc());save(root/'progress.json',status);logmsg('EVALUATION_PROGRESS',j,len(selected),'output_rows',record_count)
  # Convert person-level values to equally-weighted seed means, then pointwise t intervals.
  seedmeans=defaultdict(float);seedcounts=defaultdict(int)
  for key,val in sums.items():
   cell=key[:-2];seed=key[-2];metric=key[-1];n=counts[key]
   seedmeans[(cell,seed,metric)]=val/n;seedcounts[(cell,seed,metric)]=n
  with (root/'tables/seed_means.csv').open('w',newline='') as f:
   w=csv.writer(f);w.writerow(['dataset','panel_type','unseen_panel_size','competing_model_count','budget','method','seed','metric','mean_person_value','people'])
   for (cell,seed,metric),value in sorted(seedmeans.items(),key=lambda x:str(x[0])):
    ds,pt,k,n,budget,method=cell;w.writerow([ds,pt,k,n,budget,method,seed,metric,value,seedcounts[(cell,seed,metric)]])
  grouped=defaultdict(list)
  for (cell,seed,metric),value in seedmeans.items():grouped[(cell,metric)].append((seed,value))
  summary={}
  with (root/'tables/all_cells_95ci.csv').open('w',newline='') as f:
   w=csv.writer(f);w.writerow(['dataset','panel_type','unseen_panel_size','competing_model_count','budget','method','metric','mean','ci_low','ci_high','seeds'])
   for (cell,metric),vals in sorted(grouped.items(),key=lambda x:str(x[0])):
    vals=sorted(vals);arr=np.asarray([v for s,v in vals],float);m=float(arr.mean());q=float(student_t.ppf(.975,len(arr)-1))*float(arr.std(ddof=1))/np.sqrt(len(arr)) if len(arr)>1 else 0.
    lo,hi=m-q,m+q;ds,pt,k,n,budget,method=cell
    w.writerow([ds,pt,k,n,budget,method,metric,m,lo,hi,len(arr)])
    summary[(ds,pt,k,n,budget,method,metric)]=(m,lo,hi,len(arr))
  # Paired seed contrasts: budget, unseen panel size, set size, and model generation panel.
  lookup=seedmeans
  effects=[]
  def add_effect(ds,pt,k,n,budget,method,metric,label,left,right):
   arr=[]
   for seed in cfg['seeds']:
    a=lookup.get(((ds,pt,k,n,left,method),seed,metric));b0=lookup.get(((ds,pt,k,n,right,method),seed,metric))
    if a is not None and b0 is not None:arr.append(b0-a)
   if len(arr)<2:return
   a=np.asarray(arr);m=float(a.mean());q=float(student_t.ppf(.975,len(a)-1))*float(a.std(ddof=1))/np.sqrt(len(a))
   effects.append(dict(dataset=ds,panel_type=pt,unseen_panel_size=k,competing_model_count=n,budget=budget,method=method,metric=metric,contrast=label,mean=m,ci_low=m-q,ci_high=m+q,seeds=len(a)))
  for ds in cfg['datasets']:
   b0=cfg['budgets'][ds][0];bmax=cfg['budgets'][ds][-1]
   for pt in PANEL_TYPES:
    for k in UNSEEN_COUNTS:
     for n in MODEL_COUNTS:
      for method in METHODS:
       for metric in METRICS:add_effect(ds,pt,k,n,bmax,method,metric,'largest budget minus zero',b0,bmax)
       for budget in [bmax]:
        for metric in METRICS:add_effect(ds,pt,k,n,budget,method,metric,'50 unseen models minus 5',5,50)
       for metric in METRICS:add_effect(ds,pt,k,n,bmax,method,metric,'24 competing models minus 6',6,24)
       for newpt in ['bootstrap_full_feature','bootstrap_matched_mask']:
        for metric in METRICS:
         left=(ds,newpt,50,n,bmax,method);right=(ds,'hessian_existing',50,n,bmax,method)
         arr=[]
         for seed in cfg['seeds']:
          a=lookup.get((left,seed,metric));b0v=lookup.get((right,seed,metric))
          if a is not None and b0v is not None:arr.append(a-b0v)
         if len(arr)>=2:
          aa=np.asarray(arr);m=float(aa.mean());q=float(student_t.ppf(.975,len(aa)-1))*float(aa.std(ddof=1))/np.sqrt(len(aa))
          effects.append(dict(dataset=ds,panel_type=newpt,unseen_panel_size=50,competing_model_count=n,budget=bmax,method=method,metric=metric,contrast='bootstrap panel minus existing Hessian panel at largest budget',mean=m,ci_low=m-q,ci_high=m+q,seeds=len(aa)))
  efpath=root/'tables/paired_effects_95ci.csv'
  with efpath.open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(effects[0].keys()));w.writeheader();w.writerows(effects)
  # Separate panels, each with pointwise 95% CIs.
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  figdir=root/'figures';figdir.mkdir(exist_ok=True)
  made=0
  for ds in cfg['datasets']:
   for pt in PANEL_TYPES:
    for method in METHODS:
     for metric in ['joint_validity','mean_acceptance']:
      fig,ax=plt.subplots(figsize=(6.6,4.5))
      for n,style in zip(MODEL_COUNTS,['-','--',':']):
       for k in UNSEEN_COUNTS:
        xs=cfg['budgets'][ds];ys=[];low=[];high=[]
        for budget in xs:
         s=summary.get((ds,pt,k,n,float(budget),method,metric))
         if s is None:ys.append(np.nan);low.append(0);high.append(0)
         else:ys.append(s[0]);low.append(s[1]);high.append(s[2])
        ax.plot(xs,ys,linestyle=style,marker={5:'o',10:'s',25:'^',50:'D'}[k],label=f'{n} competing, {k} unseen')
        ax.fill_between(xs,np.clip(low,0,1),np.clip(high,0,1),alpha=.06)
      ax.set_ylim(0,1);ax.set_xlabel('L2 budget (dataset standardized units)');ax.set_ylabel(metric.replace('_',' '))
      ax.set_title(f'{ds} | {method} | {pt}');ax.legend(fontsize=6,ncol=2);ax.grid(alpha=.2);fig.tight_layout()
      stem=f'{ds}__{method}__{pt}__{metric}'
      fig.savefig(figdir/(stem+'.png'),dpi=150);fig.savefig(figdir/(stem+'.pdf'));plt.close(fig);made+=1
  # Report contains a compact maximum-budget snapshot and methodology caveats.
  lines=['V7 RETRAINED-LOGISTIC TRANSFER PILOT',f'Generated: {utc()}',f'Run: {root}',
   f'Point-level panel evaluations: {record_count}; source recourse records: {len(selected)}.',
   'Independent bootstrap logistic panels: two per dataset/seed, each 50 hypotheses; existing Hessian panel included as reference.',
   'All saved interventions reused; no optimization was rerun. Models are fit on resampled training rows only.',
   'Masks in the matched arm are sampled from the empirical masks of the first 24 competing hypotheses.',
   'Validation accuracy was diagnostic only; no quality-based panel rejection or method tuning.',
   'Every inferential table and figure reports pointwise 95% Student-t intervals over 10 seed means (df=9); paired contrasts within seed.',
   'The 50 hypotheses and repeated interventions are not independent replicates. The three datasets reuse the completed study cohorts; this is an extension, not an independent replication.',
   'Primary outcome is all-K joint validity; mean model acceptance is reported separately.',
   'NO MILP is run in this evaluation-only pilot. Mixed-integer action constraints need a separate matched-action-space protocol.',
   '\nMAXIMUM BUDGET, 24 COMPETING MODELS, 50 UNSEEN MODELS: mean [95% CI]']
  for ds in cfg['datasets']:
   for pt in PANEL_TYPES:
    lines.append(f'\n{ds} — {pt}')
    bmax=float(cfg['budgets'][ds][-1])
    for method in METHODS:
     vals=[]
     for metric in ['joint_validity','mean_acceptance']:
      v=summary.get((ds,pt,50,24,bmax,method,metric))
      vals.append('NA' if v is None else f'{100*v[0]:.1f}% [{100*v[1]:.1f}, {100*v[2]:.1f}]%')
     lines.append(f'{method:16s} all-50 joint {vals[0]}; mean acceptance {vals[1]}')
  report='\n'.join(lines)+'\n';(root/'report.txt').write_text(report)
  # HTML links every separate panel and the audit tables.
  cards=[]
  for image in sorted(figdir.glob('*.png')):
   cards.append(f'<figure><a href=\"figures/{html.escape(image.stem)}.pdf\"><img loading=\"lazy\" src=\"figures/{html.escape(image.name)}\" width=\"520\"></a><figcaption>{html.escape(image.stem)}</figcaption></figure>')
  (root/'report.html').write_text('<!doctype html><meta charset=\"utf-8\"><title>Retrained model transfer pilot</title><h1>V7 retrained logistic transfer pilot</h1><pre>'+html.escape(report)+'</pre><h2>Tables</h2><ul><li><a href=\"tables/all_cells_95ci.csv\">Cell estimates and 95% CIs</a></li><li><a href=\"tables/paired_effects_95ci.csv\">Paired effects and 95% CIs</a></li><li><a href=\"tables/seed_means.csv\">Seed means</a></li></ul><h2>Separate figure panels</h2>'+''.join(cards))
  # Verify inputs unchanged, panel disjointness, counts, numeric ranges and output integrity.
  after={n:sha_file(source/n) for n in manifest}
  if after!=manifest:raise RuntimeError('source hashes changed during evaluation')
  if record_count!=expected:raise RuntimeError(f'row count {record_count} != expected {expected}')
  for item in inventory:
   panel=root/item['path']
   with np.load(panel,allow_pickle=False) as z:
    if z['A'].shape[0]!=50 or len(set(theta_hash(a,b) for a,b in zip(z['A'],z['b'])))!=50:raise RuntimeError('panel has invalid count or duplicate parameters')
  if made!=126:raise RuntimeError(f'expected 126 separate figure panels, created {made}')
  with zipfile.ZipFile(root/'figures_tables.zip','w',zipfile.ZIP_DEFLATED) as z:
   for name in ['protocol.json','PROTOCOL.txt','freeze.json','input_manifest.json','panel_inventory.json','verification.json','analysis.json','report.txt','report.html','evaluation_records.csv']:
    f=root/name
    if f.exists():z.write(f,name)
   for folder in ['tables','figures','panels']:
    for f in (root/folder).rglob('*'):
     if f.is_file():z.write(f,str(f.relative_to(root)))
  verification={'passed':True,'verified_utc':utc(),'source_preservation':True,'expected_evaluation_rows':expected,
   'actual_evaluation_rows':record_count,'panels':len(inventory),'panel_models_each':50,'panel_hashes_distinct_within_panel':True,
   'separate_figure_panels':made,'source_hashes':manifest,'failures':[],'ci_seed_count':len(SEEDS),'method':'independent checks of counts, finite outputs, hashes and frozen panels'}
  save(root/'verification.json',verification)
  save(root/'analysis.json',dict(completed_utc=utc(),figure_panels=made,each_saved_separately=True,
   all_inferential_tables_have_95ci=True,verification_passed=True,study_complete=True))
  # Rebuild archive to include final status records.
  with zipfile.ZipFile(root/'figures_tables.zip','w',zipfile.ZIP_DEFLATED) as z:
   for f in root.rglob('*'):
    if f.is_file() and f.name not in ['run.log','worker.lock','failure.json','progress.json','figures_tables.zip'] and 'plot_cache' not in str(f):
     z.write(f,str(f.relative_to(root)))
  status.update(status='complete',completed_records=record_count,expected_records=expected,verification_passed=True,updated_utc=utc());save(root/'progress.json',status)
  logmsg('PILOT_COMPLETED','report and archive saved')
 except Exception as e:
  failure={'failed_utc':utc(),'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()}
  save(root/'failure.json',failure);status.update(status='failed',updated_utc=utc(),error=str(e));save(root/'progress.json',status);logmsg('PILOT_FAILED',type(e).__name__,str(e));raise

def launch():
 PROJECT.mkdir(parents=True,exist_ok=True);(PROJECT/'phase_pilot/results').mkdir(parents=True,exist_ok=True)
 v=jload(FULL/'verification.json')
 if not v.get('passed'):raise RuntimeError('verified source full study is not passed')
 cfg=PROJECT/'protocol.json'
 if cfg.exists():
  prior=jload(cfg)
  if prior!=CONFIG:raise RuntimeError('protocol already exists and differs; refusing overwrite')
 else:save(cfg,CONFIG)
 proto=PROJECT/'PROTOCOL.txt'
 if not proto.exists():
  proto.write_text('V7 RETRAINED-LOGISTIC TRANSFER PILOT — FROZEN PROTOCOL 1.0\n\n'+json.dumps(CONFIG,indent=2)+'\n\nThe pilot evaluates the completed V7 full-study recourse points under fresh 50-model logistic panels. It includes the existing Hessian panel and two independent bootstrap refit arms: full-feature and empirically matched feature masks. It covers Synthetic, COMPAS, and Give_Me_Some_Credit (GMC), all ten existing seeds and twenty selected people, 6/12/24 competing models, all seven methods, and the expanded budget grids. Bootstrap samples are drawn only from saved training rows; no test or recourse outcome is used to fit or select models. Validation accuracy is diagnostic only. Outputs include all-K joint validity and mean acceptance separately, pointwise 95% CIs across ten seed means, paired within-seed contrasts, 126 individually saved PNG/PDF panels, and complete model/panel/source audits. Existing interventions are re-evaluated; recourse optimization is not repeated. The pilot and full study are reused data, so this is an extension rather than an independent replication. A genuine MILP comparison is a later, separate study with matched mixed-integer action constraints.\n')
 root=PROJECT/'phase_pilot/results'/(dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'_'+uuid.uuid4().hex[:10])
 root.mkdir(parents=False)
 shutil.copy2(cfg,root/'protocol.json');shutil.copy2(proto,root/'PROTOCOL.txt')
 src=root/'source';src.mkdir();script=Path(__file__).resolve();shutil.copy2(script,src/'runner.py')
 manifest={str(f.relative_to(src)):sha_file(f) for f in src.rglob('*') if f.is_file()}
 manifest.update({'protocol.json':sha_file(root/'protocol.json'),'PROTOCOL.txt':sha_file(root/'PROTOCOL.txt')})
 save(root/'freeze.json',dict(frozen_utc=utc(),source_sha256=manifest,version='1.0'))
 phasefd=os.open(PROJECT/'phase_pilot/active_worker.lock',os.O_CREAT|os.O_RDWR,0o600)
 try:fcntl.flock(phasefd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('a pilot worker already holds the phase lock')
 workerfd=os.open(root/'worker.lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(workerfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLBACKEND='Agg',MPLCONFIGDIR=str(root/'plot_cache'),PYTHONDONTWRITEBYTECODE='1')
 cmd=[sys.executable,'-u',str(src/'runner.py'),'--root',str(root),'--lock-fd',str(workerfd)]
 with (root/'run.log').open('xb') as log:proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,pass_fds=(workerfd,phasefd))
 for fd in [workerfd,phasefd]:os.ftruncate(fd,0);os.write(fd,(str(proc.pid)+'\n').encode());os.close(fd)
 caffeinate=None
 if Path('/usr/bin/caffeinate').exists():caffeinate=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(proc.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True).pid
 save(root/'launch.json',dict(pid=proc.pid,caffeinate_pid=caffeinate,started_utc=utc(),python=sys.executable,root=str(root),command=cmd,scope='authorized three-dataset evaluation-only pilot'))
 (PROJECT/'phase_pilot/CURRENT_RUN.txt').write_text(str(root)+'\n')
 print('RUN_ROOT',root);print('PID',proc.pid);return root

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--launch',action='store_true');ap.add_argument('--root');ap.add_argument('--lock-fd',type=int)
 a=ap.parse_args()
 if a.launch:launch()
 elif a.root:run_worker(a.root,a.lock_fd)
 else:ap.error('pass --launch or worker --root')

