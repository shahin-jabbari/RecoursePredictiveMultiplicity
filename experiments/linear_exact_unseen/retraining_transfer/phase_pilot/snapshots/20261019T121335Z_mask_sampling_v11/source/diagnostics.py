#!/usr/bin/env python3
"""Correct mask sampling and independently audit the authorized saved-recourse pilot."""
import argparse, csv, datetime as dt, fcntl, hashlib, html, importlib.util, json, os, shutil, sys, time, zipfile
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import t
from threadpoolctl import threadpool_limits
threadpool_limits(1)

METRICS=['joint_validity','strict_joint_validity','mean_acceptance','minimum_logit','worst_loss']
CELL=['dataset','panel_type','unseen_panel_size','competing_model_count','budget','method']
ARMS=['hessian_existing','bootstrap_full_feature','bootstrap_matched_mask']
KS=[5,10,25,50]
LABELS={'Synthetic':'Synthetic','COMPAS':'COMPAS','Give_Me_Some_Credit':'GMC',
 'hessian_existing':'Existing Hessian panel','bootstrap_full_feature':'Full-feature bootstrap',
 'bootstrap_matched_mask':'Random-mask bootstrap'}
def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def digest(b): return hashlib.sha256(b).hexdigest()
def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(q,p)
def theta_hash(a,b): return digest(np.r_[a,b].astype('<f4').tobytes())
def load(p): return json.loads(Path(p).read_text())
def module(p):
 s=importlib.util.spec_from_file_location('frozen_generator',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def aggregate(frame,keys):
 long=frame.melt(id_vars=keys,value_vars=METRICS,var_name='metric',value_name='value')
 a=long.groupby(keys+['metric'],observed=True,sort=True)['value'].agg(['mean','std','count']).reset_index()
 assert (a['count']==10).all(), 'CI groups must each contain ten outer seeds'
 half=t.ppf(.975,9)*a['std']/np.sqrt(10)
 a['ci_low']=a['mean']-half;a['ci_high']=a['mean']+half
 return a.drop(columns='std').rename(columns={'count':'seeds'})

def run(parent,root):
 started=time.perf_counter();parent=Path(parent);root=Path(root)
 oldlock=(parent/'worker.lock').open('r');fcntl.flock(oldlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not root.exists(), 'A correction run must use a new result directory'
 root.mkdir(parents=True);(root/'source').mkdir();(root/'tables').mkdir();(root/'audits').mkdir()
 lock=(root/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.write(str(os.getpid())+'\n');lock.flush()
 logfile=(root/'run.log').open('a',buffering=1)
 def log(*args):
  msg=utc()+' '+' '.join(map(str,args));print(msg,flush=True);logfile.write(msg+'\n')
 status={'status':'preparing_corrected_panels','expected_records':1108800,'completed_records':0}
 def progress(**kwargs):status.update(kwargs,updated_utc=utc());save(root/'progress.json',status)
 cfg=load(parent/'protocol.json');src=Path(cfg['source_full_run'])
 cfg.update(version='1.1',correction_of=str(parent),
  correction='Mask identities are now sampled independently and uniformly with replacement from the first 24 saved source masks, as specified in version 1.0. Prior full-feature panels and Hessian scores are reused unchanged. No outcome-based selection or tuning.',
  mask_rng='default_rng(93000000 + seed*101 + dataset_code); dataset codes Synthetic=11001, COMPAS=22003, GMC=33007; separate from bootstrap row RNG')
 save(root/'protocol.json',cfg)
 (root/'PROTOCOL.txt').write_text('V7 RETRAINED-LOGISTIC TRANSFER PILOT — FROZEN CORRECTION 1.1\n\n'+json.dumps(cfg,indent=2)+'\n')
 generator=(parent/'source/runner.py').read_text()
 old="elif panel_type=='bootstrap_matched_mask':masks_to_fit=[masks[i%len(masks)] for i in range(50)]"
 new="elif panel_type=='bootstrap_matched_mask':\n  maskcode={'Synthetic':11001,'COMPAS':22003,'Give_Me_Some_Credit':33007}[ds]\n  mask_rng=np.random.default_rng(93000000+seed*101+maskcode)\n  masks_to_fit=[masks[int(i)] for i in mask_rng.integers(0,len(masks),size=50)]"
 assert generator.count(old)==1;generator=generator.replace(old,new)
 (root/'source/panel_generator.py').write_text(generator)
 shutil.copy2(__file__,root/'source/diagnostics.py')
 save(root/'freeze.json',{'frozen_utc':utc(),'version':'1.1','sha256':{str(p.relative_to(root)):sha(p) for p in [root/'protocol.json',root/'PROTOCOL.txt',root/'source/panel_generator.py',root/'source/diagnostics.py']}})
 save(root/'launch.json',{'pid':os.getpid(),'started_utc':utc(),'python':sys.executable,'root':str(root),'parent':str(parent),'scope':'correct the previously authorized mask sampling and finish independent diagnostics; no recourse optimization'})
 progress();log('CORRECTION_STARTED',root)
 try:
  manifest=load(parent/'input_manifest.json');shutil.copy2(parent/'input_manifest.json',root/'input_manifest.json')
  for name,h in manifest['source'].items(): assert sha(src/name)==h, f'Source changed: {name}'
  parent_csv_hash=sha(parent/'evaluation_records.csv')
  oldinv=load(parent/'panel_inventory.json');inventory=[];m=module(root/'source/panel_generator.py')
  for ds in cfg['datasets']:
   for seed in cfg['seeds']:
    item=next(x for x in oldinv if x['dataset']==ds and x['seed']==seed and x['panel_type']=='bootstrap_full_feature')
    dest=root/item['path'];dest.parent.mkdir(parents=True,exist_ok=True)
    assert sha(parent/item['path'])==item['sha256']
    assert sha((parent/item['path']).with_suffix('.json'))==item['audit_sha256']
    shutil.copy2(parent/item['path'],dest);shutil.copy2((parent/item['path']).with_suffix('.json'),dest.with_suffix('.json'));inventory.append(item)
    _,_,item=m.generate_panel(ds,seed,'bootstrap_matched_mask',root);inventory.append(item)
    progress(panels_completed=len(inventory),total_panels=60)
    log('MASK_PANEL_FROZEN',ds,seed)
  save(root/'panel_inventory.json',inventory);log('ALL_PANELS_FROZEN',60)
  save(root/'parent_reuse.json',{'parent':str(parent),'parent_evaluation_csv_sha256':parent_csv_hash,'full_feature_panels_reused':30,'prior_rows_expected_unchanged':739200,
   'explanation':'The original CSV already contained all 1,108,800 rows when report setup failed; progress.json was one checkpoint behind. No missing rows were appended. The abandoned recovery attempts produced no scientific changes.'})
  # Verify each trained panel from saved training draws and frozen coefficients, without refitting.
  panels={};quality=[];max_grad=0.;fit_seconds=0.;masked_differences=0
  for ds in cfg['datasets']:
   code={'Synthetic':11001,'COMPAS':22003,'Give_Me_Some_Credit':33007}[ds]
   for seed in cfg['seeds']:
    bank=src/'blocks'/ds/f'seed_{seed}'/'bank';meta=load(bank/'models.json')['models'][:24]
    with np.load(bank/'split_and_predictions.npz') as z:X=z['X_train'].astype(float);y=z['y_train'].astype(float);Xv=z['X_validation'].astype(float);yv=z['y_validation'].astype(float)
    originals=set()
    for i,desc in enumerate(meta):
     with np.load(bank/f'model_{i:04d}.npz') as z:
      prefix='linear.' if 'linear.weight' in z else 'base_model.linear.'
      a=z[prefix+'weight'].reshape(-1).astype(float);b=float(z[prefix+'bias'].reshape(-1)[0])
     if desc.get('feature_mask') is not None:a*=np.asarray(desc['feature_mask'])
     originals.add(theta_hash(a,b))
    with np.load(src/'blocks'/ds/f'seed_{seed}'/'panel.npz') as z:A=z['unseen_A'].astype(float);b=z['unseen_b'].astype(float)
    rng=np.random.default_rng(81000000+seed+{'Synthetic':1,'COMPAS':2,'Give_Me_Some_Credit':3}[ds]);panels[ds,seed,ARMS[0]]=(A,b,rng.permutation(50))
    for arm in ARMS[1:]:
     item=next(x for x in inventory if x['dataset']==ds and x['seed']==seed and x['panel_type']==arm);path=root/item['path'];audit=load(path.with_suffix('.json'))
     assert sha(path)==item['sha256'] and sha(path.with_suffix('.json'))==item['audit_sha256']
     with np.load(path) as z:A=z['A'].copy();b=z['b'].copy();order=z['order'].copy()
     assert A.shape==(50,X.shape[1]) and np.array_equal(np.sort(order),np.arange(50))
     hashes={theta_hash(a,bb) for a,bb in zip(A,b)};assert len(hashes)==50 and not hashes.intersection(originals)
     rng=np.random.default_rng(73000000+seed*101+code+(1 if arm==ARMS[1] else 2)*100000)
     ids=np.random.default_rng(93000000+seed*101+code).integers(0,24,size=50)
     if arm==ARMS[2]:masked_differences+=int(np.sum(ids!=np.arange(50)%24))
     for i,entry in enumerate(audit['models_audit']):
      for _ in range(1+entry['duplicate_redraws']):idx=rng.integers(0,len(y),size=len(y))
      f=entry['fit'];assert digest(idx.astype('<i8').tobytes())==f['bootstrap_indices_sha256']
      mask=None if arm==ARMS[1] else meta[int(ids[i])].get('feature_mask')
      cols=np.arange(X.shape[1]) if mask is None else np.flatnonzero(np.asarray(mask)>0)
      assert cols.tolist()==f['active_features'];assert np.all(A[i,np.setdiff1d(np.arange(X.shape[1]),cols)]==0)
      assert f['mask_sha256']==(None if mask is None else digest(np.asarray(mask,dtype='<f4').tobytes()))
      theta=np.r_[A[i,cols],b[i]];xx=np.column_stack([X[idx][:,cols],np.ones(len(idx))]);zz=xx@theta
      grad=xx.T@(expit(zz)-y[idx])/len(idx)+.001*theta
      norm=float(np.linalg.norm(grad));max_grad=max(max_grad,norm);assert norm<=2e-5
      objective=float(np.mean(np.logaddexp(0,zz)-y[idx]*zz)+.0005*(theta@theta));assert abs(objective-f['objective'])<1e-9
      assert theta_hash(A[i],b[i])==entry['theta_sha256']
      zv=Xv@A[i]+b[i];assert abs(float(np.mean((zv>=0)==yv))-entry['validation_accuracy'])<1e-12
      assert abs(float(np.mean(np.logaddexp(0,zv)-yv*zv))-entry['validation_log_loss'])<1e-10
      fit_seconds+=f['seconds']
     assert np.array_equal(rng.permutation(50),order);panels[ds,seed,arm]=(A,b,order)
    for arm in ARMS:
     A,b,_=panels[ds,seed,arm];z=Xv@A.T+b
     quality.append(dict(dataset=ds,panel_type=arm,seed=seed,validation_accuracy=float(np.mean((z>=0)==yv[:,None])),validation_log_loss=float(np.mean(np.logaddexp(0,z)-yv[:,None]*z))))
  log('MODEL_AUDITS_PASSED',3000,'maximum_gradient',max_grad);progress(status='verifying_and_scoring_saved_points')
  records=[]
  with (src/'records.jsonl').open() as f:
   for line in f:
    a=json.loads(line)
    if a['dataset'] in cfg['datasets']:records.append(a)
  assert len(records)==92400
  dtype={k:'category' for k in ['dataset','panel_type','method']}
  df=pd.read_csv(parent/'evaluation_records.csv',dtype=dtype)
  assert len(df)==1108800 and not df.duplicated(CELL+['seed','person']).any()
  for name,source_name in [('dataset','dataset'),('seed','seed'),('person','person'),('test_id','test_id'),('competing_model_count','model_count'),('budget','budget'),('budget_index','budget_index'),('method','method'),('available','available')]:
   assert np.array_equal(df[name].to_numpy(),np.repeat([r[source_name] for r in records],12)), name
  assert np.array_equal(df['panel_type'].to_numpy(),np.tile(np.repeat(ARMS,4),92400))
  assert np.array_equal(df['unseen_panel_size'].to_numpy(),np.tile(KS,92400*3))
  values=df[METRICS].to_numpy().reshape(92400,3,4,5).copy();boundary=df['boundary_count'].to_numpy().reshape(92400,3,4).copy()
  cases=defaultdict(list)
  for i,r in enumerate(records):cases[r['point_file']].append(i)
  points_manifest={};max_difference=0.;original_unaffected=values[:,:2].copy()
  for j,(rel,indices) in enumerate(cases.items(),1):
   rows=[records[i] for i in indices];ds=rows[0]['dataset'];seed=rows[0]['seed'];availability=np.asarray([r['available'] for r in rows],bool)
   path=src/rel;points_manifest[rel]=sha(path)
   with np.load(path,allow_pickle=False) as z:x=np.column_stack([z[r['method']] for r in rows]).astype(float)
   for ai,arm in enumerate(ARMS):
    A,b,order=panels[ds,seed,arm];logits=np.column_stack([A[order]@x[:,c]+b[order] for c in range(x.shape[1])])
    for ki,k in enumerate(KS):
     z=logits[:k];minimum=z.min(axis=0)
     calc=np.column_stack([availability&np.all(z>=-1e-7,axis=0),availability&np.all(z>=0,axis=0),availability*np.mean(z>=-1e-7,axis=0),minimum,np.logaddexp(0.,-minimum)])
     bc=np.sum(np.abs(z)<=1e-7,axis=0)
     if ai<2:
      diff=float(np.max(np.abs(values[indices,ai,ki]-calc)));max_difference=max(max_difference,diff)
      assert np.allclose(values[indices,ai,ki],calc,rtol=1e-11,atol=1e-9),f'Existing evaluation mismatch {rel}/{arm}/{k}'
      assert np.array_equal(boundary[indices,ai,ki],bc)
     else:values[indices,ai,ki]=calc;boundary[indices,ai,ki]=bc
   if j%3000==0:log('SAVED_POINTS_CHECKED',j,len(cases))
  assert np.array_equal(values[:,:2],original_unaffected)
  df[METRICS]=values.reshape(-1,5);df['boundary_count']=boundary.reshape(-1)
  assert np.isfinite(df[METRICS].to_numpy()).all()
  assert np.all(values[:,:,:,0]<=values[:,:,:,2]+1e-12)
  assert np.all(np.diff(values[:,:,:,0],axis=2)<=0)
  assert np.all(values[~np.asarray([r['available'] for r in records],bool),:,:,:3]==0)
  df.to_csv(root/'evaluation_records.csv',index=False,float_format='%.17g')
  save(root/'audits/current_source_point_hashes.json',points_manifest)
  log('ALL_RECORDS_COMPLETE',len(df));progress(status='analysis',completed_records=len(df),source_records_completed=len(records))
  # Independent score audit of the corrected arm from the written CSV, using per-point dot products.
  saved=pd.read_csv(root/'evaluation_records.csv',dtype=dtype)
  assert len(saved)==len(df) and not saved.duplicated(CELL+['seed','person']).any()
  assert np.allclose(saved[METRICS],df[METRICS],rtol=1e-12,atol=1e-10)
  for i in np.random.default_rng(20261019).choice(len(records),size=600,replace=False):
   r=records[i];A,b,order=panels[r['dataset'],r['seed'],ARMS[2]]
   with np.load(src/r['point_file']) as z:x=z[r['method']].astype(float)
   for ki,k in enumerate(KS):
    logits=np.asarray([float(np.dot(A[ii],x)+b[ii]) for ii in order[:k]])
    row=saved.iloc[i*12+8+ki]
    assert row.joint_validity==float(r['available'] and min(logits)>=-1e-7)
    assert abs(row.mean_acceptance-(np.mean(logits>=-1e-7) if r['available'] else 0.))<1e-12
  del saved
  sizes=df.groupby(CELL+['seed'],observed=True).size();assert (sizes==20).all()
  seed=df.groupby(CELL+['seed'],observed=True)[METRICS].mean().reset_index()
  seed.to_csv(root/'audits/seed_means.csv',index=False)
  summary=aggregate(seed,CELL);summary.to_csv(root/'tables/all_cells_95ci.csv',index=False)
  effects=[]
  def contrast(frame,axis,left,right,label):
   keys=[x for x in CELL if x!=axis]
   a=frame[frame[axis]==left].drop(columns=axis);b=frame[frame[axis]==right].drop(columns=axis)
   joined=a.merge(b,on=keys+['seed'],suffixes=('_left','_right'),validate='one_to_one')
   assert len(joined)>0
   delta=joined[keys+['seed']].copy()
   for metric in METRICS:delta[metric]=joined[metric+'_right']-joined[metric+'_left']
   table=aggregate(delta,keys);table['contrast']=label;table['contrast_axis']=axis;table['left_level']=str(left);table['right_level']=str(right);effects.append(table)
  for ds in cfg['datasets']:
   part=seed[seed.dataset==ds];budgets=cfg['budgets'][ds]
   for low,high in zip(budgets[:-1],budgets[1:]):contrast(part,'budget',low,high,'adjacent budget increase')
   contrast(part,'budget',budgets[0],budgets[-1],'largest budget minus zero')
  for left,right in [(6,12),(12,24),(6,24)]:contrast(seed,'competing_model_count',left,right,'competing set size increase')
  for left,right in [(5,10),(10,25),(25,50),(5,50)]:contrast(seed,'unseen_panel_size',left,right,'unseen panel size increase')
  for arm in ARMS[1:]:contrast(seed,'panel_type',ARMS[0],arm,'bootstrap minus Hessian')
  paired=pd.concat(effects,ignore_index=True);paired.to_csv(root/'tables/paired_effects_95ci.csv',index=False)
  # Panel-quality diagnostics use the same ten-seed inference unit.
  q=pd.DataFrame(quality).melt(['dataset','panel_type','seed'],var_name='metric',value_name='value')
  qt=q.groupby(['dataset','panel_type','metric'],observed=True)['value'].agg(['mean','std','count']).reset_index();half=t.ppf(.975,9)*qt['std']/np.sqrt(10)
  qt['ci_low']=qt['mean']-half;qt['ci_high']=qt['mean']+half;qt=qt.drop(columns='std').rename(columns={'count':'seeds'});qt.to_csv(root/'tables/panel_quality_95ci.csv',index=False)
  log('CONFIDENCE_INTERVALS_COMPLETE',len(summary),len(paired));progress(status='rendering_figures')
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
  figs=root/'figures';figs.mkdir();lookup=summary.set_index(CELL+['metric']);made=0
  colors=['#1769aa','#c64f24','#27804b'];styles=[':','--','-.','-']
  for ds in cfg['datasets']:
   for arm in ARMS:
    for method in cfg['methods']:
     for metric in ['joint_validity','mean_acceptance']:
      fig,ax=plt.subplots(figsize=(8.4,5.8));xs=cfg['budgets'][ds]
      for ni,n in enumerate([6,12,24]):
       for ki,k in enumerate(KS):
        a=np.asarray([lookup.loc[(ds,arm,k,n,float(B),method,metric),['mean','ci_low','ci_high']].to_numpy(dtype=float) for B in xs])
        ax.plot(xs,a[:,0],color=colors[ni],linestyle=styles[ki],linewidth=1.7,label=f'{n} competing / {k} unseen')
        ax.fill_between(xs,np.clip(a[:,1],0,1),np.clip(a[:,2],0,1),color=colors[ni],alpha=.055)
      ax.set(xlabel='Recourse budget (standardized L2)',ylabel='All-K joint validity' if metric=='joint_validity' else 'Mean acceptance across K models',ylim=(-.01,1.01),title=f'{LABELS[ds]} · {method}\n{LABELS[arm]}')
      ax.grid(alpha=.2);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3,fontsize=8,frameon=False)
      fig.text(.5,.015,'Bands: pointwise 95% t intervals across 10 seed means; displayed within [0, 1].',ha='center',fontsize=8)
      fig.tight_layout(rect=[0,.14,1,1]);stem=f'{ds}__{method}__{arm}__{metric}'
      fig.savefig(figs/(stem+'.png'),dpi=160);fig.savefig(figs/(stem+'.pdf'));plt.close(fig);made+=1
      if made%21==0:log('FIGURES_SAVED',made,126)
  assert made==126
  # Hashes and summary checks after analysis, including preservation of the historical failed run.
  assert sha(parent/'evaluation_records.csv')==parent_csv_hash
  for name,h in manifest['source'].items():assert sha(src/name)==h
  for rel,h in points_manifest.items():assert sha(src/rel)==h
  for item in inventory:assert sha(root/item['path'])==item['sha256']
  for ds in cfg['datasets']:
   for arm in ARMS:
    s=seed[(seed.dataset==ds)&(seed.panel_type==arm)&(seed.unseen_panel_size==50)&(seed.competing_model_count==24)&(seed.budget==cfg['budgets'][ds][-1])&(seed.method=='ORPM')].sort_values('seed')
    check=lookup.loc[(ds,arm,50,24,float(cfg['budgets'][ds][-1]),'ORPM','joint_validity')]
    a=s.joint_validity.to_numpy();assert abs(check['mean']-a.mean())<1e-14 and abs(check.ci_low-(a.mean()-t.ppf(.975,9)*a.std(ddof=1)/np.sqrt(10)))<1e-14
  verification={'passed':True,'verified_utc':utc(),'records':len(df),'source_recourses':len(records),'new_masked_models':1500,'reused_full_feature_models':1500,
   'model_panels':60,'source_manifest_hashes_checked':len(manifest['source']),'current_point_hashes_checked':len(points_manifest),
   'source_and_parent_preservation':True,'all_unaffected_records_recomputed_and_preserved':739200,
   'all_3000_model_gradients_and_bootstrap_draws_checked':True,'maximum_gradient_l2':max_grad,
   'all_random_mask_draws_replayed':True,'corrected_mask_score_independent_scalar_checks':2400,
   'all_record_keys_unique_and_complete':True,'all_seed_cells_have_20_people':True,'confidence_intervals_checked':True,
   'figure_panels':126,'each_saved_separately_png_pdf':True,'protocol_compliant':True,'failures':[],
   'scope':'Float64 diagnostics at declared tolerances; bootstrap panels conditional on saved training splits and cohorts. This is an extension, not an independent replication.'}
  save(root/'verification.json',verification)
  lines=['V7 RETRAINED-LOGISTIC TRANSFER PILOT — VERIFIED CORRECTION 1.1',f'Generated {utc()}',f'Results: {root}','',
   'STATUS: All 1,108,800 evaluation rows and 126 separate PNG/PDF figure panels are complete; independent verification passed.',
   'Scope: GMC, COMPAS and Synthetic; ten seeds; twenty original people per seed; seven methods; all frozen budgets; 6/12/24 competing models; 5/10/25/50 unseen models.',
   'The existing Hessian panel and full-feature bootstrap panel are preserved. The random-mask arm samples uniformly with replacement from each source bank’s first 24 feature masks, then fits ordinary row bootstrap logistic models using the saved training split (L2=0.001). No validation or recourse outcome is used to select a model.',
   'Correction history: The original worker completed all scoring but failed when creating its report tables. Its progress count lagged the actual CSV. Audit also found that masks had been cycled rather than randomly sampled. Version 1.1 corrects only that arm in a new run; the earlier result directory is preserved. No recourse optimization was repeated.',
   'Intervals: pointwise 95% Student-t intervals across ten equally weighted seed means (df=9). Paired effects subtract matching seed means. Curves sharing people/models/seeds are dependent. No multiplicity correction is applied. Proportion bounds are clipped to [0,1] only for report/figure display; CSVs retain raw t bounds.',
   'Primary outcomes: all-K joint validity and mean acceptance are different quantities. Unavailable methods contribute zero to both. Auxiliary minimum-logit and worst-loss averages include the stored fallback point when a method is unavailable.',
   'Limits: This reuses the completed study’s cohorts, splits and recourses and is not independent replication. Bootstrap panels measure conditional training-sample sensitivity. For Synthetic, every source feature mask is full-feature, so the two bootstrap arms are independent refit panels of the same model class. This does not evaluate mixed-integer action constraints or nonlinear unseen models.',
   '', 'AT LARGEST BUDGET, 24 COMPETING MODELS, 50 UNSEEN MODELS — mean [95% CI]']
  highlights=[]
  def fmt(row,percent=True):
   scale=100 if percent else 1;lo=max(0.,row.ci_low) if percent else row.ci_low;hi=min(1.,row.ci_high) if percent else row.ci_high
   return f'{scale*row["mean"]:.1f} [{scale*lo:.1f}, {scale*hi:.1f}]'+('%' if percent else '')
  for ds in cfg['datasets']:
   lines+=['',f'{LABELS[ds]} (budget {cfg["budgets"][ds][-1]})']
   for arm in ARMS:
    lines.append(LABELS[arm])
    for method in cfg['methods']:
     a=lookup.loc[(ds,arm,50,24,float(cfg['budgets'][ds][-1]),method,'joint_validity')];b=lookup.loc[(ds,arm,50,24,float(cfg['budgets'][ds][-1]),method,'mean_acceptance')]
     lines.append(f'  {method}: all-50 joint {fmt(a)}; mean acceptance {fmt(b)}')
     if method=='ORPM':highlights.append({'dataset':ds,'panel_type':arm,'joint_mean':float(a['mean']),'joint_ci_low':float(a.ci_low),'joint_ci_high':float(a.ci_high),'acceptance_mean':float(b['mean'])})
  lines+=['','PAIRED EFFECTS FOR ORPM (percentage-point differences; 95% CI)']
  for ds in cfg['datasets']:
   part=paired[(paired.dataset==ds)&(paired.method=='ORPM')&(paired.metric=='joint_validity')]
   for arm in ARMS:
    rows=part[(part.panel_type==arm)&(part.unseen_panel_size==50)]
    budget=rows[(rows.contrast=='largest budget minus zero')&(rows.competing_model_count==24)]
    count=rows[(rows.contrast_axis=='competing_model_count')&(rows.left_level=='6')&(rows.right_level=='24')&(rows.budget==cfg['budgets'][ds][-1])]
    for label,x in [('largest minus zero budget',budget),('24 minus 6 competing models at largest budget',count)]:
     assert len(x)==1
     a=x.iloc[0];lines.append(f'{LABELS[ds]} | {LABELS[arm]} | {label}: {100*a["mean"]:.1f} [{100*a.ci_low:.1f}, {100*a.ci_high:.1f}] pp')
  elapsed=time.perf_counter()-started
  lines+=['',f'Correction and diagnostics elapsed: {elapsed/60:.1f} minutes.','Files: tables/all_cells_95ci.csv; tables/paired_effects_95ci.csv; tables/panel_quality_95ci.csv; audits/seed_means.csv; verification.json; figures/; figures_tables.zip.']
  report='\n'.join(lines)+'\n';(root/'report.txt').write_text(report)
  cards=''.join(f'<figure><a href="figures/{html.escape(p.stem)}.pdf"><img loading="lazy" src="figures/{html.escape(p.name)}" alt="{html.escape(p.stem)}"></a></figure>' for p in sorted(figs.glob('*.png')))
  page='<!doctype html><html><head><meta charset="utf-8"><title>V7 transfer diagnostics</title><style>body{max-width:1100px;margin:40px auto;padding:0 20px;font:16px/1.6 system-ui;color:#152232}pre{white-space:pre-wrap;font:14px/1.6 ui-monospace}figure{margin:30px 0}img{width:100%;max-width:850px}</style></head><body><h1>V7 retrained-model transfer diagnostics</h1><pre>'+html.escape(report)+'</pre><p><a href="tables/all_cells_95ci.csv">All estimates and 95% CIs</a> · <a href="tables/paired_effects_95ci.csv">Paired effects and 95% CIs</a> · <a href="tables/panel_quality_95ci.csv">Panel quality and 95% CIs</a></p>'+cards+'</body></html>'
  (root/'report.html').write_text(page)
  save(root/'analysis.json',{'completed_utc':utc(),'study_complete':True,'verification_passed':True,'figure_panels':made,'all_inferential_tables_have_95ci':True,'elapsed_seconds':elapsed,'orpm_largest_budget_highlights':highlights})
  with zipfile.ZipFile(root/'figures_tables.zip','w',zipfile.ZIP_DEFLATED) as z:
   for p in sorted(root.rglob('*')):
    if p.is_file() and ('__pycache__' not in p.parts) and p.name not in ['figures_tables.zip','worker.lock','progress.json','run.log']:
     z.write(p,str(p.relative_to(root)))
  with zipfile.ZipFile(root/'figures_tables.zip') as z:assert z.testzip() is None
  progress(status='completed',verification_passed=True,elapsed_seconds=time.perf_counter()-started)
  log('PILOT_COMPLETED','all records verified; corrected mask sampling; report and archive ready')
 except Exception as e:
  import traceback
  save(root/'failure.json',{'failed_utc':utc(),'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()});progress(status='failed',error=str(e));log('DIAGNOSTICS_FAILED',type(e).__name__,e);raise

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--parent',required=True);ap.add_argument('--root',required=True);a=ap.parse_args();run(a.parent,a.root)
