"""Frozen design helpers for same-architecture unseen tree/forest evaluation."""
from common import *
from native_prediction import native_saved_prediction
from source_verify import independent_prob
from collections import Counter,defaultdict
import shutil
DATASETS=['Synthetic','COMPAS','Polish','German','Give_Me_Some_Credit','Folktables']
SOURCE=BASE/'experiments/trade-off-trees/phase_main_10x5_v1_0/results/20261023T122509.927561Z_tree_tradeoff'
LABELS={'Give_Me_Some_Credit':'GMC','Folktables':'ACSIncome'}
PANELS=['refit','bootstrap']

def fingerprint(model):
 h=hashlib.sha256()
 for t in model.trees:
  for a in [t.left,t.right,t.feature,t.threshold,t.probability]:h.update(np.ascontiguousarray(a).tobytes())
 return h.hexdigest()

def seed_for(arm,ds,seed,stream,index):
 return 10_000_000+(100_000_000 if arm=='random_forest_pilot' else 0)+DATASETS.index(ds)*10_000_000+(seed-400)*100_000+stream*10_000+index

def common_eligibility(models,X):
 pr=independent_prob(models,X);ok=np.ones(len(pr),dtype=bool)
 for m in [6,12,24]:ok &= 2*(pr[:,:m]<.5).sum(1)>=m
 return ok

def frozen_check(root):
 for rel,digest in read(root/'freeze.json')['sha256'].items():
  assert sha(root/rel)==digest,('frozen_changed',str(root/rel))
 for path,digest in read(root/'source_preservation.json')['sha256'].items():
  assert sha(path)==digest,('source_changed',path)

def progress(root,**kw):
 old=read(root/'progress.json') if (root/'progress.json').exists() else {}
 write(root/'progress.json',dict(old,utc=now(),**kw))

def evaluation_metrics(pr,seen,factual,needed):
 valid=(np.asarray(pr)>=.5);fvalid=np.asarray(factual)>=.5
 ans=dict(seen_joint=float(seen),unseen_mean=float(valid.mean()),unseen_joint=float(valid.all()),
  unseen_worst_probability=float(np.min(pr)),factual_unseen_mean=float(fvalid.mean()),factual_unseen_joint=float(fvalid.all()),
  mean_acceptance_gain=float(valid.mean()-fvalid.mean()),joint_acceptance_gain=float(float(valid.all())-float(fvalid.all())),
  seen_to_unseen_joint_drop=float(seen)-float(valid.all()),recourse_needed=bool(needed))
 for k in [5,10,25,50]:ans[f'unseen_joint_{k}']=float(valid[:k].all())
 return ans

def make_cells(p):
 result=[]
 for seed in p['seeds']:
  for m in p['model_counts']:
   for ds in p['datasets']:
    for b in p['budgets'][ds]:
     result.append(dict(key=cell_key(ds,seed,0,m,b),dataset=ds,seed=seed,person=0,models=m,budget=b,pilot=True))
 return result
