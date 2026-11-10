from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os
import numpy as np

BASE=Path('/Users/sj987/Documents/ChatGPT/Code/Predictive_Multiplicity_Recourse/V7/experiments')
TRADE=BASE/'trade-off_paper/phase_10x20/results/20261017T004739.247649Z_6ffc650a4c'
DATASETS=['Synthetic','COMPAS','Polish','German','Give_Me_Some_Credit','Folktables']
GROUPS={
 'COMPAS':dict(feature='sex=female',kind='coded',first_code=1,codes=[0,1],first='female_recorded',second='male_recorded'),
 'Folktables':dict(feature='SEX',kind='coded',first_code=2,codes=[1,2],first='female_recorded',second='male_recorded'),
 'German':dict(feature='Attribute13',kind='age',threshold=40,first='under_40',second='40_and_older'),
 'Give_Me_Some_Credit':dict(feature='age',kind='age',threshold=40,first='under_40',second='40_and_older')}
def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()
def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(p)
def append(p,obj):
 with open(p,'a',buffering=1) as f:f.write(json.dumps(obj,allow_nan=False)+'\n')
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def indices(n):return list(range(24,24+n//2))+list(range(36,36+n//2))
def labels(X,pre,ds):
 if ds not in GROUPS:return np.repeat('all',len(X))
 g=GROUPS[ds];j=pre['feature_names'].index(g['feature']);assert j not in pre['actionable_indices']
 values=X[:,j]*pre['scale'][j]+pre['mean'][j]
 assert np.max(abs(values-np.round(values)))<1e-4
 values=np.round(values).astype(int)
 if g['kind']=='coded':
  assert set(np.unique(values))<=set(g['codes']);first=values==g['first_code']
 else:first=values<g['threshold']
 return np.where(first,g['first'],g['second'])
def load_net(path):
 z=arrays(path);keys=sorted([k for k in z if k.endswith('.weight')],key=lambda k:int(k.split('.')[-2]))
 layers=[]
 for i,k in enumerate(keys):
  W=z[k].astype(float);b=z[k[:-6]+'bias'].astype(float)
  if i==0 and 'feature_mask' in z:W=W*z['feature_mask'].astype(float)
  layers.append((W,b))
 return layers
def forward(net,X):
 a=np.asarray(X,float)
 for i,(W,b) in enumerate(net):
  a=a@W.T+b
  if i<len(net)-1:a=np.maximum(a,0.)
 return np.asarray(a)[...,0]
def logits(nets,X):return np.stack([forward(net,X) for net in nets],axis=-1)
def torch_forward(path,X):
 import torch
 z=arrays(path);a=torch.tensor(X,dtype=torch.float64)
 if 'feature_mask' in z:a=a*torch.tensor(z['feature_mask'],dtype=torch.float64)
 keys=sorted([k for k in z if k.endswith('.weight')],key=lambda k:int(k.split('.')[-2]))
 for i,k in enumerate(keys):
  a=torch.nn.functional.linear(a,torch.tensor(z[k],dtype=torch.float64),torch.tensor(z[k[:-6]+'bias'],dtype=torch.float64))
  if i<len(keys)-1:a=torch.relu(a)
 return a.detach().numpy()[...,0]
def feasible(x,x0,tau,free,lower,upper,tol=1e-8):
 imm=np.setdiff1d(np.arange(len(x)),free)
 return bool(np.isfinite(x).all() and np.array_equal(x[imm],x0[imm]) and np.linalg.norm(x-x0)<=tau+tol and np.all(x>=lower-tol) and np.all(x<=upper+tol))
