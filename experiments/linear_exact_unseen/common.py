import json,hashlib,os
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

def utc():return datetime.now(timezone.utc).isoformat()
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,Path):return str(x)
    return x

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(clean(obj),indent=2,allow_nan=False)+'\n');os.replace(temporary,path)
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def theta_hash(theta):return hashlib.sha256(np.asarray(theta,dtype='<f4').tobytes()).hexdigest()
def metrics(A,b,x,available,tol):
    z=A@x+b; ok=z>=-tol; strict=z>=0
    return dict(joint_validity=float(available and ok.all()),strict_joint_validity=float(available and strict.all()),mean_validity=float(ok.mean()) if available else 0.,minimum_logit=float(z.min()),worst_loss=float(np.logaddexp(0.,-z).max()),boundary_count=int(np.sum(abs(z)<=tol)))
