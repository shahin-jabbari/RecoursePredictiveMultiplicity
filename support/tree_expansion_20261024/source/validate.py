from common import *
from sklearn.tree import DecisionTreeClassifier
from itertools import product
from runner import gap
import tempfile

def exhaustive(models,x0,b,c,w=None):
 best=float('inf');n=0
 for combo in product(*[model.trees[0].leaves(len(x0)) for model in models]):
  lo=np.maximum.reduce([c.lower,*[q[1] for q in combo]]);hi=np.minimum.reduce([c.upper,*[q[2] for q in combo]])
  if np.any(lo>hi):continue
  z=np.maximum(lo,np.minimum(hi,x0.numpy().astype(float))).astype(np.float32)
  if not point_check(z,x0,b,c):continue
  l=loss(models,z);value=float(l.max() if w is None else l@w);best=min(best,value);n+=1
 return best,n

def validate(destination):
 destination=Path(destination);destination.mkdir(parents=True,exist_ok=True);checks=[]
 rng=np.random.default_rng(2210);X=rng.normal(size=(150,2)).astype(np.float32);y=(X[:,0]+.7*X[:,1]>0).astype(int)
 models=[rt.TreeModel.from_estimator(DecisionTreeClassifier(max_depth=2,min_samples_leaf=4,max_features=1,random_state=s).fit(X,y)) for s in [1,9,21]]
 x=torch.tensor([-.3,-.4],dtype=torch.float32)
 for immutable in [[],[1]]:
  lo=np.full(2,-np.inf);hi=np.full(2,np.inf);lo[immutable]=x.numpy()[immutable];hi[immutable]=x.numpy()[immutable];c=rc.Constraints(lo,hi,immutable)
  for budget in [.1,1.,4.]:
   for objective,w in [('weighted_loss',np.array([.2,.3,.5])),('minimax',None)]:
    oracle=rt.TreeOracle(models,x,budget,c,time_limit=5,abs_gap=1e-8,seed=390)
    try:
     z,info=checked_solve(oracle,w,objective);exact,n=exhaustive(models,x,budget,c,w);audit=audit_encoding(oracle,z,objective)
     assert n>0 and audit['passed'],audit
     assert info['lower_bound']<=exact+1e-7 and info['objective']>=exact-1e-7,(info,exact)
     assert abs(info['objective']-exact)<1e-6 and info['global_certificate'],(info,exact)
     checks.append(dict(immutable=immutable,budget=budget,objective=objective,exact=exact,oracle=info['objective'],audit=audit))
    finally:oracle.close()
 # Float32 boundary neighbors and probability ties use p>=.5, not native class tie-breaking.
 clf=DecisionTreeClassifier(max_depth=1).fit(np.array([[0.],[0.],[2.],[2.]],dtype=np.float32),[0,1,1,1]);model=rt.TreeModel.from_estimator(clf);threshold=model.trees[0].threshold[0];left=float(rt.f32_floor(threshold));right=float(np.nextafter(np.float32(left),np.float32(np.inf)));points=np.array([[left],[right]],dtype=np.float32)
 assert np.array_equal(model.predict_probability(points),clf.predict_proba(points)[:,1]);assert model.predict_probability(points)[0]==.5 and (model.predict_probability(points)[0]>=.5)
 assert clf.predict(points)[0]==0;checks.append(dict(boundary_neighbors_and_tie_rule=True,left=left,right=right))
 try:gap(.1,.2,.3);raise AssertionError('Contradiction was not rejected')
 except RuntimeError:pass
 # Inject a contradictory certificate only in labeled validation; require complete witness audit and conservative rejection.
 oracle=rt.TreeOracle(models,x,1.,rc.Constraints(np.full(2,-np.inf),np.full(2,np.inf)),time_limit=5)
 original=oracle.solve
 def inject(*a,**kw):
  z=original(*a,**kw);z.recourse_info['lower_bound']=2.;return z
 oracle.solve=inject
 try:
  z,info=checked_solve(oracle,np.ones(3)/3);assert info['certificate_rejected'] and info['lower_bound']==0 and info['witness_audit']['passed'];checks.append(dict(injected_contradictory_bound_rejected=True))
 finally:oracle.close()
 from supervisor import supervise
 for mode in ['failure','deadline']:
  result=supervise(destination/('validation_'+mode),validation=mode,seconds=.3 if mode=='deadline' else 10)
  assert not result['completed'];assert result['returncode']==7 if mode=='failure' else result['runtime_limit'];checks.append(dict(supervisor_test=mode,result=result))
 write(destination/'validation.json',dict(passed=True,utc=now(),tests=checks,scope='exhaustive tiny-tree weighted and minimax problems, immutable regions, float32 boundaries/ties, rejected-certificate audit, supervisor failure/deadline'))
 print('VALIDATION_PASSED',len(checks),flush=True)
if __name__=='__main__':validate(Path(sys.argv[1]))
