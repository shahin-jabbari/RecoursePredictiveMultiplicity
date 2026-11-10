"""Synthetic integrity tests only; not scientific outcomes or pilot calibration."""
from experiment import *
from forest_methods import reference_solve
from methods import run_orpm,pack_models,outcome,direct
from legacy_verify import verify_method
from analyze_unseen import interval,paired
from scipy.stats import t as student
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def run(base):
 base=Path(base);base.mkdir(parents=True,exist_ok=False);tests=[]
 # Native sklearn forest and frozen array representation agree, including mean probabilities.
 rng=np.random.default_rng(713);X=rng.normal(size=(100,2)).astype(np.float32);y=(X[:,0]+X[:,1]>0).astype(int)
 clf=RandomForestClassifier(n_estimators=10,max_depth=4,min_samples_leaf=5,max_features='sqrt',random_state=177,n_jobs=1).fit(X,y)
 model=rt.TreeModel.from_estimator(clf)
 assert np.allclose(model.predict_probability(X),clf.predict_proba(X)[:,1],atol=1e-12,rtol=0)
 assert np.allclose(native_saved_prediction(model,X),clf.predict_proba(X)[:,1],atol=1e-12,rtol=0)
 tests.append('native_10_tree_forest_probability_averaging')
 # Exact toy: the fifth threshold is where half of the ten trees accept.
 def tree(threshold):return rt.TreeSpec(np.array([1,-1,-1]),np.array([2,-1,-1]),np.array([0,-2,-2]),np.array([threshold,-2.,-2.]),np.array([.5,0.,1.]))
 forests=[rt.TreeModel([tree(.2+.1*k+.005*j) for k in range(10)],2,'random_forest') for j in range(24)]
 x=torch.tensor([0.,0.]);con=rc.Constraints(np.array([-np.inf,0.]),np.array([np.inf,0.]),[1],'immutable')
 point,info=reference_solve(forests[:1],x,2.,con,401,base/'reference',[])
 expected=float(np.nextafter(np.float32(rt.f32_floor(.6)),np.float32(np.inf)))
 assert info['success'] and abs(float(point[0])-expected)<1e-7 and rt.probabilities(forests[:1],point)[0]==.5
 assert not all(t.probability[t.apply(point)[0]]>=.5 for t in forests[0].trees)
 tests.append('whole_forest_acceptance_not_all_constituent_trees')
 pt,no=reference_solve(forests[:1],x,.1,con,401,base/'infeasible',[])
 assert not no['success'] and no['solver_status']==3 and np.array_equal(pt,x)
 tests.append('bounded_reference_exclusion_is_not_a_successful_factual_witness')
 # Checked loss oracle rejects a contradictory numeric status only after full feasible-witness audit.
 o=rt.TreeOracle(forests[:2],x,2.,con,time_limit=1,artifact_dir=base/'oracle')
 try:
  w=np.ones(2)/2;pt= o.solve(w);raw=dict(pt.recourse_info);raw.update(solver_status=3,lower_bound=1.,bound_inconsistent_with_verified_candidate=True)
  from common import checked_solve
  pt,fixed=checked_solve(o,w,known_points=(x,),recovered_info=raw)
  assert fixed['certificate_rejected'] and fixed['lower_bound']==0 and fixed['witness_audit']['passed'] and not fixed['global_certificate']
  assert audit_encoding(o,torch.tensor([.8,1.]),'weighted_loss')['passed'] is False
 finally:o.close();pack_models(base/'oracle')
 tests.append('contradictory_loss_certificate_audited_and_invalid_witness_rejected')
 # Run/replay actual 100-round methods on labeled synthetic bank, separate from study.
 b=base/'banks/Synthetic/seed_401';b.mkdir(parents=True)
 for j,m in enumerate(forests):m.save(b/'candidates'/f'{j:04d}')
 np.savez_compressed(b/'split.npz',X_test=np.array([[0.,0.]],np.float32),test_ids=np.array([0]))
 write(b/'bank.json',dict(selected_candidate_indices=list(range(24)),immutable_indices=[1]));write(base/'cohorts.json',{'Synthetic':{'401':dict(positions=[0],source_ids=[0])}});write(base/'protocol.json',dict(absolute_deadline_epoch=None))
 cell=dict(key=cell_key('Synthetic',401,0,6,2.),dataset='Synthetic',seed=401,person=0,models=6,budget=2.,pilot=True)
 d=method_dir(base,cell,'ORPM');d.mkdir(parents=True);result=run_orpm(base,cell,d);write(d/'outcome.json',result)
 replay=verify_method(base,cell,'ORPM');assert replay['rounds']==100 and result['solver_calls']==100
 d=method_dir(base,cell,'UniformForest');d.mkdir(parents=True);result=direct(base,cell,'UniformForest',d);write(d/'outcome.json',result);verify_method(base,cell,'UniformForest')
 tests.append('100_round_whole_forest_weights_actions_and_uniform_baseline_replay')
 # Aggregation uncertainty must count seeds, not unseen models or person rows.
 a=np.arange(10)/10;ci=interval(a);half=student.ppf(.975,9)*a.std(ddof=1)/np.sqrt(10)
 assert ci['seeds']==10 and ci['df']==9 and abs(ci['ci_high']-(a.mean()+half))<1e-12
 e=evaluation_metrics(np.array([.9]*49+[.1]),True,np.zeros(50),True);assert e['unseen_mean']==.98 and e['unseen_joint']==0 and e['seen_to_unseen_joint_drop']==1
 tests.append('seed_level_t_intervals_and_mean_vs_joint_distinction')
 assert len({seed_for(arm,ds,s,stream,i) for arm in ['decision_tree','random_forest_pilot'] for ds in DATASETS for s in range(401,411) for stream in [0,1,2] for i in [0,4095]})==2*6*10*3*2
 tests.append('disjoint_new_fit_seed_namespaces')
 write(base/'validation.json',dict(passed=True,utc=now(),synthetic_only=True,not_study_outcomes=True,tests=tests))
 print(json.dumps(dict(passed=True,tests=tests)),flush=True)

if __name__=='__main__':run(Path(sys.argv[1]))
