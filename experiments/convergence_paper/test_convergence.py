"""Numerical contract checks using analytic problems, plus per-instance censoring checks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import convergence_study as study
rc,gap,mixed=study.rc,study.gap,study.mixed

def main():
    results={}
    c=rc.Constraints(lower=np.array([-1.]),upper=np.array([1.]),immutable=[])
    origin=np.array([0.]);models=rc.ModelSet()
    for sign in [1.,-1.]:
        m=rc.LogisticRegressionModel(1)
        with torch.no_grad():m.linear.weight.fill_(sign);m.linear.bias.zero_()
        models.append(m)
    oracle=gap.ConvexLogitOracle(models,origin,1.,c,maxiter=100);oracle.normalizer=1.
    x,info=oracle.solve(np.array([.5,.5]),np.array([.9]))
    exact=np.log(2.)
    assert info['lower_bound']<=exact+1e-8<=info['objective']+1e-8
    assert abs(x[0])<1e-4
    results['convex_global_minimum_enclosed']=True
    # The randomized-response gap and point gap are genuinely different quantities.
    neural=rc.ModelSet()
    for sign in [1.,-1.]:
        m=rc.NeuralNetworkModelSingleLayer(1,1)
        with torch.no_grad():
            m.net[0].weight.fill_(sign);m.net[0].bias.zero_()
            m.net[2].weight.fill_(2.);m.net[2].bias.fill_(-1.)
        neural.append(m)
    local=mixed.LocalLogisticOracle(neural,origin,1.,c,1.,100)
    exact=float((np.logaddexp(0,-1.)+np.logaddexp(0,1.))/2)
    evaluator=mixed.NeuralLogisticBound(neural,origin,1.,c,1.,time_limit=2.,seed=9901)
    try:
        point,info=evaluator.solve(np.array([.5,.5]),np.array([1.]),local)
        assert info['bound_consistent']
        assert info['lower_bound']<=exact+1e-6 and info['upper_bound']>=exact-1e-6
        assert info['upper_bound']-info['lower_bound']<.005
    finally:evaluator.close()
    deterministic=float(local.losses(origin).max())-exact
    randomized=float(np.mean([local.losses(np.array([-1.])),local.losses(np.array([1.]))],axis=0).max())-exact
    assert abs(deterministic-.5)<1e-10 and abs(randomized)<1e-10
    results.update(neural_bound_contains_analytic_minimum=True,randomized_gap=randomized,deterministic_gap=deterministic)
    g=pd.DataFrame(dict(round=[1,5,10,20],gap_lower=[.02,.001,.001,.001],gap_upper=[.03,.004,.006,.004]))
    s=study.crossing_summary(g,.005,2)
    assert s['first_certified_round']==5 and s['first_confirmed_round'] is None
    assert s['lost_certification_after_first'] and not s['recrossed_after_first']
    assert s['sustained_through_final_round']==20
    u=study.crossing_summary(pd.DataFrame(dict(round=[1,5],gap_lower=[0.,0.],gap_upper=[.1,.1])),.005,2)
    assert u['final_status']=='unresolved' and u['first_certified_round'] is None
    results['unresolved_not_misclassified_as_nonconvergence']=True
    for family in ['linear','neural','mixed']:
        assert set(study.indices(family,6))<set(study.indices(family,12))<set(study.indices(family,24))
    results['nested_shared_pool_indices_verified']=True
    # A real interrupted trajectory must resume identically to an uninterrupted one.
    import tempfile
    config=json.loads(Path(__file__).with_name('config.json').read_text())
    config.update(datasets=['Synthetic'],seeds=[9903],instances=1,rounds=10,checkpoints=[1,5,10],save_every=5)
    with torch.no_grad():
        models[0].linear.bias.fill_(.25)
        models[1].linear.weight.fill_(-2.)
    with tempfile.TemporaryDirectory(prefix='recourse_resume_') as tmp:
        runner=study.Study(Path(tmp)/'study',config)
        original_local=runner.local
        def interrupting_local(*args,**kwargs):
            oracle=original_local(*args,**kwargs);solve=oracle.solve;calls=[0]
            def wrapped(*a,**k):
                calls[0]+=1
                if calls[0]==5:raise KeyboardInterrupt('test interruption')
                return solve(*a,**k)
            oracle.solve=wrapped
            return oracle
        resume=runner.root/'resume';resume.mkdir();fresh=runner.root/'fresh';fresh.mkdir()
        runner.local=interrupting_local
        try:runner.trajectory(resume,models,origin,1.,c,3.)
        except KeyboardInterrupt:pass
        else:raise AssertionError('Expected interruption')
        assert len(study.arrays(resume/'trajectory.npz')['points'])==4
        runner.local=original_local
        resumed,_=runner.trajectory(resume,models,origin,1.,c,3.)
        uninterrupted,_=runner.trajectory(fresh,models,origin,1.,c,3.)
        assert not np.allclose(resumed['weights'][0],resumed['weights'][-1])
        for key in ['points','weights','losses','log_weights']:
            np.testing.assert_array_equal(resumed[key],uninterrupted[key])
    results['interrupted_resume_matches_uninterrupted_trajectory']=True
    from convergence_report import summarize_seed_fractions
    f=pd.DataFrame(dict(group=['a','a'],seed=[1,2],metric=['certified','certified'],difference=[.2,.4]))
    out=summarize_seed_fractions(f,['group','metric'],['difference'])
    assert out.metric.iloc[0]=='certified' and out.summary_metric.iloc[0]=='difference'
    assert abs(out['mean'].iloc[0]-.3)<1e-12
    results['paired_effect_outcome_and_seed_intervals_verified']=True
    # All six loaders, label mappings and train-only scaling are checked before launch.
    datasets={}
    for ds in study.DATA_FILES:
        X,y,action=rc.load_data_and_normalize(ds);ctx=rc.prepare_split(X,y,401,ds,action)
        assert set(ctx.train_ids).isdisjoint(ctx.test_ids) and set(ctx.validation_ids).isdisjoint(ctx.test_ids)
        np.testing.assert_allclose(ctx.scaler.mean_,X.iloc[ctx.train_ids].to_numpy().mean(axis=0),rtol=1e-10,atol=1e-10)
        np.testing.assert_array_equal(ctx.y_test,rc.favorable_labels(y,ds)[ctx.test_ids])
        datasets[ds]=dict(observations=len(y),features=X.shape[1],actionable=len(action),favorable_raw_label=rc.FAVORABLE_LABEL[ds])
    results['datasets']=datasets;results['status']='passed'
    Path(__file__).with_name('contract_verification.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):
        torch.set_num_threads(1);main()
