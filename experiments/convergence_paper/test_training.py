"""Regression checks for the weak-reference failure and admission semantics."""
from pathlib import Path
import json,tempfile
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import convergence_study as study
import model_training as training

def main():
    torch.set_num_threads(1);results={}
    X,y,action=study.rc.load_data_and_normalize('Synthetic')
    for seed in [402,404]:
        ctx=study.rc.prepare_split(X,y,seed,'Synthetic',action)
        models=[]
        for init in [seed,12345]:
            m,info=training.fit(study.rc.LogisticRegressionModel(2,init),ctx.X_train,ctx.y_train,{})
            assert info['passed'];assert info['certificate']['absolute_error_bound']<=1e-8
            assert training.accuracy(m,ctx.X_validation,ctx.y_validation)['accuracy']>.9
            models.append(training.parameters(m))
        np.testing.assert_array_equal(*models)
        results[f'bad_seed_{seed}_global_fit_passed']=True
    # An exhausted budget does not silently produce a successful training result.
    m,info=training.fit(study.rc.LogisticRegressionModel(2,404),ctx.X_train,ctx.y_train,{'linear_maxiter':1})
    assert not info['passed'];results['linear_iteration_limit_rejected']=True
    m,info=training.fit(study.rc.NeuralNetworkModelSingleLayer(2,20,404),ctx.X_train,ctx.y_train,
                        {'neural_adam_epochs':0,'neural_maxiter':1,'neural_probe_count':0})
    assert not info['passed'];assert info['certificate']['global_optimum_guarantee'] is False
    results['neural_iteration_limit_rejected_and_no_global_claim']=True
    # L2 includes the bias so the gradient bound applies to the complete parameter vector.
    th=np.array([.3,-.2,.7]);D=np.c_[ctx.X_train,np.ones(len(ctx.y_train))]
    val,grad=training.linear_objective(th,D,ctx.y_train,.001);numeric=[]
    for i in range(3):
        e=np.eye(3)[i]*1e-5
        numeric.append((training.linear_objective(th+e,D,ctx.y_train,.001)[0]-training.linear_objective(th-e,D,ctx.y_train,.001)[0])/2e-5)
    np.testing.assert_allclose(grad,numeric,atol=1e-8)
    results['regularized_gradient_matches_finite_difference']=True
    # Cached fits are protocol-scoped; stored float32 weights must retain the verified result.
    with tempfile.TemporaryDirectory() as temp:
        fn=lambda *a,**k:None
        a,ai=training.saved_fit(temp,'model',study.rc.LogisticRegressionModel(2,404),ctx,{},fn)
        b,bi=training.saved_fit(temp,'model',study.rc.LogisticRegressionModel(2,123),ctx,{},fn)
        np.testing.assert_array_equal(training.parameters(a),training.parameters(b));assert ai==bi
        try:training.saved_fit(temp,'model',study.rc.LogisticRegressionModel(2),ctx,{'l2':.002},fn)
        except ValueError:pass
        else:raise AssertionError('Expected protocol mismatch')
    results['resume_weights_identical_and_protocol_mismatch_rejected']=True
    # Every block is preflighted; one failure prevents all recourse trajectories.
    config=json.loads(Path(__file__).with_name('config.json').read_text())
    config.update(datasets=['Synthetic','German'],seeds=[9903],instances=1)
    with tempfile.TemporaryDirectory() as temp:
        runner=study.Study(temp,config);seen=[]
        def badblock(ds,seed):
            seen.append(ds);raise RuntimeError('intentional test failure')
        runner.block=badblock
        runner.trajectory=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Trajectory before preflight'))
        try:runner.execute()
        except RuntimeError as exc:assert 'preflight blocks failed' in str(exc)
        else:raise AssertionError('Expected preflight failure')
        assert seen==['Synthetic','German']
    results['all_preflight_blocks_checked_and_trajectories_gated']=True
    Path(__file__).with_name('training_test_results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
