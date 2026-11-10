"""Checked classifier training for all V7 differentiable-model experiments (separate from recourse oracles).

Convex fits certify regularized training objective error using strong convexity.
Neural fits require a measured gradient or continued-optimization stability check;
neither establishes a global optimum.
Sampled logistic hypotheses are intentionally nonoptimal; their loss excess and
validation accuracy are checked instead of falsely calling them trained optima.
"""
from pathlib import Path
import copy, hashlib, json, time
from types import SimpleNamespace
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit
from . import recourse_core as rc, recourse_results as rr

DEFAULTS=dict(l2=0.001,linear_maxiter=2000,linear_gap_tolerance=1e-8,
              neural_adam_epochs=100,neural_adam_lr=.01,neural_maxiter=2000,
              neural_gradient_tolerance=.001,neural_probe_maxiter=300,neural_probe_count=4,
              neural_stability_tolerance=1e-6,batch_size=4096,
              reference_max_attempt_factor=20)

PROTOCOL = 'checked_classifier_training_v2'


class TrainingCheckError(RuntimeError):
    """A fitted candidate exhausted its effort without passing its check."""
    def __init__(self, info):
        self.info = copy.deepcopy(info)
        super().__init__(f"Classifier training check did not pass: {info['certificate']['kind']}; "
                         f"gradient norm={info['gradient_l2']:.6g}. Candidate excluded.")


def validate_options(options=None):
    unknown = set(options or {}) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown classifier training options: {sorted(unknown)}')
    result = dict(DEFAULTS, **(options or {}))
    for key, value in result.items():
        if not np.isfinite(value):
            raise ValueError(f'{key} must be finite')
        if isinstance(DEFAULTS[key], int):
            minimum = 0 if key in {'neural_adam_epochs', 'neural_probe_count'} else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f'{key} must be an integer >= {minimum}')
        elif value <= 0:
            raise ValueError(f'{key} must be positive')
    return result


def effective_options(epochs=100, lr=.01, options=None):
    """Legacy epochs/lr configure warmup only, never acceptance of a model."""
    active = rr._ACTIVE_SETTINGS.get() or {}
    revision = active.get('training_protocol', PROTOCOL)
    if revision != PROTOCOL:
        raise ValueError('Classifier training protocol differs; start a new compatible run.')
    result = dict(neural_adam_epochs=int(active.get('training_epochs', epochs)), neural_adam_lr=lr)
    result.update(active.get('training_options', {}))
    result.update(options or {})
    return validate_options(result)


def sampled_logistic_check(model, center, X, y, tolerance):
    """Bound sampled loss excess relative to the same regularized optimum."""
    central = center.training_settings
    if not central['passed'] or central['certificate']['kind'] != 'global_regularized_training_objective_bound':
        raise ValueError('Sampling requires a checked convex reference model.')
    a, b = rc.linear_parameters(model)
    objective, gradient = linear_objective(np.r_[a.numpy(), float(b)],
        np.c_[X, np.ones(len(y))].astype(float), np.asarray(y), central['l2'])
    upper = objective - central['regularized_training_loss'] + central['certificate']['absolute_error_bound']
    info = dict(protocol=PROTOCOL, role='sampled_logistic',
        passed=bool(np.isfinite(upper) and upper <= tolerance + 1e-8),
        l2=central['l2'], objective=central['objective'], regularized_training_loss=objective,
        gradient_l2=float(np.linalg.norm(gradient)), training_excess_upper=float(upper),
        certificate=dict(kind='global_regularized_training_loss_excess_bound',
            absolute_error_bound=float(upper), criterion_tolerance=float(tolerance)),
        note='Intentionally sampled near-optimal hypothesis; not an independently optimized model')
    model.training_settings = info
    model.training_loss_history = []
    return info

def parameters(model):
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()]).astype(float)

def assign(model,theta):
    offset=0
    with torch.no_grad():
        for p in model.parameters():
            size=p.numel();p.copy_(torch.as_tensor(theta[offset:offset+size].reshape(p.shape),dtype=p.dtype));offset+=size
    assert offset==len(theta)

def neural_objective(model,X,y,l2,batch_size):
    model.zero_grad(set_to_none=True);value=0.;n=len(y)
    for start in range(0,n,batch_size):
        x=torch.as_tensor(X[start:start+batch_size],dtype=next(model.parameters()).dtype)
        target=torch.as_tensor(y[start:start+batch_size],dtype=x.dtype).reshape(-1,1)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(model.logits(x),target,reduction='sum')/n
        loss.backward();value+=float(loss.detach())
    reg=sum((p*p).sum() for p in model.parameters())*(l2/2)
    reg.backward();value+=float(reg.detach())
    gradient=np.concatenate([p.grad.detach().cpu().numpy().ravel() for p in model.parameters()]).astype(float)
    return value,gradient

def linear_objective(theta,X,y,l2):
    z=X@theta;value=np.mean(np.logaddexp(0,z)-y*z)+l2*.5*(theta@theta)
    grad=X.T@(expit(z)-y)/len(y)+l2*theta
    return float(value),grad

def fit(model,X,y,options=None):
    o=validate_options(options);X=np.asarray(X);y=np.asarray(y);
    if X.ndim!=2 or len(y)!=len(X) or not len(y) or not np.isfinite(X).all() or not np.isin(y,[0,1]).all():
        raise ValueError('Training requires finite two-dimensional features and binary labels.')
    start=time.perf_counter();model.eval();history=[]
    if rc.is_linear(model):
        d=X.shape[1];mask=model.feature_mask.numpy() if isinstance(model,rc.FeatureMaskedModel) else np.ones(d)
        design=np.c_[np.asarray(X,dtype=float)*mask,np.ones(len(y))]
        def fun(th):return linear_objective(th,design,np.asarray(y),o['l2'])
        a,b=rc.linear_parameters(model);initial=np.r_[a.numpy(),float(b)]
        # Zero initialization makes the unique regularized optimum independent of the random seed.
        initial=np.zeros_like(initial)
        result=minimize(fun,initial,jac=True,method='L-BFGS-B',
            options={'maxiter':o['linear_maxiter'],'ftol':1e-15,'gtol':1e-9,'maxls':50},
            callback=lambda th:history.append(fun(th)[0]))
        base=model.base_model if isinstance(model,rc.FeatureMaskedModel) else model
        rc.set_model_weights(base,torch.tensor(result.x,dtype=torch.float32))
        a,b=rc.linear_parameters(model);theta=np.r_[a.numpy(),float(b)]
        value,grad=fun(theta);error=float(grad@grad/(2*o['l2']))
        passed=bool(np.isfinite(value) and error<=o['linear_gap_tolerance'])
        certificate=dict(kind='global_regularized_training_objective_bound',absolute_error_bound=error,
                         criterion_tolerance=o['linear_gap_tolerance'],regularizes_intercept=True)
    else:
        model.float().train();optimizer=torch.optim.Adam(model.parameters(),lr=o['neural_adam_lr'])
        for epoch in range(o['neural_adam_epochs']):
            value,grad=neural_objective(model,X,y,o['l2'],o['batch_size']);history.append(value)
            if np.linalg.norm(grad)<=o['neural_gradient_tolerance']:break
            optimizer.step()
        model.double().eval()
        last=[None,None];iterations=[0]
        def fun(th):
            if last[0] is None or not np.array_equal(th,last[0]):
                assign(model,th);last[:]=[th.copy(),neural_objective(model,X,y,o['l2'],o['batch_size'])]
            return last[1]
        class CriterionReached(Exception):
            def __init__(self,theta):self.theta=theta.copy()
        def callback(th):
            iterations[0]+=1;value,grad=fun(th);history.append(value)
            # Stop on the actual declared L2 criterion; leave margin for float32 saving.
            if np.linalg.norm(grad)<=.8*o['neural_gradient_tolerance']:raise CriterionReached(th)
        try:
            result=minimize(fun,parameters(model),jac=True,method='L-BFGS-B',
                options={'maxiter':o['neural_maxiter'],'ftol':1e-15,'gtol':1e-7,'maxls':40},callback=callback)
        except CriterionReached as exc:
            result=SimpleNamespace(x=exc.theta,success=True,message='Measured full-data gradient criterion reached',nit=iterations[0])
        probes=[];stable=0;best=result.x.copy();assign(model,best)
        best_value,best_grad=fun(best)
        for probe in range(o['neural_probe_count']):
            if np.linalg.norm(best_grad)<=.8*o['neural_gradient_tolerance'] or stable>=2:break
            before=best_value
            refined=minimize(fun,best,jac=True,method='L-BFGS-B',
                options={'maxiter':o['neural_probe_maxiter'],'ftol':1e-15,'gtol':1e-7,'maxls':40})
            new_value,new_grad=fun(refined.x)
            if new_value<best_value:
                best=refined.x.copy();best_value,best_grad=new_value,new_grad
            decrease=before-best_value
            valid=bool(refined.success or (refined.status==1 and refined.nit>0))
            small=valid and decrease<=o['neural_stability_tolerance']*max(1.,abs(before))
            stable=stable+1 if small else 0
            probes.append(dict(before=before,after=best_value,decrease=decrease,small_improvement=small,
                optimizer_success=bool(refined.success),iterations=int(refined.nit),message=str(refined.message)))
            history.append(best_value)
        result.x=best
        assign(model,result.x);model.float().eval()
        # Verify the actual float32 weights saved/used by prediction and recourse.
        value,grad=neural_objective(model,X,y,o['l2'],o['batch_size'])
        gradient_pass=bool(np.linalg.norm(grad)<=o['neural_gradient_tolerance'])
        # Float32 conversion must preserve the objective verified by the probes.
        stability_pass=bool(stable>=2 and abs(value-best_value)<=o['neural_stability_tolerance']*max(1.,abs(best_value)))
        passed=bool(np.isfinite(value) and (gradient_pass or stability_pass))
        certificate=dict(kind='computed_gradient_criterion' if gradient_pass else 'empirical_continued_optimization_stability',
            gradient_tolerance=o['neural_gradient_tolerance'],global_optimum_guarantee=False,
            gradient_criterion_passed=gradient_pass,stability_criterion_passed=stability_pass,probes=probes,
            note='Neither criterion proves a global optimum; stability is empirical and does not prove nonsmooth stationarity')
    passed=bool(passed and np.isfinite(grad).all() and np.isfinite(parameters(model)).all())
    model.eval();model.zero_grad(set_to_none=True)
    info=dict(protocol=PROTOCOL,role='trained',passed=passed,objective='mean BCE + l2/2 * squared norm of all weights and biases',
        l2=o['l2'],regularized_training_loss=float(value),gradient_l2=float(np.linalg.norm(grad)),
        solver_success=bool(result.success),solver_message=str(result.message),solver_iterations=int(result.nit),
        seconds=time.perf_counter()-start,certificate=certificate,settings=o)
    model.training_settings=info;model.training_loss_history=history+[float(value)]
    return model,info
