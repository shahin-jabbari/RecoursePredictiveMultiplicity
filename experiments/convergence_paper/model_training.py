"""Checked classifier training for the paper study (separate from recourse oracles).

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
from utils import recourse_core as rc, recourse_diversity as dv, recourse_results as rr

DEFAULTS=dict(l2=0.001,linear_maxiter=2000,linear_gap_tolerance=1e-8,
              neural_adam_epochs=100,neural_adam_lr=.01,neural_maxiter=2000,
              neural_gradient_tolerance=.001,neural_probe_maxiter=300,neural_probe_count=4,
              neural_stability_tolerance=1e-6,batch_size=4096,
              candidate_pool_size=32,reference_max_attempt_factor=20)

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

def fit(model,X,y,options):
    o=dict(DEFAULTS,**options);start=time.perf_counter();model.eval();history=[]
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
    model.eval();model.zero_grad(set_to_none=True)
    info=dict(role='trained',passed=passed,objective='mean BCE + l2/2 * squared norm of all weights and biases',
        l2=o['l2'],regularized_training_loss=float(value),gradient_l2=float(np.linalg.norm(grad)),
        solver_success=bool(result.success),solver_message=str(result.message),solver_iterations=int(result.nit),
        seconds=time.perf_counter()-start,certificate=certificate,settings=o)
    model.training_settings=info;model.training_loss_history=history+[float(value)]
    return model,info

def accuracy(model,X,y):
    p=dv.probabilities([model],X)[:,0];pred=p>=.5
    return dict(accuracy=float((pred==y).mean()),rejection_count=int((~pred).sum()),
        negative_recall=float((~pred[np.asarray(y)==0]).mean()),positive_recall=float(pred[np.asarray(y)==1].mean()))

def saved_fit(directory,name,model,ctx,options,progress):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/name;meta=path.with_suffix('.json');weights=path.with_suffix('.npz')
    if meta.exists() and weights.exists():
        record=json.loads(meta.read_text())
        if record['settings']!=dict(DEFAULTS,**options):raise ValueError('Cached training protocol differs')
        with np.load(weights,allow_pickle=False) as z:model.load_state_dict({k:torch.tensor(z[k]) for k in z.files})
        model.training_settings=record;model.training_loss_history=record.get('loss_history',[]);model.eval()
    else:
        progress('model_training',model=name)
        model,record=fit(model,ctx.X_train,ctx.y_train,options)
        record['validation']=accuracy(model,ctx.X_validation,ctx.y_validation)
        record['loss_history']=list(model.training_loss_history)
        rr._npz(weights,{k:v.detach().numpy() for k,v in model.state_dict().items()})
        rr.atomic_json(meta,record)
    return model,record

def build_pool(ctx,config,directory,progress):
    o=dict(DEFAULTS,**config['training_protocol']);d=len(ctx.columns);directory=Path(directory)
    central,central_info=saved_fit(directory,'central',rc.LogisticRegressionModel(d,ctx.seed),ctx,o,progress)
    if not central_info['passed']:raise RuntimeError('Central logistic fit failed its global training-objective bound')
    baseline=central_info['validation']['accuracy'];floor=baseline-config['validation_tolerance']
    center=np.r_[rc.linear_parameters(central)[0].numpy(),float(rc.linear_parameters(central)[1])]
    design=np.c_[ctx.X_train,np.ones(len(ctx.y_train))].astype(float)
    prob=expit(design@center);hessian=(design.T*(prob*(1-prob)))@design/len(prob)+o['l2']*np.eye(d+1)
    reference=[];candidate_banks=[];audit=[];selections=[]
    factories=[lambda s:rc.LogisticRegressionModel(d,s),lambda s:rc.NeuralNetworkModelSingleLayer(d,20,s),
               lambda s:rc.NeuralNetworkModelDoubleLayers(d,50,100,s)]
    capacities=[24,12,12];groups=dv.feature_groups(ctx.columns)
    for kind,count in enumerate(capacities):
        refs=[];ref_info=[]
        for attempt in range(count*o['reference_max_attempt_factor']):
            if len(refs)==count:break
            seed=ctx.seed+kind*100+attempt
            if kind==0:
                model=copy.deepcopy(central)
                if attempt:
                    th=rc.sample_from_ellipsoid(torch.tensor(center),torch.tensor(hessian),config['validation_tolerance'],1,seed)[0]
                    rc.set_model_weights(model,th.float())
                th=np.r_[rc.linear_parameters(model)[0].numpy(),float(rc.linear_parameters(model)[1])]
                objective,grad=linear_objective(th,design,ctx.y_train,o['l2'])
                excess=objective-central_info['regularized_training_loss']
                global_excess_upper=excess+central_info['certificate']['absolute_error_bound']
                info=copy.deepcopy(central_info) if attempt==0 else dict(role='sampled_logistic',passed=bool(global_excess_upper<=config['validation_tolerance']+1e-8),
                    regularized_training_loss=objective,gradient_l2=float(np.linalg.norm(grad)),
                    training_excess_upper=global_excess_upper,
                    note='Intentionally sampled near-optimal hypothesis, not an independently optimized model')
                info['validation']=accuracy(model,ctx.X_validation,ctx.y_validation)
                if attempt:model.training_loss_history=[]
            else:model,info=saved_fit(directory,f'ref_{kind}_{attempt:04d}',factories[kind](seed),ctx,o,progress)
            accept=info['passed'] and info['validation']['accuracy']>=floor-1e-12
            audit.append(dict(kind=kind,group='reference',attempt=attempt,accepted=accept,**info))
            if accept:
                model.training_settings=info;refs.append(model);ref_info.append(info)
        if len(refs)!=count:
            rr.atomic_json(directory/'admission_audit.json',audit)
            raise RuntimeError(f'{ctx.name}/{ctx.seed}: only {len(refs)}/{count} admissible type {kind} references after training checks')
        candidates=[];candidate_infos=[]
        for attempt in range(o['candidate_pool_size']):
            seed=ctx.seed+10000*(kind+1)+attempt
            mask=dv.feature_mask(groups,ctx.actionable_indices,np.random.default_rng(seed),[.5,.75][attempt%2])
            model=rc.FeatureMaskedModel(factories[kind](seed),mask)
            model,info=saved_fit(directory,f'candidate_{kind}_{attempt:04d}',model,ctx,o,progress)
            audit.append(dict(kind=kind,group='candidate',attempt=attempt,accepted=bool(info['passed'] and info['validation']['accuracy']>=floor-1e-12),**info))
            if info['passed']:
                candidates.append(model);candidate_infos.append(info)
        rp=dv.probabilities(refs,ctx.X_validation);ra=np.mean((rp>=.5)==ctx.y_validation[:,None],axis=0)
        cp=dv.probabilities(candidates,ctx.X_validation) if candidates else np.empty((len(ctx.y_validation),0))
        masks=dv.masks_for(candidates,d) if candidates else np.empty((0,d))
        selected,ca=dv.select_matched(rp,cp,masks,ctx.y_validation,ra,floor,config['diversity_accuracy_caliper'],pinned_first=kind==0)
        for i,j in enumerate(selected):
            model=refs[i] if j is None else candidates[j];info=ref_info[i] if j is None else candidate_infos[j]
            assert info['passed'] and info['validation']['accuracy']>=floor-1e-12
            model.recourse_context=ctx
            selections.append(dict(kind=kind,slot=i,masked=j is not None,reference_accuracy=float(ra[i]),
                                   validation_accuracy=info['validation']['accuracy'],training=info))
            reference.append(model)
        rr.atomic_json(directory/'admission_audit.json',audit)
    pool=rc.ModelSet(reference,context=ctx);pool.metadata=dict(diversity=True,training_protocol=o,
        baseline_validation_accuracy=baseline,validation_floor=floor,selection_metadata=selections,
        trained_global_guarantee='regularized logistic fits only',neural_guarantee='computed gradient or empirical continued-optimization stability only')
    rr.atomic_json(directory/'pool_quality.json',pool.metadata)
    return pool
