import os,sys,time,traceback,shutil,json,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'frozen_utils'))
import numpy as np
import torch
from scipy.special import expit,logsumexp
from utils import recourse_core as rc,recourse_results as rr,recourse_baselines as bas
from utils.recourse_convergence import ConvexLogitOracle
from baseline_feasibility import repair_baseline_point
from common import *
import exact_linear

def load_bank(bank):
    meta=read(bank/'models.json');prep=read(bank/'preprocessing.json')
    data=dict(np.load(bank/'split_and_predictions.npz',allow_pickle=False))
    models=[]
    for i,desc in enumerate(meta['models'][:24]):
        assert desc['class']=='LogisticRegressionModel'
        model=rc.LogisticRegressionModel(desc['linear_dimensions'][0][0],seed=0)
        if desc['feature_mask'] is not None:model=rc.FeatureMaskedModel(model,desc['feature_mask'])
        with np.load(bank/f'model_{i:04d}.npz',allow_pickle=False) as f:
            model.load_state_dict({k:torch.tensor(f[k]) for k in f.files})
        model.eval();models.append(model)
    ctx=rc.SplitContext(name=prep['dataset'],columns=prep['feature_names'],scaler=None,seed=prep['seed'],actionable_indices=prep['actionable_indices'],actionability_policy=prep['actionability_policy'],**{k:data[k] for k in ['X_train','y_train','X_validation','y_validation','X_test','y_test','train_ids','validation_ids','test_ids']})
    for m in models:m.recourse_context=ctx
    models=rc.ModelSet(models,context=ctx)
    pars=[rc.linear_parameters(m) for m in models]
    A=np.stack([a.numpy().astype(float) for a,b in pars]);b=np.array([float(b) for a,b in pars])
    return models,ctx,A,b,meta,prep

def panel(ctx,A,b,meta,cfg):
    theta=np.r_[A[0],b[0]];d=len(theta)
    X=np.column_stack([ctx.X_train,np.ones(len(ctx.X_train))]).astype(float);y=ctx.y_train.astype(float)
    V=np.column_stack([ctx.X_validation,np.ones(len(ctx.X_validation))]).astype(float)
    settings=meta['models'][0]['training_settings'];l2=settings['l2'];assert l2==.001
    z=X@theta;p=expit(z);center=float(np.mean(np.logaddexp(0,z)-y*z)+l2*.5*(theta@theta))
    grad=X.T@(p-y)/len(X)+l2*theta;gap=float(grad@grad/(2*l2));assert gap<=1e-8
    acc=float(np.mean((V@theta>=0)==ctx.y_validation))
    H=(X.T*(p*(1-p)))@X/len(X)+l2*np.eye(d)
    val,vec=np.linalg.eigh(H);transform=vec@np.diag(1/np.sqrt(val))@vec.T
    hashes={theta_hash(np.r_[aa,bb]) for aa,bb in zip(A,b)};accepted=[];audit=[]
    for attempt in range(cfg['unseen_max_attempts']):
        rngseed=cfg['unseen_seed_offset']+ctx.seed*10000+attempt;rng=np.random.default_rng(rngseed)
        direction=rng.normal(size=d);direction/=np.linalg.norm(direction)
        candidate=(theta+transform@(np.sqrt(2*cfg['unseen_epsilon'])*rng.random()**(1/d)*direction)).astype(np.float32).astype(float)
        z=X@candidate;loss=float(np.mean(np.logaddexp(0,z)-y*z)+l2*.5*(candidate@candidate))
        va=float(np.mean((V@candidate>=0)==ctx.y_validation));h=theta_hash(candidate)
        excess=loss-center+gap
        passed=bool(excess<=cfg['unseen_epsilon']+1e-8 and va>=acc-cfg['validation_accuracy_allowance'] and h not in hashes)
        audit.append(dict(attempt=attempt,rng_seed=rngseed,training_loss=loss,excess_upper=excess,validation_accuracy=va,theta_sha256=h,accepted=passed))
        if passed:accepted.append(candidate);hashes.add(h)
        if len(accepted)==cfg['unseen_count']:break
    assert len(accepted)==cfg['unseen_count'], 'Unseen admission exhausted; no relaxation allowed'
    return np.array(accepted),dict(center_loss=center,center_gap_bound=gap,center_validation_accuracy=acc,l2=l2,accepted_count=len(accepted),candidate_attempts=len(audit),attempts=audit)

def prepare(root,cfg):
    inventory=read(Path(cfg['source_run'])/'model_inventory.json');out={};manifest={}
    for ds in cfg['datasets']:
        for seed in cfg['seeds']:
            key=f'{ds}/{seed}';bank=Path(inventory[key]['source_bank']);directory=root/'blocks'/ds/f'seed_{seed}';snapshot=directory/'bank';snapshot.mkdir(parents=True)
            names=['models.json','preprocessing.json','split_and_predictions.npz']+[f'model_{i:04d}.npz' for i in range(24)]
            for name in names:
                manifest[str(bank/name)]=sha(bank/name);shutil.copy2(bank/name,snapshot/name)
                assert sha(snapshot/name)==manifest[str(bank/name)]
            models,ctx,A,b,meta,prep=load_bank(snapshot)
            assert ctx.actionability_policy=='immutable'
            X=np.asarray(ctx.X_test,dtype=np.float32).astype(float)
            logits=X@A.T+b;eligible=np.flatnonzero(np.logical_and.reduce([(logits[:,:n]<0).sum(1)>=n/2 for n in cfg['model_sizes']]))
            order=np.random.default_rng(seed+cfg['cohort_order_seed_offset']).permutation(eligible)
            assert len(order)>=cfg['instances'],(key,len(order))
            selected=order[:cfg['instances']]
            cohort=dict(selection='optimization-only common majority rejection',eligible_count=len(eligible),eligible_indices=eligible.tolist(),random_order=order.tolist(),selected_indices=selected.tolist(),selected_test_ids=ctx.test_ids[selected].tolist(),future_20_prefix=order[:20].tolist(),future_20_available=bool(len(order)>=20))
            save(directory/'cohort.json',cohort)
            unseen,audit=panel(ctx,A,b,meta,cfg);save(directory/'unseen_admission.json',audit)
            np.savez_compressed(directory/'panel.npz',A=A,b=b,unseen_A=unseen[:,:-1],unseen_b=unseen[:,-1],selected_X=X[selected],selected_indices=selected,test_ids=ctx.test_ids[selected],free=np.asarray(ctx.actionable_indices,dtype=int))
            out[key]=dict(source_bank=str(bank),snapshot=str(snapshot.relative_to(root)),panel_sha256=sha(directory/'panel.npz'),cohort_sha256=sha(directory/'cohort.json'),unseen_admission_sha256=sha(directory/'unseen_admission.json'),models=24,unseen=50)
            print(utc(),'PANEL_FROZEN',key,'eligible',len(eligible),'attempts',audit['candidate_attempts'],flush=True)
            save(root/'progress.json',dict(status='preparing_panels',panels_completed=len(out),total_panels=9,completed_records=0,expected_records=cfg['expected_records'],updated_utc=utc()))
    save(root/'input_manifest.json',manifest);save(root/'model_inventory.json',out)
    save(root/'panels_frozen.json',dict(frozen_utc=utc(),count=len(out),inventory_sha256=sha(root/'model_inventory.json'),no_recourse_yet=True))
    return out

def orpm(models,x0,B,c,cfg,ds):
    oracle=ConvexLogitOracle(models,x0,B,c,maxiter=cfg['oracle_maxiter']);oracle.normalizer=1.
    T=cfg['rounds'].get(ds,cfg['rounds']['default']);logw=np.full(len(models),-np.log(len(models)));points=[];weights=[];losses=[];gaps=[]
    previous=x0.copy()
    for _ in range(T):
        w=np.exp(logw);x,info=oracle.solve(w,previous);v=oracle.losses(x)
        points.append(x);weights.append(w);losses.append(v);gaps.append(info['absolute_gap'])
        logw+=cfg['raw_outer_rate']*v;logw-=logsumexp(logw);previous=x
    return np.mean(points,axis=0),dict(rounds=T,return_rule='arithmetic_average',max_oracle_gap=float(max(gaps)),mean_oracle_gap=float(np.mean(gaps))),dict(orpm_points=np.array(points),orpm_weights=np.array(weights),orpm_losses=np.array(losses),orpm_oracle_gaps=np.array(gaps))

def run(root,cfg):
    torch.set_num_threads(1);np.random.seed(0);torch.manual_seed(0)
    save(root/'preflight.json',exact_linear.self_check(cfg['solver']))
    inventory=prepare(root,cfg);count=0;start=time.perf_counter();tol=cfg['solver']['evaluation_logit_tolerance']
    with exact_linear.environment() as env,open(root/'records.jsonl','x',buffering=1) as records:
        for ds in cfg['datasets']:
            for seed in cfg['seeds']:
                directory=root/'blocks'/ds/f'seed_{seed}';models,ctx,A,b,meta,prep=load_bank(directory/'bank');saved=dict(np.load(directory/'panel.npz',allow_pickle=False));UA,Ub=saved['unseen_A'],saved['unseen_b'];free=saved['free'].tolist()
                ellice=bas.prepare_ellice(models[0],ctx.X_train,epsilon=cfg['ellice_epsilon'])
                for person,x0 in enumerate(saved['selected_X']):
                    x0t=torch.tensor(x0,dtype=torch.float32)
                    for n in cfg['model_sizes']:
                        subset=rc.ModelSet(models[:n],context=ctx);c=rc.constraints_for(subset,x0t)
                        assert set(free)==set(range(len(x0)))-set(c.immutable)
                        assert np.isneginf(c.lower[free]).all() and np.isposinf(c.upper[free]).all()
                        minpoint,mininfo=exact_linear.min_l2(A[:n],b[:n],x0,free,cfg['solver'],env)
                        setting=directory/'cases'/f'person_{person:02d}_n_{n}';setting.mkdir(parents=True)
                        save(setting/'minimum_cost.json',dict(point=minpoint,info=mininfo))
                        for budget_index,B in enumerate(cfg['budgets'][ds]):
                            paths=setting/f'budget_{budget_index}';paths.mkdir()
                            exactpoint,exactinfo=exact_linear.max_margin(A[:n],b[:n],x0,B,free,cfg['solver'],env)
                            save(paths/'exact_margin.json',dict(point=exactpoint,info=exactinfo))
                            arrays=dict(origin=x0,exact_max_point=exactpoint,exact_min_point=minpoint);rows=[]
                            for method in cfg['methods']:
                                st=time.perf_counter();available=True;info={};trace={};raw=None
                                if B==0:point=x0.copy();info=dict(status='singleton_factual_control')
                                elif method=='ExactMaxMargin':point=exactpoint.copy();info=exactinfo
                                elif method=='ExactMinL2':
                                    available=bool(mininfo['available'] and mininfo['cost_upper']<=B+1e-8)
                                    point=minpoint.copy() if available else x0.copy();info=dict(mininfo,budget_output_available=available)
                                elif method=='ORPM':point,info,trace=orpm(subset,x0,B,c,cfg,ds)
                                elif method=='Reference':
                                    if A[0]@x0+b[0]>=0:point=x0.copy();info=dict(status='reference_already_accepts')
                                    else:
                                        local=ConvexLogitOracle(rc.ModelSet([models[0]],context=ctx),x0,B,c,maxiter=cfg['reference_maxiter']);local.normalizer=1.
                                        point,info=local.solve(np.ones(1))
                                else:
                                    loss=lambda m,x:torch.nn.functional.softplus(-m.logits(x)).sum()
                                    if method=='ADV':ret=bas.ADV(x0t,subset,B,loss,T=cfg['baseline_maxiter'],lr=cfg['baseline_lr']*B,constraints=c)[-1]
                                    elif method=='ROAR':
                                        theta=np.column_stack([A[:n],b[:n]]);delta=float(abs(theta-theta[0]).max())
                                        ret=bas.ROAR(x0t,models[0],delta,B,loss,T=cfg['baseline_maxiter'],constraints=c)[-1]
                                    elif method=='ElliCE':ret=bas.ElliCE(x0t,subset,B,loss,T=cfg['baseline_maxiter'],constraints=c,preparation=ellice)[-1]
                                    else:raise ValueError(method)
                                    raw=ret.detach().numpy().astype(float);point,repair=repair_baseline_point(raw,x0,B,c);info=dict(getattr(ret,'recourse_info',{}),numerical_repair=repair)
                                elapsed=time.perf_counter()-st
                                if method=='ExactMaxMargin' and B>0:elapsed=exactinfo['seconds']
                                if method=='ExactMinL2' and B>0:elapsed=mininfo['seconds']
                                assert np.isfinite(point).all() and np.linalg.norm(point-x0)<=B+1e-8 and np.array_equal(point[c.immutable],x0[c.immutable])
                                seen=metrics(A[:n],b[:n],point,available,tol);unseen=metrics(UA,Ub,point,available,tol)
                                row=dict(dataset=ds,seed=seed,person=person,test_id=int(saved['test_ids'][person]),model_count=n,budget=float(B),budget_index=budget_index,method=method,available=available,cost=float(np.linalg.norm(point-x0)),seconds=elapsed,point_file=str((paths/'points.npz').relative_to(root)),info=info)
                                row.update({'seen_'+k:v for k,v in seen.items()});row.update({'unseen_'+k:v for k,v in unseen.items()})
                                row['exact_loss_gap']=float(seen['worst_loss']-np.logaddexp(0.,-exactinfo['margin_lower'])) if available else None
                                row['minimum_cost_lower']=mininfo.get('cost_lower');row['minimum_cost_upper']=mininfo.get('cost_upper')
                                row['cost_excess_over_minimum']=row['cost']-mininfo['cost_upper'] if seen['joint_validity'] and mininfo['available'] else None
                                arrays[method]=point
                                if raw is not None:arrays[method+'_raw']=raw
                                arrays.update(trace);rows.append(row)
                            np.savez_compressed(paths/'points.npz',**arrays)
                            for row in rows:records.write(json.dumps(clean(row),allow_nan=False)+'\n');count+=1
                            save(root/'progress.json',dict(status='running',completed_records=count,expected_records=cfg['expected_records'],dataset=ds,seed=seed,person=person,model_count=n,budget=B,elapsed_solver_phase_seconds=time.perf_counter()-start,updated_utc=utc()))
                        print(utc(),'PROGRESS',count,'/',cfg['expected_records'],ds,seed,person,n,flush=True)
    assert count==cfg['expected_records']
    save(root/'progress.json',dict(status='verifying',completed_records=count,expected_records=count,updated_utc=utc()))
    import verify
    verification=verify.verify(root);save(root/'verification.json',verification)
    import analyze
    analyze.analyze(root)
    status='complete' if verification['passed'] else 'complete_with_readiness_failures'
    save(root/'progress.json',dict(status=status,completed_records=count,expected_records=count,verification_passed=verification['passed'],updated_utc=utc()))
    print(utc(),'PILOT_COMPLETED',status,'report and archive saved',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--lock-fd',type=int);args=ap.parse_args();root=Path(args.root)
    try:run(root,read(root/'protocol.json'))
    except BaseException as e:
        save(root/'failure.json',dict(time_utc=utc(),error=repr(e),traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True);raise
    finally:
        if args.lock_fd is not None:os.close(args.lock_fd)
