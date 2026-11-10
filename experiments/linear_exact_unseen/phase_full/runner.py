import os,sys,time,traceback,shutil,json,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'frozen_utils'))
import numpy as np
import torch
from utils import recourse_core as rc,recourse_baselines as bas
from utils.recourse_convergence import ConvexLogitOracle
from baseline_feasibility import repair_baseline_point
from common import *
from pilot_engine import load_bank,orpm
from expansion import prepare
import exact_linear

def run(root,cfg):
    torch.set_num_threads(1);np.random.seed(0);torch.manual_seed(0)
    save(root/'preflight.json',exact_linear.self_check(cfg['solver']))
    reused,skip=prepare(root,cfg);count=len(reused);new_count=0;start=time.perf_counter();tol=cfg['solver']['evaluation_logit_tolerance']
    with exact_linear.environment() as env,open(root/'records.jsonl','x',buffering=1) as records:
        for row in reused:records.write(json.dumps(clean(row),allow_nan=False)+'\n')
        records.flush()
        print(utc(),'PILOT_PRESERVED',len(reused),'records',flush=True)
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
                        setting=directory/'cases'/f'person_{person:02d}_n_{n}';setting.mkdir(parents=True,exist_ok=True)
                        if (setting/'minimum_cost.json').exists():
                            cached=read(setting/'minimum_cost.json');minpoint=np.asarray(cached['point']);mininfo=cached['info']
                        else:
                            minpoint,mininfo=exact_linear.min_l2(A[:n],b[:n],x0,free,cfg['solver'],env)
                            save(setting/'minimum_cost.json',dict(point=minpoint,info=mininfo))
                        for budget_index,B in enumerate(cfg['budgets'][ds]):
                            if (ds,seed,person,n,B) in skip:continue
                            case_start=time.perf_counter()
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
                            for row in rows:records.write(json.dumps(clean(row),allow_nan=False)+'\n');count+=1;new_count+=1
                            with open(root/'timing.jsonl','a') as timing:timing.write(json.dumps(dict(dataset=ds,seed=seed,person=person,model_count=n,budget=B,seconds=time.perf_counter()-case_start))+'\n')
                            save(root/'progress.json',dict(status='running',completed_records=count,preserved_records=len(reused),new_records=new_count,expected_records=cfg['expected_records'],dataset=ds,seed=seed,person=person,model_count=n,budget=B,elapsed_solver_phase_seconds=time.perf_counter()-start,updated_utc=utc()))
                        print(utc(),'PROGRESS',count,'/',cfg['expected_records'],ds,seed,person,n,flush=True)
    assert count==cfg['expected_records']
    save(root/'progress.json',dict(status='verifying',completed_records=count,expected_records=count,updated_utc=utc()))
    import verify
    verification=verify.verify(root);save(root/'verification.json',verification)
    if not cfg.get('validation',False):
        save(root/'progress.json',dict(status='analyzing',completed_records=count,expected_records=count,verification_passed=verification['passed'],updated_utc=utc()))
        import analyze
        analyze.analyze(root)
    status='complete' if verification['passed'] else 'complete_with_readiness_failures'
    save(root/'progress.json',dict(status=status,completed_records=count,expected_records=count,verification_passed=verification['passed'],updated_utc=utc()))
    print(utc(),'FULL_STUDY_COMPLETED' if not cfg.get('validation',False) else 'VALIDATION_COMPLETED',status,'report and archive saved',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--lock-fd',type=int);args=ap.parse_args();root=Path(args.root)
    try:run(root,read(root/'protocol.json'))
    except BaseException as e:
        save(root/'failure.json',dict(time_utc=utc(),error=repr(e),traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True);raise
    finally:
        if args.lock_fd is not None:os.close(args.lock_fd)
