import os,sys,time,json,traceback,subprocess,faulthandler
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'frozen_linear'))
import numpy as np
import torch
import pilot_engine as engine
import exact_linear
from common import read,save,sha,utc,clean
from eta import remaining


def timing(root,kind,seconds,**identity):
    with open(root/'timing.jsonl','a') as f:f.write(json.dumps(dict(kind=kind,seconds=float(seconds),utc=utc(),**identity))+'\n')
def stage(root,name,**kw):save(root/'stage.json',dict(stage=name,utc=utc(),**kw))

def setting(block,ds,seed,person,n,bi,B,models,ctx,A,b,x0,free,cfg,env):
    case=block/'cases'/f'person_{person:02d}_n_{n}'
    dest=case/f'budget_{bi}';dest.mkdir(parents=True,exist_ok=False)
    subset=engine.rc.ModelSet(models[:n],context=ctx)
    constraints=engine.rc.constraints_for(subset,torch.tensor(x0,dtype=torch.float32))
    immutable=np.setdiff1d(np.arange(len(x0)),free)
    assert set(constraints.immutable)==set(immutable)
    assert np.isneginf(constraints.lower[free]).all() and np.isposinf(constraints.upper[free]).all()
    start=time.perf_counter()
    if B:
        op,oi,trace=engine.orpm(subset,x0,B,constraints,cfg,ds)
    else:op=x0.copy();oi=dict(rounds=0,status='factual_control');trace={}
    orpm_seconds=time.perf_counter()-start
    xp,xi=exact_linear.max_margin(A[:n],b[:n],x0,B,free,cfg['solver'],env)
    save(dest/'exact_margin.json',dict(point=xp,info=xi))
    save(dest/'orpm_info.json',dict(info=oi,seconds=orpm_seconds))
    np.savez_compressed(dest/'points.npz',origin=x0,ORPM=op,ExactMaxMargin=xp,**trace)
    tol=cfg['solver']['evaluation_logit_tolerance']
    return dict(dataset=ds,seed=seed,person=person,model_count=n,budget_index=bi,budget=B,
                method='ORPM',available=True,cost=float(np.linalg.norm(op-x0)),
                seen_joint_validity=float(np.all(A[:n]@op+b[:n]>=-tol)),seconds=orpm_seconds,
                point_file=str((dest/'points.npz').relative_to(block.parents[2])))


def run(root):
    root=Path(root);cfg=read(root/'protocol.json');acfg=read(root/'analysis_protocol.json')
    torch.set_num_threads(1);np.random.seed(0);torch.manual_seed(0)
    assert read(root/'preparation_verification.json')['passed']
    for name,h in read(root/'freeze.json')['sha256'].items():assert sha(root/name)==h,'Frozen source drift: '+name
    completed=0;cached=0
    plan=read(root/'cohort_plan.json')
    with exact_linear.environment() as env,open(root/'records.jsonl','a') as records,open(root/'new_records.jsonl','x') as newrecords:
        for entry in plan:
            ds,seed=entry['dataset'],entry['seed']
            if not entry['new_people']:continue
            block=root/'blocks'/ds/f'seed_{seed}'
            stage(root,'load_saved_bank',dataset=ds,seed=seed)
            models,ctx,A,b,meta,prep=engine.load_bank(block/'bank')
            with np.load(block/'panel.npz') as p:
                assert np.array_equal(A,p['A']) and np.array_equal(b,p['b'])
                X,free,ids=p['selected_X'],p['free'],p['test_ids']
            for person in range(20,len(X)):
                x0=X[person]
                for n in acfg['model_counts']:
                    case=block/'cases'/f'person_{person:02d}_n_{n}'
                    assert not case.exists();case.mkdir()
                    stage(root,'minimum_l2',dataset=ds,seed=seed,person=person,model_count=n)
                    start=time.perf_counter();point,info=exact_linear.min_l2(A[:n],b[:n],x0,free,cfg['solver'],env)
                    save(case/'minimum_cost.json',dict(point=point,info=info));cached+=1
                    timing(root,'min_l2',time.perf_counter()-start,dataset=ds,seed=seed,person=person,model_count=n)
                    for bi,B in enumerate(cfg['budgets'][ds]):
                        stage(root,'orpm_and_max_margin',dataset=ds,seed=seed,person=person,model_count=n,budget=B,budget_index=bi)
                        start=time.perf_counter()
                        r=setting(block,ds,seed,person,n,bi,B,models,ctx,A,b,x0,free,cfg,env)
                        r['test_id']=int(ids[person])
                        line=json.dumps(clean(r),allow_nan=False)+'\n';records.write(line);newrecords.write(line);records.flush();newrecords.flush()
                        completed+=1
                        timing(root,'setting_wall',time.perf_counter()-start,dataset=ds,seed=seed,person=person,model_count=n,budget_index=bi)
                        save(root/'progress.json',dict(status='running',preserved_settings=26400,new_completed_settings=completed,expected_new_settings=14190,completed_settings=26400+completed,expected_settings=40590,new_minimum_l2_completed=cached,expected_new_minimum_l2=1899,training_required=False,current=dict(dataset=ds,seed=seed,person=person,model_count=n,budget=B),updated_utc=utc()))
                print(utc(),'PERSON_COMPLETED',ds,seed,person,'new_settings',completed,'of',14190,flush=True)
            save(root/'runtime_estimate.json',remaining(root))
    assert completed==14190 and cached==1899
    stage(root,'independent_verification')
    print(utc(),'RECOURSE_COMPLETED','new_settings',completed,flush=True)
    from verify_extra import verify
    verification=verify(root);save(root/'verification.json',verification)
    stage(root,'cost_analysis_and_reports')
    subprocess.run([sys.executable,'-u',str(HERE/'analyze_cost.py'),'--output',str(root/'cost_report')],check=True)
    stage(root,'disparity_analysis_and_reports')
    droot=Path(acfg['disparity_result']);droot.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,'-u',str(HERE/'analyze_disparity.py'),'--cost-result',str(root/'cost_report'),'--output',str(droot)],check=True)
    from verify_extra import verify_reports
    verification['report_checks']=verify_reports(root,droot);verification['completed_utc']=utc();save(root/'verification.json',verification)
    for folder in [root/'cost_report',droot]:
        # Reports include their own numeric audit. This links the full independent
        # trajectory/source audit without changing the source outputs.
        save(folder/'full_run_verification.json',verification)
        import zipfile
        with zipfile.ZipFile(folder/'figures_tables.zip','a',zipfile.ZIP_DEFLATED) as z:z.write(folder/'full_run_verification.json','full_run_verification.json')
    save(root/'analysis.json',dict(passed=True,cost_report=str(root/'cost_report'),disparity_report=str(droot),cost_panels=30,disparity_panels=44,combined_people=1833,new_people=633,settings=40590,verification_passed=True))
    save(root/'progress.json',dict(status='completed',completed_settings=40590,expected_settings=40590,new_completed_settings=14190,preserved_settings=26400,verification_passed=True,updated_utc=utc()))
    for folder in [root.parents[2],droot.parents[2]]:
        save(folder/'latest_expansion.json',dict(result=str(root) if folder==root.parents[2] else str(droot),cost_report=str(root/'cost_report'),disparity_report=str(droot),status='completed',verification_passed=True))
    stage(root,'completed');print(utc(),'LINEAR_MULTIPLICITY_EXPANSION_COMPLETED',flush=True)

if __name__=='__main__':
    root=Path(sys.argv[1]);native=open(root/'native_traceback.log','a');faulthandler.enable(file=native,all_threads=True)
    try:run(root)
    except BaseException as e:
        failure=dict(utc=utc(),error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),stage=read(root/'stage.json') if (root/'stage.json').exists() else None)
        save(root/'failure.json',failure);print(failure['traceback'],flush=True);sys.exit(1)
