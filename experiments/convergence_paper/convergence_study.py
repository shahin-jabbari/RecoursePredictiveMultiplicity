"""Paired per-person convergence study. All gap quantities are RAW logistic loss.

Global evaluation runs after each trajectory and cannot affect its iterates.
Saved upper/lower bounds describe optimization uncertainty, not sampling CIs.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, copy, fcntl, hashlib, json, os, signal, sys, time, traceback
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get('RECOURSE_V7_PROJECT', '/Users/sj987/Documents/ChatGPT/Code/Predictive_Multiplicity_Recourse/V7'))
sys.path.insert(0, str(HERE if (HERE/'utils').is_dir() else PROJECT))
from utils import recourse_core as rc, recourse_results as rr, recourse_diversity as dv
from utils import recourse_convergence as gap, recourse_gmc_mixed as mixed
from utils.recourse_duality import check_feasible
from utils.recourse_bound_checks import DIAGNOSTIC_VERSION, DIAGNOSTIC_POLICY
rc.PROJECT_ROOT = PROJECT  # Utility snapshots read the original bundled data, never cloud paths.

DATA_FILES = {'Synthetic':'Synthetic.csv', 'COMPAS':'compas.csv', 'Polish':'polish-companies_clean_uncut.csv',
              'German':'germanc.csv', 'Give_Me_Some_Credit':'GiveMeSomeCredit.csv', 'Folktables':'ACSIncome_CA_2018.csv.gz'}
CAPACITIES = [24,12,12]
_STOP_REQUESTED = False

def request_stop(*_):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True

def check_stop():
    if _STOP_REQUESTED:raise KeyboardInterrupt("Termination requested; resume from saved checkpoints")

def load(p): return json.loads(Path(p).read_text())
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
    with np.load(p, allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
def csv(frame,p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.csv.tmp')
    frame.to_csv(tmp,index=False);os.replace(tmp,p)
def indices(family,n):
    if family=='linear': return list(range(n))
    if family=='neural': return list(range(24,24+n//2))+list(range(36,36+n//2))
    if family=='mixed': return list(range(n//3))+list(range(24,24+n//3))+list(range(36,36+n//3))
    raise ValueError(family)
def subset(pool,family,n): return rc.ModelSet([pool[i] for i in indices(family,n)],context=pool.context)
def classify(lower,upper,threshold):
    return 'certified_below' if upper<=threshold else 'definitely_above' if lower>threshold else 'unresolved'

def crossing_summary(frame,threshold,confirmations=2):
    """Crossings occur at measured checkpoints; unresolved is NOT above threshold."""
    frame=frame.sort_values('round');t=frame['round'].to_numpy(int)
    hit=frame.gap_upper.to_numpy(float)<=threshold;above=frame.gap_lower.to_numpy(float)>threshold
    first=int(t[np.flatnonzero(hit)[0]]) if hit.any() else None
    windows=[i for i in range(confirmations-1,len(t)) if hit[i-confirmations+1:i+1].all()]
    confirmed=int(t[windows[0]]) if windows else None
    sustained=next((int(t[i]) for i in range(len(t)) if hit[i:].all()),None)
    return dict(threshold=threshold,first_certified_round=first,first_confirmed_round=confirmed,
        sustained_through_final_round=sustained,confirmation_checkpoints=confirmations,
        final_status=classify(frame.gap_lower.iloc[-1],frame.gap_upper.iloc[-1],threshold),
        final_gap_lower=float(frame.gap_lower.iloc[-1]),final_gap_upper=float(frame.gap_upper.iloc[-1]),
        final_round=int(t[-1]),recrossed_after_first=bool(first is not None and above[np.flatnonzero(hit)[0]+1:].any()),
        lost_certification_after_first=bool(first is not None and (~hit[np.flatnonzero(hit)[0]+1:]).any()),
        crossing_resolution='saved checkpoints only; no interpolation',
        confirmed_by_end=confirmed is not None,certified_at_end=bool(hit[-1]),
        unresolved_at_end=bool(not hit[-1] and not above[-1]))

class Study:
    def __init__(self,root,config):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.c=copy.deepcopy(config)
        self.started=time.perf_counter();self.done=0;self.task_records=[]
        self.cached_completed_paths={str(p.parent.relative_to(self.root)) for p in
            (self.root/'blocks').glob('*/seed_*/person_*/*/models_*/budget_*/completed.json')}
        self.expected=sum(len(self.c['budgets'][ds]) for ds in self.c['datasets'])*len(self.c['seeds'])*self.c['instances']*len(self.c['families'])*len(self.c['model_sizes'])
        self.utility_dir=Path(rc.__file__).parent
        self.hashes={p.name:digest(p) for p in self.utility_dir.glob('*.py')}
        self.data_hashes={ds:digest(PROJECT/'dataset'/DATA_FILES[ds]) for ds in self.c['datasets']}
        self.code_hashes={p.name:digest(p) for p in HERE.glob('*.py')}
        protocol=dict(configuration=self.c,utility_hashes=self.hashes,source_hashes=self.code_hashes,
            nonconvex_bound_policy=DIAGNOSTIC_POLICY,
            dataset_hashes=self.data_hashes,expected_trajectories=self.expected,raw_target_favorable=dict(rc.FAVORABLE_LABEL),
            loss='softplus(-logit); identical loss for all model families',
            model_pool='24 logistic + 12 single-hidden-layer + 12 double-hidden-layer; shared prefixes across families',
            synthetic_size=2000,nested=True,common_people=True,global_evaluation_does_not_modify_trajectory=True,
            raw_gap_definition='max_i loss_i(mean x) - inf_x mean_weights @ loss(x)',
            randomized_gap_definition='max_i mean_t loss_i(x_t) - inf_x mean_weights @ loss(x)',
            normalization='shared loss upper bound per person over all 48 models and the largest budget; reporting only',
            step='exp(raw_outer_rate * raw_loss); same raw rate for all conditions',
            round_oracle='raw SLSQP objective; stop at solver convergence or configured iteration cap',
            guarantees='Averaging is justified for convex logistic sets. Neural deterministic gaps need not vanish.',
            uncertainty='A feasible local solution only bounds the dual minimum from ABOVE; global lower bounds are independent.',
            domain='continuous standardized L2 ball with dataset-specific immutable coordinates, no categorical integrality')
        if (self.root/'protocol.json').exists():
            if load(self.root/'protocol.json')!=protocol:raise RuntimeError('Resume signature differs; create a new study instead.')
        else:rr.atomic_json(self.root/'protocol.json',protocol)
        if not self.c['checkpoints'] or self.c['checkpoints'][-1]!=self.c['rounds']:raise ValueError('Final checkpoint must equal rounds')
        if rc.FAVORABLE_LABEL['German']!=0:raise ValueError('German favorable raw target must be 0')

    def progress(self,stage,**kw):
        check_stop()
        rr.atomic_json(self.root/'progress.json',dict(status='running',stage=stage,completed=self.done,total=self.expected,
            updated_utc=datetime.now(timezone.utc).isoformat(),session_seconds=time.perf_counter()-self.started,**kw))

    def block(self,ds,seed):
        directory=self.root/'blocks'/ds/f'seed_{seed}';directory.mkdir(parents=True,exist_ok=True)
        if digest(PROJECT/'dataset'/DATA_FILES[ds])!=self.data_hashes[ds]:raise RuntimeError('Dataset changed after study initialization')
        X,y,action=rc.load_data_and_normalize(ds);ctx=rc.prepare_split(X,y,seed,ds,action);ctx.epsilon=self.c['validation_tolerance']
        for part in ['train','validation','test']:
            np.testing.assert_array_equal(getattr(ctx,'y_'+part),rc.favorable_labels(y,ds)[getattr(ctx,part+'_ids')])
        if (directory/'ready.json').exists():
            ready=load(directory/'ready.json');bankpath=self.root/ready['bank']
            assert ready['training_protocol']==self.c['training_protocol']
            assert ready['model_hashes']=={p.name:digest(p) for p in bankpath.glob('model_*.npz')}
            pool=rc.ModelSet(rr.load_models(bankpath),context=ctx);pool.metadata=load(bankpath/'models.json')['selection_metadata']
            for model,desc in zip(pool,load(bankpath/'models.json')['models']):
                model.training_settings=desc['training_settings'];assert model.training_settings['passed']
                model.recourse_context=ctx
            z=arrays(bankpath/'split_and_predictions.npz')
            for key in ['X_train','X_validation','X_test','train_ids','validation_ids','test_ids']:
                np.testing.assert_array_equal(z[key],getattr(ctx,key))
            chosen=np.asarray(ready['test_positions'],int)
        else:
            self.progress('training',dataset=ds,seed=seed)
            from model_training import build_pool
            start=time.perf_counter();run=rr.RunStore(directory,'model_pool',self.c);token=rr._ACTIVE_RUN.set(run)
            try:
                run.scope.update(seed=seed,fold=0);run.save_split(ctx)
                pool=build_pool(ctx,self.c,directory/'training_checks',
                    lambda stage,**kw:self.progress(stage,dataset=ds,seed=seed,**kw))
                probabilities=dv.probabilities(pool,ctx.X_test);eligible=np.ones(len(ctx.X_test),bool);rejection={}
                per_set={}
                for family in self.c['families']:
                    for n in self.c['model_sizes']:
                        rejected=(probabilities[:,indices(family,n)]<.5).sum(axis=1)
                        keep=2*rejected>=n;eligible &= keep;rejection[f'{family}_{n}']=rejected
                        per_set[f'{family}_{n}']=int(keep.sum())
                available=np.flatnonzero(eligible)
                rr.atomic_json(directory/'eligibility.json',dict(per_set=per_set,common_eligible=len(available),
                    requested=self.c['instances'],test_count=len(ctx.test_ids),training_verified=True))
                if len(available)<self.c['instances']:
                    raise RuntimeError(f'{ds}/{seed}: common cohort has {len(available)} people, fewer than {self.c["instances"]}; no silent cohort change')
                chosen=np.random.default_rng(seed+85000).permutation(available)[:self.c['instances']]
                ctx.candidate_ids=ctx.test_ids[chosen].tolist()
                ctx.candidate_selection=dict(mode='intersection_majority',ties_included=True,eligible=len(available),requested=self.c['instances'])
                saved=run.save_seed(pool,[torch.tensor(x) for x in ctx.X_test[chosen]],fold=0,seed=seed)
                rr._npz(directory/'cohort.npz',dict(test_positions=chosen,instance_ids=ctx.test_ids[chosen],eligible_positions=available,**rejection))
                ready=dict(bank=str(saved.relative_to(self.root)),test_positions=chosen.tolist(),instance_ids=ctx.candidate_ids,
                    eligible_count=len(available),raw_observations=len(y),features=len(ctx.columns),training_seconds=time.perf_counter()-start,
                    model_hashes={p.name:digest(p) for p in saved.glob('model_*.npz')},training_protocol=self.c['training_protocol'])
                run.finish();rr.atomic_json(directory/'ready.json',ready)
            except BaseException as exc:run.finish(error=str(exc));raise
            finally:rr._ACTIVE_RUN.reset(token)
        checks=[]
        assert len(pool)==sum(CAPACITIES)
        for family in self.c['families']:
            previous=set()
            for n in self.c['model_sizes']:
                current=set(indices(family,n));assert previous<=current;previous=current
                models=subset(pool,family,n);p=dv.probabilities(models,ctx.X_test[chosen])
                assert (2*(p<.5).sum(axis=1)>=n).all()
                checks.append(dict(dataset=ds,seed=seed,family=family,models=n,**dv.describe_set(models)))
        rr.atomic_json(directory/'set_diagnostics.json',checks)
        return directory,ctx,pool,chosen

    def local(self,models,origin,budget,constraints,evaluation=False):
        iterations=self.c['evaluation_maxiter'] if evaluation else self.c['oracle_maxiter']
        if all(rc.is_linear(m) for m in models):
            oracle=gap.ConvexLogitOracle(models,origin,budget,constraints,maxiter=iterations,ftol=1e-10)
            # All optimizer objectives/gradients, bounds and updates use raw loss.
            oracle.normalizer=1.;oracle.budget=budget;oracle.project=oracle._feasible_candidate
            return oracle
        return mixed.LocalLogisticOracle(models,origin,budget,constraints,1.,iterations)

    def trajectory(self,folder,models,origin,budget,constraints,common_B):
        local=self.local(models,origin,budget,constraints);n=len(models)
        path=folder/'trajectory.npz';points=[];weights=[];losses=[];infos=[];seconds=[];logw=np.full(n,-np.log(n));elapsed=0.
        if path.exists():
            z=arrays(path);points=list(z['points']);weights=list(z['weights']);losses=list(z['losses']);seconds=list(z['algorithm_seconds'])
            infos=json.loads(str(z['oracle_infos'].item()));logw=z['log_weights'];elapsed=seconds[-1] if seconds else 0.
            if len(points)==self.c['rounds']:
                check_feasible(z['points'],origin,budget,constraints)
                expected=self.c['raw_outer_rate']*np.vstack([np.zeros(n),np.cumsum(z['losses'],axis=0)[:-1]])
                expected-=logsumexp(expected,axis=1,keepdims=True)
                np.testing.assert_allclose(z['weights'],np.exp(expected),rtol=1e-9,atol=1e-11)
                return z,infos  # Diagnostic replay must not rewrite the saved trajectory.
        def save():
            rr._npz(path,dict(points=np.asarray(points).reshape(-1,len(origin)),weights=np.asarray(weights).reshape(-1,n),
                losses=np.asarray(losses).reshape(-1,n),log_weights=logw,
                oracle_infos=json.dumps(rr.jsonable(infos)),algorithm_seconds=seconds,raw_outer_rate=self.c['raw_outer_rate'],
                common_normalizer=common_B,factual=origin,budget=budget,immutable=constraints.immutable,
                lower=constraints.lower,upper=constraints.upper))
        try:
            for t in range(len(points),self.c['rounds']):
                check_stop()
                start=time.perf_counter();w=np.exp(logw);x,info=local.solve(w,points[-1] if points else origin)
                check_feasible(x,origin,budget,constraints);v=local.losses(x)
                if not np.isfinite(v).all() or (v<0).any() or v.max()>common_B+1e-6:raise RuntimeError('Loss bound violated')
                logw+=self.c['raw_outer_rate']*v;logw-=logsumexp(logw)
                elapsed+=time.perf_counter()-start
                points.append(x.copy());weights.append(w.copy());losses.append(v.copy());infos.append(info);seconds.append(elapsed)
                if (t+1)%self.c['save_every']==0:
                    save();self.progress('ORPM',current=str(folder.relative_to(self.root)),round=t+1)
            save()
        except BaseException:save();raise
        z=arrays(path)
        check_feasible(z['points'],origin,budget,constraints)
        expected=self.c['raw_outer_rate']*np.vstack([np.zeros(n),np.cumsum(z['losses'],axis=0)[:-1]])
        expected-=logsumexp(expected,axis=1,keepdims=True)
        np.testing.assert_allclose(z['weights'],np.exp(expected),rtol=1e-9,atol=1e-11)
        return z,infos

    def diagnostics(self,folder,models,origin,budget,constraints,common_B,z,infos,identity):
        if identity['family']!='linear':
            from nonconvex_diagnostics import diagnostics
            return diagnostics(self,folder,models,origin,budget,constraints,common_B,z,infos,identity)
        return self._convex_diagnostics(folder,models,origin,budget,constraints,common_B,z,infos,identity)

    def _convex_diagnostics(self,folder,models,origin,budget,constraints,common_B,z,infos,identity):
        evaluator=None;local=self.local(models,origin,budget,constraints,True)
        isconvex=identity['family']=='linear';previous=[];rows=[];diag_seconds=0.
        points=z['points'];losses=z['losses'];weights=z['weights']
        component_lower=np.array([float(np.logaddexp(0,-mixed.layer_bounds(m,origin,budget,constraints)[-1][1][0])) for m in models])
        set_B=mixed.common_normalizer([models],origin,budget,constraints)
        diag_dir=folder/'checkpoints';diag_dir.mkdir(exist_ok=True)
        try:
            for t in self.c['checkpoints']:
                saved=diag_dir/f'{t:04d}.json';vectors=diag_dir/f'{t:04d}.npz'
                if saved.exists():
                    record=load(saved);aux=arrays(vectors);rows.extend(record['rows']);diag_seconds=record['diagnostic_elapsed_seconds']
                    previous.append((aux['average_weights'],record['dual_lower'],aux['dual_response']))
                    continue
                self.progress('duality_gap',current=str(folder.relative_to(self.root)),round=t)
                start=time.perf_counter();xb=points[:t].mean(axis=0);wb=weights[:t].mean(axis=0)
                average_loss=losses[:t].mean(axis=0);point_loss=local.losses(xb)
                check_feasible(xb,origin,budget,constraints)
                options=[origin,xb,points[np.argmin(losses[:t]@wb)]]+[p[2] for p in previous]
                initial=min(options,key=lambda p:float(wb@local.losses(p)))
                response,localinfo=local.solve(wb,initial)
                upper=min(float(wb@local.losses(initial)),float(wb@local.losses(response)))
                lower=float(wb@component_lower)
                # Nonnegative component losses allow transporting previous lower bounds.
                for oldw,oldlower,_ in previous:
                    scale=float(np.min(wb[oldw>0]/oldw[oldw>0]))
                    lower=max(lower,scale*oldlower)
                globalinfo={};global_seconds=0.;build_seconds=0.
                if isconvex:
                    lower=max(lower,localinfo['lower_bound'],float(np.mean([i['lower_bound'] for i in infos[:t]])))
                elif upper-lower>1e-4 and max(point_loss.max(),average_loss.max())-lower>min(self.c['raw_gap_thresholds']):
                    if evaluator is None:
                        buildstart=time.perf_counter()
                        evaluator=mixed.NeuralLogisticBound(models,origin,budget,constraints,1.,seed=identity['seed'])
                        build_seconds=time.perf_counter()-buildstart
                    evaluator.time_limit=self.c['global_seconds_final'] if t==self.c['rounds'] else self.c['global_seconds_per_checkpoint']
                    response,globalinfo=evaluator.solve(wb,response,local)
                    if globalinfo['bound_consistent']:lower=max(lower,globalinfo['lower_bound'])
                    # An inconsistent MIP bound is discarded, retaining independent valid bounds.
                    upper=min(upper,globalinfo['upper_bound']);global_seconds=globalinfo['solver_seconds']
                else:globalinfo=dict(skipped='independent bounds already tight or both gaps below smallest threshold')
                check_feasible(response,origin,budget,constraints)
                if lower>upper+1e-6:raise RuntimeError(f'Inconsistent independent dual bounds: {lower} > {upper}')
                lower=min(lower,upper);diag_seconds+=time.perf_counter()-start
                played=float(np.mean(np.sum(weights[:t]*losses[:t],axis=1)))
                adversary_regret=float(average_loss.max())-played
                current=[]
                for kind,primal in [('averaged_point',float(point_loss.max())),('randomized_response',float(average_loss.max()))]:
                    interval=gap.gap_interval(primal,lower,upper,tolerance=1e-6)
                    error=float(np.mean([i['absolute_gap'] for i in infos[:t]])) if isconvex else None
                    theo=gap.regret_bound(len(models),self.c['raw_outer_rate'],t,error,B=set_B) if isconvex else None
                    row=dict(identity,round=t,gap_kind=kind,primal=primal,dual_lower=lower,dual_upper=upper,
                        gap_lower=interval['gap_lower_bound'],gap_upper=interval['gap_upper_bound'],dual_uncertainty=upper-lower,
                        normalized_gap_lower=interval['gap_lower_bound']/common_B,normalized_gap_upper=interval['gap_upper_bound']/common_B,
                        common_normalizer=common_B,set_loss_bound=set_B,raw_outer_rate=self.c['raw_outer_rate'],
                        adversary_regret=adversary_regret,recourse_regret_lower=played-upper,recourse_regret_upper=played-lower,
                        averaging_difference=float(point_loss.max()-average_loss.max()),
                        algorithm_seconds=float(z['algorithm_seconds'][t-1]),diagnostic_seconds=diag_seconds,
                        global_solver_seconds=global_seconds,global_build_seconds=build_seconds,
                        convex_theory_upper=theo,mean_oracle_error_bound=error,
                        mean_inner_iterations=float(np.mean([i['optimizer_iterations'] for i in infos[:t]])),
                        oracle_success_fraction=float(np.mean([i['optimizer_success'] for i in infos[:t]])),
                        oracle_hit_cap_fraction=float(np.mean([i['optimizer_iterations']>=self.c['oracle_maxiter'] for i in infos[:t]])),
                        global_bound_discarded=globalinfo.get('bound_consistent') is False,
                        numerical_tolerance=1e-6,artifact=str(folder.relative_to(self.root)))
                    if isconvex and interval['gap_upper_bound']>theo+1e-5:raise RuntimeError('Convex theory bound verification failed')
                    current.append(row)
                rr._npz(vectors,dict(average_point=xb,average_weights=wb,dual_response=response,
                    average_response_losses=average_loss,point_losses=point_loss,component_loss_lower_bounds=component_lower))
                record=dict(rows=current,dual_lower=lower,dual_upper=upper,diagnostic_elapsed_seconds=diag_seconds,
                    local_evaluation=localinfo,global_evaluation=globalinfo)
                rr.atomic_json(saved,record);rows.extend(current);previous.append((wb,lower,response))
            csv(pd.DataFrame(rows),folder/'gaps.csv')
            return rows,diag_seconds
        finally:
            if evaluator is not None:evaluator.close()

    def execute(self):
        token=rr._ACTIVE_SETTINGS.set(dict(candidate_selection='majority'))
        try:
            # Validate every planned model bank/cohort before any expensive trajectory.
            failures=[];blocks=[];preflight=self.root/'preflight.json'
            for ds in self.c['datasets']:
                for seed in self.c['seeds']:
                    self.progress('preflight',dataset=ds,seed=seed,blocks_ready=len(blocks),blocks_total=len(self.c['datasets'])*len(self.c['seeds']))
                    try:
                        directory,ctx,pool,chosen=self.block(ds,seed)
                        blocks.append(dict(dataset=ds,seed=seed,status='passed',eligible=load(directory/'ready.json')['eligible_count']))
                    except Exception as exc:
                        failures.append(dict(dataset=ds,seed=seed,error=repr(exc),traceback=traceback.format_exc()))
                        print('PREFLIGHT BLOCK FAILED',ds,seed,repr(exc),flush=True)
                    rr.atomic_json(preflight,dict(status='running',blocks=blocks,failures=failures,
                        expected=len(self.c['datasets'])*len(self.c['seeds'])))
                    dv._BANK_CACHE.clear();rc._TRAINING_CACHE.clear()
            rr.atomic_json(preflight,dict(status='failed' if failures else 'passed',blocks=blocks,failures=failures,
                expected=len(self.c['datasets'])*len(self.c['seeds'])))
            if failures:raise RuntimeError(f'{len(failures)} preflight blocks failed; see preflight.json. No ORPM trajectories launched.')
            # Calibrate one individual on every dataset, then complete the SAME cohorts.
            first=self.c['seeds'][0]
            jobs=[(ds,first,0,1) for ds in self.c['datasets']]
            jobs += [(ds,seed,1 if seed==first else 0,self.c['instances'])
                     for seed in self.c['seeds'] for ds in self.c['datasets']]
            for ds,seed,begin,end in jobs:
                if begin==end:continue
                directory,ctx,pool,chosen=self.block(ds,seed)
                for pos in chosen[begin:end]:
                    origin=ctx.X_test[pos].astype(float);iid=int(ctx.test_ids[pos]);person=directory/f'person_{iid}';person.mkdir(exist_ok=True)
                    constraints=rc.constraints_for(pool,torch.tensor(origin));B=mixed.common_normalizer([pool],origin,max(self.c['budgets'][ds]),constraints)
                    rr._npz(person/'factual.npz',dict(factual=origin,common_normalizer=B,lower=constraints.lower,
                        upper=constraints.upper,immutable=constraints.immutable))
                    for family in self.c['families']:
                        for n in self.c['model_sizes']:
                            models=subset(pool,family,n)
                            for budget in self.c['budgets'][ds]:
                                folder=person/family/f'models_{n}'/f'budget_{budget:g}';folder.mkdir(parents=True,exist_ok=True)
                                identity=dict(dataset=ds,seed=seed,instance_id=iid,family=family,models=n,budget=budget)
                                record=load(folder/'completed.json') if (folder/'completed.json').exists() else None
                                if record is None or (family!='linear' and record.get('diagnostic_version')!=DIAGNOSTIC_VERSION):
                                    replay=(folder/'trajectory.npz').exists()
                                    z,infos=self.trajectory(folder,models,origin,budget,constraints,B)
                                    rows,diagnostic_seconds=self.diagnostics(folder,models,origin,budget,constraints,B,z,infos,identity)
                                    assert len(rows)==2*len(self.c['checkpoints'])
                                    record=dict(identity,algorithm_seconds=float(z['algorithm_seconds'][-1]),diagnostic_seconds=diagnostic_seconds,
                                        artifact=str(folder.relative_to(self.root)),rounds=len(z['points']),checkpoints=len(rows)//2,
                                        verified_weights_and_feasibility=True,
                                        diagnostic_version='convex_analytic_v1' if family=='linear' else DIAGNOSTIC_VERSION,
                                        reused_saved_trajectory=replay)
                                    rr.atomic_json(folder/'completed.json',record)
                                self.done+=1;self.task_records.append(record)
                    csv(pd.DataFrame(self.task_records),self.root/'completed_trajectories.csv')
                    self.progress('person_completed',dataset=ds,seed=seed,instance_id=iid)
                    self.estimate_runtime()
                    print('PERSON COMPLETE',ds,seed,iid,self.done,'/',self.expected,flush=True)
                dv._BANK_CACHE.clear();rc._TRAINING_CACHE.clear()
            assert self.done==self.expected
            self.progress('analysis')
            from convergence_report import analyze
            analyze(self.root)
            assert self.hashes=={p.name:digest(p) for p in self.utility_dir.glob('*.py')}
            assert self.code_hashes=={p.name:digest(p) for p in HERE.glob('*.py')}
            assert self.data_hashes=={ds:digest(PROJECT/'dataset'/DATA_FILES[ds]) for ds in self.c['datasets']}
            rr.atomic_json(self.root/'verification.json',dict(status='passed',trajectories=self.done,expected=self.expected,
                nested_sets=True,common_majority_cohort=True,weights_reconstructed=True,feasible_responses=True,
                raw_loss_common_across_families=True,evaluation_separate=True,utility_and_data_hashes_unchanged=True,
                study_source_hashes_unchanged=True,all_model_banks_preflighted=True,
                nonconvex_diagnostic_version=DIAGNOSTIC_VERSION,
                training_protocol=self.c['training_protocol'],neural_global_training_guarantee=False))
            rr.atomic_json(self.root/'progress.json',dict(status='completed',completed=self.done,total=self.expected,
                finished_utc=datetime.now(timezone.utc).isoformat()))
            print('COMPLETED: trajectories, verification, report and figures_tables.zip saved.',flush=True)
        finally:rr._ACTIVE_SETTINGS.reset(token)

    def estimate_runtime(self):
        f=pd.DataFrame(self.task_records);f['total_seconds']=f.algorithm_seconds+f.diagnostic_seconds
        seen=set(f.artifact);pending_cached=self.cached_completed_paths-seen
        estimates=[]
        for (ds,family,n,budget),g in f.groupby(['dataset','family','models','budget']):
            cached=sum(p.split('/')[1]==ds and p.split('/')[4]==family and
                p.split('/')[5]==f'models_{int(n)}' and p.split('/')[6]==f'budget_{budget:g}' for p in pending_cached)
            remaining=max(0,self.c['instances']*len(self.c['seeds'])-len(g))
            # Existing linear cases are reused unchanged; existing nonconvex
            # cases need only diagnostic replay, not another ORPM trajectory.
            projected=float(g.total_seconds.mean())*max(0,remaining-cached)
            if family!='linear':projected+=float(g.diagnostic_seconds.mean())*cached
            estimates.append(dict(dataset=ds,family=family,models=int(n),budget=float(budget),observed=len(g),
                mean_seconds=float(g.total_seconds.mean()),p90_seconds=float(g.total_seconds.quantile(.9)),
                target=self.c['instances']*len(self.c['seeds']),pending_cached_trajectories=cached,
                projected_remaining_seconds=projected))
        projected=sum(r['projected_remaining_seconds'] for r in estimates)
        expected_conditions=sum(len(self.c['budgets'][ds]) for ds in self.c['datasets'])*len(self.c['families'])*len(self.c['model_sizes'])
        rr.atomic_json(self.root/'runtime_estimate.json',dict(covered_conditions=len(estimates),total_conditions=expected_conditions,
            projected_remaining_seconds_for_observed_conditions=projected,conditions=estimates,
            note='Accounts for cached completed trajectories requiring only diagnostic replay. Excludes unobserved conditions, future training, report generation, and machine-load variation; not a full ETA until every condition is sampled.'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--config',required=True);p.add_argument('--lock-fd',type=int)
    a=p.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
    lock=os.fdopen(a.lock_fd,'a+') if a.lock_fd is not None else (root/'worker.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,request_stop)
    try:
        with threadpool_limits(limits=1):
            torch.set_num_threads(1);Study(root,load(a.config)).execute()
    except BaseException as exc:
        rr.atomic_json(root/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),utc=datetime.now(timezone.utc).isoformat()))
        rr.atomic_json(root/'progress.json',dict(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc)))
        raise
    finally:lock.close()

if __name__=='__main__':main()
