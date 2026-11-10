from tradeoff_study import *

class FreshTradeoff(Study):

    def __init__(self,root,config):
        self.root=root
        self.c=config
        self.source=root/'no_previous_run'
        self.started=time.perf_counter()
        self.done=0
        self.rows=[]
        self.last_progress=0
        self.source_manifest={}
        self.expected=sum(len(config['budgets'][d]) for d in config['datasets'])*len(config['seeds'])*config['instances']*len(config['families'])*len(config['model_sizes'])*len(config['methods'])
        rr._ACTIVE_SETTINGS.set(dict(lime_samples=config['lime_samples'],candidate_selection='majority'))

    def block(self,ds,seed):
        folder=self.root/'blocks'/ds/f'seed_{seed}'
        bank=folder/'bank'
        data=arrays(bank/'split_and_predictions.npz')
        prep=load(bank/'preprocessing.json')
        keys=('X_train','y_train','X_validation','y_validation','X_test','y_test','train_ids','validation_ids','test_ids')
        ctx=rc.SplitContext(name=ds,columns=prep['feature_names'],scaler=None,seed=seed,actionable_indices=prep['actionable_indices'],actionability_policy=prep['actionability_policy'],**{k:data[k] for k in keys})
        pool=rc.ModelSet(rr.load_models(bank),context=ctx)
        for model in pool:model.recourse_context=ctx
        sel=load(folder/'selection.json')
        chosen=np.array(sel['test_positions'][:self.c['instances']])
        np.testing.assert_array_equal(ctx.test_ids[chosen],sel['instance_ids'][:len(chosen)])
        return folder,ctx,pool,chosen
    def execute(self):
        for ds in self.c['datasets']:
            for seed in self.c['seeds']:
                block,ctx,pool,chosen=self.block(ds,seed)
                preparations={}
                for refindex in [0,24]:
                    p=rb.prepare_ellice(pool[refindex],ctx.X_train,self.c['ellice_epsilon']);preparations[refindex]=p
                    rr._npz(block/f'ellice_reference_{refindex}.npz',dict(H=p['H'].numpy(),H_inverse=p['H_inverse'].numpy(),theta=p['theta'].numpy(),epsilon=p['epsilon']))
                for pos in chosen:
                    iid=int(ctx.test_ids[pos]);origin=ctx.X_test[pos].astype(float)
                    person=block/f'person_{iid}'
                    person.mkdir(exist_ok=True)
                    cons=rc.constraints_for(pool,torch.tensor(origin));check_feasible(origin,origin,0,cons)
                    rr._npz(person/'factual.npz',dict(factual=origin,lower=cons.lower,upper=cons.upper,immutable=cons.immutable))
                    self.progress('local_surrogate_preparation',dataset=ds,seed=seed,instance_id=iid)
                    sm=self.surrogates(person,pool,origin,seed+iid) if any(self.c['budgets'][ds]) else pool
                    for family in self.c['families']:
                        for n in self.c['model_sizes']:
                            models=subset(pool,family,n);surrogates=subset(sm,family,n);refindex=indices(family,n)[0]
                            for budget in self.c['budgets'][ds]:
                                identity=dict(dataset=ds,seed=seed,instance_id=iid,family=family,models=n,budget=float(budget))
                                case=person/family/f'models_{n}'/f'budget_{budget:g}'
                                for method in self.c['methods']:
                                    folder=case/method;folder.mkdir(parents=True,exist_ok=True);result=folder/'result.json'
                                    if result.exists():row=load(result)
                                    else:
                                        self.progress('recourse',force=False,**identity,method=method)
                                        if budget==0:
                                            point=origin.copy();info=dict(status='zero_budget_no_change',seconds=0.,rounds=0)
                                        elif method=='ORPM':point,info,_=self.orpm(folder,models,origin,budget,cons,identity)
                                        else:
                                            # Only share baseline computations when their actual objective and reference are identical.
                                            shared=person/'shared'/f'reference_{refindex}'/f'budget_{budget:g}'/method
                                            cache=shared if method in ['ElliCE','Reference'] else folder/'optimizer'
                                            point,info=self.baseline(cache,method,models,surrogates,origin,budget,cons,preparations[refindex])
                                        check_feasible(point,origin,budget,cons)
                                        logits=numpy_logits(models,point)[0];loss=np.logaddexp(0,-logits);prob=expit(logits)
                                        reference_valid=bool(logits[0]>=0)
                                        row=dict(**identity,method=method,reused_phase=False,joint_validity=bool((logits>=0).all()),
                                            average_validity=float((logits>=0).mean()),minimum_probability=float(prob.min()),
                                            worst_loss=float(loss.max()),cost=float(np.linalg.norm(point-origin)),reference_validity=reference_valid,
                                            reference_pool_index=refindex,feasible=True,seconds=float(info.get('seconds',0.)),
                                            new_solve_seconds=0. if info.get('shared_cache_hit') else float(info.get('seconds',0.)),
                                            reused_source=bool(info.get('reused_source',False)),
                                            original_algorithm_seconds=float(info.get('original_algorithm_seconds',0.)),
                                            feasibility_projected=bool(info.get('feasibility_repair',{}).get('applied',False)),
                                            projection_distance=float(info.get('feasibility_repair',{}).get('distance',0.)),
                                            rounds=int(info.get('rounds',0)),optimizer=info,artifact=str(folder.relative_to(self.root)))
                                        rr._npz(folder/'evaluation.npz',dict(point=point,factual=origin,probabilities=prob,losses=loss,logits=logits))
                                        rr.atomic_json(result,row)
                                    self.rows.append(row);self.done+=1
                                # Checkpoint per setting; compact CSV once per person below.
                    self.save_rows()
                    self.progress('person_completed',dataset=ds,seed=seed,instance_id=iid)
                    print('PERSON_COMPLETE',ds,seed,iid,f'{self.done}/{self.expected}',flush=True)
                self.progress('block_completed',dataset=ds,seed=seed)
        assert self.done==self.expected
        self.save_rows()
    def verify(self):
        f=pd.read_csv(self.root/'results.csv',low_memory=False);keys=['dataset','seed','instance_id','family','models','budget','method']
        assert len(f)==self.expected and not f.duplicated(keys).any() and f.feasible.all()
        assert f.groupby(['dataset','family','models','budget','method','seed']).size().eq(self.c['instances']).all()
        maximum_error=0.
        for ds in self.c['datasets']:
            for seed in self.c['seeds']:
                _,ctx,pool,chosen=self.block(ds,seed)
                for family in self.c['families']:
                    for n in self.c['model_sizes']:
                        models=subset(pool,family,n);g=f[f.dataset.eq(ds)&f.seed.eq(seed)&f.family.eq(family)&f.models.eq(n)]
                        bundles=[arrays(self.root/a/'evaluation.npz') for a in g.artifact]
                        points=np.stack([z['point'] for z in bundles]);true_logits=numpy_logits(models,points)
                        np.testing.assert_allclose(true_logits,np.stack([z['logits'] for z in bundles]),atol=1e-9,rtol=1e-9)
                        np.testing.assert_array_equal((true_logits>=0).all(axis=1),g.joint_validity.to_numpy(bool))
                        np.testing.assert_allclose(expit(true_logits).min(axis=1),g.minimum_probability,atol=1e-12)
                        # A second backend checks the vectorized NumPy prediction implementation.
                        with torch.no_grad():
                            double=[copy.deepcopy(m,{id(ctx):ctx}).double() for m in models]
                            tx=torch.tensor(points,dtype=torch.float64)
                            torch_logits=np.stack([m.logits(tx).numpy().reshape(-1) for m in double],axis=1)
                        err=float(np.max(np.abs(torch_logits-true_logits)));maximum_error=max(maximum_error,err)
                        np.testing.assert_allclose(torch_logits,true_logits,rtol=1e-8,atol=1e-8)
                        for row,z in zip(g.itertuples(),bundles):
                            cons=rc.constraints_for(models,torch.tensor(z['factual']))
                            check_feasible(z['point'],z['factual'],row.budget,cons)
                            assert abs(np.linalg.norm(z['point']-z['factual'])-row.cost)<1e-9
                            if row.budget==0:np.testing.assert_array_equal(z['point'],z['factual'])
                self.progress('independent_verification',dataset=ds,seed=seed)
        for path,h in self.source_manifest.items():assert digest(path)==h,f'Source was modified: {path}'
        return dict(rows=len(f), maximum_torch_logit_error=maximum_error)
