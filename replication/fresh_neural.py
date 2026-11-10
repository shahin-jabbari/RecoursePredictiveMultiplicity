from engine import *

def run_settings(root,cfg):
    preserved_text,count="",0;start=time.perf_counter();tol=cfg['solver']['evaluation_logit_tolerance']
    with benchmarks.environment() as env,open(root/'records.jsonl','x',buffering=1) as records:
        records.write(preserved_text)
        progress(root,cfg,completed_records=count,preserved_records=count,new_completed_records=0)
        for ds in cfg['datasets']:
            for seed in cfg['seeds']:
                directory=root/'blocks'/ds/f'seed_{seed}';models,ctx,meta,prep=load_bank(directory/'bank')
                with np.load(directory/'panel.npz',allow_pickle=False) as f:saved=dict(f)
                full={k:saved['seen_'+k] for k in ['W','b','v','c']};unseen={k:saved['unseen_'+k] for k in ['W','b','v','c']};free=saved['free'].tolist()
                prep_start=time.perf_counter()
                ellice=bas.prepare_ellice(models[0],ctx.X_train,epsilon=cfg['ellice_epsilon'])
                ellice_seconds=time.perf_counter()-prep_start
                append_time(root,dict(kind='ellice_shared',dataset=ds,seed=seed,seconds=ellice_seconds))
                for person,x0 in enumerate(saved['selected_X']):
                    x0t=torch.tensor(x0,dtype=torch.float32)
                    lst=time.perf_counter();token=rr._ACTIVE_SETTINGS.set(dict(lime_samples=cfg['lime_samples']))
                    try:surrogates=bas.linearize_models_lime(models,x0,ctx.X_train,seed=seed+900000+person)
                    finally:rr._ACTIVE_SETTINGS.reset(token)
                    lime_seconds=time.perf_counter()-lst
                    pars=[rc.linear_parameters(m) for m in surrogates];theta=np.stack([np.r_[a.numpy(),float(b)] for a,b in pars])
                    persondir=directory/'cases'/f'person_{person:02d}';persondir.mkdir(parents=True)
                    np.savez_compressed(persondir/'surrogates.npz',theta=theta)
                    save(persondir/'surrogates.json',dict(seconds=lime_seconds,samples=cfg['lime_samples'],seed=seed+900000+person,diagnostics=[m.surrogate_diagnostics for m in surrogates]))
                    append_time(root,dict(kind='lime_shared',dataset=ds,seed=seed,person=person,seconds=lime_seconds))
                    for n in cfg['model_sizes']:
                        setting_start=time.perf_counter();subset=rc.ModelSet(models[:n],context=ctx);p=sliced(full,n);c=rc.constraints_for(subset,x0t);setting=persondir/f'n_{n}';setting.mkdir()
                        progress(root,cfg,status='solving_minimum_cost',dataset=ds,seed=seed,person=person,model_count=n,completed_records=count)
                        minpoint,mininfo=benchmarks.solve(p,x0,max(cfg['budgets'][ds]),free,cfg['solver'],env,'min_l2',setting/'minimum_cost_solver');save(setting/'minimum_cost.json',dict(point=minpoint,info=mininfo))
                        append_time(root,dict(kind='miqcp_min',dataset=ds,seed=seed,person=person,model_count=n,seconds=mininfo['seconds'],certified=mininfo['certified']))
                        for bi,B in enumerate(cfg['budgets'][ds]):
                            budget_start=time.perf_counter();path=setting/f'budget_{bi}';path.mkdir();progress(root,cfg,status='solving_setting',dataset=ds,seed=seed,person=person,model_count=n,budget=B,completed_records=count)
                            maxpoint,maxinfo=benchmarks.solve(p,x0,B,free,cfg['solver'],env,'max_margin',path/'maximum_margin_solver');save(path/'maximum_margin.json',dict(point=maxpoint,info=maxinfo))
                            if B>0:append_time(root,dict(kind='miqcp_max',dataset=ds,seed=seed,person=person,model_count=n,budget=B,seconds=maxinfo['seconds'],certified=maxinfo['certified']))
                            arrays=dict(origin=x0);rows=[]
                            for method in cfg['methods']:
                                mark('method_started',dataset=ds,seed=seed,person=person,model_count=n,budget=float(B),method=method)
                                st=time.perf_counter();info={};trace={};available=True;raw=None
                                if B==0:point=x0.copy();info=dict(status='singleton_factual_control')
                                elif method=='MIQCPMaxMargin':point=maxpoint.copy();info=maxinfo
                                elif method=='MIQCPMinL2':
                                    point,available,label=benchmarks.min_budget_output(minpoint,mininfo,x0,B,cfg['solver']);info=dict(mininfo,budget_classification=label)
                                elif method=='ORPM':point,info,trace=orpm(p,x0,B,free,cfg,ds)
                                elif method=='Reference':
                                    if logits(sliced(p,1),x0)[0]>=0:point=x0.copy();info=dict(status='reference_already_accepts')
                                    else:point,info=oracle(sliced(p,1),x0,B,free,np.ones(1),x0,cfg['reference_maxiter'])
                                else:
                                    loss=lambda m,x:torch.nn.functional.softplus(-m.logits(x)).sum()
                                    if method=='ADV':ret=bas.ADV(x0t,subset,B,loss,T=cfg['baseline_maxiter'],lr=cfg['baseline_lr']*B,constraints=c)[-1]
                                    elif method=='ROAR':ret=bas.ROAR(x0t,surrogates[0],float(abs(theta[:n]-theta[0]).max()),B,loss,T=cfg['baseline_maxiter'],constraints=c)[-1]
                                    elif method=='ElliCE':ret=bas.ElliCE(x0t,subset,B,loss,T=cfg['baseline_maxiter'],constraints=c,preparation=ellice)[-1]
                                    else:raise ValueError(method)
                                    raw=ret.detach().numpy().astype(float);point,repair=repair_baseline_point(raw,x0,B,c);info=dict(getattr(ret,'recourse_info',{}),numerical_repair=repair)
                                elapsed=time.perf_counter()-st
                                if method=='MIQCPMaxMargin' and B>0:elapsed=maxinfo['seconds']
                                if method=='MIQCPMinL2' and B>0:elapsed=mininfo['seconds']
                                assert np.isfinite(point).all() and np.linalg.norm(point-x0)<=B+cfg['solver']['budget_tolerance'] and np.array_equal(point[c.immutable],x0[c.immutable]),(method,point,x0,B)
                                seen=metric(logits(p,point),available,tol);un=metric(logits(unseen,point),available,tol)
                                row=dict(dataset=ds,seed=seed,person=person,test_id=int(saved['test_ids'][person]),model_count=n,budget=float(B),budget_index=bi,method=method,available=available,cost=float(np.linalg.norm(point-x0)) if available else None,seconds=elapsed,surrogate_shared_seconds=lime_seconds if method=='ROAR' and B>0 else 0.,point_file=str((path/'points.npz').relative_to(root)),info=info)
                                row.update({'seen_'+k:v for k,v in seen.items()});row.update({'unseen_'+k:v for k,v in un.items()})
                                lower,upper=maxinfo['margin_lower'],maxinfo.get('margin_upper')
                                row.update(margin_optimum_lower=lower,margin_optimum_upper=upper,loss_gap_lower=max(0.,seen['worst_loss']-float(np.logaddexp(0.,-lower))) if available else None,loss_gap_upper=max(0.,seen['worst_loss']-float(np.logaddexp(0.,-upper))) if available and upper is not None else None,max_margin_certified=maxinfo['certified'],minimum_cost_lower=mininfo.get('cost_lower'),minimum_cost_upper=mininfo.get('cost_upper'),min_cost_certified=mininfo['certified'])
                                positive=len(cfg['budgets'][ds])-1
                                row['shared_preparation_seconds']=lime_seconds if method=='ROAR' and B>0 else (ellice_seconds if method=='ElliCE' and B>0 else 0.)
                                share=(lime_seconds/(len(cfg['model_sizes'])*positive) if method=='ROAR' and B>0 else (ellice_seconds/(cfg['instances']*len(cfg['model_sizes'])*positive) if method=='ElliCE' and B>0 else 0.))
                                row['grid_amortized_seconds']=(elapsed/positive if method=='MIQCPMinL2' and B>0 else elapsed)+share
                                row['standalone_with_preparation_seconds']=elapsed+row['shared_preparation_seconds']
                                row['joint_solution_found']=bool(available and seen['joint_validity'])
                                arrays[method]=point
                                if raw is not None:arrays[method+'_raw']=raw
                                arrays.update(trace);rows.append(row)
                            np.savez_compressed(path/'points.npz',**arrays)
                            for row in rows:records.write(json.dumps(clean(row),allow_nan=False)+'\n');count+=1
                            append_time(root,dict(kind='setting_wall',dataset=ds,seed=seed,person=person,model_count=n,budget=B,seconds=time.perf_counter()-budget_start))
                            progress(root,cfg,status='running',completed_records=count,elapsed_solver_seconds=time.perf_counter()-start)
                        print(utc(),'SETTING_GROUP_COMPLETE',count,'/',cfg['expected_records'],ds,seed,person,n,'seconds',round(time.perf_counter()-setting_start,2),flush=True)
    assert count==cfg['expected_records']
    return count

