"""Independent numeric audit: no recourse optimizer or runner evaluation imported."""
from pathlib import Path
import json
import numpy as np
from scipy.special import expit,logsumexp
from scipy.optimize import linprog
from common import read,sha,theta_hash,utc

def verify(root):
    root=Path(root);cfg=read(root/'protocol.json');s=cfg['solver'];tol=s['evaluation_logit_tolerance'];failures=[];counts=dict(records=0,margin_certificates=0,cost_certificates=0,orpm_trajectories=0,panels=0)
    def check(condition,message):
        if not bool(condition):failures.append(message)
    for path,h in read(root/'input_manifest.json').items():check(sha(path)==h,'source drift: '+path)
    freeze=read(root/'freeze.json')
    for path,h in freeze['source_sha256'].items():check(sha(root/path)==h,'frozen source drift: '+path)
    check(read(root/'preflight.json')['status']=='passed','analytic preflight')
    inv=read(root/'model_inventory.json');panels={}
    for key,item in inv.items():
        ds,ss=key.split('/');seed=int(ss);directory=root/'blocks'/ds/f'seed_{seed}';bank=directory/'bank';p=dict(np.load(directory/'panel.npz'));co=read(directory/'cohort.json');adm=read(directory/'unseen_admission.json');data=dict(np.load(bank/'split_and_predictions.npz'));pre=read(bank/'preprocessing.json')
        check(sha(directory/'panel.npz')==item['panel_sha256'],'panel drift '+key)
        check(sha(directory/'cohort.json')==item['cohort_sha256'],'cohort drift '+key)
        check(sha(directory/'unseen_admission.json')==item['unseen_admission_sha256'],'admission drift '+key)
        for original,h in read(root/'input_manifest.json').items():
            if str(Path(original).parent)==item['source_bank']:check(sha(bank/Path(original).name)==h,'snapshot drift '+original)
        A=[];b=[]
        for i in range(24):
            f=dict(np.load(bank/f'model_{i:04d}.npz'));prefix='base_model.' if 'feature_mask' in f else ''
            # Match saved model's float32 multiplication of mask and coefficients.
            a=f[prefix+'linear.weight'].reshape(-1)*f.get('feature_mask',np.ones_like(f[prefix+'linear.weight'].reshape(-1)))
            A.append(a.astype(float));b.append(float(f[prefix+'linear.bias'].item()))
        A=np.array(A);b=np.array(b);check(np.array_equal(A,p['A']) and np.array_equal(b,p['b']),'optimization parameters '+key)
        X=np.asarray(data['X_test'],np.float32).astype(float);z=X@A.T+b
        eligible=np.flatnonzero(np.logical_and.reduce([(z[:,:n]<0).sum(1)>=n/2 for n in cfg.get('cohort_model_sizes',cfg['model_sizes'])]))
        order=np.random.default_rng(seed+cfg['cohort_order_seed_offset']).permutation(eligible)
        check(co['random_order']==order.tolist() and co['selected_indices']==order[:cfg['instances']].tolist(),'cohort reproduction '+key)
        check(np.array_equal(p['selected_X'],X[order[:cfg['instances']]]),'selected features '+key)
        check(np.array_equal(p['test_ids'],data['test_ids'][order[:cfg['instances']]]),'selected identities '+key)
        check(np.array_equal(p['free'],pre['actionable_indices']),'actionability '+key)
        opt_hash={theta_hash(np.r_[aa,bb]) for aa,bb in zip(A,b)};unseen=np.column_stack([p['unseen_A'],p['unseen_b']]);uh=[theta_hash(t) for t in unseen]
        check(len(unseen)==cfg['unseen_count'] and len(set(uh))==len(uh) and not(set(uh)&opt_hash),'disjoint panel '+key)
        train=np.column_stack([data['X_train'],np.ones(len(data['X_train']))]).astype(float);y=data['y_train'];val=np.column_stack([data['X_validation'],np.ones(len(data['X_validation']))]);theta=np.r_[A[0],b[0]];l2=adm['l2'];z=train@theta;pr=expit(z);loss=np.mean(np.logaddexp(0,z)-y*z)+l2*.5*(theta@theta);g=train.T@(pr-y)/len(y)+l2*theta;gap=g@g/(2*l2);va=np.mean((val@theta>=0)==data['y_validation']);H=(train.T*(pr*(1-pr)))@train/len(y)+l2*np.eye(len(theta));e,U=np.linalg.eigh(H);transform=U@np.diag(1/np.sqrt(e))@U.T
        check(gap<=1e-8 and abs(loss-adm['center_loss'])<1e-10,'central fit '+key)
        admitted=[];hh=set(opt_hash)
        for attempt in adm['attempts']:
            k=attempt['attempt'];rng=np.random.default_rng(cfg['unseen_seed_offset']+seed*10000+k);v=rng.normal(size=len(theta));v/=np.linalg.norm(v);candidate=(theta+transform@(np.sqrt(2*cfg['unseen_epsilon'])*rng.random()**(1/len(theta))*v)).astype(np.float32).astype(float)
            zz=train@candidate;ll=np.mean(np.logaddexp(0,zz)-y*zz)+l2*.5*(candidate@candidate);aa=np.mean((val@candidate>=0)==data['y_validation']);h=theta_hash(candidate)
            ok=ll-loss+gap<=cfg['unseen_epsilon']+1e-8 and aa>=va-cfg['validation_accuracy_allowance'] and h not in hh
            check(bool(ok)==attempt['accepted'] and h==attempt['theta_sha256'],'candidate admission '+key+':'+str(k))
            if ok:admitted.append(candidate);hh.add(h)
        check(np.array_equal(np.array(admitted),unseen),'regenerated panel '+key)
        panels[key]=p;counts['panels']+=1
    check(read(root/'panels_frozen.json')['count']==cfg['expected_panels'],'all panels frozen')
    rows=(json.loads(line) for line in open(root/'records.jsonl'));seen_keys=set();files={};exact_seen=set();min_seen=set()
    for r in rows:
        key=f"{r['dataset']}/{r['seed']}";p=panels[key];A=p['A'][:r['model_count']];b=p['b'][:r['model_count']];x0=p['selected_X'][r['person']];free=p['free'];immutable=np.array(sorted(set(range(len(x0)))-set(free)),dtype=int);B=r['budget'];ident=tuple(r[k] for k in ['dataset','seed','person','model_count','budget','method'])
        check(ident not in seen_keys,'duplicate '+str(ident));seen_keys.add(ident)
        path=root/r['point_file']
        if path not in files:files={path:dict(np.load(path))}
        f=files[path];x=f[r['method']];cost=np.linalg.norm(x-x0)
        check(np.isfinite(x).all() and cost<=B+1e-8 and np.array_equal(x[immutable],x0[immutable]),'feasibility '+str(ident))
        check(abs(cost-r['cost'])<1e-9,'cost '+str(ident))
        if B==0:check(np.array_equal(x,x0),'zero-budget '+str(ident))
        for prefix,aa,bb in [('seen',A,b),('unseen',p['unseen_A'],p['unseen_b'])]:
            z=aa@x+bb;expected=dict(joint_validity=float(r['available'] and np.all(z>=-tol)),strict_joint_validity=float(r['available'] and np.all(z>=0)),mean_validity=float(np.mean(z>=-tol)) if r['available'] else 0.,minimum_logit=float(min(z)),worst_loss=float(np.logaddexp(0.,-z).max()),boundary_count=int(np.sum(abs(z)<=tol)))
            for metric,value in expected.items():check(abs(value-r[prefix+'_'+metric])<1e-8,'metric '+prefix+'_'+metric+' '+str(ident))
        if r['method']=='ORPM' and B>0:
            points=f['orpm_points'];w=f['orpm_weights'];losses=np.logaddexp(0.,-(points@A.T+b));T=cfg['rounds'].get(r['dataset'],cfg['rounds']['default'])
            check(len(points)==T and np.allclose(points.mean(0),x,atol=1e-10,rtol=0),'ORPM return '+str(ident))
            check(np.all(np.linalg.norm(points-x0,axis=1)<=B+1e-8) and np.array_equal(points[:,immutable],np.broadcast_to(x0[immutable],points[:,immutable].shape)),'ORPM responses '+str(ident))
            logw=np.vstack([np.zeros(len(A)),np.cumsum(losses[:-1],axis=0)])*cfg['raw_outer_rate'];logw-=logsumexp(logw,axis=1)[:,None]
            check(np.allclose(w,np.exp(logw),atol=1e-9,rtol=1e-8) and np.allclose(losses,f['orpm_losses'],atol=1e-9),'ORPM weights/losses '+str(ident));counts['orpm_trajectories']+=1
        if path not in exact_seen:
            q=read(path.parent/'exact_margin.json');info=q['info'];point=np.array(q['point']);w=np.array(info['dual_weights']);z=A@x0+b
            lower=float(min(A@point+b));upper=float(w@z+B*np.linalg.norm(A[:,free].T@w)) if B else float(min(z));gap=upper-lower
            check(np.all(w>=0) and abs(sum(w)-1)<1e-9,'dual simplex '+str(path))
            check(gap>=-1e-7 and gap<=s['certificate_tolerance']*(1+abs(lower)) and info['certificate_passed'],'max-margin certificate '+str(path))
            check(abs(lower-info['margin_lower'])<1e-8 and abs(upper-info['margin_upper'])<1e-8,'max-margin bound values '+str(path))
            check(np.linalg.norm(point-x0)<=B+1e-8 and np.array_equal(point[immutable],x0[immutable]),'max-margin feasible '+str(path));counts['margin_certificates']+=1;exact_seen.add(path)
        mpath=path.parent.parent/'minimum_cost.json'
        if mpath not in min_seen:
            q=read(mpath);info=q['info'];point=np.array(q['point']);z=A@x0+b;Q=A[:,free]
            if info['available']:
                lam=np.array(info['dual_multipliers']);delta=point[free]-x0[free];primal=.5*delta@delta;dual=-lam@z-.5*np.sum((Q.T@lam)**2)
                check(np.all(lam>=0) and min(A@point+b)>=-tol and np.array_equal(point[immutable],x0[immutable]),'min-cost feasibility '+str(mpath))
                check(primal-dual>=-1e-7 and primal-dual<=s['certificate_tolerance']*(1+abs(primal)) and info['certificate_passed'],'min-cost gap '+str(mpath))
                check(abs(primal-info['primal_objective'])<1e-7 and abs(dual-info['dual_objective'])<1e-7,'min-cost bound values '+str(mpath))
            else:
                lp=linprog(np.zeros(len(free)),A_ub=-Q,b_ub=z,bounds=[(None,None)]*len(free),method='highs')
                check(lp.status==2 and info['status']=='certified_infeasible' and info['certificate_passed'],'independent infeasibility '+str(mpath))
            counts['cost_certificates']+=1;min_seen.add(mpath)
        if r['method']=='ExactMinL2' and B>0:
            info=read(mpath)['info'];available=info['available'] and info['cost_upper']<=B+1e-8
            check(bool(available)==r['available'],'minimum-cost availability '+str(ident))
        counts['records']+=1
    check(counts['records']==cfg['expected_records'],'expected record count')
    expected={(ds,seed,i,n,float(B),m) for ds in cfg['datasets'] for seed in cfg['seeds'] for i in range(cfg['instances']) for n in cfg['model_sizes'] for B in cfg['budgets'][ds] for m in cfg['methods']}
    check(seen_keys==expected,'complete factorial keys')
    return dict(passed=not failures,verified_utc=utc(),counts=counts,failures=failures,numerical_scope='float64 independent certificates at frozen tolerances; not interval arithmetic',source_preservation=True if not any('drift' in f for f in failures) else False,unseen_scope='fresh full-feature conditional logistic hypotheses, shared training data',future_20_available_all_pilot_blocks=all(read(root/'blocks'/ds/f'seed_{seed}'/'cohort.json')['future_20_available'] for ds in cfg['datasets'] for seed in cfg['seeds']))

if __name__=='__main__':
    import sys
    from common import save
    root=Path(sys.argv[1]);r=verify(root);save(root/'verification_recheck.json',r);print(json.dumps(r,indent=2))
