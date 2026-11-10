"""Generate study inputs. No historical result or model repository is consulted."""
import os,sys,shutil,tempfile
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from io_tools import ROOT,read,write,snapshot,setup_source
from raw_cache import obtain
import numpy as np
os.environ.update(RECOURSE_V7_PROJECT=str(ROOT),PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg')

def mixed(root,ds,seed):
    from types import SimpleNamespace
    setup_source('non-tree-tradeoff')
    from phase_support import rc,rr,dv,cohort,torch,DATA_FILES,digest
    from model_training import build_pool
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0)
    rc.PROJECT_ROOT=ROOT
    cfg=read(snapshot('non-tree-tradeoff')/'config.json')
    opts=read(ROOT/'experiments/convergence_paper/config.json')
    f=ROOT/'dataset'/DATA_FILES[ds]
    assert digest(f)==next(r['sha256'] for r in read(ROOT/'dataset/manifest.json')['files'] if r['file']==f.name)
    X,y,act=rc.load_data_and_normalize(ds);ctx=rc.prepare_split(X,y,seed,ds,act);ctx.epsilon=cfg['ellice_epsilon']
    block=root/'blocks'/ds/f'seed_{seed}';block.mkdir(parents=True)
    with tempfile.TemporaryDirectory(dir=root) as temp:
        tmp=Path(temp);store=rr.RunStore(tmp,'raw_training',opts);token=rr._ACTIVE_RUN.set(store)
        try:
            store.scope.update(seed=seed,fold=0);store.save_split(ctx)
            pool=build_pool(ctx,opts,tmp/'fit',lambda *a,**k:None)
            prob=dv.probabilities(pool,ctx.X_test)
            available,order,counts=cohort(SimpleNamespace(c=cfg),ctx,prob);chosen=order[:cfg['instances']]
            ctx.candidate_ids=ctx.test_ids[chosen].tolist()
            ctx.candidate_selection=dict(mode='intersection_majority',ties_included=True,eligible=len(available),requested=len(chosen))
            bank=store.save_seed(pool,[torch.tensor(x) for x in ctx.X_test[chosen]],fold=0,seed=seed)
            assert len(pool)==48 and all(d['training_settings']['passed'] for d in read(bank/'models.json')['models'])
            shutil.move(str(bank),str(block/'bank'))
        finally:rr._ACTIVE_RUN.reset(token)
    write(block/'selection.json',dict(test_positions=chosen,instance_ids=ctx.test_ids[chosen],eligible_count=len(available)))
    np.savez_compressed(block/'cohort.npz',test_positions=chosen,instance_ids=ctx.test_ids[chosen],eligible_positions=available,complete_random_order=order,**counts)

def main():
    kind,ds,s,out=sys.argv[1:];root=Path(out);seed=int(s)
    print(f'Preparing {kind}: {ds}, seed {seed}',flush=True)
    if kind=='non-tree-tradeoff':mixed(root,ds,seed)
    elif kind=='non-tree-convergence':
        from raw_convergence import generate
        generate(root,ds,seed)
    elif kind in ('tree-tradeoff','tree-cost','tree-convergence','dt-generalization','rf-generalization'):
        from raw_trees import generate
        generate(kind,root,ds,seed)
    else:
        from raw_panels import generate
        generate(kind,root,ds,seed)
if __name__=='__main__':main()
