"""Use the convergence study's own training checks and its five-person cohort rule."""
from io_tools import ROOT,read
def generate(root,ds,seed):
    import sys
    sys.path.insert(0,str(ROOT/'experiments/convergence_paper'))
    from convergence_study import Study,torch,threadpool_limits
    cfg=read(ROOT/'experiments/convergence_paper/config.json')
    cfg.update(datasets=[ds],seeds=[seed])
    with threadpool_limits(1):
        torch.set_num_threads(1);Study(root,cfg).block(ds,seed)
