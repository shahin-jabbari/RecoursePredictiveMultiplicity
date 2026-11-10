"""Disjoint-seed rate calibration on saved benchmark banks; never changes those runs."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from convergence_study import Study,load,arrays,rc,rr,indices,csv,HERE

SOURCES=[HERE/'benchmark_results/20261010T212437.931999Z_1ade065471',
         HERE/'benchmark_german_results/20261010T212726.979655Z_a3f5f5357d']

def main():
    allrows=[];root=HERE/'rate_calibration';root.mkdir(exist_ok=True)
    for alpha in [.25,1.]:
        for source in SOURCES:
            config=load(source/'config.json');ds=config['datasets'][0];seed=config['seeds'][0]
            config.update(raw_outer_rate=alpha,model_sizes=[24],profile=f'rate_{alpha:g}',checkpoints=[1,100,500,1000])
            config['global_seconds_per_checkpoint']=1.;config['global_seconds_final']=5.
            subroot=root/f'{ds}_rate_{alpha:g}';study=Study(subroot,config)
            block=source/'blocks'/ds/f'seed_{seed}';ready=load(block/'ready.json')
            pool=rc.ModelSet(rr.load_models(source/ready['bank']),context=None)
            person=block/f'person_{ready["instance_ids"][0]}'
            for family in config['families']:
                previous=person/family/'models_24/budget_2';z=arrays(previous/'trajectory.npz')
                cons=rc.Constraints(lower=z['lower'],upper=z['upper'],immutable=z['immutable'].astype(int).tolist())
                models=rc.ModelSet([pool[i] for i in indices(family,24)],context=None)
                folder=subroot/family;folder.mkdir(parents=True,exist_ok=True)
                identity=dict(dataset=ds,seed=seed,instance_id=ready['instance_ids'][0],family=family,models=24,budget=2.)
                trajectory,infos=study.trajectory(folder,models,z['factual'],2.,cons,float(z['common_normalizer']))
                rows,seconds=study.diagnostics(folder,models,z['factual'],2.,cons,float(z['common_normalizer']),trajectory,infos,identity)
                allrows.extend(rows);csv(pd.DataFrame(allrows),root/'rates.csv')
                print('RATE CALIBRATION',ds,family,alpha,[(r['gap_kind'],r['gap_lower'],r['gap_upper']) for r in rows if r['round']==1000],flush=True)
    rr.atomic_json(root/'completed.json',dict(status='completed',sources=[str(p) for p in SOURCES],rows=len(allrows)))

if __name__=='__main__':
    with threadpool_limits(limits=1):torch.set_num_threads(1);main()
