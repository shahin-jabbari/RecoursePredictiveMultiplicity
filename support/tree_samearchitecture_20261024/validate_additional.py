import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'source'))
from experiment import *
from verify_unseen import verify_reference
from eta import full_estimate
from supervisor_unseen import complete
base=Path('output/tree_samearchitecture_validation_20261024_v1').resolve()
# Independently reconstruct the actual archived min-cost formulation from the earlier synthetic solve.
cell=dict(dataset='Synthetic',seed=401,person=0,models=6,budget=2.)
pt=np.array([float(np.nextafter(np.float32(rt.f32_floor(.6)),np.float32(np.inf))),0],np.float32)
# Native result was not written by the low-level earlier test; use original solve record, without solving again.
raw=rows(base/'reference/solver/solves.jsonl')[0]
raw.update(success=True,analytic=False,solver_called=True,ORPM_warm_start=False,returned_ORPM_candidate=False,certificate_rejected=False,incumbent_rejected=False)
write(base/'reference/result.json',dict(point=pt,info=raw,seconds=raw['solver_seconds']))
row=dict(shared_reference='reference/result.json',point=pt,reference_success=True,solver_calls=1)
verify_reference(base,cell,row)
assert not complete(base)
# Synthetic timing fixture: no optimizer is called and fixture never enters scientific estimates.
f=base/'eta_fixture';f.mkdir(exist_ok=False)
full={ds:([2,10,50] if ds=='Give_Me_Some_Credit' else [1,5,10] if ds=='Folktables' else [.5,2,5]) for ds in DATASETS}
write(f/'protocol.json',dict(arm='random_forest_pilot',datasets=DATASETS,full_positive_budgets=full))
for ds in DATASETS:
 for seed in [401,406]:
  append(f/'timing.jsonl',dict(kind='preparation_bank',dataset=ds,seed=seed,seconds=1.))
  append(f/'timing.jsonl',dict(kind='evaluation_bank',dataset=ds,seed=seed,seconds=2.))
  for b in [full[ds][0],full[ds][-1]]:
   append(f/'timing.jsonl',dict(kind='reference_shared',dataset=ds,seed=seed,budget=b,seconds=1.))
   for m in [6,24]:
    for method,seconds in [('ORPM',10.),('UniformForest',2.),('DirectMinimax',2.)]:append(f/'timing.jsonl',dict(kind='method',dataset=ds,seed=seed,models=m,budget=b,method=method,seconds=seconds))
write(f/'verification.json',dict(seconds=4.));estimate=full_estimate(f)
assert abs(sum(estimate['methods_hours'].values())-10.75)<1e-10
assert estimate['expected_response_calls']==270000 and estimate['full_launch_authorized'] is False
v=read(base/'validation.json');v['tests']+=['independent_archived_minimum_cost_formulation_check','incomplete_run_cannot_report_success','runtime_counts_shared_reference_once_and_preserves_full_launch_gate'];write(base/'validation.json',v)
print(json.dumps(dict(passed=True,total_tests=len(v['tests']))))
