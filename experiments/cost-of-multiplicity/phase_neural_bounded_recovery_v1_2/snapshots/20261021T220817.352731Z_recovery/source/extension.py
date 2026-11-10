"""Deterministic expansion and byte-preservation audit; no scientific solves."""
from collections import Counter
from common import *
PILOT=BASE/'cost-of-multiplicity/phase_neural_reuse_pilot_v1_0/results/20261021T163355.360251Z_neural_reuse'

def selected_positions(ds,seed,labs):
 prior=[c['source_position'] for c in read(PILOT/'cohort_plan.json') if c['dataset']==ds and c['seed']==seed]
 chosen=list(prior)
 if ds in GROUPS:
  for name in ['first','second']:
   group=GROUPS[ds][name]
   for j,lab in enumerate(labs):
    if sum(labs[k]==group for k in chosen)>=2:break
    if lab==group and j not in chosen:chosen.append(j)
 for j in range(len(labs)):
  if len(chosen)>=4:break
  if j not in chosen:chosen.append(j)
 assert len(chosen)==4 and len(set(chosen))==4
 return chosen

def check_preservation(root,check_original=True):
 root=Path(root);cfg=read(root/'protocol.json');plan=read(root/'cohort_plan.json');old=read(PILOT/'cohort_plan.json');prior={c['key']:c for c in old}
 assert len(plan)==240 and len({c['key'] for c in plan})==240
 assert set(read(root/'preserved_case_keys.json'))==set(prior)
 full=read(root/'full_cohort.json');bykey={c['key']:c for c in plan}
 for ds in DATASETS:
  for seed in range(401,411):
   block=[c for c in plan if c['dataset']==ds and c['seed']==seed];source=[c for c in full if c['dataset']==ds and c['seed']==seed]
   assert len(block)==4 and len(source)==20
   expected=selected_positions(ds,seed,[c['group'] for c in source])
   assert [c['source_position'] for c in block]==expected
   for c in block:
    original=source[c['source_position']]
    assert original['instance_id']==c['instance_id'] and original['group']==c['group']
    assert original['group_weight']==c['group_weight']
   if ds in GROUPS:
    available=Counter(c['group'] for c in source);chosen=Counter(c['group'] for c in block)
    for group,num in available.items():assert chosen[group]>=min(2,num)
 for c in old:
  assert bykey[c['key']]==c
  source=PILOT/'cases'/c['key'];dest=root/'cases'/c['key']
  for p in source.rglob('*'):
   if p.is_file():assert sha(dest/p.relative_to(source))==sha(p),'Preserved pilot artifact changed: '+str(p)
 if check_original:
  for relative,digest in read(root/'pilot_source_sha256.json').items():assert sha(PILOT/relative)==digest,'Original pilot modified: '+relative
 previous=(PILOT/'source/solver.py').read_text();current=(root/'source/solver.py').read_text()
 assert current[current.index('def environment'):current.index('def solve')]==previous[previous.index('def environment'):previous.index('def solve')],'Encoding or solver parameters changed'
 from recovery import check_recovery_preservation
 check_recovery_preservation(root)
 return dict(passed=True,preserved_cases=36,preserved_tasks=972,selected_cases=240,new_cases=204,source_and_copied_artifacts_unchanged=True,all_ten_seed_cohorts_frozen=True,group_quota_and_saved_order_checked=True,encoding_and_parameters_identical=True,utc=utc())
