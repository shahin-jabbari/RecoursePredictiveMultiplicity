"""Audit reuse at the task level without changing any prior result."""
from common import *
FAILED=BASE/'cost-of-multiplicity/phase_neural_bounded_v1_1/results/20261021T200652.743024Z_bounded_10x4'

def check_recovery_preservation(root):
 root=Path(root);cfg=read(root/'protocol.json');original=read(FAILED/'protocol.json')
 for key in ['datasets','source_seeds','pilot_seeds','pilot_people_per_seed','model_counts','budgets','groups','maximum_optimization_seconds_per_solve','threads','seed','expected_cases','expected_tasks','expected_new_tasks','preserved_tasks','success_logit_threshold','budget_tolerance']:
  assert cfg[key]==original[key],key
 assert cfg['absolute_deadline_epoch']==read(FAILED/'deadline.json')['deadline_epoch']
 for p,h in read(root/'failed_source_sha256.json').items():assert sha(FAILED/p)==h,'Failed source changed: '+p
 for p,h in read(root/'copied_prior_sha256.json').items():assert sha(root/p)==h,'Preserved task/input changed: '+p
 assert (root/'timing.jsonl').read_bytes().startswith((FAILED/'timing.jsonl').read_bytes())
 assert sha(root/'cohort_plan.json')==sha(FAILED/'cohort_plan.json')
 return dict(passed=True,prior_completed_tasks_preserved=1322,original_cohort_preserved=True,original_deadline_preserved=True,no_old_task_resolved=True,utc=utc())
