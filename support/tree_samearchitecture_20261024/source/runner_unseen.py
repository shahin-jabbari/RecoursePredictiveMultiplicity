from experiment import *
from prepare_unseen import prepare
from evaluate_unseen import evaluate
from verify_unseen import verify
from analyze_unseen import analyze
from eta import remaining,full_estimate
import traceback,faulthandler

def run(root):
 root=Path(root);p=read(root/'protocol.json')
 assert read(root.parent/'prelaunch_verification.json')['passed']
 if (root/'freeze.json').exists():raise RuntimeError('No automatic resume or rerun')
 plan=prepare(root);write(root/'runtime_estimate.json',remaining(root))
 if p['arm']=='random_forest_pilot':
  from forest_methods import run_method
  completed=0;rounds_done=0
  for cell in plan['cells']:
   for method in p['methods']:
    result=run_method(root,cell,method);append(root/'records.jsonl',result);completed+=1
    if method=='ORPM' and cell['budget']>0:rounds_done+=100
    progress(root,status='solving',completed_method_rows=completed,expected_method_rows=288,completed_rounds=rounds_done,expected_rounds=4800)
   write(root/'runtime_estimate.json',remaining(root))
   stage(root,'FOREST_PILOT_CELL_COMPLETED',completed_method_rows=completed,completed_rounds=rounds_done)
 evaluate(root);progress(root,status='verifying');stage(root,'independent_verification_started');verify(root)
 if p['arm']=='random_forest_pilot':
  estimate=full_estimate(root);write(root/'full_study_estimate.json',estimate)
  write(root/'pilot_verification.json',dict(read(root/'verification.json'),full_launch_authorized=False,continuation='STOP; report measured estimate and wait for user authorization'))
 progress(root,status='reporting');stage(root,'analysis_and_reporting');analyze(root)
 frozen_check(root);progress(root,status='completed',completed_method_rows=plan['expected_method_rows'],completed_evaluation_rows=plan['expected_evaluation_rows'],full_forest_authorized=False)
 stage(root,'DECISION_TREE_UNSEEN_COMPLETED' if p['arm']=='decision_tree' else 'RANDOM_FOREST_PILOT_COMPLETED_STOPPED_BEFORE_FULL',method_rows=plan['expected_method_rows'],unseen_evaluation_rows=plan['expected_evaluation_rows'])
 write(root/'runtime_estimate.json',remaining(root))

if __name__=='__main__':
 root=Path(sys.argv[1]);native=(root/'native_traceback.log').open('a');faulthandler.enable(native,all_threads=True)
 try:run(root)
 except BaseException as e:
  write(root/'failure.json',dict(utc=now(),type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),stage=read(root/'stage.json') if (root/'stage.json').exists() else None));traceback.print_exc();raise
