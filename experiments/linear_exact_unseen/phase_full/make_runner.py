from pathlib import Path
p=Path(__file__).parent;s=(p/'pilot_engine.py').read_text();start=s.index('def run(root,cfg):');end=s.index("if __name__=='__main__':")
body=s[start:end]
body=body.replace("inventory=prepare(root,cfg);count=0;start=time.perf_counter();tol=cfg['solver']['evaluation_logit_tolerance']", "reused,skip=prepare(root,cfg);count=len(reused);new_count=0;start=time.perf_counter();tol=cfg['solver']['evaluation_logit_tolerance']")
body=body.replace("        for ds in cfg['datasets']:","        for row in reused:records.write(json.dumps(clean(row),allow_nan=False)+'\\n')\n        records.flush()\n        print(utc(),'PILOT_PRESERVED',len(reused),'records',flush=True)\n        for ds in cfg['datasets']:",1)
old="""                        minpoint,mininfo=exact_linear.min_l2(A[:n],b[:n],x0,free,cfg['solver'],env)
                        setting=directory/'cases'/f'person_{person:02d}_n_{n}';setting.mkdir(parents=True)
                        save(setting/'minimum_cost.json',dict(point=minpoint,info=mininfo))"""
new="""                        setting=directory/'cases'/f'person_{person:02d}_n_{n}';setting.mkdir(parents=True,exist_ok=True)
                        if (setting/'minimum_cost.json').exists():
                            cached=read(setting/'minimum_cost.json');minpoint=np.asarray(cached['point']);mininfo=cached['info']
                        else:
                            minpoint,mininfo=exact_linear.min_l2(A[:n],b[:n],x0,free,cfg['solver'],env)
                            save(setting/'minimum_cost.json',dict(point=minpoint,info=mininfo))"""
assert old in body;body=body.replace(old,new)
body=body.replace("                            paths=setting/f'budget_{budget_index}';paths.mkdir()", "                            if (ds,seed,person,n,B) in skip:continue\n                            case_start=time.perf_counter()\n                            paths=setting/f'budget_{budget_index}';paths.mkdir()")
body=body.replace("                            for row in rows:records.write(json.dumps(clean(row),allow_nan=False)+'\\n');count+=1", "                            for row in rows:records.write(json.dumps(clean(row),allow_nan=False)+'\\n');count+=1;new_count+=1\n                            with open(root/'timing.jsonl','a') as timing:timing.write(json.dumps(dict(dataset=ds,seed=seed,person=person,model_count=n,budget=B,seconds=time.perf_counter()-case_start))+'\\n')")
body=body.replace("completed_records=count,expected_records=cfg['expected_records'],dataset=ds", "completed_records=count,preserved_records=len(reused),new_records=new_count,expected_records=cfg['expected_records'],dataset=ds")
body=body.replace("    import analyze\n    analyze.analyze(root)", "    if not cfg.get('validation',False):\n        save(root/'progress.json',dict(status='analyzing',completed_records=count,expected_records=count,verification_passed=verification['passed'],updated_utc=utc()))\n        import analyze\n        analyze.analyze(root)")
body=body.replace("'PILOT_COMPLETED'", "'FULL_STUDY_COMPLETED' if not cfg.get('validation',False) else 'VALIDATION_COMPLETED'")
header="""import os,sys,time,traceback,shutil,json,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'frozen_utils'))
import numpy as np
import torch
from utils import recourse_core as rc,recourse_baselines as bas
from utils.recourse_convergence import ConvexLogitOracle
from baseline_feasibility import repair_baseline_point
from common import *
from pilot_engine import load_bank,orpm
from expansion import prepare
import exact_linear

"""
footer=s[end:]
footer=footer.replace("args=ap.parse_args();root=Path(args.root)", "args=ap.parse_args();root=Path(args.root)")
(p/'runner.py').write_text(header+body+footer)
