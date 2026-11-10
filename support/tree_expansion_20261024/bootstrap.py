from pathlib import Path
import json,shutil,datetime
BASE=Path('/Users/sj987/Documents/ChatGPT/Code/Predictive_Multiplicity_Recourse/V7/experiments')
C=BASE/'convergence-tree/phase_main_10x3_v2_0/results/20261022T184643.666374Z_main_10x3'
T=BASE/'trade-off-trees/phase_main_10x5_v1_0/results/20261023T122509.927561Z_tree_tradeoff'
K=BASE/'cost-of-multiplicity-tree/phase_main_372_v1_0/results/20261024T123757.193816Z_tree_multiplicity'
U=BASE/'tree-samarchitecture/phase_dt_unseen_rf_pilot_v1_0/results/20261024T182750.996196Z_dt_unseen_rf_pilot'
W=Path(__file__).resolve().parent
S=W/'source'
shutil.copytree(C/'source',S,ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
p=json.loads((C/'protocol.json').read_text())
p.update(version='3.0',status='APPROVED_EXPANSION_10X20',people_per_dataset_seed=20,target_people=1200,expected_trajectories=10800,expected_rounds=2160000,expected_diagnostics=162000,expected_references=10800,
 approval='User authorized all existing tree studies expanded sequentially to ten seeds and twenty people per dataset/seed, preserving results, updating reports after each study and progress every two hours.',
 authorization_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),absolute_deadline_epoch=None,absolute_deadline_utc=None,max_wall_hours=None,user_limit_hours=None,
 runtime_policy='No new hard deadline authorized. Historical deadlines do not apply. Sequential stages with independent verification and concise-report update before next launch.',
 reuse_policy='Keep original complete convergence trajectories byte-for-byte. Continue compatible 100-round trade-off/cost prefixes without repeating their response optimizations. Reconstruct algorithm weights and saved actions; rebuild native optimizer state and reapply independently checked valid leaf exclusions. Independently compute the missing checkpoint diagnostics. Preserve old raw histories and reference outcomes; post-solve witness audits may conservatively reject contradictory bounds.',
 source_runs={'convergence':str(C),'tradeoff':str(T),'multiplicity':str(K),'unseen':str(U)},initial_eta_hours=None,initial_planning_hours=None,estimated_trajectory_hours=None,
 cohort_policy='First twenty people in each original eligible random order. No new models, reselection or outcome-based selection.',
 timing_policy='Historical solve timings calibrate rates only. Every new trajectory records new wall time; copied trajectory rows are explicitly excluded from expansion elapsed-work totals.')
(W/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
f=S/'common.py';s=f.read_text();s=s.replace(" if time.time()>=read(Path(root)/'protocol.json')['absolute_deadline_epoch']:raise RuntimeError('Original main-study deadline reached')", " limit=read(Path(root)/'protocol.json').get('absolute_deadline_epoch')\n if limit is not None and time.time()>=limit:raise RuntimeError('Study deadline reached')")
f.write_text(s)
f=S/'analyze.py';s=f.read_text().replace('10 seeds × 3 people','10 seeds × 20 people')
s=s.replace("description+=f\"Native solver statuses:", "description+='This expansion preserves completed trajectories and continues saved 100-round prefixes with independently evaluated checkpoints. Numerical bounds and sampling uncertainty are distinct.\\n\\n'\n description+=f\"Native solver statuses:")
s=s.replace('Total completed-trajectory wall time:', 'Historical plus newly recorded trajectory work (not expansion elapsed time):')
f.write_text(s)
print(W)
