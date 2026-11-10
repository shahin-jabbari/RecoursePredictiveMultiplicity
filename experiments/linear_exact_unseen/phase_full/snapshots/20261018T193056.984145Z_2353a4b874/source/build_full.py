from pathlib import Path
import shutil,json
base=Path('/Users/sj987/Documents/ChatGPT/Code/Predictive_Multiplicity_Recourse/V7/experiments/linear_exact_unseen')
pilot=base/'results/20261018T180330.097982Z_98de9f6e9a';stage=Path(__file__).parent
for n in ['common.py','exact_linear.py','baseline_feasibility.py']:shutil.copy2(pilot/'source'/n,stage/n)
shutil.copy2(pilot/'source/runner.py',stage/'pilot_engine.py')
shutil.copytree(pilot/'source/frozen_utils',stage/'frozen_utils',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
cfg=json.loads((base/'full_study_PROPOSED.json').read_text());cfg.update(study='V7_linear_exact_unseen_full_10x20',phase='full_10x20',status='AUTHORIZED',protocol_version='1.1-expansion',pilot_run=str(pilot),preserved_pilot_records=3780,new_records=181020,expected_panels=60,cohort_model_sizes=[6,12,24],validation=False,allow_model_training=False)
cfg['readiness']['scientific_claims']='Estimate budget/count trade-offs and conditional within-family unseen transfer; no universal generalization claim.'
cfg['pilot_estimate_note']='Three-seed pilot is reused; full-study results are an expansion, not an independent replication.'
(stage/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
text=(pilot/'PROTOCOL.txt').read_text()
intro='''V7 EXACT LINEAR AND UNSEEN-MODEL STUDY — AUTHORIZED EXPANSION v1.1

This extension was authorized on 2026-10-18 after the verified pilot. The JSON
configuration is authoritative for expanded datasets, seeds, people and budgets.
Six datasets (Synthetic, COMPAS, Polish, German, GMC, ACSIncome/Folktables),
seeds 401–410, 20 people per dataset/seed, nested 6/12/24 competing models.
Seven budgets per dataset, eight for GMC and ACSIncome; explicit grids below.
184,800 method records = 3,780 preserved pilot + 181,020 new records.
26,400 person/count/budget settings. Same seven methods, iterations, solvers,
thresholds, action space and conditional unseen-model distribution as the pilot.
50 unseen hypotheses per dataset/seed; all 60 panels frozen before new solves.
Existing nine panels are reused exactly; 51 new panels follow the same rules.
No model training. All 60 existing validated optimization banks are reused.

PRESERVATION
The pilot stays untouched. Its full saved eligible order is checked against the
recomputed order. Its five people remain the first five of each 20-person cohort.
Unseen parameters and admission logs remain identical. The 3,780 pilot outputs,
traces, exact certificates and scientific record fields are copied unchanged.
Only budget_index and point_file addresses change to fit the expanded grid.
All copied files are hash-checked, all pilot rows are compared independently,
and originals are rehashed at completion. No pilot solve is repeated.
A separate phase lock prevents duplicate live launches; no automatic resume.

UNCERTAINTY AND REPORTS
Pointwise 95% Student-t CIs over ten equally weighted seed means (df=9).
Paired method/budget/model-count contrasts are differenced within seed first.
Keep the same twenty people and same fifty unseen models across each comparison.
The pilot is included, so phase differences are descriptive, not independent
replication tests. Raw CI limits are saved; rate plots clip to [0,1]. Degenerate
boundary intervals mean identical seed means, not population certainty.
Every inferential figure/table has CIs; each figure panel is a separate PNG/PDF.
Seed/raw observations and deterministic counts are audit data, not estimates.

COMPLETION
Require all 184,800 records, passed independent verification including exact
bounds, source/pilot/cohort preservation, analysis, archive and completion log.
Uncertified solver results remain visible and prevent a clean completion claim.
Pilot numerical settings and scientific definitions below remain unchanged;
their small-sample design/counts are superseded only as explicitly stated above.

EXPANDED BUDGETS (standardized L2 units)
'''
for ds,bs in cfg['budgets'].items():intro+=ds+': '+', '.join(map(str,bs))+'\n'
(stage/'PROTOCOL.txt').write_text(intro+'\n\nORIGINAL FROZEN PILOT DEFINITIONS (retained for provenance)\n\n'+text)
(stage/'README.txt').write_text('''Full linear exact-comparison and unseen-model study

protocol.json and PROTOCOL.txt freeze the authorized 6-dataset, 10-seed,
20-person, 6/12/24-model, 7–8-budget expansion. Pilot artifacts in the parent
experiment folder are preserved. No neural models are trained.

launch.py starts one fresh full-study attempt. A phase-level lock prevents
concurrent duplication. CURRENT_RUN.txt identifies the latest full attempt.
results/<timestamp>/ stores source/protocol snapshots, copied pilot artifacts,
all intervention outputs, progress and failure records, independent verification,
95% CI tables, separate figure panels, report.txt/report.html and figures_tables.zip.
validation/ contains labeled pipeline checks, not full-study results.

The unseen panel is conditional on full-feature Hessian-sampled logistic
hypotheses. This is not a test of arbitrary unseen families or new populations.
''')
# Base independent numeric verifier, with general dimensions and streaming rows.
v=(pilot/'source/verify.py').read_text().replace("for n in cfg['model_sizes']]))","for n in cfg.get('cohort_model_sizes',cfg['model_sizes'])]))")
v=v.replace("check(read(root/'panels_frozen.json')['count']==9,'nine panels frozen')","check(read(root/'panels_frozen.json')['count']==cfg['expected_panels'],'all panels frozen')")
v=v.replace("rows=[json.loads(line) for line in (root/'records.jsonl').read_text().splitlines()];seen_keys=set();files={};exact_seen=set();min_seen=set()","rows=(json.loads(line) for line in open(root/'records.jsonl'));seen_keys=set();files={};exact_seen=set();min_seen=set()")
v=v.replace("check(len(rows)==cfg['expected_records'],'expected record count')","check(counts['records']==cfg['expected_records'],'expected record count')")
(stage/'numeric_verify.py').write_text(v)
