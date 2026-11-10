from pathlib import Path
import json,shutil
p=Path(__file__).resolve().parent
source=Path('/Users/sj987/Documents/ChatGPT/Code/Predictive_Multiplicity_Recourse/V7/experiments/trade-off_paper/phase_10x20/results/20261017T004739.247649Z_6ffc650a4c')
config=dict(study='V7_linear_exact_unseen_pilot',protocol_version='1.0',source_run=str(source),datasets=['Synthetic','COMPAS','Give_Me_Some_Credit'],seeds=[401,402,403],instances=5,model_sizes=[6,12,24],budgets={'Synthetic':[0,.5,1.5,3],'COMPAS':[0,.5,2,5],'Give_Me_Some_Credit':[0,5,25,75]},methods=['ORPM','ADV','ROAR','ElliCE','Reference','ExactMaxMargin','ExactMinL2'],rounds={'default':200,'Give_Me_Some_Credit':1000},raw_outer_rate=1.,oracle_maxiter=100,baseline_maxiter=500,reference_maxiter=300,baseline_lr=.01,ellice_epsilon=.05,cohort_order_seed_offset=85000,unseen_count=50,unseen_seed_offset=50000000,unseen_max_attempts=10000,unseen_epsilon=.05,validation_accuracy_allowance=.05,solver={'time_limit_seconds':30,'feasibility_tolerance':1e-9,'barrier_tolerance':1e-10,'certificate_tolerance':1e-5,'evaluation_logit_tolerance':1e-7},expected_records=3780,expected_settings=540,ci={'level':.95,'unit':'equally weighted seed means','method':'pointwise Student-t, df=2; paired differences within seed','independent_seeds':3},readiness={'all_records_required':True,'all_exact_certificates_required':True,'all_source_hashes_unchanged':True,'all_unseen_panels_admissible_and_disjoint':True,'all_returns_feasible':True,'all_cohort_prefixes_reproducible':True,'independent_verification_passed':True,'scientific_claims':'pilot descriptive only; no outcome threshold is used to decide engineering readiness'})
(p/'protocol.json').write_text(json.dumps(config,indent=2)+'\n')
full=json.loads((source/'config.json').read_text());proposal=dict(config);proposal.update(study='V7_linear_exact_unseen_full_PROPOSED_NOT_AUTHORIZED',datasets=full['datasets'],seeds=full['seeds'],instances=20,budgets=full['budgets'],expected_records=6*10*20*3*7*7+2*10*20*3*7,expected_settings=6*10*20*3*7+2*10*20*3,status='PROPOSAL_ONLY_DO_NOT_LAUNCH');proposal['ci']=dict(config['ci'],independent_seeds=10,method='pointwise Student-t, df=9; paired differences within seed')
(p/'full_study_PROPOSED.json').write_text(json.dumps(proposal,indent=2)+'\n')
shutil.copytree(source/'source'/'utils',p/'frozen_utils'/'utils',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
shutil.copy2(source/'source'/'baseline_feasibility.py',p/'baseline_feasibility.py')
(p/'PROTOCOL.txt').write_text('''V7 EXACT LINEAR AND UNSEEN-MODEL PILOT — FROZEN PROTOCOL v1.0

Scope and design
Three datasets: Synthetic, COMPAS, Give Me Some Credit (GMC). Three seeds
401–403; five test individuals per dataset/seed. Nested optimization sets of
6/12/24 saved logistic models; four budgets per dataset (protocol.json).
45 individuals, 540 person/count/budget settings, 3,780 method records.
The full-study proposal is saved separately and is NOT authorized to run.

Input preservation and cohort
Reuse the verified 10x20 study's split, training-only standardization and first
24 saved linear hypotheses. No neural training or existing-result modification.
Full-feature central logistic fit is model 0; feature masks in other optimization
models are retained. Eligible test individuals have at least half of each
6/12/24 optimization set rejecting at logit<0 (ties in the vote included).
Use RNG seed+85000 to permute eligible indices and take the first five. Save
full order; a future 20-person cohort must preserve this prefix. Neither unseen
predictions nor method outcomes enter selection. This is a new linear-only
cohort; it is not a paired comparison to the previous mixed-family study.

Action space and exact comparators
Continuous standardized features, L2 intervention budget and saved immutable
features. No rounding or additional plausibility bounds. Nominal favorable
logit is >=0. ExactMaxMargin maximizes the minimum optimization-model logit
under the same L2 ball, equivalently minimizing worst raw logistic loss.
ExactMinL2 minimizes L2 cost subject to every optimization logit >=0; solve once
per individual/count, return the solution when its cost is within budget.
If unavailable, record no returned intervention and validity zero; save the
factual as a placeholder with available=false. At B=0 every method returns the
factual as an explicit singleton control. Cost of unavailable interventions is
not interpreted as a successful zero-cost solution.
These are convex QP/SOCP benchmarks solved with Gurobi. There are no binary
variables for affine models and continuous actions. The repository's legacy
MILP wrapper is NOT used as an equivalent comparator: it minimizes L1 under an
L2 constraint and imposes a positive logit margin. A genuine MILP comparison
would require a different, explicitly frozen discrete/L1 action space.

Optimality and numerical evaluation
Max-margin: feasible-point lower bound and independent simplex-weight upper
bound w'z+B||A_free^T w||. Minimum cost: .5||delta||^2 primal and nonnegative
multiplier dual -lambda'z-.5||A_free^T lambda||^2. Every bound is recomputed by
an independent evaluator. Infeasibility requires Gurobi and independent HiGHS
linear-feasibility agreement. Time limit 30 seconds per solver invocation;
solver tolerances and scaled certificate gap 1e-5 are in protocol.json.
Common evaluation is float64 at logit>=-1e-7; strict >=0 is also reported,
including boundary-sensitive decisions. Numerical-only repair of baseline
float32 budget drift follows the prior study's frozen repair utility; larger
violations fail the run. Both raw and repaired points are saved.

Approximate methods
ORPM: raw logistic loss, multiplicative weights rate 1, arithmetic average of
all responses; 200 rounds, GMC 1000; SLSQP oracle cap 100 with ftol=1e-10.
ADV: 500 Adam steps, step size .01*budget; best worst-loss point returned.
ROAR: exact affine parameter-box baseline, delta=max coordinate difference
from reference over the optimization set, SLSQP cap 500, tol 1e-4.
ElliCE: deployed-reference training Hessian ellipsoid epsilon=.05 and ridge
1e-5, frozen 500-step SLSQP minimum-cost routine. Reference: reference-model
logistic solve, cap 300 (factual if already accepted). B=0 bypasses optimizers.
All reused utility source is snapshotted and hashed before pilot evaluation.

Unseen model panel and leakage prevention
For each dataset/seed, admit 50 FRESH full-feature logistic parameter draws
from the training Hessian ellipsoid (epsilon .05) around the pinned central
regularized logistic fit. Hessian includes l2=.001 on weights AND intercept.
Candidate RNG seed=50000000+10000*seed+attempt, attempt starting at zero.
Normal direction, radius sqrt(2*.05)*U^(1/d). Quantize theta to float32 before
admission. Require actual regularized training excess over the certified global
minimum upper bound <=.05+1e-8 and validation accuracy >= central accuracy-.05.
Maximum 10,000 attempts; fail rather than relaxing admission. Exact parameter
hashes must differ from all 24 optimization models and one another. Save every
attempt's admission data. All nine panels are frozen BEFORE recourse starts.
Unseen models never enter cohort selection, solver input, tuning or stopping.
The same 50-model panel is used for every method, count, individual and budget.
These are hypotheses sharing training data/center, NOT 50 independent dataset
fits. This tests conditional within-family transfer to fresh unmasked logistic
hypotheses, not unseen feature-mask distributions, neural models, independent
training samples or external populations. Optimization sets include masked
hypotheses, so seen/unseen validity differences also reflect set composition.

Outcomes and uncertainty
Report seen and unseen joint validity, mean model acceptance, minimum logit,
worst logistic loss, L2 cost, output availability and wall time. Report paired
unseen-minus-seen change, paired method differences, exact worst-loss gaps,
minimum-cost bound and zero-threshold sensitivity. Primary denominator includes
all five selected individuals per seed; failures/unavailable outputs remain.
Conditional costs are labeled; no success-only primary validity comparison.
Every inferential figure and table includes pointwise 95% Student-t CIs from
three equally weighted seed means (df=2). Pair differences within seed first.
Bounded-rate plot limits may clip CIs to [0,1]; raw limits are retained in CSV.
Counts/settings are exact design facts, not estimates needing CIs. The 50
hypotheses and repeated budgets are NOT independent replicates. With three
seeds, CIs are unstable and descriptive; no multiplicity-adjusted claims.
Each figure panel is saved separately as PNG and PDF; no combined panels.

Readiness gates (frozen before outcomes)
All 3,780 records complete; input hashes unchanged; all points feasible; all
50-model panels admissible and disjoint; cohort prefixes reproducible; all
exact bounds/infeasibility checks pass; independent verification and report
archive complete. Any unresolved gate means NOT READY without repair. Outcome
quality is summarized but not used to tune this pilot. Passing gates supports
running the proposed larger study, not claiming scientific generalization.
''')
(p/'README.txt').write_text('''V7 linear exact-comparison and unseen-model experiments

PROTOCOL.txt and protocol.json: frozen pilot design.
full_study_PROPOSED.json: future design only; do not launch without approval.
results/<timestamp>/: self-contained pilot outputs, source snapshot, input
hashes, cohorts, unseen panels, exact certificates, saved intervention points,
records.jsonl, independent verification, report.txt, report.html, individual
figure panels and CI tables. figures_tables.zip bundles presentation outputs.
CURRENT_RUN.txt points to the latest pilot attempt. Prior attempts are retained.

Pilot entry point: launch.py. It refuses to reuse a result directory. No
automatic resume or extension. Analysis and verification do not solve recourse.
''')
