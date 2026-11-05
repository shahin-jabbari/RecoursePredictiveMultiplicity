Recourse under predictive multiplicity — V7
==========================================

Open RecourseModelMultiplicity-v7.ipynb. The package contains dataset/ and
utils/ and needs no previous experimental results or dataset downloads.
Install requirements.txt in a Python 3.11 environment and restart the kernel.
Launch Jupyter from the extracted package, or set ROOT in the first code cell.

Main controls
-------------
RUN_MODE = "quick"
CANDIDATE_SELECTION = "majority"   # or "reference"
diversity = True

All notebook experiment cells use RUN_MODE. Intermediate and full profiles
remain available; none is enabled initially. The optional extra diversity-off/on
comparison starts disabled to avoid duplicating the quick campaign.
The notebook and utility entry points default to majority rejection with
diversity enabled.

reference: select test individuals rejected by the designated reference model
(the first selected competing model by default). Feature diversity pins that
model. Other competing models need not reject the individual.

majority: select test individuals rejected by at least half the actual competing
models: 2 * rejection_count >= number_of_models. An even-size tie qualifies.
With six models, three rejections qualify; with five models, three are required.
The reference model can accept an individual who qualifies in majority mode.
A forest counts as one competing model, not as many votes as its internal trees.

Both rules treat a favorable probability below 0.5 as rejection; exactly 0.5 is
acceptance. They query held-out model predictions before optimizing recourse and
do not use test labels or recourse success to select people. Differentiable
cohorts use deterministic seed-based subsampling. Tree cohorts preserve the
V6 held-out row ordering. If fewer people qualify than requested, the actual
smaller count (including zero) is recorded without substituting ineligible rows.

The rule applies to all comparison, trade-off, generalization, and duality-gap
experiments. Generalization eligibility queries only optimization models, never
unseen evaluation models. Multiplicity burden retains the saved source cohort.
Every seed saves cohort_selection.json and cohort_selection.npz, including the
rule, selected IDs, test eligibility, and model rejection counts.

Experiments
-----------
1. Convex logistic duality gap.
2. Mixed logistic/neural duality gap.
3. Logistic baselines.
4. Unseen-model generalization.
5. Neural-network baselines.
6. Mixed-model budget/validity and model-count/validity trade-offs.
7. Differentiable multiplicity burden, using the fresh trade-off outputs.
8. Decision-tree duality gap.
9. Decision-tree and random-forest baselines.
10. Tree/forest budget/validity trade-offs.
11. Tree/forest multiplicity burden, using the fresh trade-off outputs.
12. Optional diversity-off/on comparison of convex gaps and mixed trade-offs.

There is no standalone objective-loss convergence campaign. Per-round losses
and responses are still saved as intermediate data for audit and later analysis.

Experiment summary (default quick profile)
------------------------------------------
Every enabled experiment uses Synthetic and COMPAS, seeds 42 and 43, and up to
three eligible test individuals per seed/configuration. Budgets below are in
standardized L2 distance. Baseline comparisons report joint validity (all
competing models accept the same feasible recourse), worst-model squared loss,
cost, and runtime. Confidence intervals use equally weighted seed means.

Convex logistic gap: three logistic models; budget 2; 20 rounds. Measures the
duality gap using convex cross-entropy, with optimization lower/upper bounds.

Mixed logistic/neural gap: one logistic, one single-hidden-layer network, and
one two-hidden-layer network; budget 2; 20 rounds. Measures nonconvex gaps and
separates local feasible solutions from global mixed-integer bounds.

Logistic baselines: three logistic models; budget 5. Compare ORPM, ADV, ElliCE,
and ROAR. ORPM uses 10 rounds and the gradient baseline iteration cap is 20.

Unseen-model generalization: three optimization logistic models and five unseen
logistic models; budget 2. Compare ORPM and ElliCE on unseen-model validity.
Only optimization models are used for majority eligibility; ORPM uses 10 rounds.

Neural baselines: three two-hidden-layer networks; budget 5. Compare ORPM, ADV,
ElliCE, and ROAR. ORPM uses 10 rounds; the neural MIQCP comparison is off in quick.

Mixed trade-offs: nested (2,2,2), (4,4,4), and (8,8,8) logistic/single-layer/
double-layer sets, giving 6, 12, and 24 models. Budgets are 0.5, 2, and 5.
Compare ORPM and all three differentiable baselines; ORPM uses 10 rounds.
Save validity-versus-budget for each set and validity-versus-size for each budget.

Differentiable multiplicity burden: reuse those saved trade-offs, factuals, and
models. Search each model separately (40 projected-gradient steps, two restarts)
and contrast individual-model attainability with saved joint validity.

Tree gap: three depth-two decision trees; budget 2; 10 rounds. Record squared-loss
gap bounds from the mixed-integer oracle. Like the other gap sections, report both
the average feature vector and a uniform random oracle response.

Tree/forest baselines: three competing models, separately for decision trees,
probability-averaging forests, and hard-voting forests. Forests have five internal
trees; all trees have maximum depth two. Budget 2; ORPM uses five rounds. Compare
ORPM-tree, Joint-minimax, ADV/ROAR surrogates, constrained RobX, and OCEAN-style
recourse; hard-voting forests also include DirectSAA and RobustSAA adaptations.

Tree trade-offs: the same three model families, budgets 0.5, 2, and 5, and the
same baselines. Hold each model set and eligible cohort fixed across budgets.

Tree multiplicity burden: reuse tree trade-offs; search for singleton recourse
with a 0.5-second solver cap and retain unresolved searches and solver bounds.

Optional diversity comparison: disabled initially. If enabled, rerun convex
gaps and mixed trade-offs with diversity off/on, comparing actually shared
evaluated individuals. The rest of the notebook runs with diversity on by default.

Duality-gap experiments report both:
- The gap at the arithmetic average of oracle-response feature vectors.
- The gap for a uniform random response, using the average per-model losses.
Both use average pre-update adversary weights. These are different objects for
nonconvex models. Only affine-logit logistic loss supports the averaged-feature
convex guarantee. Local neural solutions provide feasible dual upper bounds,
not global lower bounds. Separate mixed-integer bounds remain explicitly
distinguishable from statistical confidence intervals.

Cross-entropy losses are normalized by a conservative feasible-domain bound per
individual/set. Raw cross-entropy gaps are also saved. The optional off/on
comparison uses raw gaps to avoid interpreting different scales as a treatment
effect. Tree losses are squared probabilities and cannot be directly compared
numerically with cross-entropy gaps.

Quick settings
--------------
Datasets: Synthetic and COMPAS.
Seeds: 2, beginning at 42.
Individuals: up to 3 per seed/configuration.
Convex gap: 3 logistic models, 20 ORPM rounds.
Mixed gap: 1 logistic, 1 hidden-layer NN, 1 two-hidden-layer NN; 20 rounds.
Tree gap: 3 depth-two trees, 10 rounds.
Comparison ORPM: 10 rounds (trees: 5); non-tree baseline cap: 20.
Gradient comparison oracle: 25 inner steps.
Neural training: checked Adam/L-BFGS fits, with the same acceptance criteria in
all modes; see the shared classifier training update below.
LIME: 500 samples.
Mixed trade-off counts: (2,2,2), (4,4,4), (8,8,8), with nested model sets.
Mixed trade-off budgets: 0.5, 2, 5 for Synthetic and COMPAS.
Tree comparison forests: 5 depth-two trees per forest; 3 competing forests.
Tree oracle cap: 0.25 seconds per solve; mixed gap diagnostic: 1 second.

These are functionality settings, not precision-oriented paper runs. Actual
runtime depends on hardware, model admission, and solver behavior. A solver
time cap is not a wall-clock estimate for the complete experiment.

Feature diversity and comparable cohorts
---------------------------------------
Differentiable sets use validation-accuracy-matched whole feature-group masks,
including actionable groups. Defaults retain the V6-diverse candidate pool of
32, bank capacity 8, 0.02 accuracy caliper, and 0.5/0.75 retained fractions.
Unmatched slots retain their reference model; actual diversity is measured.
Decision-tree gaps incorporate the more recent feature-subset tree matching.
Tree/forest baseline and trade-off generation retain the V6 procedure.

Reference cohorts stay fixed across nested mixed-set sizes and diversity
conditions. Majority cohorts can change with model count or feature diversity.
Within one set, cohorts remain identical across budgets and compared methods.
Do not interpret majority-mode size curves as holding individuals fixed.
The optional off/on analysis pairs only actually shared evaluated individuals,
reports overlap counts, and marks comparisons with no overlap unavailable.

Saving and moving files
----------------------
The archive includes dataset/, utils/, and results/ beside the notebook.
dataset/ contains Synthetic (both sample sizes), COMPAS, Polish, German,
Give Me Some Credit, and California 2018 ACSIncome (Folktables). Input-file
checksums are verified during setup; no dataset download is required.
New outputs:
results/V7/<mode>/<reference-or-majority>/<diversity_on-or-off>/<section>/<run-id>/
Each run saves configuration, code snapshots, data splits, model parameters,
selection audits, individual outcomes, and optimizer/solver traces. Figures
and tables are saved by dataset under figures/<dataset>/ and table/<dataset>/.
Burden results have their own run IDs under the source selection-rule folder.

RESUME reuses only compatible completed jobs after checking code, data, package
versions, and settings. Mode/selection/diversity changes do not reuse incompatible
results. Fresh run IDs protect existing outputs. No V4/V5 directories are
modified. The optional moved-results path is recorded in workspace_locations.json
and LEGACY_RESULTS_ROOT in the notebook; it is not needed for fresh replication.
Keep dataset/, utils/, and the notebook together when moving this package.
Restart the kernel when switching between V6 and V7 to avoid stale imports.

Gurobi
------
The tree experiments require gurobipy with a suitable license. Global mixed
neural gap diagnostics require a full-size license, including in quick mode.
The setup preflight checks this before launching a campaign. If unavailable,
set RUN_TREES=False and RUN_MIXED_GAP=False to run convex gaps and differentiable
comparisons. INCLUDE_NEURAL_MIQCP refers to the separate comparison baseline;
it defaults off for quick mode. It is not the mixed-gap bound switch.

Interpretation and reproducibility
---------------------------------
Report equally weighted seed means and pointwise 95% Student-t intervals.
Do not treat repeated individuals as independent seeds. Two seeds are
exploratory, and zero observed seed variance does not establish certainty.
Joint validity means all competing models accept the returned feasible point.
Tree baseline ORPM retains its best-candidate return rule; duality experiments
explicitly use arithmetic averaging as the primary deterministic diagnostic.
The feature domain is the inherited continuous box/L2/immutable relaxation.
Target encodings and dataset audit notes remain as in V6; see dataset/README.txt.

Validation
----------
Run: python -m unittest discover -s utils/tests -p 'test_*.py' -v
The delivered validation.json records the checks performed on this package.
Validation runs are separate from scientific results and are not bundled as
completed V7 experiments. No large experiment is launched by creating the package.

GERMAN TARGET MAPPING (2026-10-09)
German model class 1 now means good credit (local CSV y=0). The CSV is unchanged.
Restart the notebook kernel and rerun setup before training. Previously loaded
models retain their old target semantics and must not be reused or relabeled.
Existing result folders are preserved. Campaign fingerprints include the target
mapping and utility hashes, so new campaigns do not resume old-target jobs.


Shared classifier training update (2026-10-12)
--------------------------------------------
Restart the kernel once after installing these utilities, then run this notebook from the top.
All three modes use the same classifier training criteria. Quick mode reduces the experimental workload; it does not accept a classifier merely because a short epoch budget ended.

Logistic regression minimizes mean binary cross-entropy plus L2 regularization (0.001, including the intercept), with a training-objective error bound at most 1e-8. Sampled logistic hypotheses instead pass a regularized loss-excess bound and validation-accuracy filtering. Neural networks use up to 100 Adam warmup epochs, then L-BFGS refinement and continuation checks; admission requires a measured gradient norm at most 0.001 or two sufficiently small continuation improvements. These neural checks are empirical, not a global convergence guarantee. Failed candidates are saved and excluded; an insufficient competing set is reported explicitly.

These are classifier training checks, separate from recourse-oracle optimization and ORPM duality-gap convergence. Tree/forest fitting uses its existing finite construction procedure. Every run saves the training protocol and per-model checks. Resume requires matching source, settings, and data, so runs made with the earlier fixed-epoch utilities are not reused. Earlier results remain in `results/`.
