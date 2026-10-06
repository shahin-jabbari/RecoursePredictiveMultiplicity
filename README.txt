RECOURSE UNDER PREDICTIVE MULTIPLICITY — V6 REPLICATION PACKAGE

START FROM AN EMPTY RESULTS FOLDER
1. Extract the whole archive, keeping its directory structure.
2. Create a Python 3.11 environment. For example:
     python3.11 -m venv .venv
     source .venv/bin/activate
     python -m pip install -r requirements.txt
   On Windows, activate with .venv\Scripts\activate instead.
   Alternatively: conda env create -f environment.yml
3. From this directory run:
     python -m jupyter lab RecourseModelMultiplicity-v6.ipynb
4. Select the environment's Python kernel, then Restart Kernel and Run All Cells.
   The default RUN_MODE is "quick". No old results are needed.

CONTENTS
  RecourseModelMultiplicity-v6.ipynb  The only experiment notebook.
  dataset/                          All six datasets, source notes, and checksums.
  utils/                            Models, oracles, baselines, experiments,
                                    statistics, reporting, runtime, and tests.
  requirements.txt                 Pinned dependencies for Python 3.11.
  environment.yml                  Optional Conda environment definition.
  validation.json                  Replication-package verification record.
  package_manifest.json            Checksums of distributed files.

Only library dependencies need installation. Dataset loading is entirely local:
COMPAS, German, Polish, Give Me Some Credit, ACSIncome, and both Synthetic sample
sizes are included. The original 267 MB ACS source is represented by its filtered
ACSIncome task table, with source and transformation details in dataset/README.txt.
There are no model checkpoints or experimental results in the distributed package.

EXPERIMENTS IN NOTEBOOK ORDER
1. Mixed-model worst-loss convergence (a nonconvex heuristic diagnostic).
2. Convex logistic duality gap with independent lower/upper objective bounds.
3. Homogeneous logistic baseline comparison.
4. Generalization to unseen logistic models.
5. Homogeneous neural-network baseline comparison.
6. Mixed-model budget-validity and set-size trade-offs.
7. Multiplicity burden for the newly generated mixed-model trade-offs.
8. Tree loss and duality-gap convergence, including point versus mixture gaps.
9. Decision-tree and random-forest baseline comparisons.
10. Tree budget-validity trade-offs.
11. Multiplicity burden for the newly generated tree trade-offs.

The three tree families are individual decision trees, probability-averaging
forests, and hard-voting forests. Tree baseline names identify adaptations:
ADV-surrogate, ROAR-surrogate, RobX-constrained, OCEAN-style, Joint-minimax, and,
for hard-voting forests, RobustCF4RF-DirectSAA and RobustCF4RF-RobustSAA.

ONE CONTROL CELL
  RUN_MODE = "quick"       Options: "quick", "intermediate", "full".
  SEED = 42
  RESUME = True
  RUN_TREES = True
  INCLUDE_NEURAL_MIQCP = None
  SHOW_PLOTS = True

Use a fresh kernel if switching to another package version. Package discovery is
relative to the extracted folder, and an imported utils package from a different
location is rejected. There are no personal absolute data paths or embedded
utility-writing notebook cells.

MODES
                                   quick          intermediate       full
Seeds                              2              6                  10
Comparison instances/seed          3              15                 20
Mixed trade-off instances/seed     3              8                  8
Convergence instances/seed         3              10                 20
Tree instances/seed                3              10                 20
ORPM comparison rounds             10             35                 40
ADV/ROAR iteration cap             20             75                 100
Mixed-loss convergence rounds      20             50                 75
Convex-gap rounds                  20             100                200
Tree comparison rounds             5              15                 15
Tree-gap rounds                    10             50                 100
Gradient-oracle inner steps        25             200                200
Training epochs                    40             100                100
Training learning rate             0.01           0.01               0.01
LIME neighborhood samples          500            5000               5000
Budgets per mixed/tree grid        3              4                  5
Tree competing/unseen models       3 / 3          5 / 5              5 / 5
Maximum tree depth                 2              4                  4
Trees per forest                   5              15                 15
Solver time cap per tree call      0.25 seconds   1 second           1 second
Singleton burden PGD steps         40             200                400
Singleton burden PGD restarts      2              3                  5
Singleton tree solver cap          0.5 seconds    1 second           2 seconds

Quick is a functionality check, not paper-level inference. It uses Synthetic and
COMPAS and preserves all three mixed compositions: (2,2,2), (4,4,4), (8,8,8).
Their totals are 6, 12, and 24 models, with equal counts of logistic models,
one-hidden-layer networks (width 20), and two-hidden-layer networks (50,100).
Byte-identical training calls can reuse an in-memory training-only cache; recourse
outputs are never used to train models. Every run still saves its own checkpoints.

Intermediate uses Synthetic, COMPAS, Give Me Some Credit, and ACSIncome for
non-tree comparisons. Full includes all six recorded datasets for linear,
neural, and mixed trade-off comparisons. Tree sections use Synthetic, COMPAS,
and Give Me Some Credit in larger modes. Full convex-gap experiments use
Synthetic, COMPAS, German, and Give Me Some Credit, with six logistic models and
budget 2. Tree gap experiments use budget 2 on Synthetic/COMPAS and 10 on GMC.

The full generalization section deliberately reproduces the completed exploratory
protocol: Synthetic/COMPAS/German, 3 seeds, 10 instances per seed, 8 optimization
models, 50 unseen models, budget 2, 25 ORPM rounds, and 50 baseline iterations.
The full mixed GMC trade-offs use 100 ORPM rounds for budgets at least 50.

BUDGETS AND COMPARISONS
Dataset                 Full mixed-model budget grid
Synthetic               0.5, 1, 2, 5, 10
COMPAS                  0.5, 1, 2, 5, 10
German                  0.5, 1, 2, 5, 10
Polish                  0.5, 1, 2, 5, 10
ACSIncome               1, 2, 5, 10, 20
Give Me Some Credit     2, 5, 50, 100, 200

Quick selects grid positions 1, 3, and 4; intermediate selects 1, 3, 4, and 5.
Tree grids follow these grids except GMC uses 2, 5, 10, 20, 50 in full mode.
Homogeneous logistic comparisons use 3 models and budget 5.
Homogeneous neural comparisons use 3 networks in quick and 5 in larger modes.
Their larger-mode budgets are Synthetic/Polish: 5; COMPAS/German/ACSIncome: 5, 8;
GMC: 10, 50. Quick uses the first budget. Synthetic neural runs use 2,000 samples
in quick and 20,000 in larger modes. Other Synthetic experiments use 2,000.

SOLVER SETUP
Gurobi is an external solver dependency, not a bundled license. The quick tree
profile is deliberately small. Its availability is tested before any experiments
start. Larger models and the neural MIQCP reference require a suitable full-size
license; startup checks that capability when requested. No license keys or
credentials are included or written into the package.

The extra neural MIQCP reference is OFF by default in quick mode and ON for the
larger modes. It is available in the code and the same notebook. To request it
explicitly, set INCLUDE_NEURAL_MIQCP=True. It minimizes L1 movement subject to
joint classification and the L2 budget; this differs from ORPM's minimax loss.
To run only differentiable experiments without a solver, set RUN_TREES=False and
INCLUDE_NEURAL_MIQCP=False. The corresponding cells report their explicit skip.

Gurobi documents a restricted limit of 200 variables for models with quadratic
terms (and 2,000 variables/linear constraints otherwise):
https://support.gurobi.com/hc/en-us/articles/360051597492-How-do-I-resolve-a-Model-too-large-for-size-limited-Gurobi-license-error
Academic/full license setup:
https://support.gurobi.com/hc/en-us/articles/12684663118993-How-do-I-obtain-a-Gurobi-license

DATA AND TARGET INTERPRETATION
The package retains the target conventions of the saved numerical experiments.
German's recorded y=1 is bad credit, so those results are not favorable-credit
recourse. Polish label orientation remains unresolved. These cases are flagged
in the dataset inventory, source documentation, manifests, and figure titles;
neither is part of quick mode. Retaining them supports numerical replication,
not endorsement of their favorable-label interpretation. To study favorable
recourse, establish the target mapping and rerun affected experiments; changing
only a plot label is insufficient. See dataset/README.txt for precise details.

Repetitions are stratified holdout splits, approximately 64% training, 16%
validation, and 20% test, with training-only standardization. Factuals are test
instances rejected by the deployed/reference model, not selected by a majority
of the competing models. Actionability is a continuous model-space relaxation:
fixed coordinates are respected, but categorical/integer/causal feasibility and
historical immutability are not generally enforced. Do not interpret a change
in a historical predictor as an actionable change in a person's history.

OUTPUTS AND RESUMING
  results/v6/<mode>/<section>/<timestamp-and-id>/
    all_results.csv
    runs/<timestamp-and-id>_<experiment>/
      manifest.json, source/, dataset.npz, training/, seed_*_fold_*/
      instances.jsonl, per_seed_summary.csv, solver traces, trajectories
    figures/<dataset>/
    table/<dataset>/

Every new session has a unique timestamp plus random identifier. Different modes
use separate namespaces. RESUME=True can reuse a completed compatible job; code,
data, package versions, and settings are checked before reuse. Interrupted jobs
are rerun in a new run directory. Training and individual-recourse cache entries
are keyed by inputs and settings. Only locally generated compatible checkpoints
should be resumed. Changing RUN_MODE never converts or overwrites older results.

The two burden cells use the DataFrames returned by the current notebook's
trade-off cells. They do not search older package versions. Run the notebook in
order when starting with no results. Results saved in a previous V6 session can
be reused by rerunning the corresponding trade-off cells with RESUME=True.

STATISTICS AND REPRODUCIBILITY
Compute the metric for each instance, average within each seed, then average
seed means equally. Pointwise 95% Student-t intervals use the sample SD of seed
means. Missing metrics and zero observed seed variance are explicitly flagged.
Worst-individual validity takes its minimum across models within each seed;
burden intervals use paired seed differences. Solver bounds and statistical
confidence intervals remain separate. A failed heuristic search is not an
infeasibility certificate. Two-seed quick intervals are exploratory.

Full mode specifies the larger experiment configurations used in the accompanying
analysis, with the separate historical generalization exception above. Exact
floating-point values and time-limited solver incumbents can vary with hardware
and dependency versions. Use the pinned environment and inspect saved status,
seed counts, and bounds when comparing results. Quick outputs will differ from
full outputs by design.

CHECKS
Run the compact regression suite from this folder:
  python -m unittest discover -s utils/tests -v

validation.json records the delivered package's verification. It is a status
report, not precomputed experimental data. The notebook has no saved execution
outputs so a new researcher begins with a clean run.
