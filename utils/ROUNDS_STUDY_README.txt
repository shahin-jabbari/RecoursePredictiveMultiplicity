V7 rounds-to-duality-gap study
=============================

Use either notebook, not both for a duplicate run:
- RecourseModelMultiplicity-v7-Rounds.ipynb: short launcher.
- RecourseModelMultiplicity-v7-Rounds-Lab.ipynb: the same experiment with
  separate cells for configuration, execution, analysis, and inspection.

Copy these notebooks and utils/recourse_rounds_study.py into the existing V7
folder. The add-on has no replacement for the ordinary V7 notebook or existing
utilities. Keep dataset/ and utils/ beside the notebooks. The complete V7
package also contains this add-on.

A new notebook normally has a separate kernel. Its setup cell imports the
existing utilities without running the earlier V7 experiments. If attached to
the already-running V7 kernel, it reuses that package and preserves the ordinary
campaign settings. The new module can be imported without restarting that kernel.
Do not run both launchers concurrently into the same resumed study directory.

Purpose
-------
Measure when the averaged-recourse duality gap falls below 0.005, whether it
stays there, and how this depends on model count and budget. Save secondary
thresholds 0.01 and 0.02, raw cross-entropy gaps, uniform-random-response gaps,
model disagreement, oracle diagnostics, solver-bound widths, and timings.

Fixed design in ALL modes
-------------------------
Maximum ORPM rounds: 200.
Measure gaps at round 1 and every 5 rounds, including round 200.
Model counts: 3, 6, 12, 24.
Logistic family: all models are affine-logit logistic classifiers.
Mixed family: equal logistic / one-hidden-layer / two-hidden-layer counts:
(1,1,1), (2,2,2), (4,4,4), (8,8,8).
Budgets: 0.5, 2, 5 for Synthetic, COMPAS, Polish, German;
2, 10, 50 for Give Me Some Credit; 1, 5, 10 for Folktables.
Loss: normalized logistic cross-entropy for both families. It is convex in
the input only for affine-logit models, not for the mixed family.
Learning rate: sqrt(log(number of models)/200), with normalized B=1.
Oracle limits: 200 SLSQP steps for convex logits; 50 for mixed neural logits.
Mixed global-bound diagnostic: one second per checkpoint solve, plus model
construction time. A full-size Gurobi license is required.
Training: 40 epochs, learning rate 0.01. NN widths remain 20 and (50,100).
Diversity: on, feature-group masks matched within 0.02 validation accuracy;
32 masked candidates, inherited matching/fallback rules.
Selection: majority, including ties. Reference selection remains optional.

Only these three dimensions change with MODE:
quick: Synthetic; 1 seed; up to 2 individuals per seed.
intermediate: Synthetic, COMPAS, Polish; 5 seeds; up to 5 individuals per seed.
full: all six datasets; 10 seeds; up to 20 individuals per seed.
Seed numbers begin at 42. Individuals are sampled in a reproducible order.
Quick plans 48 trajectories and up to 9,600 oracle rounds. A one-second bound
solve cap is not an estimate of total study runtime. Training, model encoding,
local oracles, and diagnostic solves all contribute to runtime.

Pairing and comparability
-------------------------
Train each family's largest bank once per dataset/seed. Smaller sets are nested
subsets of that same bank. Reuse them over budgets. Select individuals that meet
the rejection rule in EVERY included set, across both families, before any
recourse search. Save eligibility counts and selected IDs. Never use test labels,
recourse success, or the eventual gap to select individuals. The common cohort
is more restrictive than ordinary V7 selection; report that distinction.

For each individual/family, calculate one loss bound from the largest model set
and largest budget, and keep it fixed over the whole count/budget grid. Otherwise
changing a normalization constant could mimic faster convergence. Different
individuals/families can still have different normalizers; retain raw gaps and
the equivalent raw threshold, and avoid treating normalized cross-family values
as identical absolute accuracy targets.

The default is a full 200-round audit, even after crossing the threshold. This
shows recrossings and later deterioration, especially for averaged neural
responses. Optional STOP_ON_GAP=True stops after two consecutive checkpoint
upper bounds are <= 0.005, or after 200 rounds. It uses a genuine bound on the
gap, not a local objective, change in loss, or a small gap lower bound. Bad bounds
cannot trigger stopping. Diagnostics do not alter the ORPM warm-start sequence.
The ordinary V7 ORPM functions are unchanged.

Interpretation
--------------
At each checkpoint classify the averaged-point gap:
- upper bound <= threshold: certified below threshold, subject to numerical tolerances;
- lower bound > threshold: above threshold at this checkpoint;
- threshold between bounds: unresolved optimization interval;
- inconsistent bounds: invalid diagnostic, not convergence.
Remaining above threshold at round 200 does not show that the algorithm can
never converge. Wide bounds can reflect limited diagnostic solver effort.
A mixed-game gap can be small while the averaged-point gap stays positive.
The local optimizer's success flag is not a global optimality guarantee.
A small duality gap measures optimization accuracy, not successful recourse:
the available budget or model conflicts may still prevent joint validity.

Report first threshold crossing, two-checkpoint confirmation, later recrossing,
and earliest checkpoint after which all remaining observed checkpoints stay
below threshold. This last quantity is available only for full-horizon runs.
Unsuccessful trajectories retain missing/censored hitting times; they are not
assigned an invented hitting time of 200 or omitted from attainment fractions.
Failed solves are reported separately and prevent a round-cap recommendation
for an incomplete configuration.

recommended_round_caps.csv finds an observed cap at which at least 90% of
individuals (averaged within seed, then across seeds) remain certified at all
later measured checkpoints. If that target is not reached, no cap is suggested.
Quick estimates are exploratory, with no confidence interval for one seed.
Recommendations are conditional on the tested settings and common cohort.

Changing the nominal horizon changes a horizon-tuned learning rate. A cap
suggested here therefore applies to a prefix of this fixed-200 schedule; either
retain that schedule or validate a newly tuned learning rate before deployment.
Existing V7 validity comparisons use squared probability loss. A cross-entropy
rounds study does not, by itself, justify their iteration cap.

Outputs and recovery
--------------------
All outputs are under the V7 package's results/ folder:
results/V7/<mode>/<selection>/<diversity>/rounds_to_gap/<unique-run-id>/

protocol.json, progress.json, blocks.json: settings, source hashes, and progress.
blocks/: models, splits, matched-set metadata, eligibility audits, normalizers,
  per-round oracle events, checkpoint JSON, and compressed response/weight traces.
all_checkpoints.csv: all scalar diagnostics.
all_outcomes.csv: individual threshold outcomes and trajectory failures.
seed_summary.csv and summary.csv: equally weighted seed estimates and intervals.
threshold_attainment.csv: fraction reaching the threshold by each checkpoint.
recommended_round_caps.csv: conservative observed cap candidates or reasons none exists.
cohorts.csv and report.txt: scope, eligibility, and interpretation notes.
figures/<dataset>/ and table/<dataset>/: independent figures and tables.

RESUME=True reuses compatible completed dataset/seed blocks. It never silently
mixes different settings, source code, dataset hashes, or package versions.
Interrupted/failed blocks keep their partial traces; resuming reruns those
blocks in new subfolders. analyze(saved_directory) regenerates reporting without
training or solving anything. It reads registered blocks; interrupted unregistered
blocks remain available for manual inspection in blocks/.
