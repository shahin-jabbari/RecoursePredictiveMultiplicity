# Recourse experiments, from scratch

This is the code, data and settings edition of the V7 experiments. It includes seven dataset files as read by the original programs, the training and experiment code, and the original study settings. It contains no trained models, precomputed splits or panels, recourse results, plots, PDFs or execution logs. Some dataset files are the original **preprocessed inputs** (for example, the Folktables extract), rather than unprocessed source-agency records; see `dataset/README.txt` and `dataset/manifest.json`.

## Start here

Extract the ZIP, open a terminal in the extracted folder, and run:

```bash
conda env create -f environment.yml
conda activate recourse-from-scratch
python run.py verify
python run.py list
python run.py prepare tree-tradeoff --dataset Synthetic --seed 401
python run.py bank tree-literature --dataset Synthetic --seed 401 --people 1 --zero-only --out ../tree-check
```

The first study command trains that seed's decision trees from the local data. The second uses them to run the native tree comparison and the four literature adaptations for one person at zero budget. A zero-budget run checks the workflow; use the complete grid for substantive results.

The environment targets Python 3.9.17 and the numerical versions used for the checks. Use macOS or Linux. A Gurobi license of sufficient capacity is required for the optimization studies. Training and experiments read the included data; installing Python packages may need Internet access. Numerical libraries and solver versions can affect fitted weights, convergence and time-limited solutions.

## Regenerate one or all studies

```bash
python run.py prepare neural-generalization --dataset Synthetic --seed 401
python run.py bank linear-cost --dataset German --seed 401 --out ../german-cost
python run.py run tree-literature --out ../tree-literature
```

`prepare` trains the models and derives the inputs for one dataset and seed without running recourse. `bank` runs that dataset and seed. `run` executes the full configured grid. Both generate any missing inputs automatically. Full runs can take substantial time and disk space. Choose a new output directory for every run. To execute every study in a clean destination, use `python run_all.py --out ../recourse-all`. An optional `--orpm-from` for a tree literature bank can point to a matching, completed tree-tradeoff bank from this package.

By default newly generated model files live in **recourse-model-cache** beside the extracted folder. Set `RECOURSE_INPUT_CACHE` to an empty directory to choose a different location or start a clean training run. A shared bank is published only after training and admission checks succeed. The cache is separate from the ZIP. The program makes working copies so limiting a run to a few people cannot change a shared panel.

## Study map

| Question | Study argument | Scope |
|---|---|---|
| Convergence | `non-tree-convergence` | Linear, neural and mixed panels. The paper configuration uses five seeds and five people per seed. |
| Convergence | `tree-convergence` | Expanded decision-tree study, 200-round trajectories and gap diagnostics. |
| Cost and disparity of multiplicity | `linear-cost`, `neural-cost`, `tree-cost` | Matched singleton and joint recourse, capped costs, budget-specific success, structural and delivered-success gaps. Original group definitions and cohort weights are included. Each full command writes cost and disparity tables. |
| Same-structure generalization | `linear-generalization` | Nested logistic panels, Hessian-sampled unseen models and exact margin/minimum-cost benchmarks. |
| Retraining transfer | `linear-retraining` | Synthetic, COMPAS and Give Me Some Credit. The same generated linear actions are evaluated on Hessian, full-feature bootstrap and matched-mask bootstrap panels of 5, 10, 25 and 50 unseen models. |
| Same-architecture generalization | `neural-generalization` | Focused ten-seed, five-person experiment with 6 and 12 shallow neural networks, frozen unseen networks and MIQCP benchmarks. |
| Unseen trees and forests | `dt-generalization`, `rf-generalization` | Refit and bootstrap panels, native-reference comparisons and joint versus mean acceptance. Forest raw solver records and audited certificate decisions have separate files. |
| Non-tree trade-off | `non-tree-tradeoff` | ORPM, ADV, ROAR, **ElliCE** and the reference method for nested linear, neural and mixed sets of 6, 12 and 24 models. |
| Native tree trade-off | `tree-tradeoff` | ORPM, SingleTreeReference, UniformTree and DirectMinimax. |
| Literature-inspired tree trade-off | `tree-literature` | ORPM, **ADV-surrogate, ROAR-surrogate, RobX-constrained and OCEAN-style** at 6, 12 and 24 decision trees. The last two methods optimize against the first tree and are evaluated across the selected panel. |

The expanded trade-off studies use COMPAS, Folktables, German, Give_Me_Some_Credit, Polish and Synthetic, seeds 401–410, and twenty people per seed. Budget grids are study- and dataset-specific. Each frozen protocol supplies its own study scope. The base `RecourseModelMultiplicity-v7.ipynb` also covers the earlier baseline comparisons, duality-gap studies, diversity options and smaller profiles. Launch it from the package root with `jupyter lab`.

## How the inputs are produced

- **Linear, neural and mixed models:** fit and select the competing bank (24 logistic, 12 shallow and 12 deeper networks) using the study's training and validation rules. The original convergence routine uses its own five-person configuration. The other studies take their required model subsets and calculate cohort eligibility on the held-out split.
- **Linear and neural generalization:** construct the Hessian-based linear panel, retrain the logistic bootstrap panels, or train 50 separately initialized shallow networks on training-row bootstrap samples. The matched-mask logistic arm uses the corrected with-replacement mask rule.
- **Decision trees and forests:** train candidate trees from the seed's raw-data split; use the original validation cutoff and distinctness checks to select the nested bank. For generalization, also build the independent refit and bootstrap panels. The forest arm trains its own forests and uses the matched decision-tree participants.
- **Cost and disparity:** derive prespecified group labels, cohort additions and weights from the regenerated split. Use the original study-specific sampling and optimization rules. Numerical lower and upper cost bounds remain separate from intervals across seeds.
- **Trade-offs:** generate the native decision-tree ORPM actions before evaluating the literature adaptations on the same people, budgets and trees. The four tree adaptations are ADV-surrogate, ROAR-surrogate, RobX-constrained and OCEAN-style. The non-tree trade-off uses its own published method set, including ElliCE. No method is silently substituted across studies.

`replication/` connects the raw-data preparation steps to the study runners. `studies.json` maps the named workflows to their scientific source and settings; `source-origins.json` records source locations. `experiments/`, `snapshots/` and `support/` retain original training, solver, verification, reporting and plotting code, including historical versions. Historical launch/recovery files and their notebooks may mention paths or receipts from the original machine: use **run.py** for a new study. The root notebook is included for the earlier V7 experiments. Notebook outputs are cleared. `SHA256.json` lists file digests for this distribution.

Full cost/disparity and tree-literature runs aggregate their fresh outputs across seeds. The other portable runners produce the study records used by the supplied analysis code. No historical solver actions, results or timings are loaded; fresh outcomes have their own provenance.
