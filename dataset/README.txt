BUNDLED DATASETS

All inputs required by the notebook are local to this folder. manifest.json
records the source and SHA-256 checksum for each file. The data are unscaled;
standardization is fit only on each repetition's training split. The loader
returns raw targets and then applies the recorded target mapping in prepare_split.

Dataset              Usable rows   Features   Recorded target used as class 1
Synthetic              2,000           2       Generated positive Gaussian class
Synthetic (neural)    20,000           2       Same distribution, larger sample
COMPAS                 6,392           7       two_year_recid = 0
German                   600          67       Local y = 1 (bad credit; audit)
Polish                   864          64       Local y = 1 (orientation unresolved)
Give Me Some Credit  120,269          10       SeriousDlqin2yrs = 0
ACSIncome            195,665          10       Annual personal income > $50,000

Synthetic.csv and Synthetic_nonlinear.csv:
  Balanced two-class Gaussian mixtures with means (-3,-3) and (3,3), covariance
  5 I, NumPy default_rng seed 0, and a seeded permutation. Both sample sizes are
  included. Quick neural comparisons use the 2,000-row table to stay small;
  the larger neural comparisons use the 20,000-row table.

compas.csv:
  The task predicts two-year recidivism; target 0 represents no recorded
  recidivism. The processed table has seven predictors: sex=female, age,
  juv_fel_count, juv_misd_count, juvenile_crimes, priors_count, and
  current_charge_degree=felony. Race is not present in this processed table.
  The loader removes non-feature index columns, parses numeric values, removes
  incomplete rows, and balances classes with seed-42 undersampling. Age and
  sex are fixed. Other coordinates are modifiable in the numerical relaxation;
  this does not mean that historical criminal-record attributes are actionable.
  Source: https://www.kaggle.com/datasets/danofer/compass

polish-companies_clean_uncut.csv:
  Company bankruptcy prediction using 64 financial indicators. The loader
  removes incomplete/non-numeric rows and uses deterministic undersampling,
  leaving 864 observations. All financial coordinates are modifiable in the
  numerical experiment; dependencies among accounting ratios are not enforced.
  The historical experiments used local y=1 as their target. Its semantic
  orientation has not been independently established. Class counts or column
  names alone do not establish whether 1 means bankruptcy or non-bankruptcy.
  The figures explicitly retain a target-audit label.
  Source: https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data

 germanc.csv:
  Credit-risk classification: good versus bad credit. The original task has
  1,000 observations and 20 attributes; this processed file has 67 coordinates.
  Numeric complete cases and seed-42 class balancing leave 600 observations.
  Age (Attribute13), dependents (Attribute18), personal-status/sex indicators
  (Attribute9_*), and foreign-worker indicators (Attribute20_*) are fixed.
  An earlier audit matched 171 records on all 20 original UCI attributes:
  UCI good-credit class 1 mapped to local 0 (130 matches), and UCI bad-credit
  class 2 mapped to local 1 (41 matches). The historical local target 1 is
  retained for numerical replication and is explicitly flagged as bad credit.
  Source: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

GiveMeSomeCredit.csv:
  Predict serious delinquency within two years. The ten original predictors
  are retained after complete-case filtering, leaving 120,269 observations;
  classes are not balanced. Target 0 is the favorable no-serious-delinquency
  outcome. Age and NumberOfDependents are fixed. Financial/historical metrics
  remain continuous modifiable coordinates in the numerical relaxation.
  Source: https://www.kaggle.com/c/GiveMeSomeCredit

ACSIncome_CA_2018.csv.gz:
  California, 2018 one-year ACS person survey, transformed with the standard
  ACSIncome task in folktables 0.0.12. The filter retains age >16, annual personal
  income >$100, positive usual weekly hours, and person weight >=1. The target
  is personal income >$50,000. Predictors are AGEP, COW, SCHL, MAR, OCCP, POBP,
  RELP, WKHP, SEX, and RAC1P. No additional one-hot encoding or class balancing
  is applied. AGEP, MAR, POBP, RELP, SEX, RAC1P are fixed; COW, SCHL, OCCP, WKHP
  are modifiable numeric coordinates. OCCP is occupation, not industry.
  This compressed task table replaces the much larger raw survey file and
  preserves all 195,665 eligible observations used by the experiments.
  Sources: https://github.com/socialfoundations/folktables
           https://www.census.gov/programs-surveys/acs/microdata.html

DATASET USE
These files preserve the local processed inputs used by the research code;
they are not claims to reimplement all upstream raw-data cleaning decisions.
Source datasets retain their original terms. The package records provenance
and transformations but does not grant new rights in third-party data.
