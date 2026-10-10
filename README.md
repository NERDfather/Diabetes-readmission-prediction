# Predicting 30-Day Hospital Readmission in Diabetic Patients

A classical machine learning project predicting whether a diabetic patient will be
readmitted to hospital within 30 days of discharge, using the
[Diabetes 130-US Hospitals (1999-2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
dataset from the UCI Machine Learning Repository.

## Why this problem matters

Hospitals in the US face financial penalties for high 30-day readmission rates
under the Hospital Readmissions Reduction Program. Being able to flag high-risk
patients before discharge lets care teams target follow-up resources where
they're most needed. This project treats it as a binary classification problem:
will this patient be readmitted within 30 days (`1`) or not (`0`)?

## Dataset

- ~101,766 hospital encounters for diabetic patients across 130 US hospitals, 1999-2008
- 50 raw features: demographics, admission/discharge details, diagnoses,
  lab results, medications, and prior utilization history
- Genuinely messy: missing values encoded as `"?"`, administrative ID columns
  requiring a separate lookup table, sparse ICD-9 diagnosis codes, and a
  known leakage trap (see **Data cleaning** below)

## Methodology

### 1. Exploratory data analysis
- Identified that `weight`, `max_glu_serum`, and `A1Cresult` are missing not
  at random — absence itself (e.g. "test not performed") carries signal
- Found `number_inpatient` (prior inpatient visits) to be the strongest
  single numeric predictor in raw EDA — readmitted patients averaged 1.22
  prior inpatient visits vs. 0.38 for those not readmitted
- Confirmed `discharge_disposition_id` includes codes for patients who died
  or were discharged to hospice — these encounters cannot be meaningfully
  "readmitted" and were excluded before modeling

### 2. Data cleaning
- Removed 2,423 encounters (2.4%) corresponding to expired/hospice discharge
- Recoded lab test results as an explicit `"Not tested"` category rather
  than imputing, preserving the missingness signal found in EDA
- Grouped sparse ICD-9 diagnosis codes into clinical categories
  (circulatory, respiratory, diabetes, etc.)
- Engineered `total_prior_utilization` (sum of prior outpatient, emergency,
  and inpatient visits)
- Dropped 15 medication columns that were >99% constant and added no signal

### 3. Modeling
- **Grouped, stratified train/test split** (`StratifiedGroupKFold`) to
  prevent the same patient's multiple encounters from leaking across
  train and test
- Addressed class imbalance (11.4% positive) via `class_weight="balanced"`
  / sample weighting rather than resampling
- Compared three model families on an identical split:

| Model | ROC-AUC | Recall (readmitted) | Precision (readmitted) |
|---|---|---|---|
| Logistic Regression | 0.655 | 0.54 | 0.17 |
| Random Forest | 0.662 | 0.53 | 0.18 |
| **Gradient Boosting** | **0.674** | 0.60 | 0.17 |

Gradient Boosting was selected as the final model based on the best AUC
and recall on the minority class.

### 4. Interpretation (SHAP)
Used SHAP to confirm the model learned clinically plausible patterns rather
than noise. Top predictors: prior inpatient utilization, discharge
disposition, total prior utilization, number of diagnoses, and diabetes
medication/insulin status — consistent with established clinical risk
factors for readmission.

**Caveat worth noting:** `payer_code` being missing ranked unexpectedly
high in feature importance. This may reflect administrative or
data-collection patterns (e.g. correlated with hospital site or
socioeconomic factors) rather than a direct clinical signal — a fairness
consideration that would need further investigation before any real-world
use.

## Results and limitations

An AUC of ~0.67 is consistent with published benchmarks on this dataset
and reflects a genuinely hard prediction problem: 30-day readmission
depends heavily on factors not captured in administrative/billing data
(e.g. social support, medication adherence, post-discharge complications).

At the default 0.5 threshold, the model catches ~60% of true readmissions
at the cost of many false positives (precision ~0.17). In a real deployment,
the decision threshold would be tuned against the actual cost of a missed
readmission vs. an unnecessary follow-up intervention — not left at a
default cutoff.

**This model is a portfolio/learning project, not a validated clinical
tool**, and should not be used for real patient care decisions without
substantial further validation.

## What I'd do with more time

- Tune the Gradient Boosting hyperparameters (currently default-ish) with
  proper nested cross-validation
- Investigate the `payer_code` finding further — check whether it's a
  hospital-site proxy and whether removing it changes fairness across
  patient subgroups
- Try threshold tuning against an explicit cost matrix
- Deploy as a simple API with monitoring (planned next step)

## Project structure

```
├── data/
│   ├── diabetic_data.csv          # raw data (not included, see Setup)
│   └── diabetic_data_clean.csv    # cleaned data (generated by scripts)
├── eda_outputs/                   # plots generated during EDA and modeling
├── 01_load_and_inspect.py
├── 02_eda.py
├── 03_clean_and_engineer.py
├── 04_baseline_model.py
├── 05_compare_models.py
├── 06_interpret_shap.py
├── requirements.txt
└── README.md
```

## Setup

1. Download `diabetic_data.csv` and `IDs_mapping.csv` from the
   [UCI repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
   and place them in a `data/` folder.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the scripts in order (01 through 06).

## Author

Franklyn Epidi
