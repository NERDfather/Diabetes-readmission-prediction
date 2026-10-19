"""
Step 7: train the deployed model, with calibrated probabilities.

What changed after the step-8 audit (08_calibration_fairness.py):
  1. payer_code is DROPPED. Removing it changed test AUC by 0.002
     (0.660 -> 0.658) and it was a possible site/socioeconomic proxy.
  2. Probabilities are CALIBRATED. Training with balanced sample weights
     inflated the raw scores (mean 0.46 vs. a true rate of 0.11), so the
     raw score was not a real probability. An isotonic regression, fit on
     patients the model never saw in training, maps raw scores to real
     observed readmission rates.
  3. Risk tiers are DATA-DRIVEN, not hand-picked. The old fixed cutoffs
     (0.15 / 0.30) put ~89% of patients in "High". Tiers are now defined
     on the calibrated score: Low = lower-risk half, Moderate = next 30%,
     High = top 20%. Observed readmission rate per tier is printed below.

Why not retrain on 100% of the data any more?
  Calibration needs patients the model has not seen. So the deployed model
  is trained on ~80% of patients and calibrated on the other ~20% (split by
  patient, so nobody appears on both sides). We trade a little training
  data for honest probabilities.

Output: model_pipeline.joblib (pipeline + calibrator + tier thresholds).
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score, brier_score_loss

df = pd.read_csv("diabetic_data_clean.csv")

NEAR_CONSTANT = [
    "nateglinide", "chlorpropamide", "acetohexamide", "tolbutamide",
    "acarbose", "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]
df = df.drop(columns=NEAR_CONSTANT)


def age_to_midpoint(age_bucket):
    low, high = age_bucket.strip("[)").split("-")
    return (int(low) + int(high)) / 2


df["age_numeric"] = df["age"].apply(age_to_midpoint)
df = df.drop(columns=["age"])

# patient_nbr is only needed for the grouped split; payer_code is dropped
# (see the note at the top of this file).
y = df["readmit_30d"]
groups = df["patient_nbr"]
X = df.drop(columns=["readmit_30d", "patient_nbr", "payer_code"])

numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
categorical_cols = [c for c in X.columns if c not in numeric_cols]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols),
])
model = Pipeline([
    ("preprocess", preprocessor),
    ("clf", GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)),
])

# ---- split BY PATIENT: ~80% to train the model, ~20% to calibrate it ----
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
fit_idx, cal_idx = next(gss.split(X, y, groups))
X_fit, y_fit = X.iloc[fit_idx], y.iloc[fit_idx]
X_cal, y_cal = X.iloc[cal_idx], y.iloc[cal_idx]

print(f"Training on {len(X_fit):,} rows; calibrating on {len(X_cal):,} rows "
      f"(different patients)...")
model.fit(X_fit, y_fit, clf__sample_weight=compute_sample_weight("balanced", y_fit))

raw_cal = model.predict_proba(X_cal)[:, 1]
calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, y_cal)
cal_scores = calibrator.predict(raw_cal)

# ---- data-driven tiers on the calibrated score ----
t_moderate = float(np.quantile(cal_scores, 0.50))
t_high = float(np.quantile(cal_scores, 0.80))
tiers = np.where(cal_scores >= t_high, "High",
                 np.where(cal_scores >= t_moderate, "Moderate", "Low"))

print("\nCalibration slice (same data the calibrator/tiers were fit on, so")
print("treat these as descriptive; 08_calibration_fairness.py gives the")
print("honest held-out check):")
print(f"  Raw score mean:        {raw_cal.mean():.3f}  (true rate {y_cal.mean():.3f})")
print(f"  Calibrated score mean: {cal_scores.mean():.3f}")
print(f"  Raw AUC {roc_auc_score(y_cal, raw_cal):.3f} | "
      f"Brier raw {brier_score_loss(y_cal, raw_cal):.4f} -> "
      f"calibrated {brier_score_loss(y_cal, cal_scores):.4f}")
print(f"  Tier thresholds on calibrated score: Moderate >= {t_moderate:.3f}, "
      f"High >= {t_high:.3f}")
for tier in ["Low", "Moderate", "High"]:
    m = tiers == tier
    print(f"  {tier:<9} {m.mean():>5.1%} of patients, "
          f"observed readmission rate {y_cal[m].mean():.3f}")

artifact = {
    "pipeline": model,
    "calibrator": calibrator,
    "tier_thresholds": {"moderate": t_moderate, "high": t_high},
    "baseline_rate": float(y.mean()),
    "feature_columns": X.columns.tolist(),
    "categorical_columns": categorical_cols,
    "numeric_columns": numeric_cols,
}
joblib.dump(artifact, "model_pipeline.joblib")
print("\nSaved: model_pipeline.joblib")
print(f"\nThe API will expect exactly these {len(X.columns)} input fields:")
for col in X.columns:
    print(f"  - {col}")
