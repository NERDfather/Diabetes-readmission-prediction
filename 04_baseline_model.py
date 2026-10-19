"""
Step 5: Baseline model.

Logistic regression as the reference point before trying anything fancier.
Two things make this non-trivial and worth doing carefully:

1. Grouped split: some patients have multiple encounters. A random
   row-level split would let the same patient appear in both train and
   test, leaking information. We split by patient_nbr instead.

2. Class imbalance: only 11.4% of rows are positive. We use
   class_weight="balanced" rather than artificially resampling, and we
   evaluate with precision/recall/ROC-AUC, never accuracy alone.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, RocCurveDisplay
)
import matplotlib.pyplot as plt

df = pd.read_csv("diabetic_data_clean.csv")

# ---------------------------------------------------------------------
# 1. Drop near-constant medication columns identified in step 4
# ---------------------------------------------------------------------
NEAR_CONSTANT = [
    "nateglinide", "chlorpropamide", "acetohexamide", "tolbutamide",
    "acarbose", "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]
df = df.drop(columns=NEAR_CONSTANT)

# ---------------------------------------------------------------------
# 2. Convert age bucket to an ordinal number (midpoint of range)
# ---------------------------------------------------------------------
# "[70-80)" -> 75. Preserves the natural ordering, which one-hot would lose.
def age_to_midpoint(age_bucket):
    low, high = age_bucket.strip("[)").split("-")
    return (int(low) + int(high)) / 2

df["age_numeric"] = df["age"].apply(age_to_midpoint)
df = df.drop(columns=["age"])

# ---------------------------------------------------------------------
# 3. Separate target, grouping key, and features
# ---------------------------------------------------------------------
y = df["readmit_30d"]
groups = df["patient_nbr"]
X = df.drop(columns=["readmit_30d", "patient_nbr"])

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()
print(f"Categorical features: {len(categorical_cols)}")
print(f"Numeric features: {len(numeric_cols)}")

# ---------------------------------------------------------------------
# 4. Grouped, stratified train/test split
# ---------------------------------------------------------------------
# StratifiedGroupKFold keeps patients' encounters together while still
# trying to preserve class balance across the split.
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"\nTrain: {len(X_train):,} rows ({y_train.mean():.1%} positive)")
print(f"Test:  {len(X_test):,} rows ({y_test.mean():.1%} positive)")

# Sanity check: no patient should appear in both sets
train_patients = set(groups.iloc[train_idx])
test_patients = set(groups.iloc[test_idx])
overlap = train_patients & test_patients
print(f"Patient overlap between train/test: {len(overlap)} (should be 0)")

# ---------------------------------------------------------------------
# 5. Build preprocessing + model pipeline
# ---------------------------------------------------------------------
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numeric_cols),
])

model = Pipeline([
    ("preprocess", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
])

# ---------------------------------------------------------------------
# 6. Fit and evaluate
# ---------------------------------------------------------------------
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT (test set)")
print("=" * 70)
print(classification_report(y_test, y_pred, target_names=["Not readmitted <30d", "Readmitted <30d"]))

print("=" * 70)
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
print("=" * 70)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("(rows = actual, columns = predicted; order: [0, 1])")

# ROC curve
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("Baseline Logistic Regression — ROC Curve")
plt.tight_layout()
plt.savefig("eda_outputs/baseline_roc_curve.png", dpi=120)
plt.close()
print("\nSaved: eda_outputs/baseline_roc_curve.png")
