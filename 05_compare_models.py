"""
Step 6: Compare model families against the same baseline split.

Builds on step 5: same cleaning, same grouped/stratified split, so results
are directly comparable. Adds:
  - StandardScaler on numeric features (fixes the logistic regression
    convergence warning from step 5)
  - Random Forest
  - Gradient Boosting
All three evaluated with identical metrics on the identical test set.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt

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

y = df["readmit_30d"]
groups = df["patient_nbr"]
X = df.drop(columns=["readmit_30d", "patient_nbr"])

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

# Same split as step 5 (same random_state) so results are comparable
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# ---------------------------------------------------------------------
# Preprocessing: now with scaling for numeric features
# ---------------------------------------------------------------------
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols),
])

# Gradient Boosting has no class_weight param, so we compute sample
# weights manually; this achieves the same effect as "balanced".
sample_weight_train = compute_sample_weight("balanced", y_train)

models = {
    "Logistic Regression": Pipeline([
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced",
            random_state=42, n_jobs=-1)),
    ]),
    "Gradient Boosting": Pipeline([
        ("preprocess", preprocessor),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)),
    ]),
}

results = {}
plt.figure(figsize=(8, 6))

for name, pipe in models.items():
    print(f"\nTraining {name}...")
    if name == "Gradient Boosting":
        pipe.fit(X_train, y_train, clf__sample_weight=sample_weight_train)
    else:
        pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    results[name] = auc

    print(f"--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["Not readmitted", "Readmitted <30d"]))
    print(f"ROC-AUC: {auc:.3f}")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Model comparison — ROC curves")
plt.legend()
plt.tight_layout()
plt.savefig("eda_outputs/model_comparison_roc.png", dpi=120)
plt.close()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {name}: AUC = {auc:.3f}")
print("\nSaved: eda_outputs/model_comparison_roc.png")
