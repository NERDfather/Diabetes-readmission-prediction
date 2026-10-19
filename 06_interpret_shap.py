"""
Step 7: Interpret the winning model (Gradient Boosting) with SHAP.

Retrains the same pipeline from step 6 (same split, same preprocessing)
so results are directly traceable, then explains predictions on a sample
of the test set.

Install if needed: pip install shap
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
import shap
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

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols),
])

sample_weight_train = compute_sample_weight("balanced", y_train)

clf = GradientBoostingClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42
)

print("Fitting preprocessing + model...")
X_train_transformed = preprocessor.fit_transform(X_train)
clf.fit(X_train_transformed, y_train, sample_weight=sample_weight_train)

feature_names = preprocessor.get_feature_names_out()

# ---------------------------------------------------------------------
# SHAP on a sample of the test set (full test set is slow; a sample of
# a few thousand rows gives stable, representative results)
# ---------------------------------------------------------------------
X_test_transformed = preprocessor.transform(X_test)
sample_size = min(2000, X_test_transformed.shape[0])
rng = np.random.RandomState(42)
sample_idx = rng.choice(X_test_transformed.shape[0], sample_size, replace=False)

X_sample = X_test_transformed[sample_idx]
if hasattr(X_sample, "toarray"):
    X_sample = X_sample.toarray()

print(f"Computing SHAP values on {sample_size} test samples...")
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_sample)

# ---------------------------------------------------------------------
# Summary plot: which features matter most, and in which direction
# ---------------------------------------------------------------------
plt.figure()
shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=15)
plt.tight_layout()
plt.savefig("eda_outputs/shap_summary_beeswarm.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: eda_outputs/shap_summary_beeswarm.png")

# Bar plot: mean absolute SHAP value per feature (simpler to read)
plt.figure()
shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                   plot_type="bar", show=False, max_display=15)
plt.tight_layout()
plt.savefig("eda_outputs/shap_summary_bar.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: eda_outputs/shap_summary_bar.png")

# ---------------------------------------------------------------------
# Print ranked feature importance as text too, for your write-up
# ---------------------------------------------------------------------
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

print("\n" + "=" * 70)
print("TOP 15 FEATURES BY MEAN ABSOLUTE SHAP VALUE")
print("=" * 70)
print(importance_df.head(15).to_string(index=False))

print("\nSanity check: do the top features make clinical sense?")
print("(e.g. prior utilization, number of diagnoses, insulin status,")
print(" discharge disposition are plausible; an ID-like column ranking")
print(" high would be a red flag worth investigating.)")
