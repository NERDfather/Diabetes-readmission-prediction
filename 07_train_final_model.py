"""
Step 1 of deployment: train the final Gradient Boosting pipeline on the
full cleaned dataset and save it as a single reusable artifact.

Why "full dataset" now, not just the training split?
Steps 5-7 used a held-out test set so we could HONESTLY measure how well
the model generalizes -- that's done, and you've already reported those
numbers (AUC 0.66-0.67) in your README. For the model you actually serve,
it's standard practice to retrain on ALL available labeled data (train +
test combined), since you're no longer evaluating it -- you're deploying
it, and more training data generally makes the final model slightly
better. The reported metrics still describe this model's expected
performance, because the pipeline and hyperparameters are unchanged.

Output: model_pipeline.joblib -- a single file containing the fitted
ColumnTransformer (encoding + scaling) AND the fitted GradientBoosting
classifier, bundled together. Loading this one file is enough to go
from raw-ish patient data to a prediction.
"""

import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

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

# patient_nbr is not a feature -- it was only needed for the grouped
# split during evaluation. Drop it here before training the final model.
y = df["readmit_30d"]
X = df.drop(columns=["readmit_30d", "patient_nbr"])

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols),
])

model = Pipeline([
    ("preprocess", preprocessor),
    ("clf", GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)),
])

sample_weight = compute_sample_weight("balanced", y)

print(f"Training final model on all {len(X):,} rows...")
model.fit(X, y, clf__sample_weight=sample_weight)
print("Done.")

# ---------------------------------------------------------------------
# Save the full pipeline, plus the exact column names/order the API
# will need to validate incoming requests against.
# ---------------------------------------------------------------------
artifact = {
    "pipeline": model,
    "feature_columns": X.columns.tolist(),
    "categorical_columns": categorical_cols,
    "numeric_columns": numeric_cols,
}
joblib.dump(artifact, "model_pipeline.joblib")
print("Saved: model_pipeline.joblib")
print(f"\nThe API will expect exactly these {len(X.columns)} input fields:")
for col in X.columns:
    print(f"  - {col}")
