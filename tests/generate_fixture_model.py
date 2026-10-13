"""
Generates a tiny, throwaway model purely for CI testing.

Why this exists: the real model_pipeline.joblib and the real data are
deliberately excluded from the repo (see .gitignore) -- they're either
too large, or in the data's case, something others should download
fresh from the source rather than trust a committed copy of. That means
GitHub's CI runners have neither when they check out this repo.

CI's job here is NOT to verify the model is accurate (that's already
honestly reported in the README from real evaluation). CI's job is to
verify the SERVICE is correct: does it start, does it validate input,
does it return a well-formed response. A tiny model trained on random
synthetic data -- with the exact same schema as the real one -- is
enough to test all of that, without needing real patient data in the
repository.

This script is only ever run inside CI (see .github/workflows/ci.yml),
never as part of the real training pipeline.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

rng = np.random.RandomState(0)
n = 200

categorical_options = {
    "race": ["Caucasian", "AfricanAmerican", "Unknown"],
    "gender": ["Female", "Male"],
    "payer_code": ["Unknown", "MC", "HM"],
    "medical_specialty": ["Unknown", "Cardiology", "Pediatrics-Endocrinology"],
    "max_glu_serum": ["Not tested", "Norm", "High"],
    "A1Cresult": ["Not tested", "Norm", "High"],
    "metformin": ["No", "Steady", "Up"],
    "repaglinide": ["No", "Steady"],
    "glimepiride": ["No", "Steady"],
    "glipizide": ["No", "Steady"],
    "glyburide": ["No", "Steady"],
    "pioglitazone": ["No", "Steady"],
    "rosiglitazone": ["No", "Steady"],
    "insulin": ["No", "Steady", "Up", "Down"],
    "change": ["No", "Ch"],
    "diabetesMed": ["No", "Yes"],
    "diag_1_group": ["Circulatory", "Diabetes", "Other"],
    "diag_2_group": ["Circulatory", "Diabetes", "Other"],
    "diag_3_group": ["Circulatory", "Diabetes", "Other"],
}

numeric_ranges = {
    "admission_type_id": (1, 8),
    "discharge_disposition_id": (1, 10),
    "admission_source_id": (1, 10),
    "time_in_hospital": (1, 14),
    "num_lab_procedures": (0, 100),
    "num_procedures": (0, 6),
    "num_medications": (0, 40),
    "number_outpatient": (0, 5),
    "number_emergency": (0, 5),
    "number_inpatient": (0, 5),
    "number_diagnoses": (1, 16),
    "weight_recorded": (0, 1),
    "total_prior_utilization": (0, 10),
    "age_numeric": (5, 95),
}

data = {}
for col, options in categorical_options.items():
    data[col] = rng.choice(options, size=n)
for col, (low, high) in numeric_ranges.items():
    data[col] = rng.randint(low, high + 1, size=n)

X = pd.DataFrame(data)
y = rng.choice([0, 1], size=n, p=[0.89, 0.11])  # mimic real class imbalance

categorical_cols = list(categorical_options.keys())
numeric_cols = list(numeric_ranges.keys())

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols),
])

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("clf", GradientBoostingClassifier(n_estimators=10, max_depth=2, random_state=0)),
])
pipeline.fit(X, y)

artifact = {
    "pipeline": pipeline,
    "feature_columns": X.columns.tolist(),
    "categorical_columns": categorical_cols,
    "numeric_columns": numeric_cols,
}
joblib.dump(artifact, "model_pipeline.joblib")
print("CI fixture model generated: model_pipeline.joblib (NOT the real trained model)")
