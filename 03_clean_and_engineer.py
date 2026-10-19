"""
Step 4: Clean and engineer features.

Takes the raw diabetic_data.csv and produces a cleaned dataset ready for
modeling, saved to data/diabetic_data_clean.csv.

Every decision below is tied to something found in EDA (step 3). Where a
decision depends on a lookup table (IDs_mapping.csv), CHECK IT YOURSELF —
the comments give my best recollection of the standard mapping for this
dataset, but you should verify against your own copy before trusting it
in a portfolio piece.
"""

import pandas as pd
import numpy as np

df = pd.read_csv("diabetic_data.csv", na_values="?", low_memory=False)
print(f"Starting rows: {len(df):,}")

# ---------------------------------------------------------------------
# 1. Remove rows that cannot meaningfully be "readmitted"
# ---------------------------------------------------------------------
# discharge_disposition_id codes for expired / hospice patients.
# VERIFY these against your own IDs_mapping.csv before trusting this list.
EXPIRED_HOSPICE_CODES = [11, 13, 14, 19, 20, 21]

before = len(df)
df = df[~df["discharge_disposition_id"].isin(EXPIRED_HOSPICE_CODES)].copy()
print(f"Dropped {before - len(df):,} rows (expired/hospice discharge) -> {len(df):,} remain")

# ---------------------------------------------------------------------
# 2. Build the target
# ---------------------------------------------------------------------
df["readmit_30d"] = (df["readmitted"] == "<30").astype(int)

# ---------------------------------------------------------------------
# 3. Drop columns that are unusable or redundant
# ---------------------------------------------------------------------
# weight: 97% missing, can't impute meaningfully.
# We keep a "weight_recorded" flag in case absence itself is informative,
# then drop the raw column.
df["weight_recorded"] = df["weight"].notna().astype(int)
df = df.drop(columns=["weight"])

# encounter_id: pure identifier, no predictive content.
# readmitted: raw string version of the target, now redundant.
df = df.drop(columns=["encounter_id", "readmitted"])

# NOTE: we deliberately KEEP patient_nbr. It's not a model input, but
# you need it in step 5 for a grouped train/test split (so the same
# patient's multiple encounters don't end up split across train and
# test, which would leak information).

# ---------------------------------------------------------------------
# 4. Recode "not tested" as an explicit category, not a missing value
# ---------------------------------------------------------------------
# EDA showed a real (if modest) difference in readmission rate by test
# status, so we preserve that information as a category rather than
# imputing or dropping.
df["A1Cresult"] = df["A1Cresult"].fillna("Not tested")
df["max_glu_serum"] = df["max_glu_serum"].fillna("Not tested")

# ---------------------------------------------------------------------
# 5. Fill remaining categorical missingness with explicit "Unknown"
# ---------------------------------------------------------------------
for col in ["race", "medical_specialty", "payer_code"]:
    df[col] = df[col].fillna("Unknown")

# ---------------------------------------------------------------------
# 6. Group ICD-9 diagnosis codes into clinically meaningful categories
# ---------------------------------------------------------------------
# diag_1/2/3 have hundreds of distinct raw ICD-9 codes — too sparse to
# one-hot encode directly. We group into standard high-level categories
# (the grouping commonly used in published analyses of this dataset).
def group_icd9(code):
    if pd.isna(code):
        return "Missing"
    code = str(code)
    if code.startswith("V") or code.startswith("E"):
        return "Other"
    try:
        num = float(code)
    except ValueError:
        return "Other"
    if 390 <= num <= 459 or num == 785:
        return "Circulatory"
    if 460 <= num <= 519 or num == 786:
        return "Respiratory"
    if 520 <= num <= 579 or num == 787:
        return "Digestive"
    if 250 <= num < 251:
        return "Diabetes"
    if 800 <= num <= 999:
        return "Injury"
    if 710 <= num <= 739:
        return "Musculoskeletal"
    if 580 <= num <= 629 or num == 788:
        return "Genitourinary"
    if 140 <= num <= 239:
        return "Neoplasms"
    return "Other"

for col in ["diag_1", "diag_2", "diag_3"]:
    df[col + "_group"] = df[col].apply(group_icd9)
    df = df.drop(columns=[col])

print("\nDiagnosis group distribution (diag_1_group):")
print(df["diag_1_group"].value_counts())

# ---------------------------------------------------------------------
# 7. Feature engineering: prior healthcare utilization
# ---------------------------------------------------------------------
# EDA showed number_inpatient was the strongest single numeric signal.
# Combine the three utilization counts into one summary feature as well
# as keeping the originals — lets models use either the detail or the
# aggregate.
df["total_prior_utilization"] = (
    df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
)

# ---------------------------------------------------------------------
# 8. Flag near-constant columns (don't drop automatically — just report)
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("NEAR-CONSTANT COLUMNS (>99% one value) — consider dropping in step 5")
print("=" * 70)
for col in df.select_dtypes(include="object").columns:
    top_freq = df[col].value_counts(normalize=True).iloc[0]
    if top_freq > 0.99:
        print(f"  {col}: {top_freq:.1%} is '{df[col].value_counts().index[0]}'")

# ---------------------------------------------------------------------
# 9. Save cleaned dataset
# ---------------------------------------------------------------------
df.to_csv("data/diabetic_data_clean.csv", index=False)
print(f"\nSaved cleaned dataset: data/diabetic_data_clean.csv")
print(f"Final shape: {df.shape}")
print(f"Target balance:\n{df['readmit_30d'].value_counts(normalize=True).round(3)}")
