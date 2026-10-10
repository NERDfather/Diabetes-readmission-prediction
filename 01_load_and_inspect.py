"""
Step 1-2: Load and inspect the Diabetes 130-US Hospitals dataset.

Download the data first from the UCI ML Repository:
  https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

You'll get a zip containing:
  - diabetic_data.csv      (the main dataset, ~100k rows)
  - IDs_mapping.csv        (lookup table decoding some ID columns)

Place both files in a `data/` folder next to this script before running.
"""

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

# ---------------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------------
# IMPORTANT: this dataset uses the literal string "?" for missing values,
# not blank cells or NaN. If you don't tell pandas this, missingness
# analysis and dtype inference will silently be wrong.
df = pd.read_csv("data/diabetic_data.csv", na_values="?")

print("=" * 70)
print("SHAPE")
print("=" * 70)
print(f"Rows: {df.shape[0]:,}   Columns: {df.shape[1]}")

# ---------------------------------------------------------------------
# 2. Basic structure
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("DTYPES")
print("=" * 70)
print(df.dtypes.value_counts())
print("\nFirst few rows:")
print(df.head())

# ---------------------------------------------------------------------
# 3. Missing values
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("MISSING VALUES (columns with any missingness)")
print("=" * 70)
missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(1)
missing_summary = (
    pd.DataFrame({"n_missing": missing, "pct_missing": missing_pct})
    .query("n_missing > 0")
    .sort_values("pct_missing", ascending=False)
)
print(missing_summary if not missing_summary.empty else "None found.")

# ---------------------------------------------------------------------
# 4. Duplicate encounters / patients
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("DUPLICATES")
print("=" * 70)
print(f"Fully duplicate rows: {df.duplicated().sum()}")
if "patient_nbr" in df.columns:
    n_patients = df["patient_nbr"].nunique()
    n_encounters = df["encounter_id"].nunique() if "encounter_id" in df.columns else len(df)
    print(f"Unique patients: {n_patients:,}   Unique encounters: {n_encounters:,}")
    print("-> Some patients appear more than once. Decide early whether you'll")
    print("   keep one encounter per patient or model at the encounter level")
    print("   (and if the latter, whether you need grouped train/test splits).")

# ---------------------------------------------------------------------
# 5. Target variable balance
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("TARGET: readmitted")
print("=" * 70)
if "readmitted" in df.columns:
    print(df["readmitted"].value_counts())
    print("\nAs proportions:")
    print(df["readmitted"].value_counts(normalize=True).round(3))
    print("\nNote: raw values are typically '<30', '>30', 'NO'.")
    print("For 30-day readmission prediction, you'll likely binarize this as:")
    print("  target = 1 if readmitted == '<30' else 0")

# ---------------------------------------------------------------------
# 6. Quick look at likely-leaky or ID-like columns worth flagging now
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("COLUMNS TO SCRUTINIZE FOR LEAKAGE / ID NOISE (flag now, handle in step 4)")
print("=" * 70)
watch_cols = [
    "encounter_id", "patient_nbr", "discharge_disposition_id",
    "admission_source_id", "admission_type_id", "payer_code",
]
for col in watch_cols:
    if col in df.columns:
        print(f"- {col}: {df[col].nunique()} unique values")
