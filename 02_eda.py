"""
Step 3: Exploratory Data Analysis.

Run this after 01_load_and_inspect.py has confirmed the data loads cleanly.
Produces plots saved to an `eda_outputs/` folder so you can review them,
and prints summary stats you'll want to reference in your write-up.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

pd.set_option("display.max_columns", None)
os.makedirs("eda_outputs", exist_ok=True)

df = pd.read_csv("data/diabetic_data.csv", na_values="?", low_memory=False)

# Binarize target for easier analysis (full 3-class version stays available too)
df["readmit_30d"] = (df["readmitted"] == "<30").astype(int)

sns.set_style("whitegrid")

# ---------------------------------------------------------------------
# 1. Does "not tested" status relate to readmission?
# ---------------------------------------------------------------------
print("=" * 70)
print("A1C TEST STATUS vs 30-DAY READMISSION")
print("=" * 70)
df["A1C_tested"] = df["A1Cresult"].notna()
print(pd.crosstab(df["A1C_tested"], df["readmit_30d"], normalize="index").round(3))
print("\n-> If these rates differ meaningfully, 'was A1C tested' is informative")
print("   and A1Cresult should be recoded as a category, not dropped.")

print("\n" + "=" * 70)
print("GLUCOSE SERUM TEST STATUS vs 30-DAY READMISSION")
print("=" * 70)
df["glu_tested"] = df["max_glu_serum"].notna()
print(pd.crosstab(df["glu_tested"], df["readmit_30d"], normalize="index").round(3))

# ---------------------------------------------------------------------
# 2. Numeric feature distributions by readmission status
# ---------------------------------------------------------------------
numeric_cols = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]

print("\n" + "=" * 70)
print("NUMERIC FEATURES: mean by readmission status")
print("=" * 70)
print(df.groupby("readmitted")[numeric_cols].mean().round(2))

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for ax, col in zip(axes.flat, numeric_cols):
    sns.boxplot(data=df, x="readmitted", y=col, ax=ax,
                order=["NO", ">30", "<30"])
    ax.set_title(col)
plt.tight_layout()
plt.savefig("eda_outputs/numeric_features_by_readmission.png", dpi=120)
plt.close()
print("\nSaved: eda_outputs/numeric_features_by_readmission.png")

# ---------------------------------------------------------------------
# 3. Age distribution vs target
# ---------------------------------------------------------------------
plt.figure(figsize=(10, 5))
age_readmit = pd.crosstab(df["age"], df["readmitted"], normalize="index")
age_order = sorted(df["age"].unique(), key=lambda x: int(x.split("-")[0][1:]))
age_readmit.loc[age_order].plot(kind="bar", stacked=True, ax=plt.gca())
plt.title("Readmission status by age group")
plt.ylabel("Proportion")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_outputs/age_vs_readmission.png", dpi=120)
plt.close()
print("Saved: eda_outputs/age_vs_readmission.png")

# ---------------------------------------------------------------------
# 4. Correlation among numeric features (check for redundancy)
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 6))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation: numeric features")
plt.tight_layout()
plt.savefig("eda_outputs/numeric_correlation.png", dpi=120)
plt.close()
print("Saved: eda_outputs/numeric_correlation.png")

# ---------------------------------------------------------------------
# 5. Missingness pattern across columns
# ---------------------------------------------------------------------
missing_cols = df.columns[df.isna().any()].tolist()
plt.figure(figsize=(10, 6))
sns.heatmap(df[missing_cols].isna(), cbar=False, yticklabels=False, cmap="viridis")
plt.title("Missingness pattern (yellow = missing)")
plt.tight_layout()
plt.savefig("eda_outputs/missingness_pattern.png", dpi=120)
plt.close()
print("Saved: eda_outputs/missingness_pattern.png")

# ---------------------------------------------------------------------
# 6. Discharge disposition sanity check (the known leakage trap)
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("DISCHARGE DISPOSITION: value counts (check for 'expired'/'hospice' codes)")
print("=" * 70)
print(df["discharge_disposition_id"].value_counts().head(15))
print("\n-> Cross-reference these IDs against IDs_mapping.csv.")
print("   Rows where the patient died or was discharged to hospice should")
print("   almost certainly be EXCLUDED before modeling readmission risk —")
print("   they cannot be meaningfully 'readmitted'.")

print("\nDone. Review the PNGs in eda_outputs/ before moving to step 4 (cleaning).")
