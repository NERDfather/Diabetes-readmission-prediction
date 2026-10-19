"""
Step 8: Calibration + fairness audit of the readmission model.

Uses the SAME cleaning, the SAME patient-grouped split (random_state=42) and
the SAME Gradient Boosting settings as 05_compare_models.py, so the numbers
describe the model reported in the README.

Part A - Calibration: are predicted probabilities trustworthy?
  - calibration curve, Brier score, expected calibration error
  - NOTE: the model is trained with balanced sample weights, which inflates
    predicted probabilities (base rate ~11% but probabilities centre much
    higher). We therefore also report a recalibrated version
    (isotonic regression fit on a held-out slice of TRAIN patients).

Part B - Fairness: does performance differ across groups?
  - per group (race, gender, age band, payer_code missing vs. present):
    n, base rate, AUC, recall and false-positive rate at the "High" tier
    threshold (>= 0.30 on the model's raw probability, as in app.py)
  - a model retrained WITHOUT payer_code, to see what it was contributing

Outputs: eda_outputs/calibration_curve.png, eda_outputs/fairness_report.csv
Run from the project root:  python 08_calibration_fairness.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score, brier_score_loss

HIGH_TIER = 0.30  # same cutoff app.py uses for risk_tier == "High"

# ---------------------------------------------------------------- data prep
df = pd.read_csv("diabetic_data_clean.csv")

NEAR_CONSTANT = [
    "nateglinide", "chlorpropamide", "acetohexamide", "tolbutamide",
    "acarbose", "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]
df = df.drop(columns=NEAR_CONSTANT)


def age_to_midpoint(bucket):
    low, high = bucket.strip("[)").split("-")
    return (int(low) + int(high)) / 2


df["age_numeric"] = df["age"].apply(age_to_midpoint)
age_band_raw = df["age"].copy()
df = df.drop(columns=["age"])

y = df["readmit_30d"]
groups = df["patient_nbr"]
X = df.drop(columns=["readmit_30d", "patient_nbr"])

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
g_train = groups.iloc[train_idx]


def build_pipe(columns):
    cat = [c for c in columns if X[c].dtype == "object"]
    num = [c for c in columns if X[c].dtype != "object"]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num),
    ])
    return Pipeline([
        ("preprocess", pre),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)),
    ])


def fit(pipe, Xtr, ytr):
    w = compute_sample_weight("balanced", ytr)
    pipe.fit(Xtr, ytr, clf__sample_weight=w)
    return pipe


# ------------------------------------------------------------ main model
print("Training the reported model (all features)...")
model = fit(build_pipe(X.columns.tolist()), X_train, y_train)
p_test = model.predict_proba(X_test)[:, 1]
print(f"Test ROC-AUC: {roc_auc_score(y_test, p_test):.3f}")

# ---------------------------------------------------------- A. calibration
# Recalibrate with isotonic regression fit on a held-out slice of TRAIN
# patients (never the test set), using a model trained on the rest.
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
fit_i, cal_i = next(gss.split(X_train, y_train, g_train))
cal_model = fit(build_pipe(X.columns.tolist()), X_train.iloc[fit_i], y_train.iloc[fit_i])
p_cal_fit = cal_model.predict_proba(X_train.iloc[cal_i])[:, 1]
iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal_fit, y_train.iloc[cal_i])
p_test_cal = iso.predict(cal_model.predict_proba(X_test)[:, 1])


def ece(y_true, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -1e-9, 1 + 1e-9
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any():
            total += m.mean() * abs(y_true[m].mean() - p[m].mean())
    return total


print("\n=== CALIBRATION (test set) ===")
print(f"Observed readmission rate:           {y_test.mean():.3f}")
print(f"Mean predicted prob (as served):     {p_test.mean():.3f}")
print(f"Brier score (as served):             {brier_score_loss(y_test, p_test):.4f}")
print(f"Expected calibration error (served): {ece(y_test.values, p_test):.4f}")
print(f"Mean predicted prob (recalibrated):  {p_test_cal.mean():.3f}")
print(f"Brier score (recalibrated):          {brier_score_loss(y_test, p_test_cal):.4f}")
print(f"Expected calibration error (recal.): {ece(y_test.values, p_test_cal):.4f}")
print(f"Brier of always predicting base rate:{brier_score_loss(y_test, np.full(len(y_test), y_train.mean())):.4f}")

fig, ax = plt.subplots(figsize=(6.5, 6))
for label, p in [("As served (balanced weights)", p_test),
                 ("Recalibrated (isotonic)", p_test_cal)]:
    frac, mean_p = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
    ax.plot(mean_p, frac, marker="o", label=label)
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed readmission rate")
ax.set_title("Calibration curve (held-out patients)")
ax.legend()
fig.tight_layout()
fig.savefig("eda_outputs/calibration_curve.png", dpi=140)
plt.close(fig)

# ------------------------------------------------------------ B. fairness
meta = pd.DataFrame({
    "race": X_test["race"].values,
    "gender": X_test["gender"].values,
    "age_band": age_band_raw.iloc[test_idx].values,
    "payer_code_recorded": np.where(X_test["payer_code"].values == "Unknown",
                                    "missing", "recorded"),
})


def group_metrics(mask, p, y_true):
    yt, pp = y_true[mask], p[mask]
    n, pos = int(mask.sum()), int(yt.sum())
    row = {"n": n, "base_rate": round(pos / n, 3) if n else np.nan}
    if pos >= 20 and pos < n:
        row["auc"] = round(roc_auc_score(yt, pp), 3)
    else:
        row["auc"] = np.nan  # too few positives to say anything
    flag = pp >= HIGH_TIER
    row["flagged_high_%"] = round(100 * flag.mean(), 1)
    row["recall_at_high"] = round(flag[yt == 1].mean(), 3) if pos else np.nan
    row["fpr_at_high"] = round(flag[yt == 0].mean(), 3) if (n - pos) else np.nan
    return row


y_arr = y_test.values
rows = []
for attr in ["race", "gender", "age_band", "payer_code_recorded"]:
    for val in sorted(meta[attr].unique()):
        r = group_metrics((meta[attr] == val).values, p_test, y_arr)
        r.update({"attribute": attr, "group": val})
        rows.append(r)
report = pd.DataFrame(rows)[["attribute", "group", "n", "base_rate", "auc",
                             "flagged_high_%", "recall_at_high", "fpr_at_high"]]
report.to_csv("eda_outputs/fairness_report.csv", index=False)

print(f"\n=== FAIRNESS (test set, 'High' tier = prob >= {HIGH_TIER}) ===")
print(report.to_string(index=False))
print("\nRead recall_at_high and fpr_at_high across groups: large gaps mean "
      "the same threshold catches / wrongly flags groups at different rates.\n"
      "AUC is blank where a group has fewer than 20 positive cases.")

# ------------------------------------ C. what does payer_code contribute?
print("\nRetraining WITHOUT payer_code...")
cols_no_payer = [c for c in X.columns if c != "payer_code"]
model_np = fit(build_pipe(cols_no_payer), X_train[cols_no_payer], y_train)
p_np = model_np.predict_proba(X_test[cols_no_payer])[:, 1]
print("\n=== PAYER_CODE ABLATION ===")
print(f"AUC with payer_code:    {roc_auc_score(y_test, p_test):.3f}")
print(f"AUC without payer_code: {roc_auc_score(y_test, p_np):.3f}")
print("If the drop is tiny, payer_code can be removed at almost no cost, "
      "which removes the proxy risk.")
for attr in ["race", "payer_code_recorded"]:
    for val in sorted(meta[attr].unique()):
        m = (meta[attr] == val).values
        a = group_metrics(m, p_test, y_arr)
        b = group_metrics(m, p_np, y_arr)
        print(f"  {attr}={val}: flagged High {a['flagged_high_%']}% -> "
              f"{b['flagged_high_%']}% (with -> without payer_code)")

print("\nSaved: eda_outputs/calibration_curve.png, eda_outputs/fairness_report.csv")
