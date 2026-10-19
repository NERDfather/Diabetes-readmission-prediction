# MLOps: Deploying the Readmission Model

This document explains how the trained model from the main project (see
[README.md](README.md)) was turned into a deployable service, and how
to run and test it yourself.

## Architecture

```
diabetic_data_clean.csv
        |
        v
07_train_final_model.py  --> model_pipeline.joblib  (model + calibrator + tier cutoffs; gitignored)
        |
        v
      app.py  (FastAPI service)
        |
        +--> /health    liveness + "did the model load" check
        +--> /predict   validates input (Pydantic) -> model -> calibrate -> tier
        |
        v
    Dockerfile  --> self-contained container, runs identically anywhere
        |
        v
.github/workflows/ci.yml  --> on every push:
                                 1. run pytest against a synthetic fixture model
                                 2. build the Docker image
                                 3. boot the container and confirm /health responds
```

## Why the trained model isn't in the repo

`model_pipeline.joblib` and the raw/cleaned CSVs are excluded via
`.gitignore`. This is deliberate, not an oversight:

- The model file is a binary blob; git handles those poorly (no useful
  diffs, repo bloat over time).
- The data belongs to UCI's repository, not here -- the main README
  already explains how to download it.
- A reader who wants the real model regenerates it in one command
  (below), from the same data and code that produced the metrics
  reported in the README.

## Why CI uses a *different*, fake model

GitHub's CI runners check out this repo fresh every time -- they never
have the real model or real data, by design (see above). So CI can't
honestly test the real model's *accuracy* (that's already measured and
reported in the README, from a proper held-out test set).

What CI *can* and does test: the **service**, independent of which
model is loaded --

- Does the API start without crashing?
- Does it reject malformed input (wrong types, missing fields) before
  it reaches the model?
- Does `/predict` return a response shaped exactly as documented?
- Does the Docker image build and the container boot successfully?

`tests/generate_fixture_model.py` creates a tiny model trained on
random synthetic data with the *same column schema* as the real one --
just enough to exercise this code path. It is never used for anything
except CI.

## Calibrated probabilities and data-driven risk tiers

An audit of the first deployed version (`08_calibration_fairness.py`)
found two problems, and this design is the fix.

**Problem 1: the score was not a probability.** The model is trained with
balanced sample weights to cope with the ~11% positive class. That
inflates the raw scores: they averaged 0.46 on held-out patients whose
true readmission rate was 0.11. A user reading "43% risk" would have been
badly misled.

**Fix:** `07_train_final_model.py` trains on ~80% of patients and fits an
isotonic calibrator on the other ~20% (split by patient, so nobody is on
both sides). The API passes every raw score through the calibrator, and
`readmission_risk_probability` is now a real estimate. Held-out
calibration error fell from 0.347 to 0.008.

**Problem 2: the hand-picked tiers were useless.** The original fixed
cutoffs (0.15 / 0.30) were applied to the inflated score and put about
89% of patients in "High". They were illustrative numbers I had not
checked against the score distribution.

**Fix:** tiers are defined on the calibrated score from its own
distribution: Low = lower-risk half, Moderate = next 30%, High = top 20%.
The cutoffs are saved in the model artifact. The response also includes
`baseline_rate`, so a reader can see how a patient's probability compares
with the average, and a `note` field explains all of this.

The binary `predicted_readmission_30d` flag was removed in v2.0.0. With
calibrated probabilities, a 0.5 cutoff almost never fires, so the flag
added confusion rather than information. The tier is the field to act on,
and only as a prioritisation aid: the model's ROC-AUC is about 0.66.

## Running it yourself

### 1. Regenerate the real model (needs the cleaned data from the main project)

```bash
python 07_train_final_model.py
# -> writes model_pipeline.joblib
```

### 2. Run the API directly

```bash
pip install -r requirements_api.txt
uvicorn app:app --reload
# -> http://127.0.0.1:8000/docs for an interactive test page
```

### 3. Run it in Docker instead

```bash
docker build -t readmission-api .
docker run -p 8000:8000 readmission-api
# -> same http://127.0.0.1:8000/docs
```

### 4. Run the test suite (as CI does)

```bash
pip install pytest httpx
python tests/generate_fixture_model.py   # creates a throwaway test model
python -m pytest tests/ -v
```

## What's deliberately out of scope here

This is a portfolio project demonstrating the deployment lifecycle, not
a production clinical system. Things a real deployment would add:

- **Authentication** on the API (currently open)
- **A real database** logging every prediction for audit/monitoring,
  rather than plain text logs
- **Model monitoring for drift** -- tracking whether incoming patient
  data or prediction distributions shift over time from what the model
  was trained on
- **Autoscaling / a managed hosting platform** rather than a single
  container
- **Tier cutoffs tuned against a real cost matrix**, done with clinical
  stakeholders; the top-20% / next-30% split is a transparent default,
  not a validated choice
- **Clinical validation** before any real-world use -- see the main
  README's limitations section
