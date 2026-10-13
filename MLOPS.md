# MLOps: Deploying the Readmission Model

This document explains how the trained model from the main project (see
[README.md](README.md)) was turned into a deployable service, and how
to run and test it yourself.

## Architecture

```
diabetic_data_clean.csv
        |
        v
07_train_final_model.py  --> model_pipeline.joblib  (gitignored -- regenerate locally)
        |
        v
      app.py  (FastAPI service)
        |
        +--> /health    liveness + "did the model load" check
        +--> /predict   validates input (Pydantic) -> runs pipeline -> returns risk
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

## The two prediction thresholds

`/predict` returns both `predicted_readmission_30d` (a plain yes/no at
the standard 0.5 probability cutoff) and `risk_tier` (Low/Moderate/High,
using lower, more sensitive thresholds). These intentionally disagree
sometimes -- e.g. a patient at 43% probability gets `false` for the
binary flag but `"High"` for the tier.

This isn't a bug. The training data is imbalanced (~11% positive
class), so most true readmissions never cross 50% predicted probability
-- a plain 0.5 cutoff misses the majority of real cases (see the main
README's limitations section for the precision/recall numbers). In a
real deployment, `risk_tier` -- not the binary flag -- is the field a
care team would act on, with its exact thresholds tuned against the
real-world cost of a missed readmission versus an unnecessary
follow-up call. The API returns both, with a `note` field explaining
this, so nothing is hidden behind a single number.

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
- **Threshold tuning against a real cost matrix**, done with clinical
  stakeholders, not the illustrative cutoffs used here
- **Clinical validation** before any real-world use -- see the main
  README's limitations section
