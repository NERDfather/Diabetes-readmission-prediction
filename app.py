"""
FastAPI service for the diabetes 30-day readmission risk model.

Run locally with:
    uvicorn app:app --reload

Then open http://127.0.0.1:8000/docs for an interactive test page
(FastAPI generates this automatically from the schema below).
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import logging

# ---------------------------------------------------------------------
# Logging: minimal but real. Every prediction gets recorded, which is
# the starting point for any monitoring you'd add later (e.g. tracking
# prediction drift over time).
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("readmission_api")

app = FastAPI(
    title="Diabetes 30-Day Readmission Risk API",
    description="Predicts the probability a diabetic patient will be "
                 "readmitted to hospital within 30 days of discharge.",
    version="1.0.0",
)

# ---------------------------------------------------------------------
# Load the model once at startup, not per-request. Loading a model from
# disk takes real time; doing it on every request would make the API
# unusably slow and is a common beginner mistake.
# ---------------------------------------------------------------------
try:
    artifact = joblib.load("model_pipeline.joblib")
    pipeline = artifact["pipeline"]
    FEATURE_COLUMNS = artifact["feature_columns"]
    logger.info(f"Model loaded successfully. Expects {len(FEATURE_COLUMNS)} features.")
except FileNotFoundError:
    pipeline = None
    FEATURE_COLUMNS = []
    logger.error("model_pipeline.joblib not found. Run 07_train_final_model.py first.")


# ---------------------------------------------------------------------
# Request schema. Pydantic validates every incoming request against
# this automatically -- wrong types or missing fields get rejected
# with a clear error before they ever reach the model. This is the
# "input validation" layer a production API needs.
# ---------------------------------------------------------------------
class PatientEncounter(BaseModel):
    race: str = Field(..., examples=["Caucasian"])
    gender: str = Field(..., examples=["Female"])
    admission_type_id: int = Field(..., examples=[1])
    discharge_disposition_id: int = Field(..., examples=[1])
    admission_source_id: int = Field(..., examples=[7])
    time_in_hospital: int = Field(..., ge=1, le=14, examples=[3])
    payer_code: str = Field(..., examples=["Unknown"])
    medical_specialty: str = Field(..., examples=["Unknown"])
    num_lab_procedures: int = Field(..., ge=0, examples=[45])
    num_procedures: int = Field(..., ge=0, examples=[1])
    num_medications: int = Field(..., ge=0, examples=[15])
    number_outpatient: int = Field(..., ge=0, examples=[0])
    number_emergency: int = Field(..., ge=0, examples=[0])
    number_inpatient: int = Field(..., ge=0, examples=[0])
    number_diagnoses: int = Field(..., ge=1, examples=[7])
    max_glu_serum: str = Field(..., examples=["Not tested"])
    A1Cresult: str = Field(..., examples=["Not tested"])
    metformin: str = Field(..., examples=["No"])
    repaglinide: str = Field(..., examples=["No"])
    glimepiride: str = Field(..., examples=["No"])
    glipizide: str = Field(..., examples=["No"])
    glyburide: str = Field(..., examples=["No"])
    pioglitazone: str = Field(..., examples=["No"])
    rosiglitazone: str = Field(..., examples=["No"])
    insulin: str = Field(..., examples=["No"])
    change: str = Field(..., examples=["No"])
    diabetesMed: str = Field(..., examples=["Yes"])
    weight_recorded: int = Field(..., ge=0, le=1, examples=[0])
    diag_1_group: str = Field(..., examples=["Circulatory"])
    diag_2_group: str = Field(..., examples=["Diabetes"])
    diag_3_group: str = Field(..., examples=["Other"])
    total_prior_utilization: int = Field(..., ge=0, examples=[0])
    age_numeric: float = Field(..., examples=[65.0])


class PredictionResponse(BaseModel):
    readmission_risk_probability: float
    predicted_readmission_30d: bool
    risk_tier: str
    note: str = (
        "predicted_readmission_30d uses the standard 0.5 probability cutoff. "
        "risk_tier uses lower, more sensitive thresholds (>=0.30 = High) because "
        "this model's training data is imbalanced (~11% positive class): most true "
        "readmissions never cross 0.5 probability, so a plain 0.5 cutoff misses "
        "many real cases (see the project README's limitations section). In a real "
        "deployment, risk_tier -- not the binary flag -- is the field to act on, "
        "with the exact thresholds tuned against the real-world cost of a missed "
        "readmission vs. an unnecessary follow-up."
    )


@app.get("/")
def root():
    return {"message": "Diabetes readmission risk API. See /docs for usage."}


@app.get("/health")
def health():
    """Basic health check -- confirms the API is up and the model loaded."""
    return {
        "status": "ok" if pipeline is not None else "model not loaded",
        "model_loaded": pipeline is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientEncounter):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")

    # Build a single-row DataFrame in the exact column order the
    # pipeline was trained on.
    input_dict = patient.model_dump()
    input_df = pd.DataFrame([input_dict])[FEATURE_COLUMNS]

    try:
        proba = pipeline.predict_proba(input_df)[0, 1]
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    # Simple three-tier risk bucketing for readability. Thresholds here
    # are illustrative, not clinically validated -- say so in your docs.
    if proba < 0.15:
        tier = "Low"
    elif proba < 0.30:
        tier = "Moderate"
    else:
        tier = "High"

    logger.info(f"Prediction made: probability={proba:.3f}, tier={tier}")

    return PredictionResponse(
        readmission_risk_probability=round(float(proba), 4),
        predicted_readmission_30d=bool(proba >= 0.5),
        risk_tier=tier,
    )
