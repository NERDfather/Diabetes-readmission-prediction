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
    version="2.0.0",
)

# ---------------------------------------------------------------------
# Load the model once at startup, not per-request. Loading a model from
# disk takes real time; doing it on every request would make the API
# unusably slow and is a common beginner mistake.
# ---------------------------------------------------------------------
try:
    artifact = joblib.load("model_pipeline.joblib")
    pipeline = artifact["pipeline"]
    calibrator = artifact["calibrator"]
    TIER_THRESHOLDS = artifact["tier_thresholds"]
    BASELINE_RATE = artifact["baseline_rate"]
    FEATURE_COLUMNS = artifact["feature_columns"]
    logger.info(f"Model loaded successfully. Expects {len(FEATURE_COLUMNS)} features.")
except FileNotFoundError:
    pipeline = None
    calibrator = None
    TIER_THRESHOLDS = {}
    BASELINE_RATE = None
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
    risk_tier: str
    baseline_rate: float
    note: str = (
        "readmission_risk_probability is a calibrated estimate of the chance of "
        "readmission within 30 days. risk_tier ranks this patient against the "
        "population the model was calibrated on: Low = lower-risk half, "
        "Moderate = next 30%, High = top 20%. The model's discrimination is "
        "modest (ROC-AUC about 0.66), so use these as a prioritisation aid for "
        "a human, never as a clinical decision. baseline_rate is the average "
        "readmission rate in the training data."
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
        raw = pipeline.predict_proba(input_df)[0, 1]
        # The raw score is inflated by class-balanced training, so it is
        # mapped to a real probability by the calibrator before use.
        proba = float(calibrator.predict([raw])[0])
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    # Tiers come from the calibrated score's distribution (see
    # 07_train_final_model.py), not hand-picked cutoffs.
    if proba >= TIER_THRESHOLDS["high"]:
        tier = "High"
    elif proba >= TIER_THRESHOLDS["moderate"]:
        tier = "Moderate"
    else:
        tier = "Low"

    logger.info(f"Prediction made: probability={proba:.3f}, tier={tier}")

    return PredictionResponse(
        readmission_risk_probability=round(proba, 4),
        risk_tier=tier,
        baseline_rate=round(float(BASELINE_RATE), 4),
    )
