"""
Smoke tests for the FastAPI service.

Run against the CI fixture model (see generate_fixture_model.py) --
these tests verify the API's CONTRACT (it starts, validates input,
returns the right shape of response), not the real model's accuracy.
That's a separate, already-documented concern (see README results).
"""

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

EXAMPLE_PATIENT = {
    "race": "Caucasian",
    "gender": "Female",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 3,
    "medical_specialty": "Unknown",
    "num_lab_procedures": 45,
    "num_procedures": 1,
    "num_medications": 15,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "number_diagnoses": 7,
    "max_glu_serum": "Not tested",
    "A1Cresult": "Not tested",
    "metformin": "No",
    "repaglinide": "No",
    "glimepiride": "No",
    "glipizide": "No",
    "glyburide": "No",
    "pioglitazone": "No",
    "rosiglitazone": "No",
    "insulin": "No",
    "change": "No",
    "diabetesMed": "Yes",
    "weight_recorded": 0,
    "diag_1_group": "Circulatory",
    "diag_2_group": "Diabetes",
    "diag_3_group": "Other",
    "total_prior_utilization": 0,
    "age_numeric": 65.0,
}


def test_root_responds():
    response = client.get("/")
    assert response.status_code == 200


def test_health_reports_model_loaded():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True


def test_predict_returns_well_formed_response():
    response = client.post("/predict", json=EXAMPLE_PATIENT)
    assert response.status_code == 200
    body = response.json()

    assert "readmission_risk_probability" in body
    assert "baseline_rate" in body
    assert "predicted_readmission_30d" not in body  # removed in v2.0.0
    assert "risk_tier" in body
    assert "note" in body

    # Probability must be a valid probability, regardless of the
    # fixture model's (meaningless) predicted value.
    assert 0.0 <= body["readmission_risk_probability"] <= 1.0
    assert body["risk_tier"] in {"Low", "Moderate", "High"}


def test_predict_ignores_removed_payer_code_field():
    # payer_code was dropped from the model; clients that still send it
    # should not break.
    patient = dict(EXAMPLE_PATIENT, payer_code="Unknown")
    response = client.post("/predict", json=patient)
    assert response.status_code == 200


def test_predict_rejects_missing_field():
    bad_patient = {k: v for k, v in EXAMPLE_PATIENT.items() if k != "age_numeric"}
    response = client.post("/predict", json=bad_patient)
    # FastAPI/Pydantic should reject this before it reaches the model.
    assert response.status_code == 422


def test_predict_rejects_wrong_type():
    bad_patient = dict(EXAMPLE_PATIENT)
    bad_patient["time_in_hospital"] = "not-a-number"
    response = client.post("/predict", json=bad_patient)
    assert response.status_code == 422
