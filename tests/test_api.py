import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_success():
    payload = {
        "payer": "Aetna",
        "provider": "Dr. Rao",
        "cpt_code": "99213",
        "diagnosis_code": "M54.5",
        "billed_amount": 220.00,
        "patient_age": 45
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_status" in data
    assert data["predicted_status"] in ["Paid", "Partially_Paid", "Denied"]
    assert "probabilities" in data
    probs = data["probabilities"]
    assert "Paid" in probs
    assert "Partially_Paid" in probs
    assert "Denied" in probs
    assert abs(sum(probs.values()) - 1.0) < 0.01

def test_predict_invalid_missing_field():
    payload = {
        "payer": "Aetna",
        "cpt_code": "99213",
        "billed_amount": 220.00
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert response.json()["error"] == "Bad Request"

def test_predict_invalid_data_type():
    payload = {
        "payer": "Aetna",
        "provider": "Dr. Rao",
        "cpt_code": "99213",
        "diagnosis_code": "M54.5",
        "billed_amount": "invalid_text_amount",
        "patient_age": 45
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400

def test_predict_out_of_bounds_age():
    payload = {
        "payer": "Aetna",
        "provider": "Dr. Rao",
        "cpt_code": "99213",
        "diagnosis_code": "M54.5",
        "billed_amount": 220.00,
        "patient_age": -5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400

def test_appeal_guidance():
    payload = {"denial_reason": "missing_auth"}
    response = client.post("/appeal-guidance", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "appeal_guidance_snippet" in data
    assert "hallucination_mitigation_rules" in data
