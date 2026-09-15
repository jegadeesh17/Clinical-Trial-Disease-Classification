"""
Unit and Integration Tests for Clinical Trial Disease Classification API.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "degraded"]
    assert "classes" in data
    assert len(data["classes"]) == 8


def test_predict_endpoint_breast_cancer(client):
    payload = {
        "summary": "Evaluation of chemotherapy combination in patients with metastatic HER2-positive breast carcinoma.",
        "top_k": 3,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_category" in data
    assert data["predicted_category"] == "breast cancer"
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["top_predictions"]) == 3
    assert "inference_time_ms" in data


def test_predict_endpoint_covid(client):
    payload = {
        "summary": "Randomized double-blind trial evaluating antiviral therapy in severe acute respiratory syndrome coronavirus 2 infection.",
        "top_k": 2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_category"] == "covid-19"
    assert len(data["top_predictions"]) == 2


def test_predict_rejects_too_short(client):
    payload = {"summary": "hi"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
