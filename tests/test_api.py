from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

patient = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_returns_prediction_and_confidence():
    r = client.post("/predict", json=patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["confidence"] <= 1
    assert body["label"] in ("Heart disease", "No heart disease")


def test_predict_rejects_missing_field():
    bad = dict(patient)
    del bad["age"]
    assert client.post("/predict", json=bad).status_code == 422


def test_metrics_endpoint_exposes_counters():
    client.post("/predict", json=patient)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "api_requests_total" in r.text
    assert "predictions_total" in r.text
