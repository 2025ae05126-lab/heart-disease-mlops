import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request, Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("heart-api")

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "heart_model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Heart Disease Prediction API")

REQUESTS = Counter("api_requests_total", "Total API requests",
                   ["endpoint", "method", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency in seconds",
                    ["endpoint"])
PREDICTIONS = Counter("predictions_total", "Predictions by label", ["label"])


class Patient(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


@app.middleware("http")
async def track_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start

    route = request.scope.get("route")
    endpoint = route.path if route else "unmatched"

    if endpoint != "/metrics":
        REQUESTS.labels(endpoint, request.method, str(response.status_code)).inc()
        LATENCY.labels(endpoint).observe(elapsed)
        logger.info("%s %s -> %s (%.1f ms)", request.method, endpoint,
                    response.status_code, elapsed * 1000)
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(patient: Patient):
    data = pd.DataFrame([patient.model_dump()])
    prediction = int(model.predict(data)[0])
    confidence = float(model.predict_proba(data)[0][prediction])
    label = "Heart disease" if prediction == 1 else "No heart disease"
    PREDICTIONS.labels(label).inc()
    logger.info("prediction=%s confidence=%.4f", label, confidence)
    return {
        "prediction": prediction,
        "label": label,
        "confidence": round(confidence, 4),
    }
