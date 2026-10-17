import joblib
import numpy as np
import pandas as pd

model = joblib.load("models/heart_model.joblib")

patient = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
}


def test_prediction_is_binary():
    pred = model.predict(pd.DataFrame([patient]).astype(float))
    assert pred[0] in (0, 1)


def test_probabilities_sum_to_one():
    proba = model.predict_proba(pd.DataFrame([patient]).astype(float))
    assert np.isclose(proba.sum(), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_handles_missing_values():
    row = dict(patient)
    row["chol"] = np.nan
    row["ca"] = np.nan
    proba = model.predict_proba(pd.DataFrame([row]).astype(float))
    assert not np.isnan(proba).any()


def test_batch_prediction_shape():
    batch = pd.DataFrame([patient, patient, patient]).astype(float)
    assert model.predict(batch).shape == (3,)
