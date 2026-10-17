import json
import sys

import joblib
import pandas as pd

model = joblib.load("models/heart_model.joblib")

# usage: python src/predict.py [path to a JSON file with one patient or a list]
path = sys.argv[1] if len(sys.argv) > 1 else "sample_input.json"
with open(path) as f:
    data = json.load(f)

patients = pd.DataFrame(data if isinstance(data, list) else [data]).astype(float)

predictions = model.predict(patients)
probabilities = model.predict_proba(patients)

print("Input file:", path)
print("Patients scored:", len(patients))
print()
for i, (pred, proba) in enumerate(zip(predictions, probabilities), start=1):
    label = "Heart disease" if pred == 1 else "No heart disease"
    print(f"Patient {i}: {label} | confidence {proba[pred]:.3f} | "
          f"P(disease) {proba[1]:.3f}")
