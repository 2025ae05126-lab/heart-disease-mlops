from pathlib import Path
import pandas as pd

columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "heart-disease/processed.cleveland.data")

Path("data/raw").mkdir(parents=True, exist_ok=True)

try:
    from ucimlrepo import fetch_ucirepo
    heart = fetch_ucirepo(id=45)
    df = pd.concat([heart.data.features, heart.data.targets], axis=1)
    df = df.rename(columns={"num": "target"})
    print("Downloaded via ucimlrepo")
except Exception as e:
    print("ucimlrepo failed, using direct URL:", e)
    df = pd.read_csv(url, header=None, names=columns, na_values="?")

df.to_csv("data/raw/heart_raw.csv", index=False)
print("Saved", df.shape[0], "rows x", df.shape[1], "columns to data/raw/heart_raw.csv")
