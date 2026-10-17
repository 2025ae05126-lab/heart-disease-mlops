from pathlib import Path
import pandas as pd

numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_features = ["sex", "cp", "fbs", "restecg", "exang",
                        "slope", "ca", "thal"]

df = pd.read_csv("data/raw/heart_raw.csv")

print("Shape before cleaning:", df.shape)
missing = df.isna().sum()
print("Missing values before cleaning:")
print(missing[missing > 0])
print("Duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()

# original target is 0-4; make it binary (0 = no disease, 1 = disease)
df["target"] = (df["target"] > 0).astype(int)

for col in categorical_features:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in numeric_features:
    df[col] = df[col].fillna(df[col].median())

Path("data/processed").mkdir(parents=True, exist_ok=True)
df.to_csv("data/processed/heart_clean.csv", index=False)

print("Missing values after cleaning:", df.isna().sum().sum())
print("Saved cleaned data:", df.shape, "-> data/processed/heart_clean.csv")
