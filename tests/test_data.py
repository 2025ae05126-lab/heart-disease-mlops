import pandas as pd

df = pd.read_csv("data/processed/heart_clean.csv")

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]


def test_columns_present():
    assert list(df.columns) == FEATURES + ["target"]


def test_no_missing_values():
    assert df.isna().sum().sum() == 0


def test_target_is_binary():
    assert set(df["target"].unique()) == {0, 1}


def test_no_duplicate_rows():
    assert df.duplicated().sum() == 0


def test_value_ranges_are_realistic():
    assert df["age"].between(20, 100).all()
    assert (df["chol"] > 0).all()
    assert (df["trestbps"] > 0).all()
