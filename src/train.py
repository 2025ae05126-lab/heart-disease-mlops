from pathlib import Path
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score,
                             ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

# ---------- data ----------
df = pd.read_csv("data/processed/heart_clean.csv")
X = df.drop(columns="target").astype(float)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
print("Train:", X_train.shape, "Test:", X_test.shape)

# ---------- preprocessing (lives inside the pipeline) ----------
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), numeric_features),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
     categorical_features),
])

# ---------- models and tuning grids ----------
experiments = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000, random_state=SEED),
        {"model__C": [0.01, 0.1, 1, 10],
         "model__class_weight": [None, "balanced"]}),
    "random_forest": (
        RandomForestClassifier(random_state=SEED),
        {"model__n_estimators": [100, 300],
         "model__max_depth": [None, 5, 10],
         "model__min_samples_leaf": [1, 3]}),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=SEED),
        {"model__n_estimators": [100, 200],
         "model__learning_rate": [0.05, 0.1],
         "model__max_depth": [2, 3]}),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = {"accuracy": "accuracy", "precision": "precision",
           "recall": "recall", "roc_auc": "roc_auc"}

# ---------- MLflow ----------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("heart-disease-classification")

rows = []
fitted = {}

for name, (model, grid) in experiments.items():
    print("\nTraining", name)
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    search = GridSearchCV(pipe, grid, cv=cv, scoring=scoring,
                          refit="roc_auc", n_jobs=-1)
    search.fit(X_train, y_train)

    best = search.best_estimator_
    i = search.best_index_
    cv_metrics = {f"cv_{m}": float(search.cv_results_[f"mean_test_{m}"][i])
                  for m in scoring}

    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]
    test_metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_prob),
    }

    # plots
    cm_path = f"reports/{name}_confusion_matrix.png"
    roc_path = f"reports/{name}_roc_curve.png"

    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_estimator(best, X_test, y_test, ax=ax, cmap="Blues")
    ax.set_title(name + " - confusion matrix")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(best, X_test, y_test, ax=ax)
    ax.set_title(name + " - ROC curve")
    fig.tight_layout()
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    # log the run to MLflow
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model_type", name)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("seed", SEED)
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(cv_metrics)
        mlflow.log_metrics(test_metrics)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact("data/processed/heart_clean.csv")
        mlflow.sklearn.log_model(best, name="model", input_example=X_train.head(3),
                                 serialization_format="cloudpickle")

    fitted[name] = best
    rows.append({"model": name, **cv_metrics, **test_metrics})
    print(name, "| CV ROC-AUC:", round(cv_metrics["cv_roc_auc"], 3),
          "| Test ROC-AUC:", round(test_metrics["test_roc_auc"], 3))

# ---------- compare and save the best model ----------
results = pd.DataFrame(rows).set_index("model").round(3)
results.to_csv("reports/model_comparison.csv")
print("\n", results.T)

best_name = results["cv_roc_auc"].idxmax()   # chosen on CV, not on the test set
joblib.dump(fitted[best_name], "models/heart_model.joblib")
print("\nBest model (by CV ROC-AUC):", best_name)
print("Saved to models/heart_model.joblib")
