# Heart Disease Risk Prediction - End-to-End MLOps Pipeline

MLOps Assignment 01 (AIMLCZG523, BITS Pilani) - Student: Vatsavayi Kaushik Chandra varma (2025AE05126)

A classifier that predicts heart disease risk from patient health data (UCI Heart Disease, Cleveland, 303 records, 13 features), served as a containerised FastAPI service on Kubernetes with CI/CD, MLflow experiment tracking and Prometheus/Grafana monitoring.

- Report: `report/2025AE05126_mlops_assignment.pdf` (also provided as .docx)
- Demo video: https://drive.google.com/file/d/1cv4eb8TOJUZ5Xupe-y9LaqLrE4RTWTMZ/view?usp=sharing

## Results

Three models were tuned with GridSearchCV (5-fold stratified CV, seed 42, 80/20 split). The final model was chosen on cross-validated ROC-AUC.

| Model | CV ROC-AUC | Test accuracy | Test precision | Test recall | Test ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression (final)** | 0.903 | 0.885 | 0.839 | 0.929 | 0.965 |
| Random Forest | 0.901 | 0.869 | 0.833 | 0.893 | 0.947 |
| Gradient Boosting | 0.874 | 0.902 | 0.867 | 0.929 | 0.961 |

## Repository structure

```
api/                 FastAPI app (/predict, /health, /metrics)
src/                 download_data.py, data_processing.py, train.py, predict.py
notebooks/           01_eda.ipynb
data/raw, processed  raw UCI data and cleaned CSV
models/              heart_model.joblib (preprocessing + classifier in one pipeline)
tests/               pytest unit tests (data, model, API)
k8s/                 deployment.yaml, service.yaml
monitoring/          prometheus.yml, grafana_dashboard.json
.github/workflows/   ci.yml (flake8, pytest, training, docker build)
reports/             model_comparison.csv, ROC curves, confusion matrices
screenshots/         EDA, MLflow, CI, Docker, Kubernetes, monitoring evidence
Dockerfile, docker-compose.yml, requirements.txt, requirements-api.txt
```

## Setup (clean clone)

```bash
git clone https://github.com/2025ae05126-lab/heart-disease-mlops.git
cd heart-disease-mlops
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt -r requirements-api.txt pytest httpx flake8
```

## Run the pipeline

```bash
python src/download_data.py        # downloads UCI data to data/raw/heart_raw.csv
python src/data_processing.py      # cleans data -> data/processed/heart_clean.csv
python src/train.py                # trains 3 models, logs to MLflow, saves models/heart_model.joblib
python src/predict.py sample_input.json   # CLI inference (use sample_batch.json for a batch)
```

Notebook: `jupyter notebook notebooks/01_eda.ipynb` (EDA plots are also in `screenshots/eda/`).

### Experiment tracking (MLflow)

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://localhost:5000. Each run logs parameters, CV and test metrics, confusion matrix, ROC curve, the cleaned dataset and the model.

### Tests and linting

```bash
flake8 .
python -m pytest -v
```

The GitHub Actions workflow runs the same steps on every push and pull request, then trains the model, uploads test results/models/reports as artifacts and builds the Docker image. It fails on any lint or test error.

## Run the API

Locally:

```bash
uvicorn api.app:app --port 8000
```

Docker:

```bash
docker build -t heart-api:1.0 .
docker run -d -p 8000:8000 --name heart-api heart-api:1.0
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_input.json
```

Example response:

```json
{"prediction": 0, "label": "No heart disease", "confidence": 0.5402}
```

Input fields (all numeric): `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`. Missing fields return HTTP 422.

## Kubernetes deployment

Tested on a local cluster (Docker Desktop / kind). Load the image into the cluster first if it is not pulled from a registry (for kind: `kind load docker-image heart-api:1.0 --name heart`).

```bash
kubectl apply -f k8s/
kubectl get pods,svc
kubectl port-forward svc/heart-api-service 8080:8080
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d @sample_input.json
```

The service is of type LoadBalancer (port 8080 to container port 8000). On a local cluster the external IP stays `<pending>`, hence the port-forward.

## Monitoring and logging

The API exposes Prometheus metrics at `/metrics` (request counter, latency histogram, predictions by class) and logs every request and prediction.

```bash
docker compose up --build
```

- API: http://localhost:8000
- Prometheus: http://localhost:9090 (target `heart-api` should be UP)
- Grafana: http://localhost:3000 (login admin/admin) - add Prometheus as a data source with URL `http://prometheus:9090`, then import `monitoring/grafana_dashboard.json` (panels: request rate, p95 latency, predictions by class, error responses).

View request logs with `docker logs heart-api` or `docker compose logs api`.

## Evidence

Screenshots for every stage (MLflow runs, CI runs and artifacts, Docker build and test, Kubernetes pods/services, Prometheus targets, Grafana dashboard) are in `screenshots/` and are referenced in the report.

## Tech stack

Python, pandas, scikit-learn, MLflow, FastAPI, prometheus-client, pytest, flake8, GitHub Actions, Docker, Kubernetes, Prometheus, Grafana.
