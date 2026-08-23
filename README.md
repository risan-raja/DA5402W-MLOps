# DA5402W — Audio Classification MLOps Pipeline

End-to-end MLOps pipeline for classifying urban sounds, built for the DA5402W course end-term project. It takes raw `.wav` clips from UrbanSound8K, extracts features with PySpark, trains and compares four models, tracks runs in MLflow, serves the best one through FastAPI, and monitors it with Prometheus and drift detection.

**Status**: architecture finalized, implementation in progress. See `docs/DESIGN.md` for the design record and why each tool was chosen.

## Overview

UrbanSound8K has 8,732 labeled clips across 10 classes: air conditioners, car horns, kids playing, dog barks, drilling, engine idling, gunshots, jackhammers, sirens, and street music. The pipeline:

1. Fetches the dataset from [`risan-raja-iitm/urbansound8K`](https://huggingface.co/datasets/risan-raja-iitm/urbansound8K) on Hugging Face Hub (a clone of `danavery/urbansound8K`, already in parquet — no Kaggle credentials needed).
2. Extracts MFCCs, spectral centroid, zero-crossing rate, and chroma features in parallel with PySpark UDFs on the host, splitting train/eval by the dataset's `fold` column per UrbanSound8K's standard protocol.
3. Trains and compares four models (XGBoost, LightGBM, Random Forest on the tabular features, and a CNN on mel-spectrograms), tuning each with Optuna and logging each run to MLflow.
4. Serves the winning model through FastAPI, containerized with Docker.
5. Runs ingestion through training as one Airflow DAG.
6. Monitors the live API with Prometheus metrics and a KS-test drift detector on incoming predictions.

## System architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────────┐     ┌─────────────┐
│  HF Hub     │────▶│   Airflow    │────▶│  PySpark (local    │────▶│   MLflow    │
│  dataset    │     │  DAG (host   │     │  mode, host venv)  │     │  (SQLite +  │
│  repo       │     │  LocalExec)  │     │  feature extraction │     │  Registry)  │
└─────────────┘     └──────┬───────┘     └───────────────────┘     └──────┬──────┘
                           │                                              │
                           ▼                                              ▼
                    ┌─────────────┐                              ┌───────────────┐
                    │  XGBoost /  │                              │   FastAPI     │
                    │  LightGBM / │─────────────registers───────▶│  /predict     │
                    │  RF / CNN   │                              │  /health      │
                    └─────────────┘                              │  /metrics     │
                                                                  └───────┬───────┘
                                                                          │
                                                          ┌───────────────┴───────────────┐
                                                          ▼                                ▼
                                                   ┌─────────────┐                ┌─────────────┐
                                                   │ Prometheus  │                │    Drift    │
                                                   │  + Grafana  │                │  Detector   │
                                                   └─────────────┘                └─────────────┘
```

Dataset and model versioning both live on Hugging Face Hub — the same public dataset repo as the UrbanSound8K mirror (`risan-raja-iitm/urbansound8K`), with pipeline outputs under `interim/` (later `processed/`), plus a separate model repo for trained artifacts. DVC stays in the repo as a documented, secondary versioning path but isn't the primary one. Airflow runs on the host (same venv as training, so ResNet can use MPS). Docker Compose runs the API, MLflow, Prometheus, and Grafana.

LocalExecutor over Celery, PySpark in local mode instead of a separate Spark cluster, and the two-HF-repo versioning setup are explained in `docs/DESIGN.md`.

## Setup and installation

Requires Python 3.13+, Java 17 (for PySpark), and Docker for the API/MLflow/monitoring stack.

```bash
git clone https://github.com/risan-raja/DA5402W-MLOps.git
cd DA5402W-MLOps
uv sync   # or: python3 -m venv .venv && source .venv/bin/activate && pip install -e .
```

You'll also need Hugging Face Hub credentials (`hf auth login`) — for downloading the dataset and for pushing dataset/model versions. No Kaggle account is needed.

## Running the pipeline

**Compose (API, MLflow, Prometheus, Grafana) and host Airflow:**

```bash
make compose    # docker compose --env-file .env -f docker/docker-compose.yml up --build -d
make airflow    # LocalExecutor UI on http://localhost:8080 (metadata in .airflow/)
```

**Fetch the dataset:**

```bash
hf download risan-raja-iitm/urbansound8K --repo-type dataset --local-dir data/raw
```

**Feature extraction on its own:**

```bash
python src/data_pipeline/spark_feature_extractor.py --input data/raw --output data/processed
```
Splitting is fold-based (the dataset's `fold` column, 1–10), matching UrbanSound8K's standard cross-validation protocol.

**Model training with MLflow tracking:**

```bash
python src/models/train.py --config config/config.yaml
mlflow ui --port 5000   # view runs, metrics, and the model registry
```

**Data and model versioning:**

```bash
# Raw stays on upstream risan-raja-iitm/urbansound8K — do not re-upload.

# Dataset repo is the same as raw: risan-raja-iitm/urbansound8K
# Push interim under interim/ (~1.8 GB). Do not re-upload raw parquet.
python -m src.data_processing.versioning push data/interim interim
# later, after Spark: python -m src.data_processing.versioning push data/processed processed

# Pull (raw and/or interim) via the downloader:
python -m src.data_pipeline.dataset_downloader --target raw
python -m src.data_pipeline.dataset_downloader --target interim
python -m src.data_pipeline.dataset_downloader --target raw --target interim

# Models: risan-raja-iitm/urbansound8k-models (all four dirs, then winner last)
python -m src.data_processing.versioning push-models models
python -m src.data_processing.versioning push-winner models
make pull-winner   # winner.json + winner/ from the model repo into models/

# DVC: kept as a documented alternative, local remote
dvc add models/
dvc push
```

## API usage

Once the container is running, the API is at `http://localhost:8000`:

```bash
# health check
curl http://localhost:8000/health

# classify a clip
curl -X POST http://localhost:8000/predict \
  -F "file=@data/sample_audio/dog_bark_sample.wav"

# send 3 clips × 10 classes so Prometheus/Grafana record request rate + latency
make demo-predict
# slower pass (15s scrape): python -m src.deployment.demo --repeat 2 --delay 1

# Prometheus scrape target
curl http://localhost:8000/metrics
```

`/predict` accepts a `.wav` file and returns the predicted class, confidence, `model_name`, `latency_ms`, and the 10-class `probabilities` map. Malformed, oversized, or non-`.wav` uploads return a structured error. If no winner is loaded, `/predict` returns 503 while `/health` stays 200.

Grafana is at [http://localhost:3000](http://localhost:3000). Sign in with `GF_SECURITY_ADMIN_USER` / `GF_SECURITY_ADMIN_PASSWORD` from `.env` (defaults: `admin` / `admin`). Open the provisioned **API Overview** dashboard.

## Docker execution

```bash
# API + MLflow + Prometheus + Grafana (Airflow is host-only: make airflow)
docker-compose --env-file .env -f docker/docker-compose.yml up --build -d

# check status
docker-compose -f docker/docker-compose.yml ps

# tail API logs
docker-compose -f docker/docker-compose.yml logs -f api

# tear down
docker-compose -f docker/docker-compose.yml down
```

Service ports: API on `8000`, MLflow UI on `5001`, host Airflow UI on `8080`, Prometheus on `9090`, Grafana on `3000`.

## Folder structure

```
DA5402W/
├── airflow/            # DAG only (host Airflow; PROJECT_ROOT = repo root)
├── Makefile            # make airflow / make compose
├── config/             # config.yaml, logging.yaml
├── data/                # raw/ (upstream cache), interim/ (versioned), processed/, sample_audio/
├── docker/             # Dockerfile.api, docker-compose.yml, prometheus/
├── docs/               # design doc, course brief
├── report/             # LaTeX technical report
├── src/
│   ├── data_pipeline/      # dataset download, PySpark feature extraction
│   ├── data_processing/    # augmentation, cleaning, preprocessing
│   ├── models/             # baseline models, CNN, training, evaluation
│   ├── deployment/         # FastAPI app + schemas
│   └── monitoring/         # prediction/latency logging, drift detection
├── tests/
├── dvc.yaml
└── pyproject.toml
```

Full stack and dependency list: `pyproject.toml`.
