# DA5402W — Audio Classification MLOps Pipeline

End-to-end MLOps pipeline for classifying urban sounds, built for the DA5402W course end-term project. It takes raw `.wav` clips from UrbanSound8K, extracts features with PySpark, trains and compares four models, tracks runs in MLflow, serves the best one through FastAPI, and monitors it with Prometheus and drift detection.

**Status**: architecture finalized, implementation in progress. See `docs/DESIGN.md` for the design record and why each tool was chosen.

## Overview

UrbanSound8K has 8,732 labeled clips across 10 classes: air conditioners, car horns, kids playing, dog barks, drilling, engine idling, gunshots, jackhammers, sirens, and street music. The pipeline:

1. Fetches the dataset from [`risan-raja-iitm/urbansound8K`](https://huggingface.co/datasets/risan-raja-iitm/urbansound8K) on Hugging Face Hub (a clone of `danavery/urbansound8K`, already in parquet — no Kaggle credentials needed).
2. Extracts MFCCs, spectral centroid, zero-crossing rate, and chroma features in parallel with PySpark UDFs, splitting train/eval by the dataset's `fold` column per UrbanSound8K's standard protocol.
3. Trains and compares four models (XGBoost, LightGBM, Random Forest on the tabular features, and a CNN on mel-spectrograms), tuning each with Optuna and logging each run to MLflow.
4. Serves the winning model through FastAPI, containerized with Docker.
5. Runs ingestion through training as one Airflow DAG.
6. Monitors the live API with Prometheus metrics and a KS-test drift detector on incoming predictions.

## System architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────────┐     ┌─────────────┐
│  HF Hub     │────▶│   Airflow    │────▶│  PySpark (local    │────▶│   MLflow    │
│  dataset    │     │  DAG (Local  │     │  mode, in-container)│     │  (SQLite +  │
│  repo       │     │  Executor)   │     │  feature extraction │     │  Registry)  │
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

Dataset and model versioning both live on Hugging Face Hub — a public dataset repo (so CI can pull it without secrets) and a separate model repo for trained artifacts. DVC stays in the repo as a documented, secondary versioning path but isn't the primary one. The runtime stack (API, MLflow, Prometheus, Grafana, Airflow) runs in one Docker Compose file.

LocalExecutor over Celery, PySpark in local mode instead of a separate Spark cluster, and the two-HF-repo versioning setup are explained in `docs/DESIGN.md`.

## Setup and installation

Requires Python 3.13+, Java 17 (for PySpark), and Docker with at least 24 GiB allocated. The full stack (Airflow, MLflow, Prometheus, Grafana, and the API) needs more than Docker Desktop's 7.75 GiB default.

```bash
git clone https://github.com/risan-raja/DA5402W-MLOps.git
cd DA5402W-MLOps
uv sync   # or: python3 -m venv .venv && source .venv/bin/activate && pip install -e .
```

You'll also need Hugging Face Hub credentials (`hf auth login`) — for downloading the dataset and for pushing dataset/model versions. No Kaggle account is needed.

## Running the pipeline

**Full stack (Airflow, MLflow, Prometheus, Grafana, API) in one command:**

```bash
docker-compose -f docker/docker-compose.yml up --build -d
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
# Dataset: pushed to a public HF Hub dataset repo
hf upload <hf-username>/<dataset-repo> data/raw --repo-type dataset
hf upload <hf-username>/<dataset-repo> data/processed --repo-type dataset

# Models: pushed to an HF Hub model repo
hf upload <hf-username>/<model-repo> models/ --repo-type model

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

# Prometheus scrape target
curl http://localhost:8000/metrics
```

`/predict` accepts a `.wav` file and returns the predicted class plus confidence score. Malformed or non-`.wav` uploads return a structured error.

## Docker execution

```bash
# build and start everything
docker-compose -f docker/docker-compose.yml up --build -d

# check status
docker-compose -f docker/docker-compose.yml ps

# tail API logs
docker-compose -f docker/docker-compose.yml logs -f api

# tear down
docker-compose -f docker/docker-compose.yml down
```

Service ports: API on `8000`, MLflow UI on `5000`, Airflow UI on `8080`, Prometheus on `9090`, Grafana on `3000`.

## Folder structure

```
DA5402W/
├── airflow/            # DAG + custom image (Airflow + Java 17 + PySpark + librosa)
├── config/             # config.yaml, logging.yaml
├── data/                # raw/, processed/, sample_audio/
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
