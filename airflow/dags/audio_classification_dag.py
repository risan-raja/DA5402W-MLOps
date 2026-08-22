from datetime import UTC, datetime
import os
import sys
from pathlib import Path

from airflow.operators.python import PythonOperator

from airflow import DAG

# Airflow mounts the repo pieces under /opt/airflow/{dags,src,data,config}.
AIRFLOW_HOME = Path("/opt/airflow")
if str(AIRFLOW_HOME) not in sys.path:
    sys.path.insert(0, str(AIRFLOW_HOME))


def run_spark_feature_extraction() -> None:
    os.chdir(AIRFLOW_HOME)
    from src.data_pipeline.spark_feature_extractor import extract_features

    push = os.environ.get("PUSH_PROCESSED", "").strip().lower() in {"1", "true", "yes"}
    extract_features(force=True, push=push)


with DAG(
    dag_id="audio_classification",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    schedule=None,
    catchup=False,
    tags=["mlops", "spark"],
) as dag:
    PythonOperator(
        task_id="spark_feature_extraction",
        python_callable=run_spark_feature_extraction,
    )
