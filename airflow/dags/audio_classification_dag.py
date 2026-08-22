import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from airflow.operators.python import PythonOperator

from airflow import DAG

# Airflow mounts the repo pieces under /opt/airflow/{dags,src,data,config}.
AIRFLOW_HOME = Path("/opt/airflow")
if str(AIRFLOW_HOME) not in sys.path:
    sys.path.insert(0, str(AIRFLOW_HOME))

logger = logging.getLogger(__name__)


# Callables import src modules inside the function body so the DAG file stays
# parseable without loading PySpark / librosa / audiomentations at scheduler start.
def run_download_raw() -> None:
    os.chdir(AIRFLOW_HOME)
    from src.data_pipeline.dataset_downloader import download_dataset, raw_data_present

    raw_dir = AIRFLOW_HOME / "data" / "raw"
    if raw_data_present(raw_dir):
        logger.info("Raw data already present at %s; skipping download", raw_dir)
        return
    download_dataset(targets=["raw"], force=False)


def run_preprocess_interim() -> None:
    os.chdir(AIRFLOW_HOME)
    from src.data_processing.preprocessor import process_raw_to_interim

    process_raw_to_interim(force=True)


def run_push_interim_hf() -> None:
    os.chdir(AIRFLOW_HOME)
    from src.data_processing.versioning import env_flag_enabled, push_dataset_tree

    if not env_flag_enabled("PUSH_INTERIM"):
        logger.info("PUSH_INTERIM not set; skipping interim Hub upload")
        return
    push_dataset_tree(AIRFLOW_HOME / "data" / "interim", path_in_repo="interim")


def run_spark_feature_extraction() -> None:
    os.chdir(AIRFLOW_HOME)
    from src.data_pipeline.spark_feature_extractor import extract_features
    from src.data_processing.versioning import env_flag_enabled

    extract_features(force=True, push=env_flag_enabled("PUSH_PROCESSED"))


with DAG(
    dag_id="audio_classification",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    schedule=None,
    catchup=False,
    tags=["mlops", "preprocess", "spark", "versioning"],
) as dag:
    download_raw = PythonOperator(
        task_id="download_raw",
        python_callable=run_download_raw,
    )
    preprocess_interim = PythonOperator(
        task_id="preprocess_interim",
        python_callable=run_preprocess_interim,
    )
    push_interim_hf = PythonOperator(
        task_id="push_interim_hf",
        python_callable=run_push_interim_hf,
    )
    spark_feature_extraction = PythonOperator(
        task_id="spark_feature_extraction",
        python_callable=run_spark_feature_extraction,
    )

    download_raw >> preprocess_interim >> push_interim_hf >> spark_feature_extraction
