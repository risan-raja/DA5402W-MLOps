import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from airflow.operators.python import PythonOperator

from airflow import DAG

# airflow/dags/this_file.py → repo root (host Airflow; not Airflow's AIRFLOW_HOME).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

TRAIN_MODELS = ("rf", "xgboost", "lightgbm", "resnet18")


# Callables import src modules inside the function body so the DAG file stays
# parseable without loading PySpark / librosa / audiomentations / torch at scheduler start.
def run_download_raw() -> None:
    os.chdir(PROJECT_ROOT)
    from src.data_pipeline.dataset_downloader import download_dataset, raw_data_present

    raw_dir = PROJECT_ROOT / "data" / "raw"
    if raw_data_present(raw_dir):
        logger.info("Raw data already present at %s; skipping download", raw_dir)
        return
    download_dataset(targets=["raw"], force=False)


def run_preprocess_interim() -> None:
    os.chdir(PROJECT_ROOT)
    from src.data_pipeline.dataset_downloader import (
        download_dataset,
        interim_data_present,
    )
    from src.data_processing.preprocessor import process_raw_to_interim
    from src.data_processing.versioning import config_enabled

    if not config_enabled("preprocessing"):
        interim_dir = PROJECT_ROOT / "data" / "interim"
        if interim_data_present(interim_dir):
            logger.info(
                "preprocessing.enabled is false; reusing interim at %s",
                interim_dir,
            )
            return
        logger.info(
            "preprocessing.enabled is false and interim incomplete; "
            "downloading from Hugging Face"
        )
        download_dataset(targets=["interim"], force=False)
        if not interim_data_present(interim_dir):
            raise FileNotFoundError(
                f"interim data still incomplete at {interim_dir} after Hub download. "
                "Push interim to the dataset repo or run with preprocessing.enabled: true."
            )
        return
    process_raw_to_interim(force=True)


def run_push_interim_hf() -> None:
    os.chdir(PROJECT_ROOT)
    from src.data_processing.versioning import (
        push_dataset_tree,
        versioning_push_enabled,
    )

    if not versioning_push_enabled("push_interim"):
        logger.info("versioning.push_interim is false; skipping interim Hub upload")
        return
    push_dataset_tree(PROJECT_ROOT / "data" / "interim", path_in_repo="interim")


def run_spark_feature_extraction() -> None:
    os.chdir(PROJECT_ROOT)
    from src.data_pipeline.dataset_downloader import (
        download_dataset,
        processed_data_present,
    )
    from src.data_pipeline.spark_feature_extractor import extract_features
    from src.data_processing.versioning import config_enabled, versioning_push_enabled

    if not config_enabled("spark"):
        processed_dir = PROJECT_ROOT / "data" / "processed"
        if processed_data_present(processed_dir):
            logger.info(
                "spark.enabled is false; reusing processed at %s",
                processed_dir,
            )
            return
        logger.info(
            "spark.enabled is false and processed incomplete; "
            "downloading from Hugging Face"
        )
        download_dataset(targets=["processed"], force=False)
        if not processed_data_present(processed_dir):
            raise FileNotFoundError(
                f"processed data still incomplete at {processed_dir} after Hub download. "
                "Push processed to the dataset repo or run with spark.enabled: true."
            )
        return
    extract_features(force=True, push=versioning_push_enabled("push_processed"))


def run_train_model(model_name: str) -> None:
    os.chdir(PROJECT_ROOT)
    from src.models.train import train_one_model

    train_one_model(model_name)


def run_select_winner() -> None:
    os.chdir(PROJECT_ROOT)
    from src.models.train import select_winner_from_artifacts

    select_winner_from_artifacts()


def run_push_all_models() -> None:
    os.chdir(PROJECT_ROOT)
    from src.data_processing.versioning import (
        push_all_trained_models,
        versioning_push_enabled,
    )

    if not versioning_push_enabled("push_models"):
        logger.info("versioning.push_models is false; skipping model Hub upload")
        return
    push_all_trained_models(PROJECT_ROOT / "models")


def run_push_winner() -> None:
    os.chdir(PROJECT_ROOT)
    from src.data_processing.versioning import (
        push_winner_artifacts,
        versioning_push_enabled,
    )

    if not versioning_push_enabled("push_models"):
        logger.info("versioning.push_models is false; skipping winner Hub upload")
        return
    push_winner_artifacts(PROJECT_ROOT / "models")


with DAG(
    dag_id="audio_classification",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    schedule=None,
    catchup=False,
    tags=["mlops", "preprocess", "spark", "training", "versioning"],
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
    train_tasks = [
        PythonOperator(
            task_id=f"train_{name}",
            python_callable=run_train_model,
            op_kwargs={"model_name": name},
        )
        for name in TRAIN_MODELS
    ]
    select_winner = PythonOperator(
        task_id="select_winner",
        python_callable=run_select_winner,
    )
    push_all_models = PythonOperator(
        task_id="push_all_models",
        python_callable=run_push_all_models,
    )
    push_winner = PythonOperator(
        task_id="push_winner",
        python_callable=run_push_winner,
    )

    (
        download_raw
        >> preprocess_interim
        >> push_interim_hf
        >> spark_feature_extraction
        >> train_tasks
        >> select_winner
        >> push_all_models
        >> push_winner
    )
