from datetime import UTC, datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def ping() -> None:
    print("airflow ok")


with DAG(
    dag_id="audio_classification",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    schedule=None,
    catchup=False,
    tags=["scaffold"],
) as dag:
    PythonOperator(task_id="ping", python_callable=ping)
