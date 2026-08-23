#!/usr/bin/env bash
# Host Airflow (LocalExecutor + sqlite). DAG tasks run in this venv (MPS on macOS).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${ROOT}/.env"
  set +a
fi

export AIRFLOW_HOME="${ROOT}/.airflow"
export AIRFLOW__CORE__EXECUTOR="${AIRFLOW__CORE__EXECUTOR:-LocalExecutor}"
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="sqlite:///${AIRFLOW_HOME}/airflow.db"
export AIRFLOW__CORE__DAGS_FOLDER="${ROOT}/airflow/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES="${AIRFLOW__CORE__LOAD_EXAMPLES:-false}"
export AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION="${AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION:-true}"
export AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_ALL_ADMINS="${AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_ALL_ADMINS:-True}"
export AIRFLOW__CORE__EXECUTION_API_SERVER_URL="${AIRFLOW__CORE__EXECUTION_API_SERVER_URL:-http://localhost:8080/execution/}"
export AIRFLOW__API_AUTH__JWT_SECRET="${AIRFLOW__API_AUTH__JWT_SECRET:-da5402w-dev-jwt-secret-change-me}"
export AIRFLOW__CORE__FERNET_KEY="${AIRFLOW_FERNET_KEY:-${AIRFLOW__CORE__FERNET_KEY:-}}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# DAG training talks to the Compose MLflow server (do not share sqlite with the container).
export MLFLOW_TRACKING_URI="${AIRFLOW_MLFLOW_TRACKING_URI:-http://localhost:5001}"

mkdir -p "${AIRFLOW_HOME}"
airflow db migrate

cleanup() {
  pids="$(jobs -p || true)"
  if [[ -n "${pids}" ]]; then
    # shellcheck disable=SC2086
    kill ${pids} 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

airflow api-server --port 8080 &
airflow dag-processor &
airflow scheduler
