#!/usr/bin/env bash
# Smoke test for the docker-compose stack: boots every service and checks
# each one reaches a healthy/reachable state. Not a substitute for pytest —
# this only proves the containers come up and talk to each other.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/docker/docker-compose.yml"
ENV_FILE="$REPO_ROOT/.env"
COMPOSE="docker-compose --env-file $ENV_FILE -f $COMPOSE_FILE"

PASS=0
FAIL=0

pass() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

wait_for() {
  # wait_for <description> <max_seconds> <check_command...>
  local desc="$1" max="$2"
  shift 2
  local waited=0
  until "$@" >/dev/null 2>&1; do
    waited=$((waited + 2))
    if [ "$waited" -ge "$max" ]; then
      fail "$desc (timed out after ${max}s)"
      return 1
    fi
    sleep 2
  done
  pass "$desc"
  return 0
}

echo "== Building and starting the stack =="
$COMPOSE up --build -d
trap '' EXIT

echo
echo "== Waiting for core services to report healthy =="
wait_for "postgres healthy" 60 bash -c "$COMPOSE ps postgres | grep -q '(healthy)'"
wait_for "airflow-webserver healthy" 120 bash -c "$COMPOSE ps airflow-webserver | grep -q '(healthy)'"

echo
echo "== Checking airflow-init ran to completion =="
if $COMPOSE ps -a airflow-init 2>/dev/null | grep -qiE 'Exit 0|exited \(0\)'; then
  pass "airflow-init exited 0"
else
  fail "airflow-init did not exit cleanly"
fi

echo
echo "== Checking HTTP endpoints =="
wait_for "Airflow api-server responds on :8080" 60 curl -sf http://localhost:8080/api/v2/monitor/health
wait_for "API /health responds on :8000" 60 curl -sf http://localhost:8000/health
wait_for "MLflow responds on :5001" 60 curl -sf http://localhost:5001/health
wait_for "Prometheus responds on :9090" 60 curl -sf http://localhost:9090/-/healthy
wait_for "Grafana responds on :3000" 60 curl -sf http://localhost:3000/api/health

echo
echo "== Checking Grafana provisioning =="
GRAFANA_AUTH="${GF_SECURITY_ADMIN_USER:-admin}:${GF_SECURITY_ADMIN_PASSWORD:-admin}"
if curl -sf -u "$GRAFANA_AUTH" http://localhost:3000/api/datasources/name/Prometheus >/dev/null 2>&1; then
  pass "Grafana Prometheus datasource provisioned"
else
  fail "Grafana Prometheus datasource not found"
fi
if curl -sf -u "$GRAFANA_AUTH" http://localhost:3000/api/search?query=API%20Overview | grep -q "api-overview"; then
  pass "Grafana API Overview dashboard provisioned"
else
  fail "Grafana API Overview dashboard not found"
fi

echo
echo "== Checking PySpark runs inside the Airflow image =="
if $COMPOSE exec -T airflow-scheduler python -c \
  "from pyspark.sql import SparkSession; s = SparkSession.builder.master('local[*]').getOrCreate(); assert s.range(5).count() == 5" \
  >/dev/null 2>&1; then
  pass "PySpark local-mode job ran inside airflow-scheduler"
else
  fail "PySpark local-mode job failed inside airflow-scheduler"
fi

echo
echo "== Checking Prometheus target status (api target may be down until /metrics ships) =="
if curl -sf http://localhost:9090/api/v1/targets | grep -q '"job":"api"'; then
  pass "Prometheus is scraping the api job (check /targets for up/down state manually)"
else
  fail "Prometheus has no api scrape target configured"
fi

echo
echo "== Summary: $PASS passed, $FAIL failed =="

if [ "$FAIL" -gt 0 ]; then
  echo
  echo "Stack left running for inspection. Tear down with:"
  echo "  docker-compose --env-file $ENV_FILE -f $COMPOSE_FILE down"
  exit 1
fi

exit 0
