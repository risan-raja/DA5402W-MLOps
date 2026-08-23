#!/usr/bin/env bash
# Smoke test for docker-compose: api, mlflow, prometheus, grafana.
# Not a substitute for pytest — only proves containers come up and talk to each other.
# On failure the stack is left running for inspection (no EXIT teardown).
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

echo
echo "== Checking HTTP endpoints =="
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
echo "== Checking Prometheus scrape targets =="
if curl -sf http://localhost:9090/api/v1/targets | grep -q '"job":"api"'; then
  pass "Prometheus is scraping the api job"
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
