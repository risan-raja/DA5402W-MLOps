#!/usr/bin/env bash
# Bootstrap: check host prerequisites, sync the lockfile, seed .env.
# Does not install system packages (no brew/apt/winget/sudo).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

ok() { printf '  [OK]   %s\n' "$1"; }
fail() { printf '  [FAIL] %s\n' "$1"; FAILED=1; }
warn() { printf '  [WARN] %s\n' "$1"; }
hint() { printf '         %s\n' "$1"; }

FAILED=0

UNAME_S="$(uname -s)"
case "${UNAME_S}" in
  Darwin) OS_LABEL="macOS" ;;
  Linux) OS_LABEL="Linux" ;;
  MINGW*|MSYS*|CYGWIN*) OS_LABEL="Windows (Git Bash)" ;;
  *)
    echo "Unsupported OS (${UNAME_S}). Use macOS, Linux, or Windows via WSL2 / Git Bash."
    exit 1
    ;;
esac
echo "Detected OS: ${OS_LABEL}"
echo

PYTHON_BIN=""
for candidate in python3 python; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v "${candidate}")"
    break
  fi
done

if [[ -z "${PYTHON_BIN}" ]]; then
  fail "Python 3.13+ not found (need python3 or python on PATH)"
  hint "Install Python 3.13+ from https://www.python.org/downloads/"
else
  PY_VER="$("${PYTHON_BIN}" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
  PY_MAJOR="$("${PYTHON_BIN}" -c 'import sys; print(sys.version_info[0])' 2>/dev/null || echo 0)"
  PY_MINOR="$("${PYTHON_BIN}" -c 'import sys; print(sys.version_info[1])' 2>/dev/null || echo 0)"
  if [[ "${PY_MAJOR}" -gt 3 ]] || { [[ "${PY_MAJOR}" -eq 3 ]] && [[ "${PY_MINOR}" -ge 13 ]]; }; then
    ok "Python ${PY_VER} (${PYTHON_BIN})"
  else
    fail "Python ${PY_VER:-unknown} found; need ≥ 3.13"
    hint "Install Python 3.13+ from https://www.python.org/downloads/"
  fi
fi

if command -v uv >/dev/null 2>&1; then
  ok "uv $(uv --version 2>/dev/null | head -n1)"
else
  fail "uv not found on PATH"
  hint "Install: https://docs.astral.sh/uv/getting-started/installation/"
fi

java_major() {
  # java -version writes to stderr; accept "17.0.x" or legacy "1.8.0_xxx"
  local raw
  raw="$(java -version 2>&1 | head -n1 || true)"
  if [[ "${raw}" =~ \"1\.([0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
  elif [[ "${raw}" =~ \"([0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo ""
  fi
}

if ! command -v java >/dev/null 2>&1; then
  fail "Java not found (need JDK 17+ for PySpark)"
  case "${OS_LABEL}" in
    macOS) hint "brew install openjdk@17  # then set JAVA_HOME (brew --prefix openjdk@17)" ;;
    Linux) hint "sudo apt install openjdk-17-jdk   # or your distro's JDK 17 package" ;;
    *) hint "Install Eclipse Temurin / Oracle JDK 17 and ensure java is on PATH" ;;
  esac
else
  J_MAJOR="$(java_major)"
  if [[ -n "${J_MAJOR}" ]] && [[ "${J_MAJOR}" -ge 17 ]]; then
    ok "Java ${J_MAJOR} ($(command -v java))"
  else
    fail "Java major version ${J_MAJOR:-unknown}; need ≥ 17"
    case "${OS_LABEL}" in
      macOS) hint "brew install openjdk@17  # then set JAVA_HOME" ;;
      Linux) hint "sudo apt install openjdk-17-jdk" ;;
      *) hint "Install JDK 17 and ensure it shadows older java on PATH" ;;
    esac
  fi
fi

if command -v docker >/dev/null 2>&1; then
  if docker compose version >/dev/null 2>&1; then
    ok "Docker + docker compose"
  else
    warn "Docker found but 'docker compose' (v2) is missing"
    hint "Needed for make compose (API / MLflow / Prometheus / Grafana), not for training alone"
  fi
else
  warn "Docker not found"
  hint "Needed for make compose (API / MLflow / monitoring), not for training alone"
fi

HF_OK=0
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_OK=1
elif [[ -f "${ROOT}/.env" ]] && grep -Eq '^[[:space:]]*HF_TOKEN=[^[:space:]#]+' "${ROOT}/.env"; then
  HF_OK=1
elif command -v hf >/dev/null 2>&1 && hf auth whoami >/dev/null 2>&1; then
  HF_OK=1
fi
if [[ "${HF_OK}" -eq 1 ]]; then
  ok "Hugging Face auth present (token or hf login)"
else
  warn "No HF_TOKEN / hf auth detected"
  hint "Run: hf auth login   # or set HF_TOKEN in .env after seeding"
fi

echo
if [[ "${FAILED}" -ne 0 ]]; then
  echo "Prerequisite checks failed. Fix the [FAIL] items above, then re-run:"
  echo "  bash scripts/setup.sh"
  echo "  # or: make setup"
  exit 1
fi

echo "Checks passed. Syncing dependencies..."
uv sync --locked --group dev

if [[ ! -f "${ROOT}/.env" ]]; then
  if [[ -f "${ROOT}/.env.example" ]]; then
    cp "${ROOT}/.env.example" "${ROOT}/.env"
    echo
    echo "Created .env from .env.example — edit HF_TOKEN and other secrets before Hub pushes."
  else
    warn ".env.example missing; skip seeding .env"
  fi
else
  echo
  echo ".env already exists — left unchanged."
fi

echo
echo "Next steps:"
case "${OS_LABEL}" in
  "Windows (Git Bash)")
    echo "  1. Activate:  source .venv/Scripts/activate"
    ;;
  *)
    echo "  1. Activate:  source .venv/bin/activate"
    ;;
esac
echo "  2. Auth:      hf auth login   # if you have not already"
echo "  3. Services:  make compose    # API :8000, MLflow :5001, Prometheus :9090, Grafana :3000"
echo "  4. Data:      python -m src.data_pipeline.dataset_downloader --target raw"
echo "  5. (Optional) make airflow  # DAG UI :8080; expects Compose MLflow at http://localhost:5001"
echo
echo "Setup complete."
