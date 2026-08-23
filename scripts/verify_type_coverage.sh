#!/usr/bin/env bash
# Type-check src/tests and gate pyrefly coverage at ≥95%.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

REPORT_DIR="${REPO_ROOT}/reports"
REPORT_JSON="${REPORT_DIR}/type_coverage.json"
FAIL_UNDER="${FAIL_UNDER:-95}"

mkdir -p "$REPORT_DIR"

echo "==> pyrefly check"
uv run --group dev pyrefly check

echo "==> pyrefly coverage report (src tests)"
uv run --group dev pyrefly coverage report src tests >"$REPORT_JSON"

python3 - "$REPORT_JSON" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
modules = payload.get("module_reports") or []

totals = {"typable": 0, "typed": 0, "untyped": 0, "any": 0}
by_root: dict[str, dict[str, float]] = defaultdict(
    lambda: {"typable": 0, "typed": 0, "untyped": 0, "any": 0}
)
strict_num = 0.0
strict_den = 0.0
untyped_mods: list[tuple[float, str, int, int]] = []

for mod in modules:
    name = str(mod.get("name") or "")
    root = name.split(".", 1)[0] if name else "other"
    typable = int(mod.get("n_typable") or 0)
    typed = int(mod.get("n_typed") or 0)
    untyped = int(mod.get("n_untyped") or 0)
    any_n = int(mod.get("n_any") or 0)
    cov = float(mod.get("coverage") or 0.0)
    strict = float(mod.get("strict_coverage") or 0.0)

    totals["typable"] += typable
    totals["typed"] += typed
    totals["untyped"] += untyped
    totals["any"] += any_n
    by_root[root]["typable"] += typable
    by_root[root]["typed"] += typed
    by_root[root]["untyped"] += untyped
    by_root[root]["any"] += any_n
    if typable:
        strict_num += strict * typable
        strict_den += typable
        if untyped or any_n:
            untyped_mods.append((cov, name, untyped, any_n))

overall = (100.0 * totals["typed"] / totals["typable"]) if totals["typable"] else 100.0
strict_overall = (strict_num / strict_den) if strict_den else 100.0

print()
print("Type coverage summary")
print("---------------------")
print(
    f"overall: {overall:.2f}%  "
    f"({totals['typed']}/{totals['typable']} typed; "
    f"{totals['untyped']} untyped; {totals['any']} Any)"
)
print(f"strict_coverage (weighted): {strict_overall:.2f}%")
for root in sorted(by_root):
    bucket = by_root[root]
    pct = (100.0 * bucket["typed"] / bucket["typable"]) if bucket["typable"] else 100.0
    print(
        f"  {root}: {pct:.2f}% "
        f"({int(bucket['typed'])}/{int(bucket['typable'])})"
    )

if untyped_mods:
    print()
    print("Top modules with gaps:")
    for cov, name, untyped, any_n in sorted(untyped_mods)[:10]:
        print(f"  {cov:6.2f}%  {name}  (untyped={untyped}, any={any_n})")

print()
print(f"Wrote {path}")
PY

echo "==> pyrefly coverage check --fail-under ${FAIL_UNDER}"
uv run --group dev pyrefly coverage check src tests --fail-under "$FAIL_UNDER"

echo "OK: type check + coverage ≥${FAIL_UNDER}%"
