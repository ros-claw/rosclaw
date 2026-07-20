#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
WORKSPACE="${ROSCLAW_ACCEPTANCE_HOME:-$(mktemp -d)}"

cleanup() {
  if [[ -z "${ROSCLAW_ACCEPTANCE_HOME:-}" ]]; then
    rm -rf "${WORKSPACE}"
  fi
}
trap cleanup EXIT

command -v uv >/dev/null 2>&1 || {
  echo "ERROR: clean install acceptance requires uv." >&2
  exit 2
}

unset PYTHONPATH
uv build --wheel --out-dir "${WORKSPACE}/dist" "${ROOT}"
uv venv --python "${PYTHON_VERSION}" "${WORKSPACE}/venv"
uv pip install --python "${WORKSPACE}/venv/bin/python" "${WORKSPACE}"/dist/*.whl

export ROSCLAW_HOME="${WORKSPACE}/home"
CLI="${WORKSPACE}/venv/bin/rosclaw"

"${CLI}" --version
"${CLI}" firstboot --yes --profile offline --no-telemetry >/dev/null
"${CLI}" demo run ur5e-reach --json >"${WORKSPACE}/receipt.json"
"${CLI}" explain latest --json >"${WORKSPACE}/explanation.json"

"${WORKSPACE}/venv/bin/python" - "${WORKSPACE}" <<'PY'
import json
import sys
from importlib.util import find_spec
from pathlib import Path

root = Path(sys.argv[1])
receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
explanation = json.loads((root / "explanation.json").read_text(encoding="utf-8"))

assert receipt["final_state"] == "COMPLETED"
assert receipt["evidence_level"] == "TASK_VERIFIED"
assert receipt["simulation_result"]["has_physics"] is True
assert receipt["verification_result"]["success"] is True
assert explanation["run_id"] == receipt["action_id"]
assert explanation["verification"]["task_verified"] is True

installed = root / "venv" / "lib"
assert any(installed.glob("python*/site-packages/rosclaw/product/status.yaml"))
assert find_spec("rosclaw_how") is None
assert find_spec("rosclaw_know") is None
print("PASS clean install", receipt["action_id"])
PY
