#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
WORKSPACE="${ROSCLAW_ACCEPTANCE_HOME:-$(mktemp -d)}"
DAEMON_PID=""

cleanup() {
  if [[ -n "${DAEMON_PID}" ]] && kill -0 "${DAEMON_PID}" 2>/dev/null; then
    kill "${DAEMON_PID}" 2>/dev/null || true
    wait "${DAEMON_PID}" 2>/dev/null || true
  fi
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
DAEMON="${WORKSPACE}/venv/bin/rosclawd"
SOCKET="${ROSCLAW_HOME}/run/rosclawd.sock"

"${CLI}" --version
"${CLI}" daemon --help >/dev/null
"${DAEMON}" --help >/dev/null
"${CLI}" firstboot --yes --profile offline --no-telemetry >/dev/null
"${CLI}" demo run ur5e-reach --json >"${WORKSPACE}/receipt.json"
"${CLI}" explain latest --json >"${WORKSPACE}/explanation.json"

"${DAEMON}" \
  --socket "${SOCKET}" \
  --robot-id clean_install \
  --log-level ERROR \
  >"${WORKSPACE}/rosclawd.stdout" \
  2>"${WORKSPACE}/rosclawd.stderr" &
DAEMON_PID="$!"

DAEMON_READY=0
for _attempt in $(seq 1 200); do
  if "${CLI}" daemon status \
    --socket "${SOCKET}" \
    --json \
    >"${WORKSPACE}/daemon-status.json" \
    2>/dev/null; then
    DAEMON_READY=1
    break
  fi
  if ! kill -0 "${DAEMON_PID}" 2>/dev/null; then
    cat "${WORKSPACE}/rosclawd.stderr" >&2
    echo "ERROR: wheel-installed rosclawd exited before becoming ready." >&2
    exit 1
  fi
  sleep 0.05
done

if [[ "${DAEMON_READY}" != "1" ]]; then
  cat "${WORKSPACE}/rosclawd.stderr" >&2
  echo "ERROR: wheel-installed rosclawd did not become ready." >&2
  exit 1
fi

"${CLI}" daemon stop \
  --socket "${SOCKET}" \
  --json \
  >"${WORKSPACE}/daemon-stop.json"
wait "${DAEMON_PID}"
DAEMON_PID=""

if [[ -e "${SOCKET}" ]]; then
  echo "ERROR: wheel-installed rosclawd left its socket behind." >&2
  exit 1
fi

"${WORKSPACE}/venv/bin/python" - "${WORKSPACE}" <<'PY'
import json
import sys
from importlib.util import find_spec
from pathlib import Path

root = Path(sys.argv[1])
receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
explanation = json.loads((root / "explanation.json").read_text(encoding="utf-8"))
daemon_status = json.loads((root / "daemon-status.json").read_text(encoding="utf-8"))
daemon_stop = json.loads((root / "daemon-stop.json").read_text(encoding="utf-8"))

assert receipt["final_state"] == "COMPLETED"
assert receipt["evidence_level"] == "TASK_VERIFIED"
assert receipt["simulation_result"]["has_physics"] is True
assert receipt["verification_result"]["success"] is True
assert explanation["run_id"] == receipt["action_id"]
assert explanation["verification"]["task_verified"] is True
assert daemon_status["running"] is True
assert daemon_status["southbound_owner"] == "rosclawd"
assert daemon_status["robot_id"] == "clean_install"
assert daemon_status["hardware_actions_executed"] == 0
assert daemon_stop["shutdown_requested"] is True

installed = root / "venv" / "lib"
assert any(installed.glob("python*/site-packages/rosclaw/product/status.yaml"))
assert find_spec("rosclaw_how") is None
assert find_spec("rosclaw_know") is None
print("PASS clean install", receipt["action_id"])
PY
