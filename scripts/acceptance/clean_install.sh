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
"${CLI}" app --help >/dev/null
"${DAEMON}" --help >/dev/null
"${CLI}" app install ros-claw/realsense-inspect --json >"${WORKSPACE}/app-install.json"
"${CLI}" app validate realsense-inspect --json >"${WORKSPACE}/app-validate.json"
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

"${WORKSPACE}/venv/bin/python" - "${SOCKET}" "${WORKSPACE}/daemon-action.json" <<'PY'
import json
import sys

from rosclaw.daemon.client import DaemonClient
from rosclaw.kernel import ActionEnvelope, ExecutionMode

client = DaemonClient(socket_path=sys.argv[1], timeout_sec=2.0)
action = ActionEnvelope(
    action_id="action-clean-install-durable",
    actor_id="clean-install",
    agent_framework="acceptance",
    session_id="clean-install",
    body_id="clean_install",
    body_snapshot_hash="sha256:clean-install-body",
    capability_id="acceptance.no-executor",
    arguments={"value": 1},
    execution_mode=ExecutionMode.SHADOW,
)
ticket = client.request_action(action)
status = client.wait_for_action(ticket["action_id"], timeout_sec=5.0)
with open(sys.argv[2], "w", encoding="utf-8") as output:
    json.dump(status, output)
PY

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

"${DAEMON}" \
  --socket "${SOCKET}" \
  --robot-id clean_install \
  --log-level ERROR \
  >>"${WORKSPACE}/rosclawd.stdout" \
  2>>"${WORKSPACE}/rosclawd.stderr" &
DAEMON_PID="$!"

DAEMON_READY=0
for _attempt in $(seq 1 200); do
  if "${CLI}" daemon status \
    --socket "${SOCKET}" \
    --json \
    >"${WORKSPACE}/daemon-restart-status.json" \
    2>/dev/null; then
    DAEMON_READY=1
    break
  fi
  if ! kill -0 "${DAEMON_PID}" 2>/dev/null; then
    cat "${WORKSPACE}/rosclawd.stderr" >&2
    echo "ERROR: wheel-installed rosclawd failed its ledger restart." >&2
    exit 1
  fi
  sleep 0.05
done

if [[ "${DAEMON_READY}" != "1" ]]; then
  cat "${WORKSPACE}/rosclawd.stderr" >&2
  echo "ERROR: restarted wheel-installed rosclawd did not become ready." >&2
  exit 1
fi

"${CLI}" daemon action-status action-clean-install-durable \
  --socket "${SOCKET}" \
  --json \
  >"${WORKSPACE}/daemon-restored-action.json"
"${CLI}" daemon stop \
  --socket "${SOCKET}" \
  --json \
  >"${WORKSPACE}/daemon-restart-stop.json"
wait "${DAEMON_PID}"
DAEMON_PID=""

if [[ -e "${SOCKET}" ]]; then
  echo "ERROR: restarted wheel-installed rosclawd left its socket behind." >&2
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
app_install = json.loads((root / "app-install.json").read_text(encoding="utf-8"))
app_validate = json.loads((root / "app-validate.json").read_text(encoding="utf-8"))
daemon_status = json.loads((root / "daemon-status.json").read_text(encoding="utf-8"))
daemon_stop = json.loads((root / "daemon-stop.json").read_text(encoding="utf-8"))
daemon_action = json.loads((root / "daemon-action.json").read_text(encoding="utf-8"))
restart_status = json.loads(
    (root / "daemon-restart-status.json").read_text(encoding="utf-8")
)
restored_action = json.loads(
    (root / "daemon-restored-action.json").read_text(encoding="utf-8")
)
restart_stop = json.loads(
    (root / "daemon-restart-stop.json").read_text(encoding="utf-8")
)

assert receipt["final_state"] == "COMPLETED"
assert receipt["evidence_level"] == "TASK_VERIFIED"
assert receipt["simulation_result"]["has_physics"] is True
assert receipt["verification_result"]["success"] is True
assert explanation["run_id"] == receipt["action_id"]
assert explanation["verification"]["task_verified"] is True
assert app_install["kind"] == "App"
assert app_install["name"] == "realsense-inspect"
assert app_validate["valid"] is True
assert app_validate["capabilities"] == [
    "camera.capture_rgbd",
    "vlm.risk_assessment",
]
assert daemon_status["running"] is True
assert daemon_status["southbound_owner"] == "rosclawd"
assert daemon_status["robot_id"] == "clean_install"
assert daemon_status["hardware_actions_executed"] == 0
assert daemon_status["ledger"]["integrity_verified"] is True
assert daemon_stop["shutdown_requested"] is True
assert daemon_action["state"] == "FINISHED"
assert daemon_action["receipt"]["final_state"] == "FAILED"
assert daemon_action["error_code"] == "EXECUTOR_UNAVAILABLE"
assert restart_status["ledger"]["integrity_verified"] is True
assert restart_status["ledger"]["write_failed"] is False
assert restart_status["recovery"]["required"] is False
assert restored_action["daemon_peer"]["pid"] != daemon_action["daemon_peer"]["pid"]
assert restored_action["daemon_peer"]["uid"] == daemon_action["daemon_peer"]["uid"]
assert {
    key: value for key, value in restored_action.items() if key != "daemon_peer"
} == {
    key: value for key, value in daemon_action.items() if key != "daemon_peer"
}
assert restart_stop["shutdown_requested"] is True

installed = root / "venv" / "lib"
assert any(installed.glob("python*/site-packages/rosclaw/product/status.yaml"))
assert any(
    installed.glob(
        "python*/site-packages/rosclaw/app/builtins/realsense-inspect/app.yaml"
    )
)
assert find_spec("rosclaw_how") is None
assert find_spec("rosclaw_know") is None
print("PASS clean install", receipt["action_id"])
PY
