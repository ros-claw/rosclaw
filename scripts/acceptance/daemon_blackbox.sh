#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
WORKSPACE="${ROSCLAW_ACCEPTANCE_HOME:-$(mktemp -d)}"
HOME_DIR="${WORKSPACE}/home"
SOCKET="${HOME_DIR}/run/rosclawd.sock"
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

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export ROSCLAW_HOME="${HOME_DIR}"
export ROSCLAW_DAEMON_SOCKET="${SOCKET}"

"${PYTHON}" -m rosclaw.daemon.cli \
  --socket "${SOCKET}" \
  --robot-id sim_ur5e \
  --log-level ERROR \
  >"${WORKSPACE}/rosclawd.stdout" \
  2>"${WORKSPACE}/rosclawd.stderr" &
DAEMON_PID="$!"

"${PYTHON}" - "${SOCKET}" "${DAEMON_PID}" <<'PY'
import os
import sys
import time
from pathlib import Path

from rosclaw.daemon.client import DaemonClient, DaemonUnavailableError
from rosclaw.kernel import ActionEnvelope, AuthorizationContext, ExecutionMode

socket_path = Path(sys.argv[1])
expected_pid = int(sys.argv[2])
client = DaemonClient(socket_path=socket_path, timeout_sec=2.0)

deadline = time.monotonic() + 30.0
while time.monotonic() < deadline:
    try:
        status = client.get_runtime_status()
        break
    except DaemonUnavailableError:
        time.sleep(0.05)
else:
    raise AssertionError("rosclawd did not become ready")

assert status["daemon_pid"] == expected_pid
assert status["daemon_pid"] != os.getpid()
assert status["southbound_owner"] == "rosclawd"
assert status["hardware_actions_executed"] == 0

action = ActionEnvelope(
    action_id="action-daemon-acceptance-forged",
    actor_id="acceptance-agent",
    agent_framework="blackbox",
    session_id="daemon-acceptance",
    body_id="sim_ur5e",
    body_snapshot_hash="sha256:acceptance-body",
    capability_id="robot.move_joints",
    arguments={"joint_positions": [0.0] * 6},
    execution_mode=ExecutionMode.REAL,
    authorization=AuthorizationContext(
        principal_id="forged-operator",
        approved=True,
        approval_id="forged-permit",
        scopes=["*"],
    ),
)
ticket = client.request_action(action)
result = client.wait_for_action(ticket["action_id"], timeout_sec=5.0)
receipt = result["receipt"]
assert receipt["final_state"] == "BLOCKED"
assert receipt["errors"][0]["code"] == "AUTHORIZATION_REQUIRED"
assert client.get_runtime_status()["hardware_actions_executed"] == 0

stop = client.emergency_stop("acceptance halt", source="daemon-blackbox")
assert stop["request_dispatched"] is False
assert stop["physical_stop_observed"] is False
assert stop["errors"][0]["code"] == "NO_STOP_TARGETS"

assert client.shutdown()["shutdown_requested"] is True
print("PASS rosclawd process boundary, forged-permit block, truthful E-stop, shutdown")
PY

wait "${DAEMON_PID}"
DAEMON_PID=""

if [[ -e "${SOCKET}" ]]; then
  echo "rosclawd left its control socket behind" >&2
  exit 1
fi

"${PYTHON}" - "${SOCKET}" <<'PY'
import sys

from rosclaw.daemon.client import DaemonClient, DaemonUnavailableError

try:
    DaemonClient(socket_path=sys.argv[1]).get_runtime_status()
except DaemonUnavailableError:
    pass
else:
    raise AssertionError("client remained usable after rosclawd stopped")
PY

"${PYTHON}" -m rosclaw.daemon.cli \
  --socket "${SOCKET}" \
  --robot-id sim_ur5e \
  --log-level ERROR \
  >>"${WORKSPACE}/rosclawd.stdout" \
  2>>"${WORKSPACE}/rosclawd.stderr" &
DAEMON_PID="$!"

"${PYTHON}" - "${SOCKET}" "${DAEMON_PID}" <<'PY'
import os
import sys
import time
from pathlib import Path

from rosclaw.daemon.client import DaemonClient, DaemonUnavailableError

socket_path = Path(sys.argv[1])
expected_pid = int(sys.argv[2])
client = DaemonClient(socket_path=socket_path, timeout_sec=2.0)

deadline = time.monotonic() + 30.0
while time.monotonic() < deadline:
    try:
        status = client.get_runtime_status()
        break
    except DaemonUnavailableError:
        time.sleep(0.05)
else:
    raise AssertionError("restarted rosclawd did not become ready")

assert status["daemon_pid"] == expected_pid
assert status["daemon_pid"] != os.getpid()
assert status["ledger"]["integrity_verified"] is True
assert status["ledger"]["write_failed"] is False
assert status["ledger"]["event_count"] >= 4
assert status["recovery"]["required"] is False

action_id = "action-daemon-acceptance-forged"
restored = client.get_action_status(action_id)
assert restored["state"] == "FINISHED"
assert restored["final_state"] == "BLOCKED"
assert restored["error_code"] == "AUTHORIZATION_REQUIRED"
receipt = client.get_execution_receipt(action_id)["receipt"]
assert receipt == restored["receipt"]
assert client.shutdown()["shutdown_requested"] is True
print("PASS rosclawd durable receipt survives a clean process restart")
PY

wait "${DAEMON_PID}"
DAEMON_PID=""

if [[ -e "${SOCKET}" ]]; then
  echo "restarted rosclawd left its control socket behind" >&2
  exit 1
fi
