#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROSCLAW_ACCEPTANCE_PYTHON:-python3}"
DAEMON_USER="${ROSCLAW_DAEMON_TEST_USER:-daemon}"
AGENT_USER="${ROSCLAW_AGENT_TEST_USER:-nobody}"
CLIENT_GROUP="${ROSCLAW_AGENT_TEST_GROUP:-nogroup}"
WORKSPACE="$(mktemp -d /tmp/rosclaw-cross-uid.XXXXXX)"
HOME_DIR="${WORKSPACE}/home"
RUN_DIR="${WORKSPACE}/run"
SOCKET="${RUN_DIR}/rosclawd.sock"
STATUS_JSON="${WORKSPACE}/status.json"
SECURITY_JSON="${WORKSPACE}/security.json"
STDOUT_LOG="${WORKSPACE}/rosclawd.stdout"
STDERR_LOG="${WORKSPACE}/rosclawd.stderr"
SUPERVISOR_PID=""
DAEMON_PID=""

fail() {
  echo "ERROR: $*" >&2
  if [[ -f "${STDERR_LOG}" ]]; then
    tail -n 100 "${STDERR_LOG}" >&2 || true
  fi
  exit 1
}

require_principal() {
  local database="$1"
  local principal="$2"
  getent "${database}" "${principal}" >/dev/null || fail "missing ${database} principal: ${principal}"
}

# Enter through root so sudoers need not authorize arbitrary Runas groups.
run_as() {
  local user="$1"
  local uid
  shift
  uid="$(id -u "${user}")"
  (
    cd /
    sudo -n "${SETPRIV}" --reuid "${uid}" --regid "${CLIENT_GID}" --clear-groups \
      env -u PYTHONPATH \
      PYTHONNOUSERSITE=1 \
      ROSCLAW_DAEMON_SOCKET="${SOCKET}" \
      ROSCLAW_DAEMON_UID="${DAEMON_UID}" \
      "$@"
  )
}

run_agent() {
  run_as "${AGENT_USER}" "$@"
}

start_daemon() {
  : >>"${STDOUT_LOG}"
  : >>"${STDERR_LOG}"
  (
    cd /
    sudo -n "${SETPRIV}" --reuid "${DAEMON_UID}" --regid "${CLIENT_GID}" --clear-groups \
      env -u PYTHONPATH \
      PYTHONNOUSERSITE=1 \
      ROSCLAW_HOME="${HOME_DIR}" \
      ROSCLAW_DAEMON_SOCKET="${SOCKET}" \
      "${PYTHON}" -m rosclaw.daemon.cli \
        --socket "${SOCKET}" \
        --socket-mode 0660 \
        --socket-group "${CLIENT_GROUP}" \
        --robot-id sim_ur5e \
        --log-level ERROR
  ) >>"${STDOUT_LOG}" 2>>"${STDERR_LOG}" &
  SUPERVISOR_PID="$!"

  for _ in $(seq 1 600); do
    if run_agent "${PYTHON}" -m rosclaw.entrypoint daemon status --json \
      >"${STATUS_JSON}" 2>/dev/null; then
      DAEMON_PID="$("${PYTHON}" - "${STATUS_JSON}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    print(json.load(stream)["daemon_pid"])
PY
)"
      return
    fi
    if ! kill -0 "${SUPERVISOR_PID}" 2>/dev/null; then
      wait "${SUPERVISOR_PID}" 2>/dev/null || true
      fail "rosclawd exited before the Agent UID could authenticate it"
    fi
    sleep 0.05
  done
  fail "rosclawd did not become ready for the Agent UID"
}

stop_daemon() {
  if [[ -z "${DAEMON_PID}" ]]; then
    return
  fi
  sudo -n kill -TERM "${DAEMON_PID}"
  for _ in $(seq 1 400); do
    if ! sudo -n kill -0 "${DAEMON_PID}" 2>/dev/null; then
      break
    fi
    sleep 0.05
  done
  if sudo -n kill -0 "${DAEMON_PID}" 2>/dev/null; then
    sudo -n kill -KILL "${DAEMON_PID}" 2>/dev/null || true
    fail "rosclawd did not stop after SIGTERM"
  fi
  wait "${SUPERVISOR_PID}" || fail "rosclawd supervisor reported a failed shutdown"
  sudo -n test ! -e "${SOCKET}" || fail "rosclawd left its socket behind"
  SUPERVISOR_PID=""
  DAEMON_PID=""
}

cleanup() {
  if [[ -n "${DAEMON_PID}" ]]; then
    sudo -n kill -TERM "${DAEMON_PID}" 2>/dev/null || true
    for _ in $(seq 1 100); do
      if ! sudo -n kill -0 "${DAEMON_PID}" 2>/dev/null; then
        break
      fi
      sleep 0.05
    done
    sudo -n kill -KILL "${DAEMON_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SUPERVISOR_PID}" ]]; then
    wait "${SUPERVISOR_PID}" 2>/dev/null || true
  fi
  sudo -n rm -rf "${WORKSPACE}" || true
}
trap cleanup EXIT

[[ "$(uname -s)" == "Linux" ]] || fail "cross-UID acceptance requires Linux SO_PEERCRED"
command -v sudo >/dev/null || fail "sudo is required"
SETPRIV="$(command -v setpriv)" || fail "setpriv is required"
sudo -n true || fail "passwordless sudo is required"
PYTHON="$(command -v "${PYTHON}")"
[[ -x "${PYTHON}" ]] || fail "Python is not executable: ${PYTHON}"
require_principal passwd "${DAEMON_USER}"
require_principal passwd "${AGENT_USER}"
require_principal group "${CLIENT_GROUP}"
DAEMON_UID="$(id -u "${DAEMON_USER}")"
AGENT_UID="$(id -u "${AGENT_USER}")"
CLIENT_GID="$(getent group "${CLIENT_GROUP}" | cut -d: -f3)"
[[ "${DAEMON_UID}" != "${AGENT_UID}" ]] || fail "daemon and Agent test UIDs must differ"
[[ "${DAEMON_UID}" != "0" && "${AGENT_UID}" != "0" && "${CLIENT_GID}" != "0" ]] || \
  fail "cross-UID acceptance refuses root principals"

IMPORT_PATH="$(run_as "${DAEMON_USER}" "${PYTHON}" - <<'PY'
from pathlib import Path

import rosclaw

print(Path(rosclaw.__file__).resolve())
PY
)" || fail "the daemon UID cannot import ROSClaw from ${PYTHON}"
case "${IMPORT_PATH}" in
  "${ROOT}"/*) fail "cross-UID acceptance requires an installed wheel, not the source tree" ;;
esac
run_agent "${PYTHON}" -c "import rosclaw" || fail "the Agent UID cannot import installed ROSClaw"
run_as "${DAEMON_USER}" test ! -w "${PYTHON}"
run_as "${DAEMON_USER}" test ! -w "${IMPORT_PATH}"
run_agent test ! -w "${PYTHON}"
run_agent test ! -w "${IMPORT_PATH}"
run_agent test ! -w "$(dirname "${PYTHON}")"
run_agent test ! -w "$(dirname "${IMPORT_PATH}")"

mkdir -p "${WORKSPACE}"
chmod 0755 "${WORKSPACE}"
sudo -n install -d -o "${DAEMON_USER}" -g "${CLIENT_GROUP}" -m 0700 "${HOME_DIR}"
sudo -n install -d -o "${DAEMON_USER}" -g "${CLIENT_GROUP}" -m 0750 "${RUN_DIR}"
touch "${STDOUT_LOG}" "${STDERR_LOG}"
chmod 0600 "${STDOUT_LOG}" "${STDERR_LOG}"

start_daemon
run_agent "${PYTHON}" -m rosclaw.entrypoint daemon security-check --json >"${SECURITY_JSON}"
"${PYTHON}" - "${STATUS_JSON}" "${SECURITY_JSON}" "${DAEMON_UID}" "${AGENT_UID}" <<'PY'
import json
import sys

status_path, security_path, daemon_uid, agent_uid = sys.argv[1:]
with open(status_path, encoding="utf-8") as stream:
    status = json.load(stream)
with open(security_path, encoding="utf-8") as stream:
    security = json.load(stream)

assert status["daemon_uid"] == int(daemon_uid)
assert status["client_peer"]["uid"] == int(agent_uid)
assert status["daemon_peer"]["uid"] == int(daemon_uid)
assert status["privilege_separated"] is True
assert status["ledger"]["integrity_verified"] is True
assert status["ledger"]["write_failed"] is False
assert security["schema_version"] == "rosclaw.daemon.security_check.v2"
for key in (
    "boundary_ready",
    "daemon_observed_client",
    "daemon_peer_matches_status",
    "daemon_uid_pinned",
    "daemon_uid_pin_matches",
    "socket_owner_matches_daemon",
    "socket_group_member",
    "socket_client_read_write",
    "runtime_directory_owner_trusted",
    "ledger_integrity_verified",
    "ledger_write_available",
    "ledger_state_private",
):
    assert security[key] is True, (key, security)
for key in (
    "socket_world_accessible",
    "runtime_directory_group_world_writable",
    "runtime_directory_client_writable",
    "client_in_dialout",
):
    assert security[key] is False, (key, security)
assert security["writable_serial_devices"] == []
assert security["readable_serial_devices"] == []
assert security["accessible_serial_devices"] == []
assert security["ledger_state_client_accessible"] == []
PY

run_agent "${PYTHON}" - <<'PY'
from rosclaw.daemon.client import DaemonClient, DaemonRequestError
from rosclaw.kernel import ActionEnvelope, AuthorizationContext, ExecutionMode

client = DaemonClient()
for operation in (
    client.shutdown,
    lambda: client.acknowledge_recovery("Agent must not review recovery"),
):
    try:
        operation()
    except DaemonRequestError as error:
        assert error.code == "PERMISSION_DENIED", error.code
    else:
        raise AssertionError("Agent UID invoked a daemon-UID-only operation")

action = ActionEnvelope(
    action_id="cross-uid-forged-real",
    actor_id="cross-uid-agent",
    agent_framework="acceptance",
    session_id="cross-uid",
    body_id="sim_ur5e",
    body_snapshot_hash="sha256:cross-uid",
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
receipt = client.wait_for_action(ticket["action_id"], timeout_sec=5.0)["receipt"]
assert receipt["final_state"] == "BLOCKED"
assert receipt["errors"][0]["code"] == "AUTHORIZATION_REQUIRED"
assert client.get_runtime_status()["hardware_actions_executed"] == 0
PY

LEDGER_DIR="${HOME_DIR}/state/daemon"
run_agent test ! -r "${LEDGER_DIR}/ledger.sqlite3"
run_agent test ! -w "${LEDGER_DIR}/ledger.sqlite3"
run_agent test ! -r "${LEDGER_DIR}/ledger.sqlite3.anchor"
run_agent test ! -r "${LEDGER_DIR}/ledger.key"
run_agent test ! -w "${RUN_DIR}"
read -r socket_mode socket_uid socket_gid < <(sudo -n stat -c '%a %u %g' "${SOCKET}")
read -r run_mode run_uid run_gid < <(sudo -n stat -c '%a %u %g' "${RUN_DIR}")
[[ "${socket_mode}:${socket_uid}:${socket_gid}" == "660:${DAEMON_UID}:${CLIENT_GID}" ]]
[[ "${run_mode}:${run_uid}:${run_gid}" == "750:${DAEMON_UID}:${CLIENT_GID}" ]]

stop_daemon
start_daemon
run_agent "${PYTHON}" - <<'PY'
from rosclaw.daemon.client import DaemonClient

client = DaemonClient()
status = client.get_action_status("cross-uid-forged-real")
assert status["state"] == "FINISHED"
assert status["final_state"] == "BLOCKED"
assert status["error_code"] == "AUTHORIZATION_REQUIRED"
assert client.get_execution_receipt(status["action_id"])["receipt"] == status["receipt"]
assert client.get_runtime_status()["hardware_actions_executed"] == 0
PY
stop_daemon

echo "PASS clean-wheel cross-UID boundary, private state, denied control operations, forged REAL block, and durable restart"
