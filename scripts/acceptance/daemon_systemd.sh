#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROSCLAW_ACCEPTANCE_PYTHON:-python3}"
DAEMON_USER="${ROSCLAW_DAEMON_TEST_USER:-daemon}"
AGENT_USER="${ROSCLAW_AGENT_TEST_USER:-nobody}"
CLIENT_GROUP="${ROSCLAW_AGENT_TEST_GROUP:-nogroup}"
UNIT="rosclawd-boundary-acceptance-$$-${RANDOM}"
RUNTIME_DIRECTORY="${UNIT}"
SOCKET="/run/${RUNTIME_DIRECTORY}/rosclawd.sock"
STATE_DIRECTORY="/var/lib/${RUNTIME_DIRECTORY}"
STATUS_JSON="$(mktemp /tmp/rosclawd-systemd-status.XXXXXX)"
SECURITY_JSON="$(mktemp /tmp/rosclawd-systemd-security.XXXXXX)"

fail() {
  echo "ERROR: $*" >&2
  sudo -n journalctl -u "${UNIT}.service" --no-pager -n 100 >&2 || true
  exit 1
}

run_agent() {
  (
    cd /
    sudo -n -u "${AGENT_USER}" -g "${CLIENT_GROUP}" env -u PYTHONPATH \
      PYTHONNOUSERSITE=1 \
      ROSCLAW_DAEMON_SOCKET="${SOCKET}" \
      ROSCLAW_DAEMON_UID="${DAEMON_UID}" \
      "$@"
  )
}

cleanup() {
  sudo -n systemctl stop "${UNIT}.service" >/dev/null 2>&1 || true
  sudo -n systemctl reset-failed "${UNIT}.service" >/dev/null 2>&1 || true
  sudo -n rm -rf "${STATE_DIRECTORY}" || true
  rm -f "${STATUS_JSON}" "${SECURITY_JSON}"
}
trap cleanup EXIT

[[ "$(uname -s)" == "Linux" ]] || { echo "ERROR: systemd acceptance requires Linux" >&2; exit 1; }
command -v sudo >/dev/null
command -v systemctl >/dev/null
command -v systemd-run >/dev/null
sudo -n true
case "$(systemctl is-system-running 2>/dev/null || true)" in
  running|degraded) ;;
  *) echo "ERROR: the system systemd manager is not running" >&2; exit 1 ;;
esac
getent passwd "${DAEMON_USER}" >/dev/null
getent passwd "${AGENT_USER}" >/dev/null
getent group "${CLIENT_GROUP}" >/dev/null
PYTHON="$(command -v "${PYTHON}")"
DAEMON_UID="$(id -u "${DAEMON_USER}")"
AGENT_UID="$(id -u "${AGENT_USER}")"
CLIENT_GID="$(getent group "${CLIENT_GROUP}" | cut -d: -f3)"
[[ "${DAEMON_UID}" != "${AGENT_UID}" ]]
[[ "${DAEMON_UID}" != "0" && "${AGENT_UID}" != "0" && "${CLIENT_GID}" != "0" ]] || \
  fail "systemd acceptance refuses root principals"

IMPORT_PATH="$(
  cd /
  sudo -n -u "${DAEMON_USER}" -g "${CLIENT_GROUP}" env -u PYTHONPATH \
    PYTHONNOUSERSITE=1 "${PYTHON}" - <<'PY'
from pathlib import Path

import rosclaw

print(Path(rosclaw.__file__).resolve())
PY
)" || fail "the daemon UID cannot import installed ROSClaw"
case "${IMPORT_PATH}" in
  "${ROOT}"/*) fail "systemd acceptance requires an installed wheel, not the source tree" ;;
esac
run_agent test ! -w "${PYTHON}"
run_agent test ! -w "${IMPORT_PATH}"
run_agent test ! -w "$(dirname "${PYTHON}")"
run_agent test ! -w "$(dirname "${IMPORT_PATH}")"

"${PYTHON}" - "${ROOT}/deploy/systemd/rosclawd.service" <<'PY'
import configparser
import sys

parser = configparser.ConfigParser(interpolation=None, strict=False)
parser.optionxform = str
with open(sys.argv[1], encoding="utf-8") as stream:
    parser.read_file(stream)
service = parser["Service"]
expected = {
    "User": "rosclaw-hw",
    "Group": "rosclaw-agent",
    "SupplementaryGroups": "dialout",
    "RuntimeDirectoryMode": "0750",
    "StateDirectoryMode": "0700",
    "UMask": "0077",
    "DevicePolicy": "closed",
    "NoNewPrivileges": "true",
    "PrivateTmp": "true",
    "ProtectSystem": "strict",
    "ProtectHome": "true",
    "ProtectClock": "true",
    "ProtectHostname": "true",
    "KeyringMode": "private",
    "RestrictAddressFamilies": "AF_UNIX AF_INET AF_INET6 AF_NETLINK",
    "RestrictNamespaces": "true",
    "SystemCallArchitectures": "native",
    "CapabilityBoundingSet": "",
    "AmbientCapabilities": "",
}
for key, value in expected.items():
    assert service.get(key) == value, (key, service.get(key), value)
PY

sudo -n rm -rf "${STATE_DIRECTORY}"
sudo -n systemd-run \
  --unit="${UNIT}" \
  --collect \
  --service-type=simple \
  --uid="${DAEMON_USER}" \
  --gid="${CLIENT_GROUP}" \
  --setenv=PYTHONNOUSERSITE=1 \
  --setenv=ROSCLAW_HOME="${STATE_DIRECTORY}" \
  --setenv=ROSCLAW_DAEMON_SOCKET="${SOCKET}" \
  --property=SupplementaryGroups=dialout \
  --property=RuntimeDirectory="${RUNTIME_DIRECTORY}" \
  --property=RuntimeDirectoryMode=0750 \
  --property=StateDirectory="${RUNTIME_DIRECTORY}" \
  --property=StateDirectoryMode=0700 \
  --property=UMask=0077 \
  --property=DevicePolicy=closed \
  --property=NoNewPrivileges=yes \
  --property=PrivateTmp=yes \
  --property=ProtectSystem=strict \
  --property=ProtectHome=yes \
  --property=ProtectControlGroups=yes \
  --property=ProtectClock=yes \
  --property=ProtectHostname=yes \
  --property=ProtectKernelLogs=yes \
  --property=ProtectKernelModules=yes \
  --property=ProtectKernelTunables=yes \
  --property=KeyringMode=private \
  --property=LockPersonality=yes \
  --property=MemoryDenyWriteExecute=yes \
  --property='RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6 AF_NETLINK' \
  --property=RestrictNamespaces=yes \
  --property=RestrictRealtime=yes \
  --property=RestrictSUIDSGID=yes \
  --property=RemoveIPC=yes \
  --property=SystemCallArchitectures=native \
  --property='CapabilityBoundingSet=' \
  --property='AmbientCapabilities=' \
  "${PYTHON}" -m rosclaw.daemon.cli \
    --socket "${SOCKET}" \
    --socket-mode 0660 \
    --socket-group "${CLIENT_GROUP}" \
    --robot-id sim_ur5e \
    --log-level ERROR

for _ in $(seq 1 600); do
  if run_agent "${PYTHON}" -m rosclaw.entrypoint daemon status --json \
    >"${STATUS_JSON}" 2>/dev/null; then
    break
  fi
  if [[ "$(systemctl is-failed "${UNIT}.service" 2>/dev/null || true)" == "failed" ]]; then
    fail "the transient rosclawd unit failed during startup"
  fi
  sleep 0.05
done
[[ -s "${STATUS_JSON}" ]] || fail "the Agent UID could not reach the transient rosclawd unit"
run_agent "${PYTHON}" -m rosclaw.entrypoint daemon security-check --json >"${SECURITY_JSON}"

run_agent "${PYTHON}" - <<'PY'
from rosclaw.daemon.client import DaemonClient, DaemonRequestError
from rosclaw.kernel import ActionEnvelope, AuthorizationContext, ExecutionMode

client = DaemonClient()
status = client.get_runtime_status()
assert status["privilege_separated"] is True
try:
    client.shutdown()
except DaemonRequestError as error:
    assert error.code == "PERMISSION_DENIED"
else:
    raise AssertionError("Agent UID shut down the systemd daemon")

action = ActionEnvelope(
    action_id="systemd-cross-uid-forged-real",
    actor_id="systemd-agent",
    agent_framework="acceptance",
    session_id="systemd-cross-uid",
    body_id="sim_ur5e",
    body_snapshot_hash="sha256:systemd-cross-uid",
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
assert security["schema_version"] == "rosclaw.daemon.security_check.v2"
assert security["boundary_ready"] is True
assert security["ledger_state_private"] is True
assert security["runtime_directory_client_writable"] is False
assert security["accessible_serial_devices"] == []
assert security["writable_serial_devices"] == []
PY

LEDGER_DIR="${STATE_DIRECTORY}/state/daemon"
run_agent test ! -r "${LEDGER_DIR}/ledger.sqlite3"
run_agent test ! -r "${LEDGER_DIR}/ledger.key"
run_agent test ! -w "/run/${RUNTIME_DIRECTORY}"

PROPERTIES="$(systemctl show "${UNIT}.service" \
  -p User -p Group -p SupplementaryGroups \
  -p RuntimeDirectoryMode -p StateDirectoryMode -p UMask \
  -p DevicePolicy -p NoNewPrivileges -p PrivateTmp \
  -p ProtectSystem -p ProtectHome -p ProtectClock -p ProtectHostname \
  -p KeyringMode -p RestrictAddressFamilies -p RestrictNamespaces \
  -p SystemCallArchitectures -p CapabilityBoundingSet)"
expect_property() {
  local expected="$1"
  grep -Fqx "${expected}" <<<"${PROPERTIES}" || fail "missing systemd property: ${expected}"
}
expect_property "User=${DAEMON_USER}"
expect_property "Group=${CLIENT_GROUP}"
expect_property 'SupplementaryGroups=dialout'
expect_property 'RuntimeDirectoryMode=0750'
expect_property 'StateDirectoryMode=0700'
expect_property 'UMask=0077'
expect_property 'DevicePolicy=closed'
expect_property 'NoNewPrivileges=yes'
expect_property 'PrivateTmp=yes'
expect_property 'ProtectSystem=strict'
expect_property 'ProtectHome=yes'
expect_property 'ProtectClock=yes'
expect_property 'ProtectHostname=yes'
expect_property 'KeyringMode=private'
ADDRESS_FAMILIES="$(sed -n 's/^RestrictAddressFamilies=//p' <<<"${PROPERTIES}")"
[[ "$(tr ' ' '\n' <<<"${ADDRESS_FAMILIES}" | sort | xargs)" == \
  "AF_INET AF_INET6 AF_NETLINK AF_UNIX" ]]
expect_property 'RestrictNamespaces=yes'
expect_property 'SystemCallArchitectures=native'
expect_property 'CapabilityBoundingSet='

echo "PASS real systemd cross-UID sandbox, pinned daemon identity, private state, and forged REAL denial"
