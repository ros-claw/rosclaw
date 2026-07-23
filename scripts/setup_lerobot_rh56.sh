#!/usr/bin/env bash
# ROSClaw × LeRobot clean-room install for the RH56 reference body (P5-E).
#
# Repeatable on a clean machine with a rosclaw checkout:
#   1. install rosclaw into a venv (or reuse ROSCLAW_PYTHON)
#   2. rosclaw setup lerobot --reference-policy rh56
#      (isolated py3.12 runtime + lerobot 0.6.x + worker plugin + smoke)
#   3. rosclaw lerobot rollout proposal-only smoke
#   4. rosclaw lerobot rollout preflight (RH56 binding + calibration)
#
# Usage:
#   scripts/setup_lerobot_rh56.sh [--rosclaw-python PATH] [--upgrade]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPGRADE="${UPGRADE:-0}"
ROSCLAW_PYTHON="${ROSCLAW_PYTHON:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rosclaw-python) ROSCLAW_PYTHON="$2"; shift 2 ;;
    --upgrade) UPGRADE=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

log() { echo "[setup-lerobot-rh56] $*"; }

# 1. rosclaw itself -----------------------------------------------------------
if [[ -z "${ROSCLAW_PYTHON}" ]]; then
  if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    log "creating rosclaw venv at ${REPO_ROOT}/.venv"
    python3 -m venv "${REPO_ROOT}/.venv"
    "${REPO_ROOT}/.venv/bin/pip" install --quiet --upgrade pip
    "${REPO_ROOT}/.venv/bin/pip" install --quiet -e "${REPO_ROOT}"
  fi
  ROSCLAW_PYTHON="${REPO_ROOT}/.venv/bin/python"
fi
log "rosclaw python: ${ROSCLAW_PYTHON}"

# 2. isolated lerobot runtime + RH56 reference policy -------------------------
SETUP_ARGS=(setup lerobot --mode isolated --reference-policy rh56 --json)
if [[ "${UPGRADE}" == "1" ]]; then
  SETUP_ARGS+=(--upgrade)
fi
log "rosclaw ${SETUP_ARGS[*]}"
"${ROSCLAW_PYTHON}" -m rosclaw.cli "${SETUP_ARGS[@]}"

# 3. proposal-only smoke -------------------------------------------------------
# proposal-only requires an observation fixture — generate a 5-step RH56
# hold_current fixture locally (no dev-machine assumptions).
SMOKE_FIXTURE="$(mktemp /tmp/rh56_smoke_obs.XXXXXX.json)"
trap 'rm -f "${SMOKE_FIXTURE}"' EXIT
"${ROSCLAW_PYTHON}" - <<'PYEOF' > "${SMOKE_FIXTURE}"
import json
obs = [
    {
        "observation.state": [1000.0] * 6,
        "state_names": ["little", "ring", "middle", "index", "thumb", "thumb_rot"],
        "task": "hold_current",
    }
    for _ in range(5)
]
print(json.dumps({"observations": obs}))
PYEOF
log "proposal-only smoke"
"${ROSCLAW_PYTHON}" -m rosclaw.cli lerobot rollout proposal-only \
  --policy.path "${REPO_ROOT}/worker_plugins/lerobot_policy_rosclaw_rh56/policies/rh56_reference_policy_v1" \
  --observation-fixture "${SMOKE_FIXTURE}" \
  --steps 5 --json || {
    echo "[setup-lerobot-rh56] proposal-only smoke FAILED" >&2; exit 1; }

# 4. RH56 preflight ------------------------------------------------------------
log "rh56 preflight"
"${ROSCLAW_PYTHON}" -m rosclaw.cli lerobot rollout preflight \
  --transport-profile "${REPO_ROOT}/configs/rh56_left_rs485_v1.yaml" \
  --calibration "${REPO_ROOT}/configs/rh56_left_01_calibration.yaml" \
  --json || {
    echo "[setup-lerobot-rh56] preflight FAILED" >&2; exit 1; }

log "OK: clean-room install complete"
