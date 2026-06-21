#!/usr/bin/env bash
# ROSClaw First Boot Acceptance Test
# This script validates the full curl|bash + firstboot + doctor flow.
#
# For local development (uses current source instead of remote package):
#   ROSCLAW_DEV_SOURCE=1 bash scripts/acceptance_firstboot.sh

set -euo pipefail

# Keep pip, venv, and mktemp build artifacts off the often-tiny /tmp filesystem
# when /data is available (e.g. the rosclaw lab host); otherwise fall back to a
# directory inside the temporary workspace.
if [ -d /data ] && [ -w /data ]; then
  export TMPDIR="${TMPDIR:-/data/tmp}"
else
  export TMPDIR=""
fi

TMP_HOME="$(mktemp -d)"
if [ -z "${TMPDIR:-}" ]; then
  export TMPDIR="$TMP_HOME/tmp"
fi
mkdir -p "$TMPDIR"
export ROSCLAW_HOME="$TMP_HOME/.rosclaw"
export ROSCLAW_CHANNEL="${ROSCLAW_CHANNEL:-dev}"
export PIP_CACHE_DIR="$TMP_HOME/.pip-cache"
mkdir -p "$PIP_CACHE_DIR"

echo "== ROSClaw First Boot Acceptance =="
echo "ROSCLAW_HOME=$ROSCLAW_HOME"

if [ "${ROSCLAW_DEV_SOURCE:-0}" = "1" ]; then
  echo "[dev] Installing from local source"
  VENV_DIR="$TMP_HOME/.rosclaw-venv"
  python3 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install --no-cache-dir --upgrade pip wheel setuptools
  "$VENV_DIR/bin/python" -m pip install --no-cache-dir -e .
  export PATH="$VENV_DIR/bin:$PATH"
  mkdir -p "$ROSCLAW_HOME/state"
  "$VENV_DIR/bin/python" - <<'PY' > "$ROSCLAW_HOME/state/install_id"
import uuid
print(uuid.uuid4())
PY
else
  bash scripts/get.sh
fi

command -v rosclaw
rosclaw --version

rosclaw firstboot --yes --profile offline --no-telemetry --robot sim_ur5e --safety strict

test -f "$ROSCLAW_HOME/config/rosclaw.yaml"
test -f "$ROSCLAW_HOME/config/mcp.json"
test -f "$ROSCLAW_HOME/config/telemetry.yaml"
test -f "$ROSCLAW_HOME/state/install.json"
test -f "$ROSCLAW_HOME/state/workspace.json"

rosclaw doctor --bootstrap
rosclaw doctor --full --json > "$ROSCLAW_HOME/artifacts/reports/doctor.json"

python3 -m json.tool "$ROSCLAW_HOME/artifacts/reports/doctor.json" >/dev/null

rosclaw config show > /dev/null
rosclaw config path | grep -q "rosclaw.yaml"
rosclaw profile current | grep -q "offline"

echo "✅ Acceptance passed"
echo "Workspace: $ROSCLAW_HOME"
