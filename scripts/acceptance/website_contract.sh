#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WEBSITE_ROOT="${ROSCLAW_WEBSITE_ROOT:-$(cd "${ROOT}/../.." && pwd)/rosclaw-website}"
PYTHON="${PYTHON:-python3}"
WORKSPACE="$(mktemp -d)"

cleanup() {
  rm -rf "${WORKSPACE}"
}
trap cleanup EXIT

[[ -f "${WEBSITE_ROOT}/content/product-status.json" ]] || {
  echo "ERROR: ROSClaw website checkout not found: ${WEBSITE_ROOT}" >&2
  exit 2
}

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON}" "${ROOT}/scripts/product/export_status.py" \
  --output "${WORKSPACE}/product-status.json"
cmp "${WORKSPACE}/product-status.json" "${WEBSITE_ROOT}/content/product-status.json"

ROSCLAW_CORE_ROOT="${ROOT}" node "${WEBSITE_ROOT}/scripts/check-product-status.mjs"

run_help() {
  "${PYTHON}" -m rosclaw.entrypoint "$@" --help >/dev/null
}

run_help firstboot
run_help body init
run_help test realsense
run_help doctor
run_help robot list
run_help agent install
run_help agent test codex
run_help hub schema export
run_help hub validate
run_help hub policy check
run_help hub publish

"${PYTHON}" -m rosclaw.entrypoint demo list --json >"${WORKSPACE}/demos.json"
"${PYTHON}" -m rosclaw.entrypoint status capabilities --json >"${WORKSPACE}/status.json"

"${PYTHON}" - "${WORKSPACE}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
demos = json.loads((root / "demos.json").read_text(encoding="utf-8"))
status = json.loads((root / "status.json").read_text(encoding="utf-8"))

assert demos["demos"][0]["id"] == "ur5e-reach"
assert status["release"]["version"] == "1.0.1"
assert status["golden_paths"]["ur5e_reach"]["dimensions"]["simulation"] == "verified"
assert status["golden_paths"]["rh56_single_step"]["agent_ready"] is False
print("PASS website contract")
PY
