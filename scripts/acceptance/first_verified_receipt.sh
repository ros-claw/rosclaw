#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
WORKSPACE="${ROSCLAW_ACCEPTANCE_HOME:-$(mktemp -d)}"

cleanup() {
  if [[ -z "${ROSCLAW_ACCEPTANCE_HOME:-}" ]]; then
    rm -rf "${WORKSPACE}"
  fi
}
trap cleanup EXIT

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export ROSCLAW_HOME="${WORKSPACE}"

"${PYTHON}" -m rosclaw.entrypoint demo run ur5e-reach --json >"${WORKSPACE}/receipt-output.json"
"${PYTHON}" -m rosclaw.entrypoint explain latest --json >"${WORKSPACE}/explanation.json"

"${PYTHON}" - "${WORKSPACE}" <<'PY'
import json
import sys
from pathlib import Path

home = Path(sys.argv[1])
receipt = json.loads((home / "receipt-output.json").read_text(encoding="utf-8"))
explanation = json.loads((home / "explanation.json").read_text(encoding="utf-8"))

assert receipt["final_state"] == "COMPLETED"
assert receipt["evidence_level"] == "TASK_VERIFIED"
assert receipt["verified"] is True
assert receipt["simulation_result"]["has_physics"] is True
assert receipt["verification_result"]["success"] is True
assert explanation["run_id"] == receipt["action_id"]
assert explanation["verification"]["task_verified"] is True
assert Path(explanation["receipt_path"]).is_file()

print(
    "PASS",
    receipt["action_id"],
    receipt["evidence_level"],
    explanation["receipt_path"],
)
PY
