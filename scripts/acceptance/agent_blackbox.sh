#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
WORKSPACE="${ROSCLAW_ACCEPTANCE_HOME:-$(mktemp -d)}"
PROJECT="${WORKSPACE}/agent-project"

cleanup() {
  if [[ -z "${ROSCLAW_ACCEPTANCE_HOME:-}" ]]; then
    rm -rf "${WORKSPACE}"
  fi
}
trap cleanup EXIT

mkdir -p "${PROJECT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export ROSCLAW_HOME="${WORKSPACE}/home"
export ROSCLAW_AGENT_CLIENT="codex"
export CODEX_HOME="${WORKSPACE}/codex-home"

"${PYTHON}" -m rosclaw.entrypoint agent install universal \
  --project-root "${PROJECT}" \
  --skip-secrets >/dev/null

"${PYTHON}" - "${CODEX_HOME}" "${PROJECT}" <<'PY'
import sys
from pathlib import Path

codex_home = Path(sys.argv[1])
project = Path(sys.argv[2]).resolve()
codex_home.mkdir(parents=True, exist_ok=True)
(codex_home / "config.toml").write_text(
    f'[projects."{project}"]\ntrust_level = "trusted"\n',
    encoding="utf-8",
)
PY

"${PYTHON}" -m rosclaw.entrypoint agent test codex \
  --project-root "${PROJECT}" \
  --quick \
  --mcp-probe | tee "${WORKSPACE}/probe.txt"

"${PYTHON}" - "${WORKSPACE}" "${PROJECT}" <<'PY'
import json
import sys
from pathlib import Path

workspace = Path(sys.argv[1])
project = Path(sys.argv[2])
home = workspace / "home"

for relative in (
    ".mcp.json",
    "AGENTS.md",
    "CLAUDE.md",
    "ROSCLAW.md",
    ".agents/skills/rosclaw/SKILL.md",
    ".codex/config.toml",
    ".rosclaw/agent/context.snapshot.json",
):
    assert (project / relative).is_file(), relative

probe = (workspace / "probe.txt").read_text(encoding="utf-8")
assert "MCP stdio probe: OK" in probe
assert "MCP tools discovered: 22" in probe
assert "MCP verified simulation run:" in probe
assert "Codex project trust: yes" in probe

latest = json.loads((home / "runs" / "latest.json").read_text(encoding="utf-8"))
run_id = latest["run_id"]
receipt = json.loads(
    (home / "runs" / run_id / "receipt.json").read_text(encoding="utf-8")
)
assert receipt["final_state"] == "COMPLETED"
assert receipt["evidence_level"] == "TASK_VERIFIED"
assert receipt["execution_mode"] == "SIMULATION"
assert receipt["verification_result"]["success"] is True

audit_path = home / "logs" / "mcp" / "audit.jsonl"
entries = [
    json.loads(line)
    for line in audit_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
tools = {entry["tool"] for entry in entries}
assert {
    "get_product_status",
    "list_product_demos",
    "run_product_demo",
    "get_execution_receipt",
    "explain_execution",
}.issubset(tools)
assert all(entry["agent_client"] == "codex" for entry in entries)
assert all(entry["tool"] not in {"serial_write", "execute_raw", "move_robot"} for entry in entries)

print(f"PASS Agent/MCP protocol black-box {run_id} (simulation only)")
PY
