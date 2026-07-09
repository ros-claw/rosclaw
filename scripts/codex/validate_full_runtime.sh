#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

REPORT_DIR="${REPORT_DIR:-reports/codex/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$REPORT_DIR"

if [[ -x ".venv-codex/bin/python" ]]; then
  PYTHON="${PYTHON:-.venv-codex/bin/python}"
  PIP="${PIP:-.venv-codex/bin/pip}"
  RUFF="${RUFF:-.venv-codex/bin/ruff}"
  MYPY="${MYPY:-.venv-codex/bin/mypy}"
  PYTEST="${PYTEST:-.venv-codex/bin/pytest}"
else
  PYTHON="${PYTHON:-python}"
  PIP="${PIP:-pip}"
  RUFF="${RUFF:-ruff}"
  MYPY="${MYPY:-mypy}"
  PYTEST="${PYTEST:-pytest}"
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${ROSCLAW:-}" ]]; then
  ROSCLAW_CMD=("$ROSCLAW")
else
  ROSCLAW_CMD=("$PYTHON" -m rosclaw.cli)
fi

failures=0

run_required() {
  echo "===== $* =====" | tee -a "$REPORT_DIR/commands.log"
  "$@" 2>&1 | tee -a "$REPORT_DIR/commands.log"
  local rc=${PIPESTATUS[0]}
  echo "----- rc=$rc" | tee -a "$REPORT_DIR/commands.log"
  if [[ "$rc" -ne 0 ]]; then
    failures=$((failures + 1))
  fi
  return 0
}

run_required "$PYTHON" --version
run_required "$PIP" check
run_required "$PYTHON" -m compileall -q src tests
run_required "$RUFF" check .
run_required "$RUFF" format --check .
run_required "$MYPY" src/rosclaw
run_required "$PYTEST" -q
run_required "$PYTEST" -q tests/integration/test_physical_ai_agent_acceptance.py

run_required "${ROSCLAW_CMD[@]}" --help
for cmd in doctor firstboot body provider sandbox practice memory know how auto darwin skill hub mcp; do
  if "${ROSCLAW_CMD[@]}" "$cmd" --help > "$REPORT_DIR/cli_${cmd}.txt" 2>&1; then
    echo "CLI OK: $cmd" | tee -a "$REPORT_DIR/cli.log"
  else
    echo "CLI MISSING/FAILED: $cmd" | tee -a "$REPORT_DIR/cli.log"
    failures=$((failures + 1))
  fi
done

run_required "${ROSCLAW_CMD[@]}" provider health --json
run_required "${ROSCLAW_CMD[@]}" provider route --capability vlm.scene_graph --json
run_required "${ROSCLAW_CMD[@]}" provider benchmark --dry-run --json
run_required "${ROSCLAW_CMD[@]}" sandbox verify \
  --case ur5e-joint-preview \
  --steps 8 \
  --json

AGENT_ROOT="$(mktemp -d /tmp/rosclaw-codex-agent.XXXXXX)"
run_required "${ROSCLAW_CMD[@]}" agent install universal \
  --project-root "$AGENT_ROOT" \
  --skip-secrets
run_required "${ROSCLAW_CMD[@]}" agent test universal \
  --project-root "$AGENT_ROOT" \
  --quick \
  --mcp-probe

run_required "$PYTHON" - <<'PY'
import socket
import sys

failures = 0
for name, port in [
    ("ros2_humble", 9090),
    ("ros1_noetic", 9091),
    ("ros2_smoke", 32887),
    ("rosclaw_api", 8000),
    ("redis", 6379),
    ("seekdb", 2881),
]:
    sock = socket.socket()
    sock.settimeout(2)
    try:
        sock.connect(("127.0.0.1", port))
        print(f"OK {name} {port}")
    except Exception as exc:
        failures += 1
        print(f"FAIL {name} {port}: {exc}")
    finally:
        sock.close()
sys.exit(1 if failures else 0)
PY

for endpoint in ws://127.0.0.1:9090 ws://127.0.0.1:9091 ws://127.0.0.1:32887; do
  run_required "${ROSCLAW_CMD[@]}" ros ping --endpoint "$endpoint" --json
  run_required "${ROSCLAW_CMD[@]}" ros discover --endpoint "$endpoint" --json
  run_required env ROSCLAW_ROS_TEST_ENDPOINT="$endpoint" "$PYTEST" -q -m integration \
    tests/connectors/ros/test_turtlesim_integration.py \
    -k "ping_reaches or discovery_finds or compile_manifest or subscribe_once"
done

if [[ "${ROSCLAW_VALIDATE_REMOTE_HUB:-0}" == "1" ]]; then
  REMOTE_HUB_ROOT="$(mktemp -d /tmp/rosclaw-codex-remote-hub.XXXXXX)"
  run_required env \
    ROSCLAW_HOME="$REMOTE_HUB_ROOT/home" \
    ROSCLAW_MCP_HUB="${ROSCLAW_MCP_HUB:-https://www.rosclaw.io}" \
    "${ROSCLAW_CMD[@]}" mcp install ros-claw/g1-mcp \
    --dry-run \
    --project-root "$REMOTE_HUB_ROOT/project"
  run_required "$PYTHON" -c \
    'import pathlib, sys; root = pathlib.Path(sys.argv[1]); files = [p for p in root.rglob("*") if p.is_file()]; print(f"REMOTE HUB DRY-RUN FILES: {files}"); raise SystemExit(1 if files else 0)' \
    "$REMOTE_HUB_ROOT"
fi

run_required "$PYTHON" - <<'PY'
from pathlib import Path
import sys

bad = []
for path in Path(".").rglob("*"):
    if any(
        part.startswith(".venv")
        or part
        in {
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "__pycache__",
            "build",
            "dist",
        }
        for part in path.parts
    ):
        continue
    if path.is_file() and path.suffix in {".py", ".md", ".yaml", ".yml", ".json", ".toml"}:
        text = path.read_text(errors="ignore")
        for index, char in enumerate(text):
            if ord(char) in list(range(0x202A, 0x202F)) + list(range(0x2066, 0x2070)):
                bad.append((str(path), index, hex(ord(char))))
if bad:
    print("BIDI/HIDDEN FOUND")
    for item in bad[:200]:
        print(item)
    sys.exit(1)
print("OK: no bidi control chars found")
PY

PRACTICE_ROOT="$(mktemp -d /tmp/rosclaw-codex-practice.XXXXXX)"
PRACTICE_HOME="$PRACTICE_ROOT/home"
PRACTICE_DATA="$PRACTICE_ROOT/practice"
SEEKDB_PATH="$PRACTICE_ROOT/seekdb.sqlite"
PARQUET_OUT="$PRACTICE_ROOT/export/parquet/rh56_minimal_loop.parquet"
LEROBOT_OUT="$PRACTICE_ROOT/export/lerobot/rh56_minimal_loop"
mkdir -p "$PRACTICE_HOME"

run_required env ROSCLAW_HOME="$PRACTICE_HOME" "${ROSCLAW_CMD[@]}" practice record \
  --fixture tests/fixtures/practice/rh56_minimal_loop.json \
  --out "$PRACTICE_DATA" \
  --json
run_required env ROSCLAW_HOME="$PRACTICE_HOME" "${ROSCLAW_CMD[@]}" practice verify \
  practice_rh56_minimal_loop \
  --data-root "$PRACTICE_DATA" \
  --strict \
  --json
run_required env ROSCLAW_HOME="$PRACTICE_HOME" "${ROSCLAW_CMD[@]}" practice distill \
  practice_rh56_minimal_loop \
  --data-root "$PRACTICE_DATA" \
  --json
run_required env ROSCLAW_HOME="$PRACTICE_HOME" "${ROSCLAW_CMD[@]}" practice ingest-seekdb \
  practice_rh56_minimal_loop \
  --data-root "$PRACTICE_DATA" \
  --seekdb-path "$SEEKDB_PATH" \
  --json
run_required env ROSCLAW_HOME="$PRACTICE_HOME" "${ROSCLAW_CMD[@]}" practice query failures \
  --robot-id rh56 \
  --data-root "$PRACTICE_DATA" \
  --seekdb-path "$SEEKDB_PATH" \
  --json
run_required env ROSCLAW_HOME="$PRACTICE_HOME" "${ROSCLAW_CMD[@]}" practice query body-cognition \
  --body-id body_rh56_left \
  --data-root "$PRACTICE_DATA" \
  --seekdb-path "$SEEKDB_PATH" \
  --json
run_required env ROSCLAW_HOME="$PRACTICE_HOME" "${ROSCLAW_CMD[@]}" practice query sim2real \
  --body-id body_rh56_left \
  --data-root "$PRACTICE_DATA" \
  --seekdb-path "$SEEKDB_PATH" \
  --json
run_required env ROSCLAW_HOME="$PRACTICE_HOME" "${ROSCLAW_CMD[@]}" practice query candidates \
  --skill-id skill_ok_contact \
  --data-root "$PRACTICE_DATA" \
  --seekdb-path "$SEEKDB_PATH" \
  --json
run_required env ROSCLAW_HOME="$PRACTICE_HOME" "${ROSCLAW_CMD[@]}" practice query interventions \
  --failure-id fail_rh56_over_contact_1 \
  --data-root "$PRACTICE_DATA" \
  --seekdb-path "$SEEKDB_PATH" \
  --json

if [[ -n "${ROSCLAW_VALIDATE_SEEKDB_DSN:-}" ]]; then
  run_required env ROSCLAW_HOME="$PRACTICE_HOME" "${ROSCLAW_CMD[@]}" practice ingest-seekdb \
    practice_rh56_minimal_loop \
    --data-root "$PRACTICE_DATA" \
    --seekdb-url "$ROSCLAW_VALIDATE_SEEKDB_DSN" \
    --json
  run_required env ROSCLAW_HOME="$PRACTICE_HOME" "${ROSCLAW_CMD[@]}" practice ingest-seekdb \
    practice_rh56_minimal_loop \
    --data-root "$PRACTICE_DATA" \
    --seekdb-url "$ROSCLAW_VALIDATE_SEEKDB_DSN" \
    --json
  run_required env SEEKDB_DSN="$ROSCLAW_VALIDATE_SEEKDB_DSN" "$PYTHON" - <<'PY'
import os
import sys

from rosclaw.memory.seekdb_client import SeekDBMySQLClient

client = SeekDBMySQLClient(os.environ["SEEKDB_DSN"])
client.connect()
counts = {
    table: client.count(table)
    for table in [
        "episodes",
        "failures",
        "how_interventions",
        "body_cognition",
        "sim2real_deltas",
        "skill_candidates",
        "promotion_results",
    ]
}
client.disconnect()
print(f"REAL SEEKDB COUNTS: {counts}")
sys.exit(0 if set(counts.values()) == {1} else 1)
PY
  run_required env ROSCLAW_HOME="$PRACTICE_HOME" "${ROSCLAW_CMD[@]}" practice query failures \
    --robot-id rh56 \
    --data-root "$PRACTICE_DATA" \
    --seekdb-url "$ROSCLAW_VALIDATE_SEEKDB_DSN" \
    --json
fi

run_required env ROSCLAW_HOME="$PRACTICE_HOME" "${ROSCLAW_CMD[@]}" practice export \
  practice_rh56_minimal_loop \
  --format parquet \
  --data-root "$PRACTICE_DATA" \
  --output "$PARQUET_OUT"
run_required env ROSCLAW_HOME="$PRACTICE_HOME" "${ROSCLAW_CMD[@]}" practice export \
  practice_rh56_minimal_loop \
  --format lerobot \
  --data-root "$PRACTICE_DATA" \
  --output "$LEROBOT_OUT"

find "$PRACTICE_ROOT" -maxdepth 5 -type f | sort > "$REPORT_DIR/practice_artifacts.txt"

echo "REPORT_DIR=$REPORT_DIR"
echo "FAILURES=$failures"
if [[ "$failures" -ne 0 ]]; then
  exit 1
fi
