---
name: rosclaw
description: Use when operating, validating, or changing ROSClaw physical-AI runtime workflows, especially CLI smoke tests, Practice evidence loops, body/runtime checks, MCP integration, MuJoCo sandbox verification, and safe ROS or hardware boundaries.
---

# ROSClaw Agent Skill

## Safety

- Treat ROSClaw as physical-AI infrastructure. Do not publish ROS topics, actuate hardware, run real robot skills, or mutate a live workspace unless the user explicitly asks for that specific action.
- Prefer dry-run, read-only, mock, fixture, simulation, or temp-workspace commands for validation.
- Use a temporary `ROSCLAW_HOME` for CLI smoke tests that write persistent state.
- Prefer `--json` for machine checks, then validate the JSON parses.

## First Checks

Run from the repo root after installing editable dev dependencies:

```bash
python -m compileall -q src tests
ruff check .
ruff format --check .
mypy src/rosclaw
pytest tests/practice -q
```

Useful health commands:

```bash
TMP=$(mktemp -d /tmp/rosclaw-health.XXXXXX)
rosclaw doctor --json
rosclaw runtime backends
rosclaw body init --robot unitree-g1 --workspace "$TMP/ws" --force --validate --render
rosclaw provider health --json
rosclaw provider route --capability vlm.scene_graph --json
rosclaw provider benchmark --dry-run --json
rosclaw sandbox verify --case ur5e-joint-preview --json
rosclaw hub search realsense
rosclaw mcp list
rosclaw skill list
```

## Agent Integration

For a project that should expose ROSClaw to Codex, Claude Code, OpenClaw, or
another MCP-aware agent, use the cross-agent installer:

```bash
rosclaw agent install --project-root . --skip-secrets
rosclaw agent test universal --project-root . --quick --mcp-probe
```

## Practice Evidence Loop

Use the RH56 fixture for a safe end-to-end loop:

```bash
TMP=$(mktemp -d /tmp/rosclaw-practice.XXXXXX)
export ROSCLAW_HOME="$TMP/home"
rosclaw practice record --fixture tests/fixtures/practice/rh56_minimal_loop.json --out "$TMP/practice" --json
rosclaw practice verify practice_rh56_minimal_loop --data-root "$TMP/practice" --strict --json
rosclaw practice distill practice_rh56_minimal_loop --data-root "$TMP/practice" --json
rosclaw practice ingest-seekdb practice_rh56_minimal_loop --data-root "$TMP/practice" --seekdb-path "$TMP/seekdb.sqlite" --json
rosclaw memory ingest --episode-id episode_rh56_minimal_loop --data-root "$TMP/practice"
rosclaw know compile "recover from over contact" --episode-id episode_rh56_minimal_loop --data-root "$TMP/practice" --json
rosclaw how advise --body body_rh56_left --failure over_contact --episode-id episode_rh56_minimal_loop --data-root "$TMP/practice" --json
```

Expected bridge signals:

- `practice record` writes events under `sessions/practice_rh56_minimal_loop/raw/events.jsonl`.
- `memory ingest` reports `Events: 9` and `Outcome: success`.
- `know compile --json` reports `robot_id: rh56` and `evidence.event_count: 9`.
- `how advise --json` reports intervention `how_rh56_001` from `practice_how_intervention`.

## ROS Smoke

For ROS bridge discovery, use read-only commands:

```bash
rosclaw ros ping --endpoint ws://127.0.0.1:9090
rosclaw ros discover --endpoint ws://127.0.0.1:9090
```

Do not use command endpoints or publish topic messages without explicit user confirmation.

## When Changing Code

- Keep changes close to the module interface implied by the command.
- Add regression tests for any CLI-visible behavior or cross-module contract.
- For Practice layout changes, verify both old `sessions/{episode_id}` and v2 catalog-backed `episode_id -> session_id -> practice_id` evidence resolution.
- Re-run the focused test plus the broader gate that covers the touched subsystem.
