# Regression Plan

## P0: Merge Gate Hygiene

- Decide whether to apply repo-wide `ruff format .` in a dedicated formatting PR, or narrow formatter scope in project config. Current `ruff format --check .` fails on 499 files.
- Fix `mypy src/rosclaw` virtualenv/package discovery. The current command follows `.venv-codex` and fails on `mcap` before checking ROSClaw source.
- Add or document the top-level `rosclaw darwin` CLI. Current docs/task expect it, but smoke fails.

## P1: Practice Closed Loop

- Keep the RH56 fixture as a regression seed:
  - `record --fixture`
  - `verify --strict`
  - `distill`
  - `ingest-seekdb`
  - all five query modes
  - `export parquet`
  - `export lerobot`
  - artifact tamper detection
- Add catalog backward-compatibility tests requested by the task:
  - v1 catalog read under v2 code
  - old event schema still readable in non-strict mode
- Keep backend failure tests:
  - invalid `--seekdb-path` must return rc 1 with a clear message
  - invalid query backend must return rc 1 with a clear message

## P2: Real SeekDB/OceanBase

- Add a backend option for the task's `--seekdb-url http://localhost:2881` flow or explicitly rename the CLI to `ingest-sqlite` to avoid false claims.
- Add idempotent repeated ingest tests against the real backend.
- Add query filters for robot_id, task_id, skill_id, and time range against the real backend.

## P3: ROS Bridge Integration

- Extend `scripts/codex/validate_full_runtime.sh` from socket smoke to real rosbridge list/read operations.
- Add tests that prove unsafe direct `/cmd_vel` or equivalent commands are blocked by sandbox/firewall and recorded to Practice.

## P4: Runtime Self-Evolution Loop

- Add dry-run tests for:
  - Practice failure -> Memory evidence
  - Memory evidence -> How intervention
  - How intervention -> Auto proposal
  - Auto proposal -> Darwin dry-run
  - Darwin result -> Skill candidate/champion gate
- Do not mark any candidate as champion without sandbox, Darwin, and human/promotion gate evidence.

