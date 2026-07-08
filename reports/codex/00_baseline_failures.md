# Baseline Failures

Audit target: PR #55 local branch `codex/pr55-review`.

Date: 2026-07-08 local run.

## Environment

- Repository: `/home/ubuntu/rosclaw/rosclaw/rosclaw-v1.0`
- Branch: `codex/pr55-review`
- PR #55 audit started from: `4a28d0f6bd13efcae661729ed440d9074ca7244e`
- Before push, local fixes were rebased onto latest PR head: `0ba634b1a891331560d1085a8020d1625d0a4d47`
- Merge base with `origin/main`: `0916efc0a8d6c90a104e14d43ca7f594d478e132`
- Python: `.venv-codex/bin/python`, Python 3.11.15
- Install commands completed:
  - `pip install -e ".[dev]"`
  - `pip install -e ".[practice-export]"`
  - `pip check`

## Commands

| Command | Result |
|---|---|
| `.venv-codex/bin/python -m compileall -q src tests` | pass |
| `.venv-codex/bin/ruff check .` | pass |
| `.venv-codex/bin/ruff format --check .` | fail, 498 files would be reformatted |
| `.venv-codex/bin/mypy src/rosclaw` | fail before checking source because `.venv-codex/lib/python3.11/site-packages/mcap/__init__.py` is mapped as both `mcap` and `mcap.__init__` |
| `.venv-codex/bin/pytest tests/practice -q` | pass, 147 passed, 3 skipped |
| `.venv-codex/bin/pytest -q` | pass after fixes, 3672 passed, 26 skipped, 15 deselected |

## Initial Failures Found

- `ruff check .` failed on an unused `pyarrow` binding in `tests/practice/test_artifact_store.py`.
- `pytest tests/practice` failed during collection because `tests/practice/test_rh56_practice_emitter.py` hardcoded `/home/nvidia/.../rosclaw-rh56-runtime/src` and imported optional `rosclaw_rh56` unconditionally.
- Full pytest initially failed with 26 failures:
  - skill template copy tried to decode generated `.pyc` files under templates as UTF-8
  - dashboard CLI test could hang because `uvicorn.run` was not stubbed
  - docs asset test required ignored local MCP config files
  - RealSense runtime handler test did not mock local discovery
  - memory empty event id was not retrievable by the generated id
- `rosclaw darwin --help` is missing as a top-level CLI command.
- PR #55 `practice ingest-seekdb` uses SQLite `--seekdb-path`; there is no `--seekdb-url http://localhost:2881` path for the task's real SeekDB/OceanBase container gate.

## Fixes Applied

- Added `rosclaw practice record --fixture ... --out ... --json`.
- Added deterministic fixture `tests/fixtures/practice/rh56_minimal_loop.json`.
- Made Practice strict verifier detect missing event envelope fields: `event_id`, `event_type`, `trace_id`, `timestamp_ns`, `timestamp_utc`.
- Added regression coverage for fixture record, strict envelope validation, clear SeekDB/query backend failures, and docs asset handling.
- Made optional RH56 runtime tests skip cleanly when the external package is absent.
- Made skill template copying ignore `__pycache__` and `.pyc/.pyo`.
- Fixed generated id persistence in `SeekDBMemoryClient.insert`.
- Fixed dashboard smoke test by stubbing `uvicorn.run` and printing a stable dashboard title.

## Remaining Failures

- `ruff format --check .` is still a repo-wide blocker: 498 files would be reformatted.
- `mypy src/rosclaw` is still blocked by the local virtualenv package discovery issue in `mcap`.
- `rosclaw darwin --help` is still absent.
- Real SeekDB/OceanBase at `localhost:2881` was socket-smoked only; PR #55 does not implement real ingest/query against that endpoint.
