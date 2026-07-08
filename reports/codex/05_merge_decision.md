# PR #55 Merge Decision

DO NOT MERGE

## Reasons

- `ruff format --check .` fails repo-wide: 498 files would be reformatted.
- `mypy src/rosclaw` does not complete because `.venv-codex/lib/python3.11/site-packages/mcap/__init__.py` is discovered twice as `mcap` and `mcap.__init__`.
- The task requires `rosclaw darwin --help`; the top-level `darwin` command is missing.
- The task requires real SeekDB/OceanBase ingest/query evidence at `localhost:2881`; PR #55 currently implements and validates a local SQLite `--seekdb-path` backend.
- ROS bridge Loop B was only endpoint-smoked, not validated with topic list/read and sandbox-blocked unsafe action.

## Evidence

- Practice local closed loop passes:
  - 9-event RH56 fixture recorded through `RuntimeBus -> PracticeRecorder`
  - `verify --strict` passes valid fixture and fails invalid envelope tests
  - sha256 manifest detects artifact tampering
  - `distill` writes derived summaries
  - SQLite-backed `ingest-seekdb` writes 7 records
  - query modes return failures, body cognition, sim2real deltas, candidates, and interventions
  - Parquet and LeRobot exports generate real files
- Test status after fixes:
  - `pytest tests/practice -q`: 147 passed, 3 skipped
  - `pytest -q`: 3672 passed, 26 skipped, 15 deselected
  - `ruff check .`: pass
  - hidden Unicode scan: pass
- Blocker status:
  - `ruff format --check .`: fail
  - `mypy src/rosclaw`: fail
  - `rosclaw darwin --help`: fail
  - real `--seekdb-url http://localhost:2881`: unavailable

## Remaining Risk

- The Practice loop is now credible for local deterministic data, but it should not be represented as verified against the live SeekDB/OceanBase container.
- The broader runtime loop from Provider through Sandbox, Practice, Memory, How, Auto, Darwin, and Skill Registry is not fully proven by one executable integration scenario.
- Formatting debt is large enough that merging PR #55 now would leave the repository in a state that fails the documented contribution gate.

## If This Is Merged Later

- Merge only after either repo-wide formatting is intentionally applied or the format gate is changed.
- Add a real SeekDB backend path or change the CLI/docs to say the current implementation is SQLite-local only.
- Add the missing Darwin CLI or remove it from advertised acceptance.
- Promote the RH56 fixture loop into CI.
