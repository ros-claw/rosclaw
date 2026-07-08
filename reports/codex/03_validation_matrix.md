# ROSClaw Codex Validation Matrix

| Area | Command/Test | Expected | Actual | Status | Evidence |
|---|---|---|---|---|---|
| Install | `pip install -e ".[dev]"` | success | success | pass | editable install completed in `.venv-codex` |
| Export deps | `pip install -e ".[practice-export]"` | success | success | pass | pandas/pyarrow installed |
| Dependency check | `pip check` | success | success | pass | no broken requirements |
| Compile | `python -m compileall -q src tests` | success | success | pass | rerun after fixes |
| Lint | `ruff check .` | success | success | pass | `All checks passed!` |
| Format | `ruff format --check .` | success | rc 1, 498 files would be reformatted | fail | `/tmp/rosclaw_ruff_format.out` |
| Type | `mypy src/rosclaw` | success or documented non-blocking | rc 2, `mcap` source file found twice | fail | `/tmp/rosclaw_mypy.out` |
| Practice unit | `pytest tests/practice -q` | success | 147 passed, 3 skipped | pass | local run |
| Full unit | `pytest -q` | success | 3672 passed, 26 skipped, 15 deselected | pass | local run after fixes |
| Docs assets | `pytest tests/test_docs_assets.py::test_referenced_files_exist -q` | success | 7 passed | pass | rerun after removing ignored MCP placeholders |
| CLI root | `rosclaw --help` | success | success | pass | CLI smoke |
| CLI modules | `doctor firstboot body provider sandbox practice memory know how auto skill hub mcp --help` | success | success | pass | CLI smoke |
| CLI Darwin | `rosclaw darwin --help` | success | command missing | fail | CLI smoke |
| Docker endpoints | socket connect to 9090, 9091, 32887, 8000, 6379, 2881 | success | success | pass | socket smoke only |
| ROS bridge Loop B | list/read topics through rosbridge | success | not completed | fail | only endpoint connectivity was checked |
| Practice record | `rosclaw practice record --fixture tests/fixtures/practice/rh56_minimal_loop.json --out /tmp/... --json` | real artifacts | 9 events recorded | pass | raw `events.jsonl` generated |
| Practice verify | `rosclaw practice verify practice_rh56_minimal_loop --strict --json` | pass valid fixture | passed true | pass | no issues |
| Strict invalid event | verifier test removes `event_id`, `trace_id`, `timestamp_ns` | strict fails | strict fails | pass | `tests/practice/test_practice_verifier.py` |
| Artifact tamper | append bytes to summary then `verify --strict` | sha256 mismatch | rc 1 with mismatch | pass | manual tamper run |
| Distill | `rosclaw practice distill ... --json` | summary artifacts | failure/how/candidate/promotion/sim2real counts all 1 | pass | manual loop |
| SeekDB ingest local | `rosclaw practice ingest-seekdb ... --seekdb-path /tmp/...sqlite --json` | write records | total_records 7 | pass | local SQLite backend |
| SeekDB real 2881 | `--seekdb-url http://localhost:2881` | real write/read | not implemented | fail | CLI has no `--seekdb-url` |
| Query failures | `practice query failures --robot-id rh56` | failure result | `fail_rh56_over_contact_1` | pass | manual loop |
| Query body cognition | `practice query body-cognition --body-id body_rh56_left` | body trait | `over_contact_risk` | pass | manual loop |
| Query sim2real | `practice query sim2real --body-id body_rh56_left` | delta result | `delta_rh56_thumb_force` | pass | manual loop |
| Query candidates | `practice query candidates --skill-id skill_ok_contact` | candidate result | `candidate_rh56_thumb_backoff`, promoted | pass | manual loop |
| Query interventions | `practice query interventions --failure-id fail_rh56_over_contact_1` | intervention result | `how_rh56_001`, resolved | pass | manual loop |
| Export parquet | `practice export ... --format parquet` | parquet file | file generated | pass | `/tmp/rosclaw-codex-practice/export/parquet/rh56_minimal_loop.parquet` |
| Export LeRobot | `practice export ... --format lerobot` | LeRobot tree | data and meta files generated | pass | `/tmp/rosclaw-codex-practice/export/lerobot/rh56_minimal_loop` |
| Hidden Unicode | git-tracked source/doc scan | none | none | pass | no bidi controls found |
| PR55 merge gate | all required gates | all pass | blockers remain | fail | see `05_merge_decision.md` |
