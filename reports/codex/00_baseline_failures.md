# Current Baseline

Date: 2026-07-09

Repository: `/home/ubuntu/rosclaw/rosclaw/rosclaw-v1.0`

Branch: `main`

PR #55 merge commit: `342d81735df7ad03c6ffa8346fb473ce1f5457dc`

## Current Gates

| Command | Result |
|---|---|
| `python -m compileall -q src tests` | pass |
| `ruff check .` | pass |
| `ruff format --check .` | pass, 891 files formatted |
| `.venv-codex/bin/mypy src/rosclaw` | pass, 457 source files |
| `pytest -q` | pass, 3712 passed, 30 skipped, 15 deselected |
| `scripts/codex/validate_full_runtime.sh` | pass, `FAILURES=0` |

Full validation evidence: `reports/codex/20260709_074712/commands.log`.

## Resolved Historical Failures

- Repo-wide format debt is cleared.
- Top-level `rosclaw darwin --help` is present.
- Provider `health`, explainable `route`, and safe `benchmark --dry-run` commands are present.
- Universal agent installation and MCP stdio discovery pass with 13 tools.
- UR5e MuJoCo verification passes with real physics state.
- ROS2 rosbridge discovery passes on ports 9090 and 32887.
- ROS1 Noetic ping, discovery, manifest compilation, and read-only pose
  subscription pass through the published port 9091.
- Public Hub `owner/repo` resolution produces a remote install plan without
  writing cache, project, or telemetry files in dry-run mode.
- DeepSeek CLI invocation uses the built-in provider path, preserves upstream
  failure status, and passes a real local HTTP protocol test.
- Practice record, strict verify, distill, query, Parquet export, and LeRobot export pass.
- Real SeekDB/OceanBase ingestion and query pass through the native MySQL-compatible port 2881.
- Repeated real SeekDB ingestion is idempotent across all seven Practice tables.
- Runtime safety failure now flows through Practice, Memory, How, Auto,
  sandbox/Darwin evaluation, and simulated Skill Registry promotion.

## Remaining External Conditions

- The supplied DeepSeek account reaches the official endpoint but returns
  `402 Insufficient Balance`; a paid-provider success response requires account
  balance outside this repository.
- Authenticated Hub publish was not attempted because no Hub write token was
  supplied. Public registry resolution and read-only dry-run are verified.
- No real robot hardware action was attempted.
