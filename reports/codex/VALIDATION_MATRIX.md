# ROSClaw Validation Matrix

Date: 2026-07-09

| Area | Actual | Status | Evidence |
|---|---|---|---|
| Install/dependencies | `pip check` clean | pass | `20260709_074712/commands.log` |
| Lint/format/type | Ruff pass; mypy pass on 457 source files | pass | `00_baseline_failures.md` |
| Unit/integration | 3712 passed, 30 skipped, 15 deselected | pass | `20260709_074712/commands.log` |
| CLI | All required module help commands return zero | pass | `20260709_074712/cli.log` |
| Agent integration | Universal install and MCP stdio discovery, 13 tools | pass | `20260709_074712/commands.log` |
| Provider | Contracts plus real HTTP invoke; official account reports insufficient balance | code pass / account blocked | `06_nonhardware_closure.md` |
| Sandbox | UR5e MuJoCo advances 8 steps with real qpos/qvel | pass | `20260709_074712/commands.log` |
| Physical-AI loop | Runtime block -> Practice/Memory -> How -> Auto -> Darwin -> simulated champion | pass | `tests/integration/test_physical_ai_agent_acceptance.py` |
| Practice | Record, strict verify, distill, query, Parquet, LeRobot | pass | `20260709_074712/commands.log` |
| SeekDB | Native 2881 SQL ingest/query; repeated ingest remains 1 row/table | pass | `20260709_074712/commands.log` |
| ROS2 | Required read-only tests on 9090 and 32887 | pass | `20260709_074712/commands.log` |
| ROS1 | Required Noetic read-only tests on published port 9091 | pass | `20260709_074712/commands.log` |
| Public Hub | Remote `owner/repo` plan; dry-run writes zero files | pass | `20260709_074712/commands.log` |
| Hidden Unicode | No bidi control characters | pass | `20260709_074712/commands.log` |

The detailed matrix is maintained in `03_validation_matrix.md`.
