# ROSClaw Codex Validation Matrix

Date: 2026-07-09

| Area | Evidence | Status |
|---|---|---|
| Compile | `python -m compileall -q src tests` | pass |
| Lint | `ruff check .` | pass |
| Format | `ruff format --check .`, 891 files | pass |
| Type | `.venv-codex/bin/mypy src/rosclaw`, 457 files | pass |
| Full tests | 3712 passed, 30 skipped, 15 deselected | pass |
| CLI modules | doctor/firstboot/body/provider/sandbox/practice/memory/know/how/auto/darwin/skill/hub/mcp help | pass |
| Provider health | 8 built-in contracts with capabilities/runtime/safety | pass |
| Provider route | `vlm.scene_graph -> vlm`, fallback `world`, guard required | pass |
| Provider benchmark | dry-run plan for LLM/VLM/VLA/critic, no external calls | pass |
| Provider invoke | DeepSeek real local HTTP success; official endpoint returns actionable `402 Insufficient Balance` | code pass / account blocked |
| Agent install | generated MCP/guidance/skill/context files in temp project | pass |
| MCP probe | stdio server advertised and returned 13 tools | pass |
| MuJoCo | UR5e, 8 steps, qpos=6, qvel=6, final time 0.016 | pass |
| Physical-AI acceptance | unsafe Runtime action -> BLOCKED Practice/Memory -> How -> Auto -> sandbox -> 3-seed Darwin -> simulated Skill Registry champion | pass |
| Docker endpoints | 9090, 9091, 32887, 8000, 6379, 2881 TCP | pass |
| ROS2 read-only | ping/discover/manifest/pose on 9090 and 32887 | pass |
| ROS1 read-only | Noetic ping/discover/manifest/pose on published port 9091 | pass |
| Public Hub | `ros-claw/g1-mcp@0.1.0` remote plan, zero dry-run files | pass |
| Practice record | RH56 fixture, 9 events | pass |
| Practice strict verify | catalog/envelopes/artifacts valid | pass |
| Practice distill | failure/intervention/candidate/promotion/sim2real counts all 1 | pass |
| Local SeekDB | SQLite ingest/query, 7 records | pass |
| Real SeekDB | MySQL DSN on OceanBase seekdb-v1.2.0.0 | pass |
| SeekDB idempotency | repeated ingest leaves all seven table counts at 1 | pass |
| Practice queries | failures/body cognition/sim2real/candidates/interventions | pass |
| Parquet export | file generated | pass |
| LeRobot export | dataset tree generated | pass |
| Hidden Unicode | no bidi control characters | pass |
| Full validation script | `reports/codex/20260709_074712`, `FAILURES=0` | pass |

## Notes

- ROS ping, discovery, manifest compilation, and pose subscription are required
  by this full-runtime script because the documented containers are part of the
  acceptance environment.
- Public Hub validation is enabled with `ROSCLAW_VALIDATE_REMOTE_HUB=1`.
