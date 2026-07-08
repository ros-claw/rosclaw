# Module Audit

This audit focuses on repository reality observed during PR #55 validation. It does not claim full architectural completion outside the commands and tests listed here.

| Area | Current implementation | What runs | Gaps | Fixes in this pass |
|---|---|---|---|---|
| Runtime | `src/rosclaw/core/runtime.py`, `src/rosclaw/runtime/bus.py` own runtime wiring and runtime event transport. | Full pytest passes after targeted fixes. | More lifecycle idempotency/failure-shutdown tests requested by the task are still not added. | No runtime ownership refactor; only Practice fixture uses `RuntimeBus` through the normal recorder path. |
| EventBus | `src/rosclaw/core/event_bus.py`, `src/rosclaw/runtime/bus.py`, and `src/rosclaw/schemas/events.py`. | Existing tests pass. Practice recorder writes event envelopes. | Task-requested eventbus-specific tests are still missing. | Practice verifier now checks raw JSONL envelope fields in strict mode. |
| Body / e-URDF | `src/rosclaw/body/*`, `e-urdf-zoo/`, body CLI wiring in `src/rosclaw/cli.py`. | CLI help smoke for `body` passes. | Effective body constraints into all sandbox/provider paths not fully proven in this pass. | Practice fixture carries `body_id` and body cognition output. |
| Provider | `src/rosclaw/provider/*`, CLI help under `provider`. | CLI help smoke passes. | Provider route to sandbox cannot be claimed fully closed without additional route/validation tests. | No provider code changed. |
| Sandbox / Firewall | `src/rosclaw/firewall/*`, sandbox CLI. | CLI help smoke passes; full pytest passes. | Required direct ROS command block and missing-body fail-closed tests remain future work. | No sandbox code changed. |
| Practice | `src/rosclaw/practice/*`, CLI commands in `src/rosclaw/cli.py`. | `record -> verify --strict -> distill -> ingest-seekdb -> query -> export parquet/lerobot` passes on the deterministic RH56 fixture. | Real SeekDB URL backend is absent; query is SQLite-backed. | Added fixture recorder command, strict envelope checks, deterministic fixture, and regression tests. |
| Memory | `src/rosclaw/memory/*`. | Full pytest passes; empty event id edge case fixed. | Evidence promotion from Practice to How/Auto/Sandbox is not fully validated. | Generated ids are stored back into records before insert/index. |
| Know | `src/rosclaw/know/*`. | Full pytest passes. | TaskCard/EvidenceTrace tests requested by the task remain future work. | No Know code changed. |
| How | `src/rosclaw/how/*`, How CLI. | CLI help smoke passes. | Human-readable intervention evidence/risk/rollback gate not fully audited. | No How code changed. |
| Auto | `src/rosclaw/auto/*`, Auto CLI. | CLI help smoke passes; full pytest passes. | End-to-end dry-run chain through Darwin and skill promotion not fully proven. | No Auto code changed. |
| Darwin | `src/rosclaw/darwin/*` exists. | Tests can import modules, but CLI smoke for top-level `rosclaw darwin --help` fails. | Top-level Darwin CLI is missing. | No Darwin CLI added; this remains a merge blocker. |
| Skill Registry | `src/rosclaw/skill/*`. | Skill tests pass after template copy fix. | Candidate/champion safety gate still needs explicit end-to-end validation. | `_copy_template` skips bytecode/cache files. |
| Hub / MCP | `src/rosclaw/hub/*`, `src/rosclaw/mcp/*`. | Full pytest passes; docs asset test now treats root MCP config as ignored local state. | Real remote Hub behavior not exercised in this pass. | Docs asset regression now aligns with `.gitignore` for `.mcp.json` and `mcp.json`. |
| CLI | `src/rosclaw/cli.py`. | `rosclaw --help` and most subcommand help smoke pass. | `darwin` is missing; docs/CLI consistency is not complete. | Added `practice record` and clearer Practice backend errors. |

## Boundary Notes

- No true hardware write, ROS topic publish, DDS publish, serial write, or motor command was executed.
- Docker endpoints were checked by socket connection only: 9090, 9091, 32887, 8000, 6379, 2881.
- ROS bridge topic listing and read tests were not completed in this pass, so Loop B is not fully satisfied.

