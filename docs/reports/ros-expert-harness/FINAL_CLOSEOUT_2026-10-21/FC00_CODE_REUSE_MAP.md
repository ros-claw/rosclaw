# Existing code and responsibility map

| Existing path | Responsibility | Final-closeout treatment |
| --- | --- | --- |
| `src/rosclaw/connectors/ros/engineering/` | Cases, evidence refs, semantic repairs, staging, source consumers | Reuse unchanged; no sixth repair mechanism |
| `benchmarks/ros_expert_ablation/physical_aci.py` | Whitelist-bound proposals and explicit validation boundary | Reuse unchanged; no evaluator in decision calibration |
| `aci_runner.py`, `aci_schema.py`, `aci_context.py` | Workbook, provider-compatible schema and sealed context | Reuse; model notes remain untrusted |
| `rsk_physical_aci.py` | Typed B/C/E interface and old pilot callback gate | Reuse B proposal-only; old callback entry untouched |
| `rsk_episode_runner.py` | Original bounded RSK runner and provider pinning | Read/audit, do not reuse its incompatible pilot allocation |
| `rsk_deterministic_baseline.py` | Rules based on current supported staging options | Reuse compiler logic only if the same original inputs are admitted |
| `rsk_operational_aci.py` | Operational findings and gated diagnostic plans | No new active probe provider |
| `rsk_pilot_protocol.py` | Five-stratum, one-seed historical pilot | Unchanged; not reinterpreted as closeout protocol |
| `sdk_operator.py` and existing operator launchers | Existing owned SDK/rosclawd verification | No replacement execution path; FC-02 separately gated |
| `integrations/ros_probe/acceptance/` | Native observation, source/body verification, canonical simulation fixture | Unchanged; no direct Agent ROS access |
| `src/rosclaw/daemon`, `core`, `body`, `feedback`, `mcp` | Execution, permissions, safety and production timing | No changes |
| `docs/ROS_CONNECTOR.md`, `PHYSICAL_ACI/`, `RSK/` | Existing operation and evidence contracts | Read first; final documentation complements historical reports |

Only `closeout_protocol.py` and `closeout_aci.py` are new benchmark adapters.
The former records the fixed matrix, missing-original gate and budgets. The
latter adds a citation-bound terminal report to an existing proposal-only B
PhysicalACI. It rejects evaluator injection and exposes no physical validation,
Memory, ROS transport, Runtime or execution operation. Reports are model
assertions awaiting independent scoring, never readiness or success receipts.

The operator maintains a new as-of `ROSClaw_CURRENT_VERIFIED_MAIN.json` alongside
local evidence. It includes a timestamp and original verification reference;
consumers must re-audit if remote main differs. Historical v59 JSON remains
unchanged, including its stale #759 top-level version fields. Final source/main
identities must be recorded after the actual PR merges, not predicted here.
