# ROS Expert Harness — implementation and verification report

## 1. Executive Summary

**Overall status: PARTIAL. The requested v1 acceptance is not achieved.** Work is isolated in `feat/ros-expert-harness-v1`, based on `origin/main` (`0e0d8e45`). The original checkout is unchanged. This implementation establishes the read-only ROS expert foundation and independently testable coverage calculations; it does not complete a robot cleaning mission.

Implemented: versioned system snapshots, optional native ROS2 probe, 26 deterministic diagnostic codes, semantic readiness, solution catalog/resolution, opt-in Python ContextCompiler integration, four read-only canonical MCP tools, CLI, existing TaskGraph proposals, missed-region recovery proposals, coverage calculations, canonical receipt evidence gate, and existing Practice/KNOW/HOW extension points.

Verified separately: fixture replay and unit/component contracts; a real Jazzy DDS graph populated by synthetic sensors, TF, lifecycle, QoS and an action server. The latter verifies native introspection, not navigation or cleaning. **Nav2/Gazebo execution, >=98% cleaning coverage with zero measured collisions, dynamic obstacle acceptance, Native Agent automatic grounding and same-model A/B are NOT_RUN.**

Machine-readable evidence: [test summary](final_test_summary.json), [benchmark status](benchmark_results.json), [coverage status](coverage_results.json), [diagnostic evidence](diagnostic_results.json), [architecture check](architecture_boundary_check.json). Stage-specific implementation, tests and failures are under RH00–RH15. Failures and incomplete runs remain recorded.

## 2. Architecture Before / After

Before: existing rosbridge transport/discovery, capability manifest, Body, daemon, TaskGraph, ContextCompiler, EventBus, Practice/Memory, KNOW/HOW and MCP server. [RH00 audit](00_CURRENT_STATE_AUDIT.md) identifies the active Pi context path and missing action-type/readiness evidence.

After: these existing extension points remain authoritative. New intelligence modules are inside the ROS connector; native ROS imports are isolated under `integrations/ros_probe/ros2`. There is no second agent framework, motion runtime, safety policy, knowledge database or memory store. Physical authority still belongs to rosclawd. Proposed decisions are not permits. [ADR](../../adr/0016-ros-expert-harness-boundaries.md).

## 3. Added Modules

`intelligence`: system model/builder/evidence; `probe`: rosbridge/native snapshot collection; `diagnosis`: deterministic predicates; `resolver`: semantics and catalog resolution; `context`: existing source adapters and compact summary; `mission`: existing TaskGraph compilation and bounded recovery proposals; `verification`: cleaning-footprint rasterization and independent mission evidence gate. `knowledge/catalog.yaml` records implementation/version/source requirements. Standalone native probe, live DDS fixture, offline replay script and focused tests accompany them.

## 4. RosSystemModel

`rosclaw.ros_system_model.v1` preserves robot/body identity, graph, runtime environment, timestamps, topic health, TF, lifecycle, QoS, navigation observations, inference, completeness and collection errors. Content hash and major-version validation reject mutated saved snapshots; a hash is integrity metadata, not a trusted measurement signature. Native observations retain their capture time and graph; combining captures does not relabel old measurements as fresh. Semantic inference is separate from supplied Body truth. Historical replay remains historical. [Contract](../../ROS_SYSTEM_MODEL.md).

## 5. ROS Expert Diagnosis

26 evidence-bearing codes cover collection/stale graph; missing, stale, future, cyclic and ambiguous TF; topic presence/freshness/rate/publishers; incompatible QoS; clock mismatch; inactive lifecycle; localization, costmap, controller and collision-monitor conditions; coverage implementation/evidence gaps. Unit tests inject every code. A diagnostic confidence of 1 describes satisfaction of a deterministic predicate, not certainty about an underlying cause. Missing observations yield UNKNOWN. Unknown TF clock age does not become a missing-topology claim. [Codes](../../ROS_DIAGNOSTICS.md).

## 6. Semantic Capability System

Legacy capability IDs remain stable. Additive semantic/implementation/alias fields map ROS1 MoveBase and ROS2 NavigateToPose to the same intent. AVAILABLE, UNKNOWN, BLOCKED and MISSING separate interface presence from prerequisites. Readiness considers fresh signals, configured minimum rates, TF timing/topology, namespace-local lifecycle and runtime diagnoses. Multiple implementations are ranked without hiding a ready alternative. No status authorizes REAL execution. [Details](../../ROS_CAPABILITY_RESOLUTION.md).

## 7. Solution Resolver

English/Chinese navigation and complete-cleaning intents select source-linked existing components, including opennav_coverage. Unsupported requests and missing cleaning actuator/footprint prerequisites remain explicit. The resolver neither installs nor configures a stack. Catalog compatibility is recorded, not live acceptance. No new planner was written. Actual package pinning/building and daemon execution bindings remain pending.

## 8. ContextCompiler Integration

The existing Python compiler accepts `SourceBundle.extra['ros_system_model']`, verifies type/hash/body/freshness, then uses existing SelfAugmentingSource/CapabilitySource protocols for L2/L3 additions. Frozen context contracts and Body are unchanged. Replay is deterministic with an explicit clock. The active Pi envelope does **not** automatically collect/populate this source. Native Chat grounding and closed-loop agent acceptance remain unverified.

## 9. Nav2 Integration

**NOT_RUN.** This host has ROS Jazzy and can run rclpy with the host ROS Python. Nav2, opennav_coverage, Gazebo and rosbridge were not available in the inspected installation/cache. In addition to installing a pinned simulation stack, daemon executor bindings, lifecycle/configuration, protected cancellation and independent evidence observation must be implemented. No NavigateToPose was dispatched, and README's existing Nav2 limitation remains unchanged.

## 10. Coverage Cleaning

Existing TaskGraphV1 compilation proposes precheck, localization, coverage planning, cleaning enable/execute, verify/recover, disable/finalize/remember. All nodes are PENDING; compilation dispatches nothing. Cleanup descriptions are not implemented failure handlers or guaranteed actuator shutdown.

CoverageVerifier calculates cell-center coverage from a supplied accessible grid, cleaning polygon and stamped poses. It checks finite data, frames, trace gaps/speed, enabled cleaning, interpolation and rotation; reports missed components and repeat visits. Synthetic replay retains temporarily blocked cells in the denominator: 80% before revisiting, 100% afterward. **These numbers are arithmetic fixtures, not robot performance.** Collision count for that replay is null. Real coverage, collision count and completion time are null/NOT_RUN in the root coverage result.

Mission verification additionally requires canonical daemon receipts matching mission/body/action/mode/domain and binding the exact independent evidence hash. Perfect supplied traces alone return NOT_VERIFIED. No current coverage executor produces these required bindings. [Evidence rules and limitations](../../ROS_COVERAGE_CLEANING.md).

## 11. Dynamic Obstacle Tests

Offline tests prove fixed-denominator accounting, bounded per-cell recovery attempts, idempotent action IDs and temporary-block waiting. They do not measure obstacle avoidance, recovery timing or collisions. Dynamic obstacles in Gazebo, multiple repeated runs and a second Body are NOT_RUN.

## 12. Fault Injection

Fixture replay records normal plus missing TF, stale sensor, QoS mismatch, inactive lifecycle, missing obstacle source and mixed clocks. Unit tests exercise all 26 fault codes and stale/unknown readiness. Native DDS acceptance measures a deliberately incompatible QoS pair and inactive synthetic lifecycle node. Live Nav2 failures, disconnects during execution, localization recovery, collision observer interruption and daemon restart during coverage remain NOT_RUN.

## 13. ROS1 Compatibility

Semantic equivalence and legacy manifest serialization are fixture-tested. There is no new ROS1 native probe; no live ROS1 master or move_base was run. ROS1 readiness/performance and cross-version mission acceptance remain pending.

## 14. Isaac ROS Integration

**NOT_RUN / deferred until Nav2 acceptance.** No Isaac ROS 5.0 support, capability readiness or performance claim is made. [Integration prerequisites](../../ISAAC_ROS_INTEGRATION.md) describe the required next investigation.

## 15. Memory / KNOW / HOW

Existing EventBus events expose snapshot, diagnosis and verification observations. The existing RosPracticeAdapter records them; unverified outcomes cannot become success. Tests exercise diagnosis-to-Practice flow. KNOW seeding uses evidence-qualified interfaces/requirements, and HOW emits intervention proposals. There is no new store. The optional canonical thin MCP path does not yet automatically wire durable runtime memory; actual mission receipt persistence, retrieval influence and memory A/B are pending.

## 16. Safety Audit

Core imports remain ROS-free. Native probe publishers are limited to snapshot/status, its refresh service schedules observation, and outbound RPCs are limited to GetState/ListParameters/GetParameters. It sends no action goal, command topic, parameter mutation or motion RPC. The live fixture publishes synthetic sensor/TF data in ROS_DOMAIN_ID 173 and creates an action server; it sends no goals. New canonical tools have read-only annotations; legacy tools are not added by this extension.

Tests reject forged supplied evidence, wrong daemon modes/domains, unbound hashes and mixed hardware/simulation receipts. Existing security/daemon-boundary regression is included in the affected suite. Actual coverage execution through request_action and independent zero-collision acceptance are NOT_RUN, so the end-to-end safety gate is not PASS.

## 17. Test Matrix

Authoritative counts and individual failures are in [final_test_summary.json](final_test_summary.json), with original JUnit/log paths. New features, affected connector/security/context/Practice regression, broader Body/Self/contracts regression, native DDS introspection, repository lint, focused types and compilation are distinct runs. Do not add their counts as unique tests.

Final CI selection (`-n4 --dist loadfile -m 'not slow and not integration and not deployment and not perf_serial'`): **7912 passed, 112 skipped, zero failures**. New-feature tests: **61 passed**. Affected connector/security/context/Practice regression: **387 passed, 11 skipped, 11 deselected**. The last additional KNOW/HOW evidence-retention test was run after the full CI selection; production code did not change after that selection. Native DDS introspection, repository lint, changed-file formatting, focused existing/new-module typing and compilation passed.

The initial full baseline and an unrestricted parallel run were interrupted and are not full-suite passes. Concurrent release tests raced over a shared dist directory; that failed run is retained. Its serial retest built the release but failed with PI_ENGINE_MISSING during native-agent initialization, so installed-agent acceptance remains FAIL.

The first CI-group regression had 7897 passes and 12 failures. Three documentation regressions were repaired; nine storage/firewall failures did not recur in the final run, isolation or baseline comparison. Their cause is not established. The clean archived baseline produced 7837 passes, 112 skips and one failure because the wheel test requires Git metadata. That test passed separately at the same unmodified commit in a proper detached worktree. Repository-wide format checking fails on hundreds of baseline files; all 34 changed Python files are formatted without reformatting the rest of the repository.

## 18. Benchmark Results

**NOT_RUN.** No same-model, same-map, same-Body and same-budget Codex/ROSClaw agent experiment was conducted. Token usage, tool calls, time, success rates and relative gain are null. Synthetic diagnosis/coverage tests are not benchmark substitutes. [Benchmark result](benchmark_results.json).

## 19. Codex vs ROSClaw

No empirical superiority claim is supported. The implementation exposes structured evidence and reusable diagnostics; whether this improves agent decisions requires the specified controlled comparison. Research questions Q1–Q8 remain UNANSWERED as empirical questions, including memory contribution, cross-Body transfer and expert-vs-code-generation value.

## 20. Known Limitations

Snapshot dictionaries are extensible and observations are not authenticated. Probe health measures receive freshness/rate; it does not derive navigation safety. Native Nav2 readiness lacks independent live localization/costmap validation. Clock configuration can be incomplete; unknown timing fails readiness conservatively. Fixed thresholds need robot-specific validation. The accessible mask is supplied, not derived from a real map. Cleaning rasterization uses cell centers and supplied geometry; polygon self-intersection and physical cleaning effectiveness are not verified. Independent collision/cleaning observers are absent. Mission runtime, conditional recovery and guaranteed cleanup are absent. Context collection and durable memory need active Pi wiring. Catalog pins, ROS1 native and Isaac gates remain unfinished.

## 21. Remaining Work

1. Provision source-backed, pinned ROS Jazzy/Nav2/Gazebo/opennav_coverage and rosbridge using the repository's host/integration lifecycle, then run the RH08 NavigateToPose gate through rosclawd.
2. Implement daemon coverage/cleaning executor bindings with bounded actions, protected cancellation, receipts and independent map/pose/cleaning/collision evidence hashing. Execute the RH09 TaskGraph with failure cleanup.
3. Derive the reachable denominator from observed map/Body geometry. Run independent RH10 >=98%/zero-collision acceptance, then RH11 dynamic obstacles/faults, repeated runs and another Body.
4. Populate the active Native Agent context source and connect canonical MCP evidence to durable Practice/Memory/KNOW/HOW. Verify black-box reasoning and recovery.
5. After physical/simulation gates, validate live ROS1 and Isaac integrations. Run the same-model A/B/Darwin protocol and answer Q1–Q8 from recorded evidence.

The code and evidence are reviewable foundations. They must not be reported as completion of the full implementation plan or its MVP acceptance.
