# ROS Expert Harness — implementation and verification report

## 1. Executive Summary

**Overall status: PARTIAL; full acceptance and merge remain pending.** Work is isolated in `feat/ros-expert-harness-v1`, based on `origin/main` (`0e0d8e45`). The original checkout remains unchanged. The read-only expert foundation now has a real Jazzy/Gazebo/Nav2 fixture, daemon-owned simulation execution, independent physics observations, active Native context collection, native ROS1 introspection and source-backed KNOW/HOW consultation.

The normal Gazebo cleaning acceptance **passed at 98.0425% independent coverage, zero collisions, complete observations and zero trace gaps** in 626.64 seconds within a 900-second budget. The canonical daemon receipt is COMPLETED / SIMULATION / TASK_VERIFIED; existing Memory persisted success and Practice closed with SUCCESS. [Archived acceptance](RH10/gazebo-normal-008/acceptance.json) includes exact artifact hashes, coverage mask, trajectory and receipts. Earlier failures at 97.04%, 97.68% and transport timeouts are retained. This scripted black-box run is not autonomous model evidence. Native single-input acceptance and a dynamic-obstacle complete mission are being rerun after fixing the long-execution RPC timeout. No empirical agent advantage or hardware verification is claimed. A PR has not yet been merged.

Existing RH00–RH15 evidence records the foundation. Later acceptance work and commands are under `integrations/ros_probe/acceptance`; live artifacts are retained in the isolated `/tmp/ros-expert-acceptance` workspace until selected, redacted evidence is archived. Older machine-readable foundation summaries describe their own earlier runs and are not current physical acceptance results.

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

English/Chinese navigation and complete-cleaning intents select source-linked existing components, including opennav_coverage. Missing prerequisites remain explicit. The resolver neither installs a stack nor grants execution authority. The acceptance fixture builds upstream opennav_coverage at `65a6598c3587cb947978227c01af421e18576f0a`, compatible with the official Jazzy Fields2Cover package; the newer upstream revision failed to compile against that API. Nav2 plans every connecting path; the harness proposes measured missed-cell goals without inventing coverage paths.

## 8. ContextCompiler Integration

Existing Python ContextCompiler source adapters remain supported. The active Pi envelope now receives an opt-in compact ROS observation summary, snapshot hash/time and readiness. A bounded read-only connector subprocess owns ROS introspection; Agentd itself opens no ROS/DDS connection. Static interface declarations come from the existing compiled Body and must match its mission-bound hash; runtime observations do not overwrite Body.

A real Native SDK/PTy journey delivered live Jazzy observations to the model prompt using a deterministic fake model. A separate actual `gpt-6.1-sol` preflight returned the expected response. These prove prompt delivery and model availability separately; autonomous cleaning by the actual model remains pending.

## 9. Nav2 Integration

A disposable ARM64 Docker image contains official Jazzy/Gazebo/Nav2/rosbridge packages plus pinned opennav_coverage. Host APT and CUDA/OpenCV installations were not modified. The configured action path is Agent MCP request → existing DaemonClient → rosclawd Runtime gateway → Nav2 action server. Gazebo observes physical motion independently.

An earlier NavigateToPose run returned canonical COMPLETED/SIMULATION/TASK_VERIFIED with measured final distance **0.109225 m** to the requested goal and zero geometric collisions. The subsequently strengthened observer requires live wheel-contact streams plus non-floor contact monitoring; a fresh navigation run with that observer is still required. Initial localization is daemon-owned, restricted to a stationary, independently observed configured fixture spawn, and produces a canonical receipt.

## 10. Coverage Cleaning

The compiled existing TaskGraphV1 still contains PENDING stage proposals; it is not evidence that TaskKernel executed those stages. The daemon simulation executor now runs the official coverage action, Nav2 missed-cell repair, cleaning enable/disable, lease monitoring and independent coverage accumulation. Every counted pose is stamped and measured after Gazebo physics with the cleaning state enabled. The fixed cleanable denominator is derived from the measured occupancy map and configured physical/cleaning geometry.

The normal physical simulation passed at **98.0425%** with zero contacts and complete observations. The 900-second budget accommodates the measured 626.64-second mission; the 98% threshold, cleaning geometry and fixed denominator are unchanged. The earlier 600-second attempt failed and remains a failure.

Mission verification requires canonical matching receipts that bind the exact independent artifact hash. The accepted normal episode persisted success into existing Memory and closed the existing Practice episode with SUCCESS. Supplied perfect trajectories remain NOT_VERIFIED.

## 11. Dynamic Obstacle Tests

Offline tests prove fixed-denominator accounting, bounded per-cell recovery attempts, idempotent action IDs and temporary-block waiting. They do not measure obstacle avoidance, recovery timing or collisions. A 20-second passive obstacle injection in Gazebo recorded zero contacts, a closest robot-center distance of 0.392 m to the obstacle box and brief observed pauses before acknowledged removal. That Native episode later failed at the communication boundary and is not a full-mission PASS. A complete dynamic-obstacle Native rerun is in progress; multiple successful repeats and a second Body are pending.

## 12. Fault Injection

Fixture replay covers all 26 fault codes, including TF, stale sensors, QoS, lifecycle, obstacle-source and clock faults. Real Jazzy DDS introspection observes synthetic incompatible QoS and inactive lifecycle. Actual Nav2 cleaning runs exposed heartbeat RPC contention, shared receive/send lock starvation and deadline failures; targeted regressions cover the repaired transport behavior. The latest timed-out episode returned a terminal daemon receipt and recorded Practice FAILURE.

Gazebo contact-observer injection now physically intersects the stationary robot, observes non-floor contact count 0→1 and removes the actor with acknowledgement. The initial misplaced injection failed to create a contact and is not counted as success. Dynamic full-mission acceptance, pause/disconnect/observer-loss safety and daemon restart during coverage remain pending.

## 13. ROS1 Compatibility

A native ROS1 read-only probe runs against a real Noetic master in a disposable container. Actual synthetic LaserScan publishers and an actionlib server are discovered; ROS1 latch semantics are observed and ROS2 QoS/lifecycle are explicitly unsupported. The probe uses getTopicTypes so subscriber-only action goal interfaces remain visible. Missing native tf2 bindings are reported as collection errors rather than healthy TF. Domain: NATIVE_ROS1_SYNTHETIC_PUBLISHERS. No ROS1 move_base physical task or cross-version mission has passed.

## 14. Isaac ROS Integration

Source-backed Isaac ROS 5.0 option analysis and performance topology now exist. They preserve UNKNOWN for absent PID/backend/copy/lifetime evidence and never claim measured optimization from topology alone. Official Isaac ROS 5.0 targets ROS2 Lyrical and native rosidl::Buffer CUDA IPC; this fixture's Jazzy stack is incompatible. Live Isaac/GPU transport, lifetime, copying, before/after performance and Isaac Sim cleaning remain NOT_RUN. See [integration prerequisites](../../ISAAC_ROS_INTEGRATION.md).

## 15. Memory / KNOW / HOW

Existing EventBus and RosPracticeAdapter record system/diagnosis/verification observations. Memory preserves observed/failure outcomes instead of defaulting them to success. The daemon memory executor replays canonical bound mission evidence and requires persistence into existing Memory before it returns success; that physical success path remains unaccepted.

The new ROS knowledge projection uses the existing modern KnowledgeFacade, ReferenceContextV2 and HowAdviceRequestV2. Actual pinned upstream coverage source was ingested by the existing research pipeline into a separate seekdb_embedded KNOW store; it returned **3 sourced reference items** and non-abstaining advisory HOW output. No new database implementation, motion authority or automatic rule promotion was added. Verified physical feedback, memory reuse latency and contribution A/B remain pending.

## 16. Safety Audit

Core imports remain ROS-free. Native probes use only observation RPCs (GetState/ListParameters/GetParameters) and diagnostic publication. Active Native context delegates observation to a bounded read-only connector worker. All fixture motor commands pass through daemon-owned Nav2 execution and a 1.5-second monotonic deadman; expiry disables cleaning and publishes zero velocity, even if the simulation clock pauses. Actual pause/orphan acceptance is still required.

Compiled Body hash matching, SIM-only executor registration, fixed service/action endpoints, serialized motion, strict observer freshness/completeness and immediate contact rejection are enforced. Generic SetBool does not imply a cleaning actuator; typed Body binding plus fresh independent state is required. Readiness never grants REAL authority. Normal end-to-end zero-collision SIM cleaning is accepted; additional live fault scenarios remain pending.

## 17. Test Matrix

The latest completed broad CI selection used `-n4 --dist loadfile -m 'not slow and not integration and not deployment and not perf_serial'`: **7984 passed, 91 skipped, zero failures**, 439.22 seconds. It predates the latest full-duplex transport, Body readiness, probe worker and Native action-deadline changes; a final rerun is required after those stabilize.

The isolated installed product journey passed in clean and contaminated-PYTHONPATH cases. Native Node tests reported 238 passed, 3 skipped. Real Pi prompt delivery through the read-only worker passed; native ROS1 and Jazzy DDS introspection passed within their stated synthetic-publisher domains. The current ROS-focused suite reported 234 passed before the latest additional binding tests; the latest targeted binding suite reported 33 passed. Counts overlap and must not be summed.

Historical baseline/CI failures remain in earlier reports and logs, including wrong shared environment/import origin, fake-model proxy routing and release-build races. Repository-wide formatting still contains unrelated baseline violations; changed-file lint is checked separately.

## 18. Benchmark Results

**NOT_RUN.** No same-model, same-map, same-Body and same-budget Codex/ROSClaw agent experiment was conducted. Token usage, tool calls, time, success rates and relative gain are null. Synthetic diagnosis/coverage tests are not benchmark substitutes. [Benchmark result](benchmark_results.json).

## 19. Codex vs ROSClaw

No empirical superiority claim is supported. The implementation exposes structured evidence and reusable diagnostics; whether this improves agent decisions requires the specified controlled comparison. Research questions Q1–Q8 remain UNANSWERED as empirical questions, including memory contribution, cross-Body transfer and expert-vs-code-generation value.

## 20. Known Limitations

Normal scripted cleaning is accepted. Complete dynamic-obstacle recovery, repeated missions, second Body transfer, autonomous actual-model operation and empirical A/B are not yet accepted. Existing TaskGraph stages are proposals rather than executed TaskKernel stage evidence. Isaac live deployment and ROS1 physical navigation are absent. Modern KNOW/HOW reference retrieval works, but verified physical feedback and memory latency comparisons remain unfinished.

Observations are integrity-hashed rather than authenticated. Cleaning effectiveness represents an explicitly simulated attachment, not real dirt removal. Fixed-grid cell centers and configured geometry define the coverage metric; physical hardware promotion is disabled. Contact pipeline completeness and failure cleanup have implementation and targeted tests, but require additional live fault runs.

## 21. Remaining Work

1. Repeat the accepted >=98% zero-contact cleaning through the actual Native model with dynamic-obstacle injection and canonical receipt, mask, trajectory, verification artifact, Practice and Memory persistence.
2. Complete dynamic obstruction/withdrawal, disconnect/lease/clock/observer faults, repeated runs and second-Body transfer.
3. Execute the actual Native Agent model from the single cleaning request through trusted capability discovery, operator approval, daemon action and verified completion.
4. Close modern KNOW/HOW feedback and measure memory contribution; run same-model Codex/ROSClaw cases through existing Darwin contracts.
5. Validate required ROS1/Isaac live gates, finish final checks, archive redacted evidence, create the PR, address CI/review issues and merge only after the mandatory gates pass.

This report records ongoing work. It must not be presented as completion of the full implementation plan or MVP acceptance.
