# ROS Expert Harness — implementation and verification report

## 1. Executive Summary

**Overall status: PARTIAL; full acceptance and merge remain pending.** Work is isolated in `feat/ros-expert-harness-v1`, based on `origin/main` (`0e0d8e45`). The original checkout remains unchanged. The read-only expert foundation now has a real Jazzy/Gazebo/Nav2 fixture, daemon-owned simulation execution, independent physics observations, active Native context collection, native ROS1 introspection and source-backed KNOW/HOW consultation.

The normal Gazebo cleaning acceptance **passed at 98.0425% independent coverage, zero collisions, complete observations and zero trace gaps** in 626.64 seconds within a 900-second budget. The canonical daemon receipt is COMPLETED / SIMULATION / TASK_VERIFIED; existing Memory persisted success and Practice closed with SUCCESS. [Archived acceptance](RH10/gazebo-normal-008/acceptance.json) includes exact artifact hashes, coverage mask, trajectory and receipts. Earlier failures at 97.04%, 97.68% and transport timeouts are retained. This scripted black-box run is not autonomous model evidence. Actual `gpt-6.1-sol` Native single-input acceptance now passed at **98.0145%**, zero physics contacts and complete observations after a 20-second dynamic obstacle. Three independent operator approvals led to canonical localization, coverage and Memory completion, Practice SUCCESS and the existing TaskKernel SUCCEEDED. [Complete model-run evidence](RH10/native-004-full-pass/acceptance.json) includes exact artifacts, SDK usage and kernel artifact registration. No empirical agent advantage or hardware verification is claimed. Draft [PR #617](https://github.com/ros-claw/rosclaw/pull/617) is open; it has not yet been merged.

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

A real Native SDK/PTy journey delivered live Jazzy observations to the model prompt using a deterministic fake model. A separate actual `gpt-6.1-sol` preflight returned the expected response. The later actual-model journey selected observation, localization, coverage and Memory tools itself and completed the existing root task. Its eight SDK turns are separately metered; the existing Core model_usage table has no Native SDK token records, so it must not be used as a zero-token claim.

## 9. Nav2 Integration

A disposable ARM64 Docker image contains official Jazzy/Gazebo/Nav2/rosbridge packages plus pinned opennav_coverage. Host APT and CUDA/OpenCV installations were not modified. The configured action path is Agent MCP request → existing DaemonClient → rosclawd Runtime gateway → Nav2 action server. Gazebo observes physical motion independently.

An earlier NavigateToPose run returned canonical COMPLETED/SIMULATION/TASK_VERIFIED with measured final distance **0.109225 m** to the requested goal and zero geometric collisions. The subsequently strengthened observer requires live wheel-contact streams plus non-floor contact monitoring; a fresh run with the stronger contact observer and official controller watchdog passed at **0.0445654m** target distance with zero contacts. [Receipt](RH10/navigation-watchdog-001/golden-navigation.receipt.json). Initial localization is daemon-owned, restricted to a stationary, independently observed configured fixture spawn, and produces a canonical receipt.

## 10. Coverage Cleaning

The compiled existing TaskGraphV1 still contains PENDING stage proposals; it is not evidence that TaskKernel executed those stages. The daemon simulation executor now runs the official coverage action, Nav2 missed-cell repair, cleaning enable/disable, lease monitoring and independent coverage accumulation. Every counted pose is stamped and measured after Gazebo physics with the cleaning state enabled. The fixed cleanable denominator is derived from the measured occupancy map and configured physical/cleaning geometry.

The normal physical simulation passed at **98.0425%** with zero contacts and complete observations. The 900-second budget accommodates the measured 626.64-second mission; the 98% threshold, cleaning geometry and fixed denominator are unchanged. The earlier 600-second attempt failed and remains a failure.

Mission verification requires canonical matching receipts that bind the exact independent artifact hash. The accepted normal episode persisted success into existing Memory and closed the existing Practice episode with SUCCESS. Supplied perfect trajectories remain NOT_VERIFIED.

## 11. Dynamic Obstacle Tests

Offline tests prove fixed-denominator accounting, bounded per-cell recovery attempts, idempotent action IDs and temporary-block waiting. They do not measure obstacle avoidance, recovery timing or collisions. A 20-second passive obstacle injection in Gazebo recorded zero contacts, a closest robot-center distance of 0.392 m to the obstacle box and brief observed pauses before acknowledged removal. That Native episode later failed at the communication boundary and is not a full-mission PASS. Native003 subsequently passed the coverage subtest at 98.0705% but did not call Memory, so its root task remained RUNNING and the full episode is PARTIAL. Native004 passed the complete dynamic-obstacle mission at 98.0145% and closed Memory, Practice and TaskKernel. Multiple complete repeats and a second Body are pending.

## 12. Fault Injection

Fixture replay covers all 26 fault codes, including TF, stale sensors, QoS, lifecycle, obstacle-source and clock faults. Real Jazzy DDS introspection observes synthetic incompatible QoS and inactive lifecycle. Actual Nav2 cleaning runs exposed heartbeat RPC contention, shared receive/send lock starvation and deadline failures; targeted regressions cover the repaired transport behavior. The latest timed-out episode returned a terminal daemon receipt and recorded Practice FAILURE.

Gazebo contact-observer injection now physically intersects the stationary robot, observes non-floor contact count 0→1 and removes the actor with acknowledgement. The initial misplaced injection failed to create a contact and is not counted as success. Live mid-motion daemon SIGKILL, a two-second clock pause/resume and rosbridge SIGKILL now passed independent three-second standstill measurements, cleaning disabled and zero contacts. Pause/disconnect receipts are FAILED; SIGKILL has no terminal receipt and is not represented as canonical completion. The first bridge test is retained as FAIL because the fixture supervisor also stopped independent observation; the repeat explicitly kept physics alive. Observer-process SIGSTOP acceptance subsequently passed using a second passive ground-truth observer and the official diff_drive_controller 0.2-second timeout. The canonical action failed; after observation resumed, cleaning remained disabled. Restart safety remains pending. [Fault evidence](RH12/live-safety/).

## 13. ROS1 Compatibility

A native ROS1 read-only probe runs against a real Noetic master in a disposable container. Actual synthetic LaserScan publishers and an actionlib server are discovered; ROS1 latch semantics are observed and ROS2 QoS/lifecycle are explicitly unsupported. The probe uses getTopicTypes so subscriber-only action goal interfaces remain visible. Missing native tf2 bindings are reported as collection errors rather than healthy TF. Domain: NATIVE_ROS1_SYNTHETIC_PUBLISHERS. No ROS1 move_base physical task or cross-version mission has passed.

## 14. Isaac ROS Integration

Source-backed Isaac ROS 5.0 option analysis and performance topology now exist. They preserve UNKNOWN for absent PID/backend/copy/lifetime evidence and never claim measured optimization from topology alone. Official Isaac ROS 5.0 targets ROS2 Lyrical and native rosidl::Buffer CUDA IPC; this fixture's Jazzy stack is incompatible. An official digest-pinned ARM64 FastOS image now runs on GB10/CUDA13.0 in an owned Lyrical container. Official cuda_buffer_backend 0.1.2 source was built against that runtime; nine native test targets (18 individual cases) passed, including multiprocess CUDA transport, GPU relay and CPU fallback. Three full-HD 20Hz comparisons per backend each received 300/300 content-validated frames: CUDA IPC median latency 0.605–0.667ms versus CPU fallback 16.33–22.82ms. The failed 50Hz CPU case (286/300) is retained. Subscriber pixel validation performs a DtoH copy, and no complete copy profiler trace exists; full zero-copy and cleaning-performance claims remain unsupported. Actual upstream ResizeNode subsequently passed 300/300 frames at 20Hz with observed 480×270 output, complete native content/backend validation and CUDA IPC. Initial missing exact-time CameraInfo, per-DSO allocation pools and stale recycled descriptors are retained as failures. Preloading the installed shared CUDA allocation library binds component and plugin to one pool; a real validated cold-start frame precedes the measured episode without changing the recycling interval. Input metadata observation forces CPU fallback and the validator copies output pixels to CPU. [Resize evidence](RH13/isaac-live/resize/acceptance.json) proves this GPU graph subtest, not zero-copy or Isaac Sim cleaning. Isaac Sim cleaning remains NOT_RUN. [GPU evidence](RH13/isaac-live/source-lock.json). See [integration prerequisites](../../ISAAC_ROS_INTEGRATION.md).

## 15. Memory / KNOW / HOW

Existing EventBus and RosPracticeAdapter record system/diagnosis/verification observations. Memory preserves observed/failure outcomes instead of defaulting them to success. The daemon memory executor replays canonical bound mission evidence and requires persistence into existing Memory before it returns success; both the scripted and actual-model SIM success paths are accepted. The actual-model Memory artifact was registered by the existing task coordinator and closed the root task.

The new ROS knowledge projection uses the existing modern KnowledgeFacade, ReferenceContextV2 and HowAdviceRequestV2. Actual pinned upstream coverage source was ingested by the existing research pipeline into a separate seekdb_embedded KNOW store; it returned **3 sourced reference items** and non-abstaining advisory HOW output. No new database implementation, motion authority or automatic rule promotion was added. Verified physical feedback, memory reuse latency and contribution A/B remain pending.

## 16. Safety Audit

Core imports remain ROS-free. Native probes use only observation RPCs (GetState/ListParameters/GetParameters) and diagnostic publication. Active Native context delegates observation to a bounded read-only connector worker. All fixture motor commands pass through daemon-owned Nav2 execution and a 1.5-second monotonic deadman; expiry disables cleaning and publishes zero velocity, even if the simulation clock pauses. Actual pause/orphan/disconnect acceptance now passed. The additional official controller watchdog passed observer-process-loss physical-stop acceptance. Subsequent watchdog-equipped complete runs are not yet accepted. Native005 failed at the four-second observation deadline; authoritative native graph reuse removed duplicate rosapi discovery and measured capture at 0.238s. Native006 safely rejected a context change; a bounded SIM-only proposal refresh preserves Body/mode/mission/session and requires a new independent approval. Native007 failed on source freshness. Dedicated observer callbacks and atomic pose/receive-time capture fixed the observed scheduling path; Native008 retained complete observations throughout but failed at redundant repair service confirmation while the actual cleaner and lease remained active. Recovery now reuses independently observed active state and serializes shared-connection RPC readers. Native009 failed with one 489.7ms ground-truth age during concurrent regression; the strict 300ms limit correctly rejected it. An idle-load complete repeat is still required. All failed artifacts remain archived.

Compiled Body hash matching, SIM-only executor registration, fixed service/action endpoints, serialized motion, strict observer freshness/completeness and immediate contact rejection are enforced. Generic SetBool does not imply a cleaning actuator; typed Body binding plus fresh independent state is required. Readiness never grants REAL authority. Normal end-to-end zero-collision SIM cleaning is accepted; additional live fault scenarios remain pending.

## 17. Test Matrix

The latest correct isolated Python CI selection used `python -m pytest -n8 --dist loadfile -m 'not slow and not integration and not deployment and not perf_serial'`: **8025 passed, 91 skipped, zero failures**, 272.67 seconds. Four additional malformed copy-event cases subsequently passed in the focused performance/receipt suite (23 passed). Native Node tests passed with 240 passed, 3 skipped. Counts overlap and must not be summed.

GitHub Python 3.11/3.12/3.13, type checks, Node tests, Cross-UID boundary/operator E2E, installed product journey and remaining completed checks passed. Lint found a test import ordering defect, now fixed locally. The latest installed journey retry **passed (209.59s)** after repairing the test driver proxy isolation for installed CLI argv as well as Python argv. The earlier connection failure remains retained.
Historical baseline/CI failures remain in earlier reports and logs, including wrong shared environment/import origin, fake-model proxy routing and release-build races. Repository-wide formatting still contains unrelated baseline violations; changed-file lint is checked separately.

## 18. Benchmark Results

**NOT_RUN.** No same-model, same-map, same-Body and same-budget Codex/ROSClaw agent experiment was conducted. Token usage, tool calls, time, success rates and relative gain are null. Synthetic diagnosis/coverage tests are not benchmark substitutes. [Benchmark result](benchmark_results.json).

## 19. Codex vs ROSClaw

No empirical superiority claim is supported. The implementation exposes structured evidence and reusable diagnostics; whether this improves agent decisions requires the specified controlled comparison. Research questions Q1–Q8 remain UNANSWERED as empirical questions, including memory contribution, cross-Body transfer and expert-vs-code-generation value.

## 20. Known Limitations

Normal scripted and autonomous actual-model dynamic-obstacle cleaning are accepted. Repeated complete missions, second Body transfer and empirical A/B are not yet accepted. Existing TaskGraph stages are proposals rather than executed TaskKernel stage evidence. Isaac live deployment and ROS1 physical navigation are absent. Modern KNOW/HOW reference retrieval works, but verified physical feedback and memory latency comparisons remain unfinished.

Observations are integrity-hashed rather than authenticated. Cleaning effectiveness represents an explicitly simulated attachment, not real dirt removal. Fixed-grid cell centers and configured geometry define the coverage metric; physical hardware promotion is disabled. Contact pipeline completeness and failure cleanup have implementation and targeted tests, but require additional live fault runs.

## 21. Remaining Work

1. Repeat the accepted Native full loop after the additional controller-watchdog change and archive its independent fault-stop proof.
2. Complete dynamic obstruction/withdrawal, disconnect/lease/clock/observer faults, repeated runs and second-Body transfer.
3. Extend live fault categories and unknown-Body solution provisioning beyond the preinstalled golden fixture.
4. Close modern KNOW/HOW feedback and measure memory contribution; run same-model Codex/ROSClaw cases through existing Darwin contracts.
5. Validate required ROS1/Isaac live gates, finish final checks, archive redacted evidence, address PR #617 CI/review issues and merge only after the mandatory gates pass.

This report records ongoing work. It must not be presented as completion of the full implementation plan or MVP acceptance.
