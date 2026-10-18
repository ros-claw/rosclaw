# RK-00 architecture gap audit — 2026-10-17

Baseline fetched from origin/main: `be5d36ffdc31f2a9e47e330acd0e9203390a074c` (PR #711). This audit is committed before RK-01 code. RSK is a read-only enhancement of the existing Connector, not a second Runtime. Frozen v1/1016 negative results are not recomputed, retried or upgraded to positive results.

## Existing implementation and boundaries

`engineering/case_compiler.py` computes six diagnostic constraints (source plus five navigation mechanisms) from sealed RosSystemModel, original URDF and explicit requirements. Evidence pointers and hashes resolve original documents. Conditional candidates are not admission. Missing/old/contradictory facts remain UNKNOWN.

`staging_plan.py` binds case/model/URDF/requirements and recomputes geometry, using existing option and repair compilers. Unsupported violations block the whole plan. No unsupported patch invention. `mechanism.py` binds historical before/after/rollback evidence; current applicability and geometry cannot be transferred as authority.

`benchmarks/ros_expert_ablation` contains the same-model protocols, TaskTools whitelist, typed facade, explicit SDK callbacks and separate operator evidence. Those are experiment adapters, not a general ROS diagnosis engine; the five-rule deterministic baseline must remain distinguishable from model reasoning.

`intelligence/system_model.py` has typed Signals/Transforms/Lifecycle but generic graph/qos/observations dictionaries. It seals snapshots without assuming new snapshots are current. `builder.py` uses native graph and measurements from the same capture. `integrations/ros_probe/ros2/probe.py` already reads real endpoint QoS, topic types, observed parameters, ROS/steady ages and messages. It omits endpoint GIDs, liveliness lease duration and per-TF publisher identity; TF cache is keyed by parent/child and may hide competing publishers. Do not interpret absence of those facts as absence of faults. No probe expansion is necessary for the historical RK-01 replay; unsupported measurements must become explicit read-only requests.

## Five supported repair mechanisms

|Mechanism|Required original facts|False trigger / abstention|
|---|---|---|
|Enable localization TF|missing required map/odom link, fresh disabled broadcast parameter|competing authorities, ambiguous frame/namespace, old parameters|
|Resume scan bridge|stale selected sensor, explicitly declared paused owned staging bridge|driver/QoS/clock failure without paused bridge evidence|
|Manage lifecycle nodes|observed inactive required node, fresh manager list and unique parameter path|ambiguous manager, stale readback, unknown state|
|Cover current collision envelope|unique original expanded URDF, complete geometry, current costmap shape|unsupported geometry, mixed polygon/circle, inferred-only Body|
|Remove owned extra route|actual extra publisher, current independent operator ownership|unowned endpoint, expected controller, missing current ownership|

## General faults outside the five repair templates

|Fault / competing explanation|Missing discriminating evidence|Safe next measurement|
|---|---|---|
|Requested reliable / offered best-effort QoS|paired actual endpoint identities and QoS|native endpoint information + RMW compatibility result|
|Deadline or liveliness incompatibility|resolved requested/offered durations and defaults|read full endpoint QoS; no guessing middleware defaults|
|TF future timestamp / clock-domain mismatch|per-node use_sim_time, ROS now, transform stamp, receive steady age|same-capture clock/TF window|
|Duplicate TF authorities|per-message publisher GID/authority and complete window|read-only TF publisher attribution; topic publisher count alone insufficient|
|Sensor frame / extrinsic mismatch|message frame, original URDF sensor joint, current TF chain|read message header and immutable sensor transform|
|Planner/controller plugin mismatch|installed version, parameter readback, actual plugin load status|read package inventory, lifecycle diagnostics and plugin errors|
|Remap / namespace or wrong message type|resolved names and endpoint types at both ends|native graph endpoints with fully qualified names|
|Executor starvation / excessive callback delay|timestamped callback scheduling/trace correlation|bounded existing ros2_tracing provider if installed; otherwise report missing dependency|
|ROS clock stopped or jumped backwards|ROS samples and independent steady window, node domains|read clock receive window and domain alignment|
|Costmap subscription/plugin not consuming scans|actual subscription, QoS, input header/TF and consumer diagnostics|read-only endpoint plus costmap diagnostics; scan rate alone is insufficient|
|Action server discovered but wrong capability/ABI|actual action type, server identity and version|read native action introspection; never send an action goal|
|DDS discovery partition or inaccessible source|domain/RMW configuration and complete source-pinned endpoint capture|operator-side read-only environment/source capture|

## RK-01 design and limits

A pure compiler will reference the sealed source model and original Case, and project typed entities, endpoints, dataflows, transforms, time facts, Body/source bindings and three-valued findings. Per-field JSON pointers and value hashes remain resolvable. Content hashes include explicit historical review time and source identity. Bounds reject oversized graphs rather than truncate into a false complete graph. Freshness is separate from clock-domain correctness; graph satisfaction never grants motion.

`describe_missing_evidence` will return a bounded, deterministic read-only plan with missing finding IDs, explicit target identities, required measurements and timeout. It does not open DDS, run shell commands, repair staging, or start Worlds. Availability of a proposed measurement is not evidence it has executed. New realistic native-shape fixtures are synthetic protocol tests, not new physical faults. RK-02 active hypothesis execution and RK-03 interventions are NOT_RUN in this phase.

## Official semantics checked

ROS 2 Jazzy QoS documentation (official jazzy source): https://github.com/ros2/ros2_documentation/blob/jazzy/source/Concepts/Intermediate/About-Quality-of-Service-Settings.rst . Requested/offered compatibility is directional and all matching policies matter; defaults are middleware dependent.

ROS clock domains: https://design.ros2.org/articles/clock_and_time.html . ROS/System/Steady clocks must not be interchanged; replay does not become current evidence. TF time: https://github.com/ros2/ros2_documentation/blob/jazzy/source/Tutorials/Intermediate/Tf2/Learning-About-Tf2-And-Time-Cpp.rst . No new ROS dependency is introduced into Core.

## Frozen baseline

See `RK00_BASELINE_FREEZE.json` for SHA256-pinned existing reports and decision artifacts. All new work uses separate 1018 numbering. Old G5 54 episodes, 1016 studies and training artifacts remain unchanged. No new World/model/REAL work is authorized by this audit.
