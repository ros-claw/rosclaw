# RH-00 current-state audit

Base: origin/main `0e0d8e45`, isolated branch `feat/ros-expert-harness-v1`. The plan directory had no repository. Historical ROSClaw test/development copies were left untouched. Core baseline was started before implementation; the full baseline was interrupted after failing unrelated installation/PTY acceptance tests.

## CURRENT

| Area | Existing implementation / evidence boundary |
|---|---|
| Transport / rosapi | RosbridgeTransport, MockTransport, RosApiResolver; ROS1/ROS2 service normalization; no core ROS imports |
| Graph | RosGraphSnapshot + RosGraphDiscovery; topic/service/node/parameter lists; action names but `_get_action_type` always returns empty |
| Graph detail | get_topic_details can query publisher/subscriber lists; basic discover does not measure topic rates |
| QoS / Lifecycle / TF | No unified native endpoint QoS, lifecycle readiness or TF topology/timing model in connector |
| Capability compiler | CapabilityManifestCompiler and embodiment cards; interface enumeration, high-level preference, risk classification and safety contracts |
| ROS provider | RosCapabilityProvider refuses non-dry-run execution with ROSCLAWD_REQUEST_ACTION_REQUIRED; safety-contract ALLOW is not authorization |
| CLI / MCP | Existing discovery/compile/validate/dry-run tools. Connector MCP registration was not attached to canonical server by default |
| Body / Sense | Existing ROS introspection and collector layers remain separate from Body truth; no system-wide readiness synthesis |
| Action lifecycle | Ros2ActionClient and OperationManager tests cover goal/feedback/result/cancel bookkeeping. This does not establish an authorized Nav2 physical executor |
| rosclawd | Existing bounded request_action, policy/permits/leases, canonical receipts and process boundary |
| Gazebo / SimForge | Prior README evidence for guarded base, ROS2 turtlesim/Gazebo, deadman/reconnect/no-replay. Historical stack scripts target Humble/Fortress |
| Native Context | Python ContextCompiler with source protocols and frozen bundle; active Pi embodied envelope is a different existing consumer and needs explicit ROS grounding wiring |
| Practice / Memory | Existing RosPracticeAdapter forwards praxis.recorded to existing consumers; avoid a new memory store |
| KNOW / HOW | Existing seed helpers; capability triples and string recovery hints, without semantic dependency/readiness evidence |
| Darwin | Existing scenario/runner/metrics infrastructure; no complete ROS coverage acceptance suite |
| Nav2 | Existing nav2 embodiment card; no NavigateToPose live acceptance or coverage mission evidence on this host |

## TARGET

Graph + independently bound Body + native measurements → RosSystemModel → deterministic doctor → semantic readiness → compatible mature solution → existing TaskGraph → canonical daemon execution → independent coverage/collision artifact → existing Practice/Memory/KNOW/HOW.

## GAP

First gaps: measurement-backed runtime model, native action types/QoS/lifecycle/TF/receive health, stable fault codes, semantic readiness and compact context. Later blocking gaps: Nav2/Gazebo/coverage dependencies, daemon Nav2/coverage executors and receipt observation binding, cleaning footprint/actuation, collision observer, live recovery runtime, Native Chat grounding, ROS1 native collector, same-model A/B and transfer benchmarks.

## Baseline

Targeted ROS/security/body/simforge/action tests: **121 PASS**, **11 deselected** by the existing integration/deployment configuration; 11 is not a pass count.

Full baseline: **8 FAIL, 1 PASS, 7 existing SKIP, 59 deselected, interrupted after 702.81 seconds**. Failure records cite PI_ENGINE_MISSING in install/probe flows and PTY connection timeout. Remaining tests are NOT_RUN. See RH00/evidence/full_baseline.log and XML. No skip/xfail was added to conceal these failures.

Environment: Ubuntu 24.04, aarch64, system ROS2 Jazzy under /opt/ros/jazzy; system Python3.12 imports rclpy after sourcing ROS, project Python3.11 does not require it. Nav2/coverage/Gazebo packages/binaries and rosbridge are absent; cached Docker images contain no robot simulation stack. See baseline_environment.json.
