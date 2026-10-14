# Current ROS and daemon generation: original passive dual supervision

Local implementation and independent actual SIM verification passed. Full P03 source/main integration and G3 L2 remain unaccepted. Goals, velocity publications, active leases and actual model experiment calls are zero.

## Exact source and scope

Actual frozen source: `80eb8e0dbbb31bd528628ea2a6898fcb86297a79`. The C++ source observer and its pinned Jazzy/Gazebo 8.15 SDK image/binaries are unchanged from the prior accepted source-stream scope. The new shared current-health helper, independent SDK rosclawd role and terminal dual observer are described in ADR 0022. No new product MCP tool, second Runtime route or generic actuator authority was added.

Twelve final fresh known-fixture Worlds were executed on that source, each with independently compiled Body and run/generation pins. Initial G1 MATCHED/L1 READY was recomputed exactly, as were all 644 current-health rows from original Native bytes. The normal dual window was 5 seconds with a longer independent source window; it is not a 75-second continuous L1 claim. The prior 75-second source-only proof remains a separate narrower result. All 75303 original contacts were independently classified, with zero unexpected contacts. Every owned World has stop/cleanup records.

| Case | Healthy joint rows before termination | Original contact messages | Actual joint outcome/reason |
|---|---:|---:|---|
| positive | 59 | 4425 | PASS_LOCAL_DUAL_READONLY_WINDOW |
| daemon_restart | 36 | 3369 | rosclawd is unavailable at /evidence/supervision/run/rosclawd.sock: [Errno 32] Broken pipe |
| ros_nav2_inactive | 72 | 6688 | CURRENT_ROS_HEALTH:UNKNOWN |
| ros_tf_loss | 55 | 5995 | CURRENT_ROS_HEALTH:UNKNOWN |
| ros_sensor_stop | 61 | 5730 | CURRENT_ROS_HEALTH:UNKNOWN |
| reset | 44 | 4014 | CURRENT_SOURCE:source stopped, stale or future |
| observer_kill | 37 | 3955 | CURRENT_SOURCE:source stopped, stale or future |
| stale | 42 | 4110 | CURRENT_SOURCE:source stopped, stale or future |
| entity_replaced | 38 | 4776 | CURRENT_SOURCE:unique dynamic robot required |
| sensor_drift | 35 | 3882 | CURRENT_SOURCE:unresolved loaded sensor type |
| clock_freeze | 37 | 3510 | CURRENT_SOURCE:incomplete, paused or reset source |
| budget_exhaustion | 117 | 6016 | CURRENT_SOURCE:source evidence capacity exhausted; recapture required |

The `ros_tf_loss` actor pauses the whole owned Nav2 process tree; its observed first rejection is costmap/localization staleness. This proves current ROS-loss detection and does not isolate TF failure. Reset, observer kill and stale also have their recorded collateral effects; the observer shares the Gazebo server. No isolated-fault claim is inferred from a case label. The daemon replacement was separately observed with a different actual PID/instance ID, and independently rejected against the old pins after the original observer latched on socket loss. Budget exhaustion was verified in both source readers.

## Failures preserved and repairs

33 actual dual-supervision Worlds were started in total, including 21 intermediate Worlds, failures and earlier narrower passes. Early startup failed because an unused Practice import required requests; the passive role now avoids that unused import without installing workstation packages. Another early failure correctly refused Runtime's canonical sandbox registrations under an overly broad zero-executor assertion; the passive contract now forbids robot executors/drivers and explicitly discloses the two sandbox registrations. Neither change enables a robot action.

Original operator failures include an inherited rollover assertion inappropriate to the preregistered shorter dual window, a ROS actor environment that hid rclpy, missing explicit owned-process fault fixture setup, and an independent audit that found premature stale costmap rejection before clock injection. The actual cadence repair and fresh complete matrix addressed the last issue. All original failed data remains available. Independent recomputation also exposed case-label assumptions: stale lidar is reported by typed lidar checks, while the whole-Nav2 pause first fails costmap checks. These are reported as observed, not promoted to isolated TF acceptance.

The current-health timing budget is measured from the start of recomputation, including its cost and subsequent daemon/source checks. Native/model/hash/policy, exact source chains, physical contacts, current-health rows, actual daemon identity/ledger/watchdog, immutable Body/config and cleanup were independently rechecked from originals.

## Software checks and evidence

Core/Body/MCP/daemon: 1340 passed, 4 skipped, 11 deselected on 9b6843f3; current Core/tests blobs are identical. Focused current/source/daemon contracts: 191 passed before the later stricter timer/watchdog changes; focused affected suites passed again, and the broad regression includes those changes. Practice: 183 passed, 9 skipped (5 optional pyarrow, 3 optional rosclaw_practice, 1 optional RH56 runtime), with reasons recorded. Required mypy: 121 files passed. Changed-file Ruff/format and compileall passed. Whole-repository Ruff retains exactly the same 290 baseline diagnostics, and format retains the same ordered 606 baseline files; neither whole-repository check is claimed green.

Final original data and source/test/operator logs: `evidence/2026-10-14/G3-dual-source80EB8E0D-twelve-original-passive-Worlds`, 2008 files, manifest SHA256 `5c056190c57dfc47218227f8e890acf52abb1fb856f9ed6f0b6646cfcd5b24cc`.

Canonical intermediate archive: `evidence/2026-10-14/G3-dual-source-original-intermediates-final-immutable-v2`, 3363 files, manifest SHA256 `54f1ff7bb8bc61fa85acf28645157b50e6844645b78af8b1ed73e9fb04eb73c1`. Runtime homes, ledgers, keys and symlinks are excluded. Every canonical gzip and final original file was rehashed. Earlier sealer pipeline stdout/stderr had been snapshotted while live; those two entries are excluded from the canonical immutable intermediate archive, with completed pipeline logs separately sealed and the earlier archive preserved.

Operator-local independent review: `recovery_2026-10-13/G3-dual-matrix-v4-independent-original-raw-review.json`. These local archive paths are not assumed GitHub-hosted attachments.

## Remaining gates

Normal PR/source CI, exact merged tree and new actual-main dual Worlds/main CI must pass before full P03 acceptance. G3 then requires explicit governed SIM admission and leased sink, three J100 position/yaw seeds, ten execution faults and canonical receipts/physical-stop evidence. Five G4 repair/rollback episodes and all 54 actual same-model A/B/C G5 episodes remain NOT_RUN. No REAL, zero-shot, model-advantage or full v1 claim is made. `v1_done=false`.
