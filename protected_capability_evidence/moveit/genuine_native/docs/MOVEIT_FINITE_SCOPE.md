# MoveIt finite fixture-freshness repair and software RUN

- case_id: MOVEIT_FIXTURE_FRESHNESS_CHATGPT_G11_74057e13020449a586e3ca33968f8b6e
- nonce: 74057e13020449a586e3ca33968f8b6e
- scope: SOURCE_RUN_finite_MoveIt_CPP_software_not_robot_execution
- Current outcome: **SOURCE_VALIDATED** for these ten generated finite software cases only.

## Authorship and precise repair

The fixture header began as copied original operator interface-carrier bytes, not a historical Native algorithm. Contract-declared seed SHA-256: `797887063696bafcce2480965c66b43b1c2e3f6c19af65a213c6fe49647ba509`.

Only `MoveItFixture::state_row(const RobotState&)` was changed in the header. It now copies the supplied state, calls `updateLinkTransforms()` on that copy, then reads joint values, tool transform and bounds through a const reference to the refreshed copy. This preserves the queried joint values (including the floating base), does not reset or clamp positions, and leaves caller-owned waypoint states untouched. Dirty transforms are recomputed from their current joints; clean transforms remain available. No const_cast, assertion suppression, synthetic transform, replacement path or change to a validity callback is used.

Installed `/opt/ros/jazzy/include/moveit_core/moveit/robot_state/robot_state.hpp` was inspected at lines 1205–1314. Its non-const global-link accessor invokes `updateLinkTransforms()`, whereas its const accessor asserts `checkLinkTransforms()`. Refreshing an owned copy before the const access honors that interface. The successful current batch demonstrates this repair for the finite outputs; it does not retrospectively identify the exact stack/case of the prior abort.

Protected Native C++ SHA-256 remains `0d247686f672fbb9f3cc6ae42ff25a4c59de529e10e8ae83444392d53653f93f`. Repaired header SHA-256 is `636f188c85e818877131a722b07a8f784725b52f2e8962a995fb1dbd1cdaf40e`. Both public receipts and their current registrations agree on these hashes. Tests, inputs, independent expectations, generated model, world semantics, SDK and compiler recipe were not modified.

## Actual current execution and persisted evidence

The two exact published SOURCE_CONTRACT commands were invoked in order, without a retry:

| Stage | Actual helper result | Full evidence |
|---|---|---|
| check | exit 0; 9.281 s; PASS_MOVEIT_FIVE_CHECK | `reports/operator_checks/moveit_1791444364363074121_check/receipt.json`, `compile/argv.json`, compiler stdout/stderr and dependencies |
| RUN | exit 0; 1.480 s; PASS_MOVEIT_FIVE_RUN | `reports/operator_checks/moveit_1791444380987634887_run/receipt.json`, `complete_result.json`, raw stdout/stderr and births |

Binary SHA-256: `7d0774096960785eef7544ddc4705dd0a37f267d58aded18768275189cdc8198`. The helper checks source and binary stamps before RUN and validates the authorized finite gate. Full output SHA-256: `091094241e5eeb2ff89c25fe1d585149a55d14dd3794333ac5786d02e87a5b6f`.

All ten actual answers and exact-input interface traces are preserved in `complete_result.json`; independent oracle verdicts are preserved in the RUN receipt. Native criteria are self-reports, not the independent verdict. The oracle independently binds exact requests to recorded MoveIt calls and checks generated mathematics/results. The observer is fixture instrumentation, not an independent hardware sensor.

| Case | Current independent result and actual output summary |
|---|---|
| F38_float_arm_variable_map | PASS: actual 13-variable model, six named group indices, KDL names and FK |
| F38_unknown_variable_reject | PASS: UNKNOWN_VARIABLE, no unknown state assignment |
| F39_world_object_identity_free | PASS: exact sphere identity/pose/radius; collision=false |
| F39_changed_pose_collision | PASS: same object moved to requested pose; collision=true |
| F40_obstacle_replan | PASS: baseline 7 points; obstacle path 19 points; endpoints/bounds/FK and swept clearance verified |
| F40_colliding_goal_no_plan | PASS: solved=false, zero points, actual error_code=-27 |
| F41_link_contact_frame | PASS: measured tool rotation and transformed point/normal/lever match independent geometry |
| F41_unknown_contact_frame | PASS: UNKNOWN_FRAME without identity fallback |
| F42_cartesian_path_finite | PASS: fraction=1, 22 points, terminal position [0.2,0,0.5], bounded coordinate jumps |
| F42_cartesian_out_of_bounds | PASS: fraction=0.5405405405405406, 61 valid points; not falsely reported as a complete path |

One RUN used 0.689784687012434 s in the helper's cumulative child-run accounting. Helper wall times are not exact child durations. Limits remain eight RUN attempts/cumulative 90 s, child 10 s, separate compiler child 90 s, six OMPL calls each <=0.5 s, two Cartesian calls, state quota 10000 and path quota 1000. This batch's recorded planning/Cartesian calls comprise three OMPL requests and two Cartesian requests. Child birth evidence reports no same-birth process alive after termination.

## Delivery and limitations

Only the authorized header, this document and `reports/evidence.json` were changed, plus the required six-key `delivery.json`. Published helper-generated outputs remain under `reports/operator_checks/`. The explicit role map registers the protected CPP and repaired header as diagnostic_source, this document and evidence as diagnostic_report, and final delivery.json as progress_report. Genuine current artifact references are supplied by registration, not fabricated or borrowed from the prior episode.

The previous LINK_BLOCKED and dirty-abort wholeFAIL remain historical failures and are not rescored. This is a new, bounded actual C++ software batch using the generated model and installed MoveIt/KDL/OMPL/FCL interfaces. No robot execution success, dynamics rollout, hardware operation, NN, practical force/grasp result, arbitrary-scene guarantee, deployment safety or learned-planner reliability is certified. Hardware=0, robot_physics_STEP=0, NN=0. Library initialization and software planning/collision mathematics did occur; they are not claimed absent. Final progress_report registration is followed by natural STOP.
