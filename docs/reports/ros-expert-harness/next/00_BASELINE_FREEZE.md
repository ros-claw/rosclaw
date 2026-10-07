# Next-phase baseline freeze

Original plan: the user's `ros_rosclaw_harness实施1008.md` and its detailed
2026-10-08 execution document, in the parent workspace. Repository baseline:
`dad31022c2b6d9aed183f5335d835ebff92ddb4f`, initially clean main. PR #617, #618
and #619 are merged; the latest saved merge tree matches its tested head.

The existing complete image is `rosclaw/ros-expert-rebuilt:dad31022`, image ID
`sha256:c31355f34739eb4ea8b60de1414c57ce0eea8225aa9e3ce854b7ceb66ca4376e`.
The coverage planner revision remains
`65a6598c3587cb947978227c01af421e18576f0a`. No dependency upgrade, speed increase,
denominator reduction, observer freshness relaxation, permit extension or
failure-sample deletion is part of N01.

## Replayed historical Waffle

The canonical artifact SHA and exact CoverageVerifier replay match. Coverage is
3506/3576 = 98.04250559284117%, distance 59.54904843349339 m, SIM span 556 s,
canonical wall duration 561.236771 s. The first recovery receipt checkpoint is
2048/3576 = 57.27069351230425%. Exact replay reaches that checkpoint across sample
indices 1282–1316 (zero based). This reproduces the coverage value, not an exact
historic phase timestamp. Existing path snapshots omitted yaw and overwrote
earlier versions; feedback and internal planner-stage provenance cannot be
retroactively invented. Missing historical causal evidence stays UNKNOWN.

The historical audit is in the parent machine's
`/tmp/ros-expert-next/historical-dad31022-audit/`; it is historical computation,
not a new episode. Parent archive:
`image_rebuild_dad31022/waffle-cleaning-acceptance/`.

## New N01 runs

Every new run uses a new evidence directory/container/loopback port and isolated
DDS domain. `n01-waffle-001` failed fixture startup because an eager package
initializer pulled Pydantic into the ROS-only witness. It dispatched no cleaning
task and is retained as FAIL. Diagnosis exports were made lazy; passive audit
imports now work in the existing image without installing host/container Python
dependencies. `n01-waffle-002` is the instrumentation pilot; results and source
limitations are recorded in N01's report after completion.

Neither the original cleaning algorithm nor coverage credit rules were changed
for this phase. Task correctness and audit completeness are separate gates.
Subsequent efficiency, online occupancy, unknown-Body and causal benchmark gates
remain open; `v1_done=false`.
