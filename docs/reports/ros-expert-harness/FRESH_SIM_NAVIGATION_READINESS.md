# Fresh SIM navigation readiness: implementation and evidence boundary

G1 source matching deliberately defers localization freshness. This change
adds `review_sim_navigation_readiness`, a pure read-only reviewer that consumes
the original independently pinned G1 inputs again and removes all deferrals.
`READY_FOR_SIM_TEST` is a point-in-time observation with no execution authority.
It grants neither daemon admission nor a permit; L2 remains NOT_RUN.

The operator separately pins a closed policy containing rationale, capture-age
bound, explicit costmap/footprint associations and planar variance limits. The
limits cannot exceed the existing 0.25 ceiling or squared immutable Body goal
tolerances. The reviewer requires all 36 covariance entries, finite symmetric
planar covariance, positive semidefiniteness, consistent summaries and fresh
localization. Symmetry uses only 1e-12 relative/1e-15 absolute serialization
precision, rather than widening uncertainty limits.

Fresh costmap GetParameters reads must agree with the observed publisher,
SIM clock, fixed Body base transform and lidar obstacle source. Configured
circles/strictly convex polygons and the actual PolygonStamped footprint must
contain the complete conservative URDF collision circle. Original numeric TF
transforms the effective polygon to the Body base; frame names are never
assumed equivalent. G1 independently checks the Gazebo loaded collision
lineage. Unsupported meshes, missing readback, stale data and ambiguous
publishers stay UNKNOWN. Unsafe geometry/covariance or source mismatch blocks.
The independent source-only physics witness must show no contact, a closed
lease and a stopped Body.

## Actual known-fixture acceptance

Frozen implementation: `7ebd1b9f4ccabb78138ceb238b0e8c95cf9622e9`.
Environment: existing ARM64 Jazzy/Gazebo 8.15 SDK image
`sha256:d16320799584a60035548fb298243bdd8b0d2cb5760675a9cb46b608ea211e56`.
Each case starts a new isolated World with read-only source assets and observer,
network none, no navigation goals and no velocity publisher in the fault actor.

| Case | Actual injection | Required result |
| --- | --- | --- |
| Positive | Fresh known fixture | READY_FOR_SIM_TEST |
| Sensor stop | SIGSTOP owned isolated scan bridge subtree | UNKNOWN or BLOCKED |
| Clock freeze | Owned Gazebo WorldControl pause | UNKNOWN or BLOCKED |
| TF loss | SIGSTOP owned Nav2 launch subtree | UNKNOWN or BLOCKED |
| Nav2 inactive | Typed collision_monitor ChangeState/deactivate RPC | UNKNOWN or BLOCKED |
| QoS mismatch | TRANSIENT_LOCAL scan subscriber against VOLATILE publisher | UNKNOWN or BLOCKED |
| Namespace mismatch | Pause owned scan bridge; replacement remaps scan to /misplaced/scan | UNKNOWN or BLOCKED |

The TF fault also stops localization/costmap updates; it is not an isolated
TF-only causal experiment. Deactivating collision_monitor can cascade through
Nav2 lifecycle management and reset localization, so a stricter BLOCKED due
to covariance is a legitimate fail-closed result. All faults are operator-owned
SIM fixture operations, separate from the Agent's read-only reviewer.

## Validation and original failures

Source 7ebd1b9f local ROS/MCP/architecture/Practice regression:
**1163 passed, 11 skipped, 11 deselected** (not integration/acceptance/slow).
There are 36 new readiness contracts; suite counts overlap and are not summed.
Compileall, changed-file Ruff/format, `ruff check src tests`, and focused mypy
(122 files) pass. Full-repository Ruff/format still fail: 290 diagnostics and
606 files, exactly matching main cc10afeb (original comparison retained).

Original failures are preserved: d6095c40's live review over-rejected AMCL
floating-point serialization and fixed base-frame differences; source assets
correctly resisted a fixture attempt to overwrite read-only bridge YAML;
the initial lifecycle CLI acknowledgment failed; duplicate rclpy shutdown
failed actor cleanup. These were corrected and new Worlds were launched.
An earlier protocol incorrectly required UNKNOWN for every negative and
rejected a valid BLOCKED covariance result after lifecycle deactivation. The
new protocol declares UNKNOWN/BLOCKED before launch, without weakening checks.

Known-fixture live success and synthetic contracts do not complete G2.
Held-out selection and acceptance have not run. G3 canonical guarded navigation,
G4 five staging repairs and G5 paired actual model/Memory experiments remain
separate gates. REAL and coverage tuning are outside this change. See
[ADR 0018](../../adr/0018-fresh-sim-navigation-readiness.md).

## Final known-fixture result

Protocol v5 completed seven fresh Worlds on source 7ebd1b9f. The positive
was READY_FOR_SIM_TEST; all six injected faults were UNKNOWN/BLOCKED. Each
original Native→RosSystemModel and L1 result was independently recomputed
as of its original review time (REPLAY, not seven additional World runs).
All seven owned containers stopped; all six fault actors exited 0; all
physics traces retained source-only mode, lease false, zero received control
and zero published velocity messages. G2_done=false and v1_done=false.

Public evidence: local `evidence/2026-10-12/G2-known-v5-original-Worlds-and-independent-recomputation/`.
The archive manifest records original/stored SHA-256 and byte counts; private
fixture homes and control ledgers are excluded. GitHub CI and normal merge
are separate integration gates recorded in the local delivery review.
