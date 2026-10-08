# Controlled boundary candidate

User plan section5.1 permits legal Edges navigation targets. The pinned upstream
Fields2Cover main route is unchanged. An optional daemon-owned boundary action
uses five targets (four existing legal-center corners plus closing corner) with
Nav2 NavigateThroughPoses. It computes no coverage path and changes no inset,
speed, guard, denominator, deadline or repair algorithm. Disabled by default;
enabled only by the preregistered SIM `perimeter` preset. Unknown/nonrectangular
corner geometry skips; upstream main failure skips. Nav2 failure is recorded
and the existing measured repair loop remains responsible for completion.

Boundary has a180s goal limit within the immutable mission deadline. The existing
fresh witness, latched contact/completeness fault, daemon heartbeat and stop
cleanup apply. Installed image action schema was read and checked before use.

Audit reports upstream MAIN_COVERAGE and BOUNDARY_PASS separately, plus actual
primary-before-repair checkpoint. Only independent observed footprints enter
CoverageVerifier. Waypoint_count5 is exposed even though this is one action.
A synthetic partition regression proves .25upstream, .50afterboundary, 1.0final
and exact segment distance sum, without conflating these checkpoints.

Physical pilot status: NOT_RUN at code freeze; no claimed efficiency gain.
The revised protocol is registered before the first perimeter dispatch.
Five pilot and ten frozen evaluation pairs per profile are still required.
N01 latest-head CI full regression:8066passed/114skipped; DataFlywheel still pending.
`v1_done=false`.

## Sequential target pilot amendment

The first through-poses live run successfully completes its main and boundary
Nav2 actions but reaches only76.9016% before repair. Recorded Nav2 paths bend
inside the target envelope; this is evidence about actual behavior, not a claim
that every missed cell has one known causal source. That run remains frozen at
6d6cec87 and is retained in full. Its final efficiency/safety gates are pending.

A separate preregistered `perimeter_sequential` candidate uses existing legal
corner and edge-midpoint centers (9 targets including closure). Every target is
a separate NavigateToPose action and has its own audit/action id. Original
xy_goal_tolerance0.025/yaw0.1, speed and safety guards remain unchanged. A failed
or timed-out target stops the boundary stage; original measured repair remains.
Per-goal budget is at most45s, cumulative boundary budget180s, within the mission
deadline. No connecting motion path is hand-built.

This candidate's separate boundary centers use the original measured map and
existing cleanable_cells geometry with radius=max(original recovery radius,
physical radius+0.05). This increases Burger boundary clearance without changing
its legacy repair centers or verifier denominator. Static screen on the actual
map yields at least0.0749999m circle-to-wall clearance for both fixtures. This
is no guarantee of live tracking/reachability. See the preflight JSON.

Tests:326 ROS contract tests pass/10deselected; boundary failure/budget/target
geometry checks, old verifier replay compatibility, mypy and Ruff pass.
Sequential physical status: NOT_RUN at this code freeze.
