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
