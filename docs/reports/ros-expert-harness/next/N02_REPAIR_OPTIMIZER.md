# N02 bounded pose-aware repair optimizer

Status: implementation and focused regression PASS; physical optimization
NOT_RUN. This change remains behind the explicit SIM `pose_aware` strategy.
The default remains the original `greedy` strategy. Main planning PR must
merge before this PR; no evaluation acceptance or efficiency gain is claimed.

The pure ranking module uses the actual remaining eligible missed cells,
existing legal robot centers and configured cleaning polygon. It considers
current heading, main-swath heading, reverse, perpendicular and diagonal
orientations. Static eight-neighbor center costs prohibit corner cutting.
A small bounded lookahead compares up to two predicted targets; only the
first is dispatched through the existing daemon-owned Nav2 client. All
subsequent selection is rebuilt from newly measured witness poses.

Predicted footprint cells never modify CoverageVerifier, Runtime, Memory
or canonical success. Repeated failures retain the existing per-cell retry
bound. Candidate generation, grid search and geometry precomputation have
a wall budget; incomplete search returns no partial selection and falls back
to the original conservative greedy selection. No control speed, lease,
stop watchdog or fixed verification denominator is reduced.

The [cost calibration](N02_REPAIR_COST_CALIBRATION.json) derives a2.8961555609 s
residual per-goal bias from the26 independently recorded baseline repair
segments; speed estimates0.2 m/s and1.8 rad/s come from unchanged controller
settings. This is a single-map pilot cost estimate, not a promise of controller
execution time. It must be assessed and frozen using paired pilots before any
held-out evaluation. A static grid route is prediction only; Nav2 still owns
real path calculation, dynamic obstacle handling and every motion.

Incremental tests cover asymmetric brushes requiring perpendicular yaw,
immutable masks and unchanged measured credit, disconnected diagonal graph
costs, exhausted retries, time-budget fallback, map mismatch, invalid costs,
and executor fallback retaining45-degree goals until an independent measured
pose arrives. Seven focused tests pass. The earlier full ROS run passed313
with11 deselected before the two additional fallback parameter cases; targeted
mypy and lint checks pass. Counts overlap and are not summed.

The optimizer is prepared in an isolated worktree while unchanged main-planning
pairs run; it is not used in those physical episodes. `v1_done=false`.
