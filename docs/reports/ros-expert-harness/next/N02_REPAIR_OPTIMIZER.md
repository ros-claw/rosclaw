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
# Current integration and remaining physical gate

The repair engine is prepared on the reviewed stateless/sequential boundary
candidate. Defaults remain greedy, and all legal boundary target/controller
options are preserved. Updated evaluation admission binds the actual image,
source, selected main and repair strategies, and original per-Body deadlines.
Statistics reject mismatched arm presets, altered baseline repair strategies,
mixed candidate strategies or missing complete independent observations. Final
analysis records its own source hash as well as every input hash.

Native supplemental acceptance supports current append-only path logs without
requiring removed overwritten snapshot filenames. Its still-running observer
prefix is explicitly LIVE_PREFIX_NOT_FINAL_AUDIT; complete shutdown/EOF/hash
validation remains separate and required. Neither this compatibility fix nor
the offline engine tests count as a Native physical acceptance run.

Physical pose-aware repair status remains NOT_RUN until the main PR gate passes.
The latest main Waffle pilot's primary91.1633% still needed23 greedy repairs
and achieved only13.7724% distance/20.5538% SIM reductions. That motivates the
repair trial; predictions cannot establish a gain. All original pilot failures
and unconsumed evaluation seeds are retained.

## Candidate diversity regression

The bounded shortlist represents distinct four-neighbor components of the
measured missed mask and removes equivalent predicted footprints. Otherwise,
multiple yaw variants covering the same cells can fill the entire shortlist
and hide the next hole from the two-goal lookahead. A narrow legal corridor
with two adjacent distant missed cells reproduces that failure: the previous
shortlist found only one target; the corrected search considers both. These
are predicted goals, not coverage credit, and only the first is dispatched.

The immutable Waffle stateless pilot's primary mask contains316 missed cells
after91.1633% coverage. Three read-only rankings evaluated20736 poses each,
returning READY in215.19,210.84 and210.30 ms, below the500 ms bound. Input
evidence SHA256 is
`bca5084b5f923ff764ced9c53fcc9f7c910021e3d545aed4d1e08801c8e5add2`.
These timings describe computation on a recorded mask, not physical efficiency.
The complete local ROS suite passed344 tests with10 integration cases
deselected; targeted mypy, Ruff and formatting passed. Physical pose-aware
acceptance remains NOT_RUN.
