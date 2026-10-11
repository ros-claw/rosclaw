# One-cell corner inset development candidate

The source62e2 Waffle107002 development pair completed both coverage tasks,
but its nine-waypoint boundary action hit the unchanged 180-second limit.
The candidate used 60.55202788732395 m and 637.28 SIM seconds, compared with
51.782128733411184 m and 493.0 seconds for its own baseline. Distance and
time increased by 16.94% and 29.27%; this candidate is not promoted.
These are one training pair's results, not formal-series statistics.

The closed Waffle feedback first changed from nine remaining poses to eight
at 159.64 navigation seconds, after eleven recoveries. Its first original
corner's minimum sampled planar distance was about 88.4 mm. The old trace
entered the hypothetical inset corner's 100 mm neighborhood at 1.94 seconds.
This static comparison motivates a new test; changing the target changes
the actual route, so it predicts neither arrival time nor efficiency.
Feedback poses do not establish each behavior-tree tick or the controller
root cause. Historical Burger105002 geometry supplies only a second closed
legal mask for static screening.

Burger107002 subsequently closed with candidate 84.00313569013865 m /
1039.0 SIM seconds versus its baseline 103.8235397693506 m / 1122.92 seconds:
19.09% / 7.47% reductions, still below the joint target. Its boundary action
also timed out. Remaining poses first changed nine to eight at 86.81 seconds
and reached three at 178.54 seconds. Its original first target was sampled
within 100 mm at 86.83 seconds; the hypothetical inset target was sampled
within 100 mm on the old trajectory at 2.43 seconds. These remain sampled
feedback and static comparisons, not tick-level or future-path evidence.
Both107002 pairs passed independent canonical phase/coverage/stop replay.

## Explicit known-fixture candidates

`perimeter_stateless_overlap_boundary_tracking_inset_corners` is limited to
Waffle; `perimeter_stateless_clearance_boundary_tracking_inset_corners` is
limited to Burger. Both select `through_poses_tracking_inset_corners`.
Exactly four corner coordinates move one existing grid cell inward in each
axis. The four original edge midpoint coordinates remain, and the first
target repeats to close nine poses. Tangent orientation hints are recomputed.
Every target must already belong to the unchanged approved boundary mask.
Missing inset corners, irregular spacing, or insufficient grid extent produce
no boundary action. This does not compute or authorize a motion path.

The executor records both original and actual targets, grid resolution and
`boundary_corner_inset_cells:1`. The generated experiment and preregistered
protocol must agree on the integer one-cell inset, in addition to the existing
strategy, nine poses, 180-second group budget, 100 mm intermediate pruning,
precise-repair flag and actual generated boundary BT SHA256. Boolean, float,
missing or different inset declarations fail closed before World startup.
Original presets reject the new inset metadata and retain their geometry.

Both candidates reuse the existing boundary-only BT and fixed container path
`/evidence/boundary-through-poses.xml`, verified against its host-side bytes
before startup and dispatch. Global repair pruning and the final goal checker
remain 25 mm. Planner parameters, Body limits, velocity limits, recoveries,
leases, fixed verifier denominator and cleaning footprint are unchanged.
Waffle retains its 0.35 m swath/0.50 m headland and robust sequence repair;
Burger retains its existing clearance preset and single-goal repair.
Failed or timed-out stage results remain visible even if later repair makes
the complete coverage task pass.

## Acceptance status

At preparation, scoped style/syntax and static geometry on two closed masks
passed. Those masks were checked with an explicitly synthetic entry pose;
they prove only legal target construction and input preservation. No new
physical seed is registered here. Behavioral tests, actual SDK generation,
Native, full strict regression and new physical development pairs must bind
to the final committed source before any promotion. Existing negative pairs
and the halted formal series remain retained without retry or replacement.

Preparation behavioral tests then passed113 cases, and mission mypy passed
seven source files. The first run retained112 passes and one test fixture
constructor-keyword error; the fixture was corrected without changing the
executor or any acceptance assertion. Actual committed-source broad software
and physical acceptance remain separate gates.

An isolated pinned-SDK component then exercised the installed Nav21.3.13
RemovePassedGoals and PipelineSequence with BehaviorTree.CPP4.10.0: ten
expected goal-count checks passed across25 mm and100 mm radii, repeated ticks,
and retention of the final goal. This used one fixture node with a directly
populated synthetic static TF cache, no spinning executor/action client/World.
The original logs retain ten dedicated-TF-thread timeout diagnostics; the
cached-transform pruning checks still passed. This is not proof of live TF
discovery, the complete navigation tree, physical arrival or efficiency.
Earlier diagnostic header/link/log-directory failures remain archived.
