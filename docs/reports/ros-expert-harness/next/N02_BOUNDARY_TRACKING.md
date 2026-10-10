# Boundary-only intermediate checkpoint experiment

The continuous nine-waypoint development pair at source
`f2e7d865f0bef4821fbde2bd63e5c83ea08b9ff8`, seed105002, completed both
cleaning tasks, but both boundary actions were canceled at the unchanged
180-second wall-clock limit. Waffle distance/SIM reductions were
12.24%/1.37%; Burger reductions were 23.07%/12.00%. Neither pair met the
joint 30% target. These are training observations, not formal-series medians.

Waffle feedback retained nine remaining poses throughout 17,992 samples.
The first target's minimum reported sampled distance was about 71.4 mm,
above the 25 mm pruning radius. The robot did move; sampled feedback cannot
prove every BT tick or establish the controller root cause.

## Opt-in candidate

The known-fixture presets `perimeter_stateless_overlap_boundary_tracking`
(Waffle) and `perimeter_stateless_clearance_boundary_tracking` (Burger)
keep the original nine legal corner/midpoint targets. A separate
`boundary-through-poses.xml` changes only intermediate pruning from 25 mm
to the unchanged controller lookahead of 100 mm. The final goal checker
and global precise repair BT remain 25 mm. Recovery nodes, planner rate,
velocities, Body limits, collision handling, leases, task budgets, and
independent brush-footprint coverage accounting are preserved.

The executor verifies a fixed daemon-owned BT path and its SHA256 before
startup and again before dispatch; missing, redirected, oversized or modified
files are rejected. Only the boundary goal uses its explicit `behavior_tree`
field. Repair goals and existing presets retain their original behavior.
The generated BT hash, strategy, radius, waypoint count, stage budget and
precise-repair flag must match the frozen protocol before Gazebo/ROS children
start. SDK prepare-only configuration generation creates no World.

The protocol requires `candidate_boundary_strategy` equal to
`through_poses_tracking_midpoints`, `candidate_boundary_stage_budget_sec:180`,
`candidate_boundary_waypoint_count:9`,
`candidate_boundary_tracking_prune_radius_m:0.1`,
`candidate_boundary_tracking_bt_sha256` equal to the actual SDK-generated
lowercase SHA256, and `precise_repair_waypoints:true`.

## Evidence boundary

Preparation ROS offline regression: 554 passed, 10 deselected; widened mypy:
130 source files passed. The first focused run retained 100 passes and one
report-field naming failure; the field was corrected before the554-test run.
Tests compare the complete BT tree after undoing the sole radius change,
verify unchanged original repair bytes and goal/deadline/result semantics,
and reject invalid registrations and tampered files. These checks do not
prove waypoint arrival, coverage improvement or efficiency. Native, SDK,
full strict boundary and fresh physical acceptance must bind to the final
committed source. No new physical seed is registered by this document.

Version-aligned primary references (not an installed-binary equivalence claim):
[Nav2 1.3.13 RemovePassedGoals](https://raw.githubusercontent.com/ros-navigation/navigation2/1.3.13/nav2_behavior_tree/plugins/action/remove_passed_goals_action.cpp),
[Nav2 1.3.13 NavigateThroughPoses](https://raw.githubusercontent.com/ros-navigation/navigation2/1.3.13/nav2_bt_navigator/src/navigators/navigate_through_poses.cpp).
