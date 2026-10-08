# Goal-rotation controller hypothesis and preregistered candidate

The sequential pilot at source955ff360 reaches its first main navigation start
and then stops translating; its main action aborts105. It never dispatches the
boundary stage. This is not evidence that sequential boundary goals fail.

Recorded SIM interval25–44s contains380 `/nav_cmd_vel` messages, all with
|linear_x|<=0.001; angular commands vary approximately[-0.1963,0.2028]rad/s.
Both smoothed and final velocity streams are recorded. No Collision Monitor
state transition occurs inside the main action interval6.75–45.87s. This does
not prove absence of every unexposed controller condition. Measured robot
position stays near(-0.903,-0.696), with small yaw changes. Nav2 logs report
Failed to make progress; source artifacts and command hashes remain unmodified.

Installed RPP/controller/NavFn version is1.3.13-1noble.20260921; the installed
RPP header includes `stateful`. The matching upstream
[RPP1.3.13 implementation](https://github.com/ros-navigation/navigation2/blob/1.3.13/nav2_regulated_pure_pursuit_controller/src/regulated_pure_pursuit_controller.cpp)
latches XY arrival when stateful is enabled and then chooses goal-heading
rotation. **Inference/hypothesis:** controller/goal-checker timing near arrival
may leave rotation active while the action cannot finish. Local causal proof is
still UNKNOWN; upstream code alone does not establish this run's internal state.

A separately preregistered `perimeter_stateless` candidate keeps the original
main coverage planner, then sequential9 boundary targets, and sets only
`controller_server.FollowPath.stateful=false`. It does not alter SimpleGoalChecker
statefulness or xy/yaw tolerances, speed, acceleration, progress-check timeout,
collision detection/monitor, source freshness, leases, bottom-level watchdog,
mission deadline, original repair centers, Body or verifier denominator.
The controller recomputes whether to enter goal rotation at each control step.
Only physical paired trials can accept/reject its behavior.

Actual pinned-image config-only preflight covers both Body profiles. Every
non-stateful Nav2 parameter compares exactly equal and world/URDF/SDF/bottom-level
controller files are byte-identical. No ROS action was dispatched in preflight.
First preflight omitted the existing /ws ROS overlay and failed before setup;
its directory is retained, and corrected preflight uses a distinct attempt002.

Passive observation now also captures the upstream published RPP rotation flag,
body-frame received plan and two lookahead targets. The rotation flag does not
distinguish goal versus path rotation and is not reported as causal proof. None
of these debug topics grants measured coverage. Full ROS contract suite327pass/
10deselected, Ruff/format pass. Physical status: NOT_RUN at this code freeze;
no efficiency benefit claimed and no evaluation seeds consumed.

Post-freeze Waffle seed100801 paired outcome: original main and all9 boundary
goals succeeded, primary measured coverage91.1633%, final98.1823%, zero contacts/
gaps, complete audit/replay and independent stop. Distance reduces13.7724% and
SIM span20.5538% against this pair's baseline; the30% target remains unmet.
This observed improvement does not prove the internal controller hypothesis.
Burger acceptance and full fixed-seed evaluation remain outstanding.

## Burger pilot: collision-ahead and planning-clearance combination

The Burger stateless pilot's original main aborts104 after579 measured samples;
the boundary stage is skipped. The Nav2 log explicitly emits repeated RPP
collision-ahead warnings, then controller patience exceeded. This is distinct
from the Waffle rotation/progress-stall hypothesis. The exact projected costmap
collision cells are unavailable, so their per-cell cause remains UNKNOWN.

Before dispatching another combination, preregister Burger-only
`perimeter_stateless_headland`: existing stateless controller and sequential
boundary handling, but use the already screened0.35m headland. Pinned planner
predictions give0.074037m conservative polyline wall clearance versus~0.024m
at baseline0.3; no physical tracking guarantee is inferred. Config-only pinned
image preflight proves that only the headland and RPP stateful flag differ from
baseline, with identical world/URDF/SDF/watchdog files and all other Nav2 params.
No speed, goal tolerance, collision detection/monitor, source health, progress
deadline, lease, mission denominator or original per-Body deadline is changed.
Waffle cannot select this Burger-only combination. Evaluation seeds remain
unused and the ongoing failed-main pilot is completed and preserved.
