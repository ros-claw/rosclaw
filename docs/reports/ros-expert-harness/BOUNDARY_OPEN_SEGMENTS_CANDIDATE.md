# Two open boundary segments: development candidate

The closed training run at `e280f77f`, seed `111002`, reported a successful
Waffle boundary action after only 0.3431 m of observed boundary travel and
11.76 SIM seconds. The latest temporally associated public `/plan` message
still contained 9.8685 m of path. Both the first and final requested targets
were the same corner. Burger traversed 9.8499 m in its boundary stage in the
same training experiment. These records demonstrate that action success
alone does not certify a completed boundary circuit. Public plans have no
action identifier; the precise controller cause remains unproven.

The new explicit `*_boundary_tracking_open_segments` presets split the
existing nine-target inset-corner route into targets 0–4 and 4–8. Each
through-poses action has different start and final corners. The shared
corner is repeated at the segment boundary. Both actions use the existing
source-bound 100 mm tracking BT and a single 180-second boundary deadline,
capped by the original action deadline. Cancellation, timeout, failure or
changed BT bytes prevents the next dispatch. The protocol must register
exactly two segments before any World launch.

Body geometry, legal target domain, planner/controller parameters, brush
footprint, denominator and actual coverage verification remain unchanged.
The previous presets retain their original behavior. Open segments reduce
endpoint ambiguity but do not prove intermediate waypoint traversal or
coverage; only independent observed footprints contribute coverage credit.

Initial verification: 162 focused tests and 649 offline ROS tests pass;
mission typing checks seven files. The new nine tests include the shared
deadline, failed/cancelled first segment, changed BT between segments and
strict preregistration. The original five-test RED and the later test-clock
fixture failure are retained in local recovery logs. SDK configuration,
full software gates, fresh fault injection and paired physics must still
validate the committed source before any efficiency claim or merge.

The previous `111002` training pairs pass physical cleaning acceptance but
miss the joint 30% distance/time target: Waffle 33.89% / 17.70%, Burger
37.10% / 23.68%. They remain original evidence, not tests of this candidate.
No formal holdout is resumed and `v1_done` remains false.
