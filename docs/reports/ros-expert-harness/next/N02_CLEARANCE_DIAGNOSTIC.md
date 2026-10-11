# Known-Burger turn-clearance diagnostic candidate

This source adds `perimeter_stateless_clearance` for the known Burger fixture
only. It changes the preceding stateless-headland candidate's headland from
0.35 m to 0.50 m. Robot/brush geometry, operation width, controller parameters,
speed, collision guards, repair budgets, lease/deadline and measured coverage
denominator remain unchanged. The same sequential boundary pass is requested
after successful main coverage. Unknown Bodies cannot use this preset.

Motivation is diagnostic, not an established controller root cause. Source7a
Burger100902 candidate achieved final measured coverage98.0212%, contact0 and
complete stop/evidence acceptance, but its main action aborted104 at31.5775%
after RPP predicted-collision refusal and controller patience timeout. It needed
89 repair requests/119 waypoints. Its paired baseline was still running when
this source was prepared; no paired efficiency conclusion is inferred here.

Closed original audit and time-matched independent poses give main Nav2/GT XY
discrepancy median0.06237 m, p950.08493 m, maximum0.09255 m. Actual GT distance
to the main planned polyline has maximum0.14937 m. These measurements assume
the existing known-fixture map/world identity, and apply no GT correction.
Costmap/predicted-arc evidence is insufficient to establish the exact cause.

The unchanged installed Fields2Cover SDK predicts minimum static circle-to-wall
clearance0.07404/0.12404/0.17404/0.22404 m for headlands0.35/0.40/0.45/0.50 m.
Predicted main lengths are21.4282/17.9813/17.2813/16.5813 m. The0.50 m source is
an explicit conservative hypothesis; it may leave more boundary work. Offline
plans do not bound physical tracking or establish measured coverage/efficiency.

Validation:422 ROS tests passed/10 deselected. Fixed-image actual configuration
generation proves only the headland value changes in Nav2, and the sequential
boundary configuration stays identical. Eleven other source files match after
normalizing only their different output directory references. The first helper
attempt failed because it compared those World argv output paths literally;
its original source/log/output remain retained. No World or ROS Node started.
Core source bytes match source7a, whose Practice183 and required mypy121 gates
are retained as preceding unchanged-Core checks rather than new runs.

This candidate has no physical diagnostic or new five-pilot/evaluation series
yet. Source7a100902 and all earlier failures remain separate and untouched.
No source/configuration change can alter a running or frozen series. Any new
physical work needs a fresh prospective protocol, original safety gates and
explicit source freeze. P0 remains open; v1_done=false.
