# Public observation coverage library — development candidate

These algorithms consume online occupancy, optional public navigation costmap,
public odometry, independent cleaner state and a registered map-from-odom
transform. They contain no house, saved waypoint, world parser or Judge input.
`Grid(..., clearance=...)` must receive the configured robot footprint.

`AnchoredCoverage.next_work` proposes finite interior/boundary/repair work and
frontier viewpoints. It cannot execute an action, approve it, mark a commanded
path as cleaned, or declare whole-house success. Existing Runtime and public
SDK remain the execution routes. A caller must revalidate the current geometry
before dispatch, stream actual brush/odom samples during execution, persist the
ledger and apply a finite operation budget and independent cancellation.

The ledger keeps observed debt and executed brush estimates in one continuous
odom epoch, then projects into the current map. Global translation/rotation
corrections preserve the same evidence. Odom resets, nonrigid map deformation
and independently moving submaps require explicit registration and are **not**
solved by applying a single SE(2) transform. Unknown space is not a known empty
room. Unknown, blocked and retry debt remains present. Missing sensor/brush
state earns no swept bridge; restart also breaks interpolation.

**Do not treat ordinary map->odom updates as global map frame changes.** They
also compensate odometry drift. The shared coverage App uses
`GraphAnchoredCoverage`: observed floor/brush patches attach to nearby stable
IDs from the public SLAM graph. Each node pair defines a local rigid basis;
corrections can differ between patches. Missing anchors retain unlocalized debt
and award no credit. Fine deformation within a patch remains an estimation
uncertainty, and a changed graph/odom epoch requires explicit registration.
The original single-rigid-frame class is retained for inputs whose registration
is explicitly known, not as a general live-SLAM correction model.

SLAM Toolbox 2.8.5 publishes node IDs and corrected XY positions in
`/slam_toolbox/graph_visualization`; its sphere marker orientation is not node
yaw, so the adapter does not use it as such. See the pinned upstream
[publisher](https://github.com/SteveMacenski/slam_toolbox/blob/2.8.5/src/loop_closure_assistant.cpp).

`provenance.json` identifies the unchanged source algorithms from HomeClean.
The port changes import namespace, formatting/lint and an explicit maximum
work length parameter in `select_phase`. `online.py` adds anchored persistence.
Ported geometry tests and new registration/brush/debt tests exercise this layer.
This is a library candidate, **not yet an installed coverage executor or H1
acceptance**. Resolver recommendations stay unconfigured until a real provider
is bound and exercised. Do not reinterpret existing A/B/C results.
