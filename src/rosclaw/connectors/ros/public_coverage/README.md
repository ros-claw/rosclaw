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


Regional native costmaps can supplement the mandatory global geometry through
`Grid(..., local_navigation=costmap, map_from_local=[x, y, yaw])`. Registration
must be explicit and current at the caller. Unknown cells inside the observed
window stay blocked. Outside that finite window the unchanged global occupancy
and costmap checks remain mandatory; a missing local window never clears global
obstacles. Both physical and tracking-margin circles use exact closed cell
squares. An unsafe current local footprint rejects the Grid before planning.


`FootprintPathSpace` reuses Grid's continuous physical/planning footprint
predicates without computing room components or frontiers. Supply the same
explicit bound spaces when checking a currently observed path suffix. It does
not establish source freshness or grant authority. A driver must retain the
original action identity, cancel through its existing native interface when a
path becomes invalid, and verify that action's terminal result before replanning.


`online.propose_transfer(grid, observed_xy, target_xy, max_target_shift=0.05)`
proposes one bounded transfer prefix. It preserves an available local detour;
only when none exists does it reuse the original occupancy/navigation snapshot
and configured body radius/margin for a global intent. Every returned prefix
is checked against the original complete live grid and is at most 0.60m.
Unknown/lethal cells remain blocked. The immutable result includes the selected
target, intent source/length and target shift, not the full global route.

An existing return contract may explicitly request up to 0.10m target snapping;
this is independent of tracking error. This function establishes neither
observation freshness nor execution admission, arrival, cancellation, charging
or mission success. Callers must use current public observations and the
existing canonical authority/ownership/result checks. Snapshot message copies
preserve accepted content if callers later mutate their inputs; copies never
renew the original timestamps. Source fixtures and geometry regression are not
an installed App or a HomeClean mission acceptance.


`AnchoredCoverage.next_work(..., prepare_path=body_work_adapter)` applies an
optional pure Body/controller path adapter **before** gain scoring, direction
selection, geometry validation and retry matching. The returned work path is
the same geometry that was scored; callers must not clip it afterwards. Each
orientation is prepared independently. A clipped, already-cleaned point cannot
borrow the unexecuted lane's expected gain, while an executable reverse prefix
can remain eligible. The adapter may return `None` for untrackable work; this
retains outstanding obligations and does not claim mission completion.

Prepared paths must remain finite, at most 1024 points, within the original
work-length bound, and safe in the complete current grid. Candidate input
arrays are copied before adaptation. Admission, critical observation freshness,
canonical result/stop verification and finite no-progress limits remain owned
by the existing executor. This API change has source/fixture evidence only
until a new installed and physical condition has been qualified.
