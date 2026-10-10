# Read-only Body discovery

`ros discover-body --urdf expanded.urdf --snapshot sealed-system.json --json`
produces a candidate using captured graph, sensor frames, odometry, map, TF and
URDF-description provenance. It does not install a Body or grant action authority.
For a read-only live inspection, omit `--snapshot` and use the existing native
probe connection and `--deep`; missing observations stay UNKNOWN.

The discovery and collision-envelope implementation is extracted from the prior
N04 development branch `ac778601`, rather than importing its dynamic-cleaning
and unactivated-bootstrap experiments wholesale. The current main diagnostic
fix is retained. No unfamiliar robot asset was used to implement this change.

The candidate selects one fresh typed odometry/map/lidar stream, requires a unique
matching captured URDF description and fresh directed TF chains, and retains the
actual namespace and frames. Duplicate signal records and multiple publishers
remain ambiguous; a last observation cannot silently select the binding.
A type-matching navigation action is an interface observation, not readiness.

Collision geometry uses expanded URDF primitives and joint transforms to derive
a conservative envelope. Unresolved meshes, invalid topology, unsupported
articulation and partial evidence retain UNKNOWN without a guessed radius.
Neither drive nor cleaning capability is inferred from the model's name.

The passive ROS probe records original message-header receive times and bounded
URDF-description digests, and recognizes a latched map by observed type and
publisher durability even after topic renaming. Existing action-status reads,
parameter/plugin diagnostics, integer parameter capture and runtime control
boundaries are retained. The CLI reads only a bounded regular URDF file and
refuses symlinks/FIFOs/directories before opening a live discovery connection.

`integrations/ros_probe/acceptance/readonly_discovery_fixture.py` is an isolated
SDK contract using explicitly synthetic sensor, TF, map and description sources.
Its typed navigation server rejects goals, and the contract sends no goals.
It checks a fresh candidate, mismatched URDF, multiple lidar publishers and
stopped odometry. Run two independently named/frame-renamed/size-varied fixtures
without changing the discovery code. Actual SDK execution must be recorded
separately; successful fixtures are not unseen-robot or physical L1/L2 acceptance.

Further gates remain: original robot asset provenance after feature freeze,
actual sensors/TF/geometry verification, immutable approved Body binding,
rosclawd-guarded point navigation with independent stop and endpoint receipts,
and controlled migration tasks. Cleaning optimization is paused per the updated
user priority; its historical failed efficiency gates remain open.
