# N04 generic collision geometry preparation

Status: OFFLINE_PREPARATION. No held-out robot has been chosen or inspected.
No L0–L4 physical acceptance, Body binding or cleaning capability is claimed.

`derive_collision_envelope(expanded_urdf_bytes, base_frame=...)` reads named
collision primitives and fixed XYZ/RPY joint transforms. The requested base
frame must exist and have only fixed ancestors. Boxes, cylinders and spheres
use containing corner boxes; movable child subtrees use containing 3D spheres,
including full rotational sweeps and bounded prismatic travel. This can
overestimate clearance substantially. It never produces an optimistic mesh
approximation or infers geometry from visual links.

All collision links must belong to a single valid tree. An unsupported mesh,
unresolved frame/articulation, missing collision geometry, unexpanded expression,
XML entity declaration, nonfinite value or arithmetic overflow makes the
numeric envelope UNKNOWN. Partial successfully resolved primitives cannot
promote a whole-body envelope. Source bytes and the requested base frame are
recorded; output is candidate evidence, not verified Body truth.

The output always grants no capabilities and infers no cleaning attachment.
This parser uses no vendor whitelist, known-robot profile, topic-name guess,
room geometry, default radius, ROS transport or runtime action.

Validation: 25 anonymous synthetic-URDF cases pass, including frame inversion,
rotated/off-center collisions, sampled articulated sweeps, nested fixed links,
bounded prismatic travel, unsupported meshes mixed with valid collisions,
disconnected cycles, incomplete sources, UTF-16 entity inputs and overflow.
The ROS connector suite passes 368 tests (10 deselected); Ruff/format pass.

`discover_body_candidate()` now consumes a sealed fresh graph and actual typed
message frame metadata, fresh TF chains and uniquely matching live URDF digest.
No default topic/frame names are used. The optional read-only ROS2 host probe
captures original message receive times and robot_description parameter hashes;
it does not export URDF payload in the periodic snapshot. Stale or ambiguous
streams/descriptions, incomplete TF and mismatched provenance remain UNKNOWN.
A PROPOSED candidate never authorizes navigation, grants a capability or binds
a cleaner. There is no drive-topic publication or writable robot RPC.

Nineteen additional synthetic tests cover namespace/sensor/frame renames,
expired signal/header/TF/URDF evidence, ambiguous sensor/description sources,
source/hash tampering and actual probe metadata extraction. The combined ROS
suite passes 387 tests (10 deselected), both modules pass mypy and Ruff/format.
These are offline tests, not held-out L0–L4 robot evidence.

`validate_sim_attachment()` requires a named explicit SIMULATED_CLEANING
declaration and a finite simple polygon. It derives a conservative inscribed
radius and records its declaration hash; drive observations cannot supply it.
`derive_allowed_region_grid()` supports bounded arbitrary simple allowed task
polygons and observed spawn positions, including a shifted L-shaped region.
It checks the task/map/spawn frames and observed map orientation, applies
clearance/connectivity only inside the declared region and produces a frozen
initial denominator plus legal centers with provenance hashes. Temporary
occupancy cannot redefine that initial denominator. Rotated/unknown map
orientations and offset-only cleaning attachments explicitly remain unsupported.
The original known-Body run.py denominator is unchanged. The observer adds actual
origin quaternion metadata for later generic preflight.

Sixteen attachment/region tests and the complete current ROS suite pass:
416 passed /10 deselected, with three-module mypy and changed-file Ruff passing.
These remain offline preflight results and grant no physical capability.

Remaining before the N04 feature freeze: verified fixture-policy binding,
generic simulator workspace generation and live region/Body integration.
Only after that freeze may the actual held-out
asset be chosen. Physical acceptance must then retain separate L0–L4 results,
including two namespace/sensor/frame/spawn perturbations without core changes.
