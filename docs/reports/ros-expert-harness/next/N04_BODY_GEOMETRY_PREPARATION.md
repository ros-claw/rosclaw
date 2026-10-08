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

Remaining before the N04 feature freeze: provenance-bound read-only Graph/TF/
sensor candidate construction, explicit SIM attachment declaration, verified
fixture-policy binding, generic simulator workspace generation and allowed
polygon/denominator integration. Only after that freeze may the actual held-out
asset be chosen. Physical acceptance must then retain separate L0–L4 results,
including two namespace/sensor/frame/spawn perturbations without core changes.
