# ADR 0047: independent SIM source binding, without execution authority

Date: 2026-10-12. Status: proposed implementation contract; G1 NOT RUN.
Scope: G1 only, following accepted G0 integration. No held-out model selection,
L1 readiness, L2 action, Memory experiment, hardware, or coverage tuning.

## Decision and reuse

Keep compiled navigation workspaces immutable. Reuse the existing workspace
inspector with separately supplied manifest/Body pins, navigation review,
RosSystemModel and SimulationNavigationBinding. Emit a separate source-binding
evidence object. Do not change the twelve-file artifact inventory, create a
Runtime, approve policy/Permits, add a robot profile, or expose raw ROS tools.

Three facts stay separate: historical artifact integrity; simulator-observed
source correspondence; current ROS navigation readiness. G1 addresses the
second. Its strongest result is SIM_SOURCE_MATCHED_NOT_AUTHORIZED, with
execution_authority=NONE, real_authority=false and L1/L2=NOT_RUN.

## Pinned implementation evidence

Existing acceptance image is sha256:d16320799584a60035548fb298243bdd8b0d2cb5760675a9cb46b608ea211e56,
ARM64 ROS Jazzy, actual `gz sim --versions` = 8.15.0. Installed development
headers expose Model, Collision, Geometry, ParentEntity, Pose, Sensor, World,
Name and SourceFilePath. No ModelSdf header exists in this pinned Gazebo 8
installation. Never claim a ModelSdf/raw-loaded-SDF hash was observed here.

Use a small Gazebo System with const ECM PostUpdate access to capture actual
entity hierarchy, collision Geometry and relative Pose, sensor mount metadata,
World identity and simulator iteration/time. No transport publishers, action
clients or mutation interfaces. Optional SourceFilePath is informative and may
be empty for models spawned from strings. Effective collision observations are
ECM geometry provided to the physics system; this does not inspect private
physics-engine internals or certify physical safety.

References: [Gazebo 8 Model / SourceFilePath](https://gazebosim.org/api/sim/8/classgz_1_1sim_1_1Model.html),
[Gazebo 8 components](https://gazebosim.org/api/sim/8/namespacegz_1_1sim_1_1components.html).
Header and package-version probes are retained outside the repository under
recovery_2026-10-12/G1-pinned-Gazebo8-*.original.log.

## Trust anchors and observation path

An operator-owned launch inventory, reviewed before simulation starts, pins:
compiled manifest/Body, exact URDF and expanded SDF, world bytes, conversion
command/version, all resolved resources, image digest, observer source/library,
expected entity/namespace/domain/frames, witness identity and fresh run ID.
The expected launch-inventory hash is supplied independently, never taken from
the same untrusted workspace as its own trust root.

A separate bounded supervisor creates a fresh isolated simulation instance,
mounts pinned inputs and observer read-only, records actual container/process
identity and command, checks mounts/asset identities before and after capture,
and captures observer output through an instance-owned channel. The verifier
must consume this checked capture, not promote caller-provided JSON containing
`trusted=true`, `GAZEBO_PHYSICS`, a Body hash, or a run ID. Operational trust is
operator-owned process/container separation. Same-UID development is not an
adversarial security boundary; root/daemon compromise is outside this claim.

Source correspondence requires both independently pinned startup inputs and
observed effective geometry. URDF strings and publisher names alone are
insufficient. The expected geometry/sensor inventory is obtained from the
pinned conversion/resolution output, compared to live ECM, and related to the
Body URDF/frame contract. Unresolved meshes, includes, poses or resource
identities remain UNKNOWN. Unsupported geometry must not be approximated to
obtain a positive result. No assumption that URDF and an unrelated SDF match.

The existing Native Probe captures actual ROS graph endpoint identities and
clock observations in the same bounded run. Check explicit domain/namespace,
frames, unique action and velocity-route endpoints against the operator launch
inventory. Endpoint names are not authenticated process identities; ambiguous
or duplicate publishers are UNKNOWN/BLOCKED. Do not claim DDS authentication.
A separately supervised existing physics witness must be bound to the same
World/entity/run. Self-reported witness metadata alone is insufficient.

## Interface and state

Request: schema_version=rosclaw.sim_source_binding_request.v1;
manifest_hash, body_snapshot_hash, launch_inventory_hash, source_asset_digest,
sim_world_id, sim_run_id, expected_robot_entity, expected_ros_domain,
expected_namespace, expected_frames, bounded freshness policy and evidence refs.
No `allow_unknown`, guessed entity or default real authority.

Evidence: schema_version=rosclaw.sim_source_binding_evidence.v1;
result=MATCHED|MISMATCH|UNKNOWN; detailed deterministic reason codes;
manifest_hash, body_hash, world_id, sim_run_id, model_entity_id,
loaded_geometry_fingerprint, source_provenance_class, ROS snapshot hash,
observed_at, capture bounds, evidence refs, execution_authority=NONE,
real_authority=false. Fingerprints describe captured semantics, not an
unobserved raw ModelSdf byte stream. Missing facts cannot become MATCHED.

MATCHED requires complete independently anchored observations. Observed
contradictions produce MISMATCH. Missing, stale, unresolved, unsupported,
reset/reloaded or unverifiable provenance produces UNKNOWN. Pin mismatches and
malformed inputs fail closed. Returning MATCHED never changes historical Body
status, authorizes an action or declares current L1 readiness.

## Acceptance matrix before G1 can close

Use a known Body in a new actual Gazebo instance, zero action goals and zero
velocity commands. Retain original runtime logs, effective collision samples,
process/mount/asset attestations, native ROS snapshots and independently
recomputed results. SDK/synthetic contract tests alone cannot close G1.

| Case | Required result |
| --- | --- |
| Independently pinned known Body/source, unique live entities/routes | MATCHED, no authority |
| Asset replacement before/during run | MISMATCH or UNKNOWN |
| Smaller effective collision despite unchanged Body | MISMATCH |
| Wrong Body/manifest/run/world pins | MISMATCH |
| Observer/witness renamed, replaced, exited or stale | UNKNOWN or MISMATCH |
| Wrong ROS domain/namespace/frame contract | MISMATCH or UNKNOWN |
| Duplicate entity or publisher / unexpected control endpoint | UNKNOWN or MISMATCH |
| World reload, simulation reset, clock regress/freeze | UNKNOWN |
| Missing SourceFilePath with otherwise checked string-spawn lineage | Not sufficient alone; checked lineage still required |
| Unresolved mesh/include/pose, missing geometry or sensor mount | UNKNOWN |
| Only self-consistent JSON hashes / publisher claims | UNKNOWN |

G1 remains NOT_DONE until its actual positive and negative matrix passes.
G0 external dependency failures do not permit bypass or an alternative live
readiness claim. Hold implementation PR/integration until #670 has normal
source checks, merge-tree proof and actual-main checks.
