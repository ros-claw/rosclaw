# SIM scene actors and mission body authority

A WorldSpec composite scene previously discarded source model references after
attaching multiple bodies. The formal mission retained a single body binding
(for the current tennis task, sim/ur5e). Neither an external composite MJCF nor
actor names G1/M20 update that binding or prove robot capability qualification.

WorldSpec compilation now persists source model refs/digests, namespaced prefixes
and attachment offsets. A separate immutable `simart` scene-actor manifest binds
the final scene model ref/digest. Verification independently compiles actual
source and scene MjModels and derives body subtrees, joint/geom owners, DOF indices,
actuator indices and joint transmission targets. Cross-actor actuator retargeting
fails closed. Source physics fields and integer ownership are verified, with one
explicit numeric exception: MjSpec serialization rounds UR5 body inertia by up to
1.56e-9 absolute / 3.52e-7 relative. Only body_inertia permits rtol=1e-6,
atol=1e-12, recorded with measured differences; other checked fields remain exact.
The manifest calls this source-binding-with-declared-rounding.

`sim inspect` exposes verified `detail.simulation_scene`; Pi context's existing
body is explicitly labeled `mission_body`. Scene manifests have physical authority
NONE, capability qualification NOT_EVALUATED, SIMULATED trust, and no real execution
eligibility. They neither change mission body nor add tools/control permissions.
Task-local sources remain LOCAL_UNREGISTERED regardless of actor name. A catalog
source additionally has to match current resolved catalog MJCF and actual assets,
so forged source metadata cannot promote an external robot to the catalog model.
A catalog-source label is provenance, not calibration or physical qualification.

Tests compile two separate generic one-joint actors without physics stepping,
verify real MjModel name/index/trnid relationships, and reject content-addressed
manifest tampering, bad source refs/digests, forged catalog identities, actual
scene inertia/mass/axis/friction/control-cap changes and cross-actor motor owners.
An actual catalog UR5 attachment confirms the declared rounding and compatibility.
These fixtures are not Unitree models or tennis controllers.

Joint and joint-in-parent actuator ownership are qualified by actual indices.
Other transmission types retain previously valid WorldSpec compilation, but expose
ownership NOT_QUALIFIED/UNSUPPORTED_TRANSMISSION, no invented joint owner, and actor
source_binding_verified=false. Partial provenance never grants physical authority.
The initial source and serialized scene MjModels are separately pinned by complete
MJB SHA256, size and backend version. Recompilation must match both captured
signatures, including compiled mesh/texture/heightfield arrays, before any mapping
is returned. Metadata cannot quietly bind a changed current source and changed
current scene to the original compiled model. Existing registered proof continues
to reference immutable original model/asset objects; raw blob corruption is also
rejected by SimStore content addressing.

Compatibility tests first reproduced two real SITE/TENDON regressions: native XML
compiled successfully but initial provenance threw UNSUPPORTED_TRANSMISSION.
They now compile and expose honest partial qualification. A read-only frozen
794443b4 verifier accepted simultaneously changed source and scene mesh assets
(new content refs/digests, same declared actor); the new initial source/scene
signature tests reject this. All checks run with zero physics steps. Imported
external composite worlds do not acquire actor bindings merely from names or
metadata, and are not silently converted into WorldSpecs. Native ROSClaw/Kimi must
provide true source relationships for the M22 scene in a subsequent explicit
registration path. Scene placement, task qualification, collision-free motion,
robot calibration and tennis success remain separate validation obligations.
