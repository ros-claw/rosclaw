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

This minimal version handles WorldSpec attachments with joint or joint-in-parent
actuator transmissions. Unsupported transmission types fail closed. Imported
external composite worlds do not acquire actor bindings merely from names or
metadata, and are not silently converted into WorldSpecs. Native ROSClaw/Kimi must
provide true source relationships for the M22 scene in a subsequent explicit
registration path. Scene placement, task qualification, collision-free motion,
robot calibration and tennis success remain separate validation obligations.
