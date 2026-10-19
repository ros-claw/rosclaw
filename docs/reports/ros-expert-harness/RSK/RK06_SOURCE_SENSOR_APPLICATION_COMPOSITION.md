# Source-bound optional sensor components in generated SIM applications

The registered source316 pilot exposed an application gap: driver bootstrap
generated a complete Nav2 configuration, but its new ROSbotXL World timed out.
The earlier accepted operator application used the same Nav2 bytes and Body
limits plus an explicit source-envelope scan-filter declaration. That comparison
supports a controlled composition experiment, not a causal claim about the
pilot failure. All original pilot outcomes remain frozen and negative.

The new opt-in `rosclaw.rsk_application_inputs.v2` keeps the original six fields
and adds `schema_version` and `sensor_processing`. The latter is a closed,
operator-owned SIM component catalog: a fixed stock filter artifact reference,
exact bytes/digest, original URDF digest, and original one-way scan bridge bytes
and digest. It explicitly represents an available artifact, not an installed
driver, current package admission or permission to install anything on a host.
Legacy six-field inputs retain their existing Core output and reader schema.
The component explicitly pins the reconstructed SDK8cc3 image; its ARM64 stock
dependency does not establish compatibility with another SDK or architecture.

The frozen Core compiler runs first. If it cannot compile the base application,
composition cannot turn that failure into a proposal. For a conditional plan,
the composition layer derives base and scan frames from the current original
Body candidate and reuses the SDK's source/mesh/geometry/fixed-transform and
typed bridge checks. There is no robot-ID branch and no caller-supplied polygon,
padding, speed, TF timeout or protection setting.

Only the generated `bringup.json` gains the existing `source_body_scan_filter`
declaration. Nav2, requirements, the eight navigation limits and the immutable
URDF remain unchanged. The existing implementation remaps only Collision
Monitor's sensor input; raw scan configuration and other consumers remain.
The envelope does not establish physical occupancy of every masked point, and
this optional profile remains restricted to operator-owned disposable SIM.

The composed plan records its Core parent hash, component input hash, file
hashes, exact dependency and source geometry, plus a component-only rollback
that restores the uncomposed application. This differs from rolling back the
entire application to driver-only operation. Both Generic A and Typed B/C/E
receive the same component original and use the same output contract. Generic
A still supplies its four file strings; mismatch returns no generated answer.

Composition is pure staging: no ROS, Runtime, subprocess, model, Memory query or
package installation starts. Actual artifact loading, raw-source/CM routing,
fresh Source/Body/full L1, two canonical receipts and independent stop/contact/
age evidence remain the existing SDK owner's responsibility.

Acceptance is a separate prospectively bounded component experiment: at most
four Worlds, fixed seed202, source Before/After/Rollback and exact-main After.
All source phase assets and source revision freeze before the first World.
There is no replay of a registered model allocation, no formal-study expansion,
no new model calls and no blind new-robot/task generalization claim.
