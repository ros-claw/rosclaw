# Private simulation surface inspection

`rosclaw.sim.backends.mujoco.surface.surface_snapshot` inspects explicit named
geometry pairs on a private kinematic copy. It does not step the simulator,
refresh live solver caches, mutate a body, choose actions, or authorize hardware.
The simulation owner must serialize live stepping and model mutation around it.

By default the v1 response retains signed surface distance and nearest points.
Body origins are not substituted for collision surfaces. A native distance
cutoff is `UNKNOWN`, not certified free space.

Set `include_distance_jacobian=True` for the opt-in v2 response. Each supported
nondegenerate pair adds `distance_jacobian_qvel`, a local tangent-space derivative:
instantaneous signed-distance rate is approximately the vector dotted with
MuJoCo's generalized velocity. The derivative includes both bodies. It is not
a derivative with respect to raw quaternion components, a globally smooth
gradient, a contact-force measurement, or a swept-clearance certificate.

Cutoff/unsupported results, near-zero distances and degenerate native segments
return an unknown derivative, never a fabricated zero. The optional workload is
bounded to 32 geometry pairs and 4096 velocity coordinates. Closest-feature
switches can be nonsmooth; consumers must qualify their local approximation and
retain physical replay and normal execution limits.

Tests compare separated and penetrating sphere pairs and an articulated
capsule/sphere pair against tangent finite differences. The articulated example
also checks a contact-point derivative that a body-origin target would miss.
Live state/cache noninterference and the default response remain tested.

This task-neutral helper contains no robot names, task rewards, football rules,
optimizer, motor executor, promotion policy, or MCP hardware action path.
