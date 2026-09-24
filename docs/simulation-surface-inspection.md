# Current-pose surface inspection

`rosclaw.sim.backends.mujoco.surface.surface_snapshot` is an experimental,
simulation-only Python utility for a synchronous simulation owner. It is not
an Agent hardware interface, a new MCP action, or a motion permit.

Pass a matching compiled MuJoCo model/data pair and up to 32 named geometry
pairs. The utility copies pose and mocap values into private storage, refreshes
only private kinematics, and queries native signed geometry distances. It does
not step physics or refresh live derived geometry, contacts, solver warm-start,
or forces. The caller must serialize stepping **and model mutation**; a final
pose/time consistency check is diagnostic, not a concurrency lock.

```python
from rosclaw.sim.backends.mujoco.surface import surface_snapshot

measurement = surface_snapshot(
    model, simulation_data, (("end_effector_collision", "floor"),),
    maximum_distance_m=1.0,
)
```

`MEASURED` includes signed distance and the native closest-point segment.
`UNKNOWN` returns no distance/segment when the native query hits its cutoff
or does not support the pair. Unknown is never replaced by free space.
These are geometry queries independent of collision-filter configuration;
they do not certify physical contact, support loads, stability, swept motion,
or real-robot clearance.

A high body origin does not prove its collision surface is clear: a tilted
long end effector can have an elevated origin while still penetrating a plane.
Tests cover this case, stale live caches, nonmutation, cutoff semantics,
malformed identities/poses, and an interleaved-step rejection.

## Optional local surface derivatives

Set `include_distance_jacobian=True` for the opt-in v2 response. Each supported,
nondegenerate pair adds `distance_jacobian_qvel`: instantaneous signed-distance
rate is approximately this vector dotted with MuJoCo generalized velocity.
Both bodies contribute. This is a tangent-space local linearization, not a
raw-quaternion derivative, globally smooth gradient, or motion plan. Default
calls retain the original v1 response without derivative fields.

Cutoff/unsupported results, near-zero distances and degenerate native segments
return an unknown derivative, never a fabricated zero. The optional workload is
bounded to 32 pairs and 4096 velocity coordinates. Closest-feature switches may
be nonsmooth; consumers must qualify the approximation and retain physical
replay and ordinary execution limits.

Tests compare separated and penetrating spheres and an articulated
capsule/sphere pair against tangent finite differences, including a
contact-point derivative that a body-origin target would miss. Private
kinematics and center-of-mass kinematics never refresh live solver caches.

No robot-specific vocabulary, task reward, optimizer, executor or promotion
policy is introduced by this task-neutral helper.
