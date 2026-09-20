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
