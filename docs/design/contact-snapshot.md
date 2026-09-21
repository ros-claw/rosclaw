# Read-only MuJoCo contact evidence

`rosclaw.sim.backends.mujoco.contact.contact_snapshot` is an optional simulation
backend helper, not an MCP motor tool or a new authority path. It contains no
robot, sport, role, scoring, or promotion assumptions.

The simulation owner supplies explicit geometry names and immutable evaluated
qpos/qvel/time. After a single Euler or implicit step, these are the **pre-step**
values, not the integrated values now in live qpos/qvel. Private kinematics and
body velocities must agree with native evaluation caches. RK4 is rejected rather
than guessing which intermediate evaluation the contact cache represents.

Each rigid contact reports force and contact-point torque **on the second geom**
in world coordinates. The first geom receives the opposite wrench. The normal
points first-to-second; relative point velocity is second-minus-first, including
rotation. Negative normal velocity means closing, not a negative contact force.
Forces are not impulses, contact-point torque is not torque about a body origin,
and a zero or absent contact is not a successful task.

The owner must serialize stepping and model mutation and establish that the
native force solve actually occurred. Cache consistency is not a cryptographic
freshness proof or hostile-code isolation. The helper never refreshes the live
solver, integrates physics, clears faults, or grants permission. Common
interleaved state/contact changes are detected without rolling back the caller.
Flex contacts involving requested geoms are outside this rigid-contact contract.

Unit checks cover world-force signs against generalized constraint forces,
rotating point velocities, stale post-integration state rejection, input bounds,
nonfinite caches/results, interleaving, and 200-step exact noninterference.
This component alone does not establish a task teacher or learning breakthrough.

Semantics: [MuJoCo contacts](https://mujoco.readthedocs.io/en/stable/computation/index.html#contact)
and [contact force API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-contactforce).
