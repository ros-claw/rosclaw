# Measured post-event motion

`growth.recovery_motion_metrics.recovery_motion_metrics` summarizes an explicit
post-event window of measured root positions, XYZW quaternions and joint angles.
The caller supplies the sample interval, event index and forward direction in
the same coordinate frame as the positions. It reports retreat, final forward
displacement, total root path length, orientation excursion and RMS angular,
joint speed, acceleration and jerk, with explicit SI units.

Quaternion sign changes do not create artificial angular motion. Inputs are
not mutated. Invalid shapes, nonfinite measurements, nonunit quaternions,
missing post-event support, zero direction and derived overflow are rejected.
Sampling frequency limits what can be observed; finite differences of a
50 Hz recording are not a certificate about unrecorded 500 Hz joint motion.

This is an offline numerical measurement, not an outcome verifier, physical
safety predicate, naturalness score, trained self-model or consciousness claim.
An intentional recovery step or dive can move substantially. Lower movement
alone is not better, and role-specific objectives must remain explicit. The
function has no simulator, checkpoint, actuator or promotion authority.

For paired comparisons, declare equal post-event durations before reading
results. Keep missing events and insufficient windows visible rather than
assigning zero movement to failed episodes. Authenticate the source reports,
traces, coordinate convention and joint mapping outside this pure function.
Changes to training rewards require a new declared protocol; existing
experiments should not be hot-patched to use these diagnostic measurements.
