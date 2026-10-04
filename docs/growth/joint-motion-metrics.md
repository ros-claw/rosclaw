# Per-joint offline motion diagnostics

`growth.joint_motion_metrics.joint_motion_metrics` measures every joint of
an aligned recorded window. It is task-neutral: it has no simulator, robot,
football, policy update, file access, or execution authority.

Inputs are measured positions and velocities, commanded position targets,
recorded actuator forces at every substep, positive per-joint torque limits,
and an explicit control-frame sample interval. Forces retain their recorded
substep resolution when measuring slew, including frame boundaries. Position
and velocity derivatives remain at control-frame resolution; the function
cannot recover unrecorded motion between those observations.

Outputs retain all joints, in input order, with SI units: measured speed,
acceleration and velocity-derived jerk, target speed, target tracking error,
actuator force and slew, fraction at or above 99% of the supplied limit, and
maximum limit fraction. Exceedances are not clipped or hidden. These are not
energy estimates: actual substep velocities are not supplied.

Finite numeric inputs, shapes, sampling interval, bounds and positive limits
are mandatory. Boolean arrays, incomplete alignment, nonfinite inputs and
derived overflow are rejected. Inputs are not mutated. Scaled RMS avoids
unnecessary overflow from squaring large otherwise finite values.

This does not declare an athlete natural, stable, safe, or successful. Intended
follow-through, running and diving can have large derivatives. It does not
define a reward or promotion threshold; any later training use needs a separate
declared experiment and full task/retention validation.

Ten focused tests cover units, tracking versus measured motion, force-substep
resolution, saturation/exceedance, input preservation and invalid data.
