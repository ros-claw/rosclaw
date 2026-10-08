# Offline action-update transport diagnostics

`rosclaw.growth.action_update_transport.bounded_residual_update_transport`
diagnoses a policy update behind a bounded residual envelope. It has no robot,
football, simulator, Torch, driver, runtime, executor or actuator dependency.

Supply all observed rows, both latent policies evaluated on the **same observed
states**, the actual previous residual, nominal target, measured behavior
residual, absolute action limits, and the original residual/slew caps. Inputs
must be authenticated by the caller. The helper checks:

1. Both latent outputs pass through `cap * tanh`.
2. Both local projections use the **same measured previous residual** and
   original slew cap.
3. Both use the same zero-inclusive nominal residual envelope and absolute
   action limits.
4. Behavior projection must match the measured behavior residual exactly.

The result reports the intended and transported update RMS/max, their L2
ratio, and changed coordinates completely masked by projection. Constraint
activity uses the actual bounds, not floating cancellation differences. The
complete numeric input hash includes all six arrays, shapes and dtypes; caps
and source hash are explicit result fields. The caller seals the surrounding
receipt and binds its policy lineage, observations, artifact pins and limits.

All rows must be retained, including failures. Allocation is bounded to 65,536
rows, 64 dimensions and 1,048,576 coordinates. No optimizer is run and no
runtime targets are returned. An identical update has no L2 ratio (`None`), not
an invented perfect score.

This is a **local, same-observation diagnostic**, not a counterfactual rollout.
It does not propagate changed actions into later physical observations or
prove task improvement. A low ratio suggests investigating projection-aware
learning and action timing; it never authorizes relaxing a safety envelope.
Real task gain still requires independent simulation/hardware evidence through
the appropriate execution and promotion boundaries. Hardware authorization,
runtime execution, policy gain and promotion remain false.
