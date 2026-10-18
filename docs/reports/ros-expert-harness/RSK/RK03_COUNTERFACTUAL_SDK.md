# RK-03 isolated SIM configuration experiment prerequisites

This increment extends only the closed SDK acceptance roles. It does not extend
Agent execution privileges, Runtime, Body, physical controllers or the five-rule
repair compiler. Actual experiment evidence remains separate from implementation.

Three source-only raw capture cases are admitted: `rsk_node_clock_mismatch`,
`rsk_controller_plugin_unavailable`, and `rsk_sensor_frame_unresolvable`.
These are operator-assigned experiment labels, never hypothesis-engine inputs.
Before and rollback captures can precede localization initialization: an invalid
plugin may prevent lifecycle configuration and activation itself. Their metadata
explicitly records `NOT_RUN_PREACTIVATION_RAW_CAPTURE`, L1 NOT_EVALUATED,
L2 NOT_RUN, no navigation workspace, and a never-admitted zero-command sink.
Capturing facts does not establish the new mechanism or physical task acceptance.

The existing Nav2 launch previously forced `use_sim_time=True` after YAML.
An optional `counterfactual_sim` bringup descriptor now carries the explicit
`rosclaw.sim_counterfactual_launch.v1` schema and at most two boolean
`node_clock_overrides` for existing SDK Nav2 server names. It is accepted only
inside the explicitly owned isolated clock-fault before/rollback capture role.
Default launch parameters remain unchanged. Readback, not YAML, must show that
an override actually took effect. Repaired After staging omits this descriptor.

The ROS-host probe additionally reads observed plugin instance lists,
`.sensor_frame`, and `.data_type` parameters through the existing bounded
read-only parameter service. Core receives original values/times; it does not
infer installed plugin classes or valid frames from configuration strings.

Acceptance still requires a pre-registered exact source, immutable physics and
single intended staging difference, genuinely observed failure and negative
hypotheses, independent new World After with fresh full L1 and canonical guarded
L2 receipts, independent stop/collision/source continuity, and rollback in another
new World. A fresh costmap publication alone does not prove obstacle consumption;
a nonexistent sensor frame must be checked against actual TF failure evidence.
No source/main CI or native experiment completion is implied by this document.
