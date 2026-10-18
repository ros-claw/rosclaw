# Optional native TF message attribution observer

This ROS-host C++ sidecar uses actual `rclcpp::MessageInfo` publisher GIDs.
It has no ROSClaw Runtime, Body, driver, action server or actuator publisher.
It discovers TFMessage topics by type, including renamed topics, or accepts an
explicit bounded `topics` parameter. It publishes only read-only JSON diagnostics
on `/rosclaw_probe/tf_attribution`. The existing Python probe reads that reserved
String topic; arbitrary String subscriptions are not enabled.

Build with existing ROS dependencies, without installing anything on a robot:

```bash
cmake -S integrations/ros_probe/ros2/tf_attribution -B /tmp/tf-observer-build
cmake --build /tmp/tf-observer-build
```

Run it only as an operator-owned isolated ROS-host observation role. Match its
`use_sim_time` setting to the observed stack so its own diagnostic node does not
introduce an unrelated clock mismatch. Its timer and retention use steady time.

The default receive window is 500 ms; `window_ms` accepts 100–5000. At most 64
TF topics and 1024 distinct topic/parent/child/GID records are retained, with
explicit invalid/drop/discovery counters. Both volatile sensor and transient-local
subscriptions are used, so callback counts can include duplicate delivery through
the two readers. A latest record is stored per key. Old records expire; new JSON
envelopes never refresh their receive times. Unknown/all-zero message GIDs are
null. Unsupported non-ASCII/control frame identities are rejected and counted.

`completeness=NOT_ESTABLISHED`, `execution_authority=NONE`, and payload source
authentication is explicitly not established by the payload itself. SDK ownership,
image/code/binary provenance and source continuity need independent review. The
observer does not assert unique or future TF authority.

The Core reads current original receive records, preserving individual JSON
pointer/time/hash evidence. It can witness a same-topic child with competing
parents or attributed publishers. Separate TF topics are not assumed to share a
consumer buffer. One observed publisher remains UNKNOWN: neither a bounded window
nor missing observations establishes uniqueness. These new facts do not modify
the original Case compiler or grant L1/Body/motion authority.

This is a prerequisite measurement provider for subsequent RK-03 work. Compilation
and synthetic tests are component evidence; actual DDS provider acceptance and
physical counterfactual before/after/rollback are separate gates.
