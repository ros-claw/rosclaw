# Optional native TF message attribution observer

This ROS-host C++ sidecar uses actual `rclcpp::MessageInfo` publication identity bytes.
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
TF topics and 1024 distinct topic/parent/child/token/reader records are retained, with
explicit invalid/drop/discovery counters. Both volatile sensor and transient-local
subscriptions are used, so callback counts can include duplicate delivery through
the two readers. A latest record is stored per key. Old records expire; new JSON
envelopes never refresh their receive times. Unknown/all-zero callback identity bytes are
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

## Identity domains (v2)

Callback tokens are opaque and scoped by `observer_instance`, `reader_scope` and
topic. They are never joined to endpoint discovery GIDs or reused across readers
or observer lifetimes. In the accepted SDK CycloneDDS 2.2.4 implementation,
`MessageInfo.publisher_gid.data` contains `publication_handle`, while discovery
reports a different GUID. The first direct-equality acceptance attempt is retained
as a failure. See the [exact primary RMW implementation](https://github.com/ros2/rmw_cyclonedds/blob/2.2.4/rmw_cyclonedds_cpp/src/rmw_node.cpp).

Version 2 therefore exposes `publisher_token`, explicitly declares
`RMW_MESSAGEINFO_OPAQUE_READER_LOCAL`, and keeps discovery mapping
`NOT_ESTABLISHED`. Two different local tokens in one current reader/topic/child
can witness competing observed writers. Two readers are not combined into a
single identity namespace. Independent alternating-source tests are needed to
validate the mapping between these local observations and the controlled fixture
writers; no global node or Body identity is claimed.
