# ADR-0016: ROS Expert Harness boundaries

Status: Proposed (2026-10-03; offline and native-probe evidence only)

The existing ROS Connector already owns discovery, manifests, safety contracts, providers and Practice/KNOW/HOW integration. Creating another ROS runtime would duplicate those boundaries.

Decisions:

1. Extend `connectors/ros`; keep the current rosbridge transport and discovery compatible.
2. Keep `rclpy`/`rospy` out of connector core. Optional ROS-host sidecars live under `integrations/ros_probe`.
3. Native probes subscribe, inspect graph and call allowlisted read RPCs. Their only publishers are diagnostic topics. They cannot publish commands, mutate parameters or send motion goals.
4. Physical ROS effects continue through Capability → ActionEnvelope → rosclawd → policy/session/lease/permit → executor → observation → ExecutionReceipt. Readiness never grants permission. No additional executor authority is added in this change.
5. Semantic capability IDs abstract implementation differences while preserving legacy IDs.
6. Keep the frozen context and receipt schemas. ROS source adapters augment existing L2/L3; mission verification is a separate artifact with canonical receipt/evidence bindings.
7. Fixture/native graph results do not promote product support or substitute for Nav2/Gazebo physical-task acceptance.

Consequences: offline intelligence can be used without ROS installed; native deployment needs ROS-compatible Python. Live coverage and Native Chat integration remain gated work. This ADR does not approve a second runtime, safety system, memory or agent framework.
