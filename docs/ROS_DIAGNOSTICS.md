# ROS Expert Diagnostics

`diagnose(model, profile=..., now=...)` produces `rosclaw.ros_diagnosis.v1`. Profiles: all, navigation, tf, qos, sensors, time. Results include issue codes, severity, evidence references/observations/timestamps, deterministic confidence, read-only next checks and repair suggestions. Suggestions do not execute configuration changes. Confidence 1.0 refers to an observed deterministic predicate, not a probability of an inferred root cause.

| Family | Checks |
|---|---|
| ROS_ENV_001 / ROS_GRAPH_001 | Collection errors / stale or future snapshot |
| ROS_TF_001–007 | Missing map→odom, odom→base, sensor link; stale TF; multiple parents; cycle; multiple roots |
| ROS_TOPIC_001–004 | Required topic missing; receive freshness; configured minimum rate; no publisher |
| ROS_QOS_001 | Offered/requested reliability or durability mismatch; native compatibility errors |
| ROS_TIME_001–003 | Mixed sim time; future transform; nonadvancing observed clock |
| ROS_LIFECYCLE_001 | Observed inactive/unconfigured/finalized lifecycle node |
| NAV2_LOCALIZATION_001 | Explicit localization-not-ready evidence |
| NAV2_COSTMAP_001–002 | Explicit obstacle-source/config or costmap freshness failure |
| NAV2_CONTROLLER_001–002 | Supplied oscillation/progress failure evidence |
| NAV2_COLLISION_001 | Supplied collision-monitor failure evidence |
| COVERAGE_PLAN_001 / COVERAGE_VERIFY_001 | Explicit invalid plan / incomplete coverage |

Navigation diagnostics include unknown TF/QoS/signal/lifecycle/time checks, rather than reporting health from lifecycle alone. Some Nav2 checks currently consume explicitly supplied observations; the native collector does not yet measure all controller/localization/costmap predicates. There are 26 fault-code fixture cases and six replay scenarios. Native lifecycle/QoS/TF/sensor collection is separately tested against synthetic DDS nodes.

HOW integration offers `diagnostic_intervention(issue)` with evidence and suggested read-only checks. It does not auto-apply repairs. Live fault injection into a Nav2 robot is NOT_RUN.
