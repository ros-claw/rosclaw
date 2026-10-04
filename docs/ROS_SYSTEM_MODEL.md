# ROS System Model v1

`RosSystemModel` uses `rosclaw.ros_system_model.v1`, the existing ContractModel/content_hash conventions and a sealed snapshot hash. It combines RosGraphSnapshot with optional native observations and an independently supplied Body binding. ROS interfaces never define Body truth.

The model contains identity (robot/snapshot/time/hash), environment, body, graph, signals, transforms, lifecycle, QoS, navigation evidence, observations, inference, completeness and collection errors. Signals include rate/jitter/receive age and endpoint counts. Transform observations include static/dynamic status and ROS-clock age. Every typed measurement retains source and timezone-aware capture time. Missing rate, freshness or state remains null/UNKNOWN. Raw graphs and native captures remain inspectable artifacts.

`from_dict` rejects unknown major versions, naive observation timestamps and hash mismatches. Replaying a model preserves its original time. `seal()` is a content-integrity operation, **not** a trust signature or authorization. MCP-supplied snapshots are explicitly supplied data; physical mission success still requires independently bound daemon evidence.

Body frame names and rate thresholds can be supplied without editing harness core. Action namespaces are retained when resolving lifecycle dependencies. Native graph and measurements are from the same capture; a fresh rosapi capture is not merged into stale native facts.

The native probe can report ROS distro, actual RMW, Domain ID, overlays, installed package inventory, endpoint QoS, lifecycle state, TF observations and topic receive timing. GPU transport, node composition, full software manifests, covariance-based localization quality and automatic e-URDF resolution are not yet implemented.
