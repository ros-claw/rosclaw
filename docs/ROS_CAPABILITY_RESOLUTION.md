# Semantic Capability Resolution

Existing capability IDs remain unchanged. RosCapability adds `semantic_id`, `implementation_id`, and aliases; legacy manifests round-trip and namespace deduplication keeps implementation IDs aligned.

Resolution uses message/action **types**, preserving concrete endpoints. ROS1 move_base and ROS2 NavigateToPose both map to `navigation.navigate_to_pose`. Unknown action types are not guessed from names. Sensing, odometry, pose/map, navigation and coverage types are normalized into semantic candidates.

Readiness is separate from execution authority. AVAILABLE means the implemented checks have evidence; MISSING means no recognized implementation; UNKNOWN means required observations are absent; BLOCKED means a required measured predicate failed or the snapshot is stale. Multiple implementations are ranked so a broken namespace does not hide a ready implementation. Body thresholds, graph presence, signal age/rate, TF connectivity, lifecycle state and deterministic runtime faults participate. No readiness result is usable as a REAL permit.

Coverage cleaning intent is recognized in Chinese or English and resolved against versioned [technology cards](../src/rosclaw/connectors/ros/knowledge/catalog.yaml). Recommendations include mature Nav2, opennav_coverage and Collision Monitor components with official sources and compatibility constraints. Unknown environments produce alternatives with compatibility unestablished, not a forced selection.

The resolver does not install, configure, build packages or execute actions. It keeps cleaning enable/disable and a cleaning footprint as explicit requirements; driving a brushless mobile base is not claimed as cleaning. The local verifier is suggested separately from ROS interface discovery.

Official coverage interfaces and distribution branches were checked against [Open Navigation's repository](https://github.com/open-navigation/opennav_coverage). Jazzy integration still requires a live build/acceptance gate; cards are not proof of installation or working compatibility.
