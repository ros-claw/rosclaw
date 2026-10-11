# Navigation observation review

This component prepares evidence for SIM Body binding review without requiring
a cleaning attachment. It does not bind a Body, configure Nav2, approve a
permit, dispatch a goal or establish physical L1/L2 acceptance.

```bash
.venv/bin/python -m rosclaw.entrypoint ros review-navigation \
  --snapshot sealed-system.json --urdf expanded.urdf \
  --requirements navigation-requirements.json --json
```

The requirements file contains exactly these fields. Names identify intended
roles; every role is checked against original observations. These are synthetic
examples, not a robot profile:

```json
{
  "interface_names": {"mapping.occupancy_map": "/robot/floor"},
  "lifecycle_nodes": {
    "localization": "/robot/localizer",
    "planner": "/robot/path_planner",
    "controller": "/robot/path_controller",
    "collision_monitor": "/robot/guard",
    "navigator": "/robot/navigator"
  },
  "costmap_topics": ["/robot/near_grid", "/robot/far_grid"],
  "localization_topic": "/robot/estimated_pose",
  "velocity_input_topic": "/robot/unfiltered",
  "velocity_output_topic": "/robot/filtered",
  "velocity_consumer_node": "/robot/drive_adapter"
}
```

`interface_names` may select odometry, occupancy map, lidar or NavigateToPose
by semantic role. `ros discover-body --interfaces interfaces.json` provides the
same optional disambiguation. With no selection, multiple typed streams remain
UNKNOWN. Selection narrows observations; it cannot supply missing types,
publishers, frames, URDF provenance, fresh samples or transforms. This matters
when a map and two costmaps all use OccupancyGrid. Selection is operator input
recorded in the result, not verified Body truth or execution authority.

The navigation review checks the fresh complete discovery candidate, advancing
ROS2 SIM clock, five distinct ACTIVE lifecycle services, selected localization
publisher, map frame and finite nonnegative x/y/yaw variances at most 0.25,
two fresh typed costmap streams, captured collision monitor topic configuration,
and the exclusive observed controller → monitor → consumer velocity chain.
Velocity publishers may be idle; graph presence does not prove actuation.
Mixed Twist/TwistStamped chains, observed QoS incompatibility and incomplete
QoS capture remain UNKNOWN. No conventional topic, frame or node suffix is
used to fill a missing role. Relative parameter topics remain UNKNOWN pending
actual ROS-host expansion and remapping evidence.

The probe preserves both position covariance axes, so a negative axis cannot
hide behind the previous maximum-of-two summary. It reads arbitrary `.topic`
parameters and passively observes Twist/TwistStamped graph streams.

`PROPOSED` means the listed observations support submission for binding review.
It does not establish collision polygon geometry, obstacle response, speed or
acceleration limits, immutable binding, daemon admission or actual navigation.
Those gates are explicitly listed in `unverified_checks`; physical acceptance
remains NOT_RUN. `ok: true` only means the CLI completed its read-only review.
Old snapshots remain historical and become UNKNOWN without restamping.

The SDK fixture is a network-isolated synthetic contract: typed fake lifecycle
services, covariance, clock and costmap sources. It sends no goal or velocity;
its action server rejects goals and its command subscribers fail on any
velocity. It is not a Nav2 server, World, unseen robot or L2 receipt. Freeze
generic functionality before selecting the later held-out Body asset.
