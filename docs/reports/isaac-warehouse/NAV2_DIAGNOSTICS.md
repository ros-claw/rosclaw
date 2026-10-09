# Warehouse-derived read-only diagnosis

The Isaac Sim 6.1 / ROS 2 Jazzy warehouse case exposed an actual obstacle-cost loss when plugin order was `voxel → inflation → static`. The local adapter was repaired to `static → voxel → inflation`; actual LiDAR, costmap and independent PhysX evidence were used to validate that application configuration. This upstream change diagnoses a configuration risk, not a collision or a proven missing obstacle.

`ros diagnose` now checks fresh `observations.node_parameters` plus per-node `parameter_captured_at` already produced by the canonical read-only ROS probe. Plugin **classes**, enabled state and explicitly observed StaticLayer `use_maximum=false` determine `NAV2_COSTMAP_003`. An inflation layer before a contributing obstacle layer produces `NAV2_COSTMAP_004`. Missing, stale or custom-layer evidence produces no speculative overwrite finding. Warnings include official sources and an isolated-simulation validation recommendation. No parameters are written and no action is authorized.

The probe reads plugin class/enable/combination fields and decodes integer ROS parameters. Body `required_topic_types` may additionally bind a topic to an exact message type (`ROS_TOPIC_005`); a `/scan` name alone never implies LaserScan.

For review/reproduction:

```bash
PYTHONPATH=src python -m pytest tests/connectors/ros/test_warehouse_diagnostics.py tests/connectors/ros/test_expert_harness.py -q
# In a ROS Jazzy environment, without a live robot or DDS connection:
python integrations/ros_probe/acceptance/probe_cache.py
```

Configuration findings are not automatic repair, physical-stop proof, or REAL readiness. Compare timestamp-aligned obstacle observations and master-costmap cells, then validate a proposed ordering against measured footprint clearance and independent collisions in an isolated simulation. Only reviewed configuration changes may enter a robot deployment.

Sources: [Nav2 Jazzy Costmap 2D](https://docs.nav2.org/jazzy/configuration_and_development/configuration_guide/core_servers/costmap_2d/), [Static Layer](https://docs.nav2.org/jazzy/configuration_and_development/configuration_guide/core_servers/costmap_2d/costmap_plugins/static/). Lab evidence: [warehouse failures](https://github.com/ros-claw/rosclaw-robotics-lab/blob/main/challenges/01-isaac-warehouse-patrol/docs/failures.md).
