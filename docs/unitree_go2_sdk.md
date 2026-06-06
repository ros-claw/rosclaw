# Unitree Go2 SDK Quick Reference

This document provides a minimal SDK reference for the Unitree Go2 quadruped robot, used by `rosclaw forge sdk-to-mcp` to generate an MCP-compatible capability bundle.

## Capabilities

- `locomotion/v1` — velocity commands (`vx`, `vy`, `yaw_rate`) in body frame
- `pose/v1` — stand height, roll/pitch adjustment
- `gait/v1` — trot, walk, climb, bound
- `perception/v1` — 3D LiDAR point cloud, front-facing depth camera
- `state/v1` — IMU, joint positions, foot force sensors, battery

## Safety Limits

| Parameter | Value |
|-----------|-------|
| Max linear velocity | 3.5 m/s |
| Max angular velocity | 2.0 rad/s |
| Min stand height | 0.25 m |
| Max stand height | 0.45 m |
| Pitch limit | ±20° |
| Roll limit | ±15° |

## Example Velocity Command

```python
{
  "capability": "locomotion/v1",
  "input": {
    "vx": 1.0,
    "vy": 0.0,
    "yaw_rate": 0.3
  }
}
```

## ROS 2 Topics

- `/cmd_vel` — geometry_msgs/Twist
- `/go2_states` — unitree_go2_msgs/Go2State
- `/utlidar/cloud` — sensor_msgs/PointCloud2
- `/front_cam/depth` — sensor_msgs/Image

See the generated bundle under `./generated/unitree_go2_bundle` for the full MCP manifest, skill stubs, and validation tests produced by `rosclaw forge`.
