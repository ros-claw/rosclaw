# Legacy rclpy-based ROS2 tests

This directory holds the legacy ROS2 integration tests that require a real
rclpy / ROS2 Humble environment. They test the old `rosclaw.mcp_drivers.ros2_driver`
and `rosclaw.mcp.ur5_server` rclpy-based layer.

The new canonical ROS connector is rosbridge-based and lives in:

- `src/rosclaw/connectors/ros/`
- `tests/connectors/ros/`

It intentionally does **not** import `rclpy` or `rospy`.

## Running legacy tests

Because `pyproject.toml` sets `norecursedirs = ["tests/legacy"]`, these tests
are excluded from the default `pytest` run. To execute them explicitly:

```bash
source /opt/ros/humble/setup.bash
export LD_LIBRARY_PATH="/tmp/ros2-local/opt/ros/humble/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/tmp/ros2-local/opt/ros/humble/local/lib/python$(python3 --version | cut -d' ' -f2 | cut -d. -f1-2)/dist-packages:$PYTHONPATH"
pytest tests/legacy/ros2_rclpy/ -v
```

## Contents

| File | Purpose |
|---|---|
| `test_ros2_driver_ros2.py` | Direct rclpy tests for `ROS2Driver` |
| `test_ur5_server_ros2.py` | Direct rclpy tests for `UR5MCPServer` / `UR5ROSNode` |
| `test_mcp_drivers_init_ros2.py` | Import smoke tests for `rosclaw.mcp_drivers` |
| `test_ros2_*_wrapper.py` | Subprocess wrappers that run scripts in `scripts/legacy/ros2_rclpy/` |

When the legacy rclpy layer is removed, this directory and
`scripts/legacy/ros2_rclpy/` can be deleted together.
