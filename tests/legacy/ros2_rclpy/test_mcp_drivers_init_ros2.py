"""
This is part of the legacy rclpy-based ROS2 layer. The new rosbridge-based
ROS connector lives in tests/connectors/ros/ and does not require rclpy.
ROS2-native tests for mcp_drivers/__init__.py import behavior.

Verifies that all drivers are available in a real ROS2 environment.

Run with:
    source /opt/ros/humble/setup.bash
    export LD_LIBRARY_PATH="/tmp/ros2-local/opt/ros/humble/lib:$LD_LIBRARY_PATH"
    export PYTHONPATH="/tmp/ros2-local/opt/ros/humble/local/lib/python$(python3 --version | cut -d' ' -f2 | cut -d. -f1-2)/dist-packages:$PYTHONPATH"
    pytest tests/test_mcp_drivers_init_ros2.py -v -p no:xdist
"""

import sys

import pytest

from tests._ros2_env import ros2_available

# Skip entire module if ROS2 is not available
if not ros2_available():
    pytest.skip(
        "ROS2 environment not available",
        allow_module_level=True,
    )

pytestmark = pytest.mark.legacy_rclpy

# Clean up sys.modules mocks from test_mcp_server.py so real ROS2 imports work.
for _mod in list(sys.modules.keys()):
    if _mod.startswith(("rclpy.", "rosclaw.", "geometry_msgs", "sensor_msgs",
                        "std_msgs", "trajectory_msgs", "control_msgs",
                        "builtin_interfaces", "unique_identifier_msgs",
                        "action_msgs", "rcl_interfaces")):
        sys.modules.pop(_mod, None)
sys.modules.pop("rclpy", None)


class TestMCPDriversInitInROS2Env:
    def test_ros2_driver_importable(self):
        from rosclaw.mcp_drivers import ROS2Driver
        assert ROS2Driver is not None

    def test_mujoco_driver_importable(self):
        from rosclaw.mcp_drivers import MuJoCoSimDriver
        assert MuJoCoSimDriver is not None

    def test_serial_driver_importable(self):
        from rosclaw.mcp_drivers import SerialDriver
        assert SerialDriver is not None

    def test_base_classes_importable(self):
        from rosclaw.mcp_drivers import BaseDriver, DriverState, TrajectoryCommand
        assert BaseDriver is not None
        assert DriverState is not None
        assert TrajectoryCommand is not None

    def test_all_exports_present(self):
        import rosclaw.mcp_drivers as m
        assert hasattr(m, "BaseDriver")
        assert hasattr(m, "DriverState")
        assert hasattr(m, "TrajectoryCommand")
        assert hasattr(m, "ROS2Driver")
        assert hasattr(m, "MuJoCoSimDriver")
        assert hasattr(m, "SerialDriver")

    def test_ros2_driver_not_none(self):
        import rosclaw.mcp_drivers as m
        assert m.ROS2Driver is not None

    def test_mujoco_driver_not_none(self):
        import rosclaw.mcp_drivers as m
        assert m.MuJoCoSimDriver is not None

    def test_serial_driver_not_none(self):
        import rosclaw.mcp_drivers as m
        assert m.SerialDriver is not None
