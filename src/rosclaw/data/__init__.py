"""
ROSClaw Data Layer (Layer 2)

The data layer provides high-performance data capture and export capabilities
for robot learning and skill transfer.

Key Components:
    RingBuffer: High-performance circular buffer for time-series data
    DataFlywheel: Event-driven data capture with automatic export
    LeRobotExporter: Export to LeRobot dataset format

Usage:
    from rosclaw.data import DataFlywheel, EventType

    flywheel = DataFlywheel(robot_id="ur5e_001", joint_dof=6)

    # In control loop
    flywheel.on_control_cycle(robot_state)

    # When event occurs
    flywheel.trigger_event(EventType.SUCCESS, metadata={"task": "pick"})
"""

from .ring_buffer import RingBuffer, MultiChannelRingBuffer, RingBufferConfig
from .flywheel import DataFlywheel, DataEvent, EventType, RobotState

__all__ = [
    "RingBuffer",
    "MultiChannelRingBuffer",
    "RingBufferConfig",
    "DataFlywheel",
    "DataEvent",
    "EventType",
    "RobotState",
]