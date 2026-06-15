"""Tests for Data Layer (RingBuffer, DataFlywheel)."""

from pathlib import Path

import numpy as np
import pytest

from rosclaw.data.flywheel import DataFlywheel, EventType, RobotState
from rosclaw.data.ring_buffer import MultiChannelRingBuffer, RingBuffer


class TestRingBuffer:
    def test_basic_append_and_retrieve(self):
        buf = RingBuffer(capacity=10, shape=(3,))
        buf.append(np.array([1.0, 2.0, 3.0]), timestamp=0.0)
        buf.append(np.array([4.0, 5.0, 6.0]), timestamp=1.0)
        values, timestamps = buf.get_last_n(2)
        assert len(values) == 2
        assert np.allclose(values[0], [1.0, 2.0, 3.0])
        assert np.allclose(values[1], [4.0, 5.0, 6.0])

    def test_buffer_full_overwrite(self):
        buf = RingBuffer(capacity=3, shape=(2,))
        buf.append(np.array([1.0, 1.0]), timestamp=0.0)
        buf.append(np.array([2.0, 2.0]), timestamp=1.0)
        buf.append(np.array([3.0, 3.0]), timestamp=2.0)
        buf.append(np.array([4.0, 4.0]), timestamp=3.0)
        values, timestamps = buf.get_last_n(3)
        assert len(values) == 3
        assert np.allclose(values[0], [2.0, 2.0])

    def test_get_last_n_more_than_size(self):
        buf = RingBuffer(capacity=3, shape=(2,))
        buf.append(np.array([1.0, 1.0]), timestamp=0.0)
        values, timestamps = buf.get_last_n(10)
        assert len(values) == 1

    def test_wrapped_buffer_retrieval(self):
        buf = RingBuffer(capacity=4, shape=(2,))
        for i in range(6):
            buf.append(np.array([float(i), float(i)]), timestamp=float(i))
        values, timestamps = buf.get_last_n(4)
        assert len(values) == 4
        assert np.allclose(values[-1], [5.0, 5.0])

    def test_latest(self):
        buf = RingBuffer(capacity=5, shape=(3,))
        buf.append(np.array([1.0, 2.0, 3.0]), timestamp=1.0)
        val, ts = buf.latest()
        assert np.allclose(val, [1.0, 2.0, 3.0])
        assert ts == 1.0

    def test_clear(self):
        buf = RingBuffer(capacity=5, shape=(3,))
        buf.append(np.array([1.0, 2.0, 3.0]), timestamp=1.0)
        buf.clear()
        assert buf.size == 0

    def test_get_range_by_timestamp(self):
        buf = RingBuffer(capacity=10, shape=(2,))
        for i in range(5):
            buf.append(np.array([float(i), float(i)]), timestamp=float(i))
        values, timestamps = buf.get_range(1.0, 3.0)
        assert len(values) == 3

    def test_performance_1khz(self):
        buf = RingBuffer(capacity=10000, shape=(6,))
        for i in range(1000):
            buf.append(np.random.randn(6), timestamp=i * 0.001)
        assert buf.size == 1000


class TestMultiChannelRingBuffer:
    def test_multi_channel_append(self):
        buf = MultiChannelRingBuffer(
            positions=(100, (6,)),
            velocities=(100, (6,)),
        )
        buf.append({
            "positions": np.ones(6),
            "velocities": np.zeros(6),
        }, timestamp=0.0)
        data = buf.get_last_n(1)
        assert "positions" in data
        assert np.allclose(data["positions"][0][0], np.ones(6))

    def test_channel_size_mismatch(self):
        buf = MultiChannelRingBuffer(positions=(10, (6,)))
        with pytest.raises(ValueError):
            buf.append({"positions": np.ones(3)}, timestamp=0.0)


class TestDataFlywheel:
    def test_initialization(self):
        fw = DataFlywheel(robot_id="test", joint_dof=6, storage_path=Path("/tmp/rosclaw_test"))
        assert fw.robot_id == "test"
        assert fw.joint_dof == 6

    def test_control_cycle(self):
        fw = DataFlywheel(robot_id="test", joint_dof=6, storage_path=Path("/tmp/rosclaw_test"))
        state = RobotState(
            timestamp=0.0,
            joint_positions=np.zeros(6),
            joint_velocities=np.zeros(6),
            joint_torques=np.zeros(6),
        )
        fw.on_control_cycle(state)
        assert fw._cycle_count == 1

    def test_trigger_event(self):
        fw = DataFlywheel(robot_id="test", joint_dof=6, storage_path=Path("/tmp/rosclaw_test"))
        state = RobotState(
            timestamp=0.0,
            joint_positions=np.zeros(6),
            joint_velocities=np.zeros(6),
            joint_torques=np.zeros(6),
        )
        fw.on_control_cycle(state)
        event_id = fw.trigger_event(EventType.SUCCESS, {"task": "test"})
        assert event_id != ""
        assert len(fw._events) == 1

    def test_invalid_state_dimensions(self):
        fw = DataFlywheel(robot_id="test", joint_dof=6, storage_path=Path("/tmp/rosclaw_test"))
        state = RobotState(
            timestamp=0.0,
            joint_positions=np.zeros(3),
            joint_velocities=np.zeros(3),
            joint_torques=np.zeros(3),
        )
        fw.on_control_cycle(state)
        assert fw._dropped_cycles == 1

    def test_get_stats(self):
        fw = DataFlywheel(robot_id="test", joint_dof=6, storage_path=Path("/tmp/rosclaw_test"))
        stats = fw.get_stats()
        assert stats["robot_id"] == "test"
        assert stats["total_cycles"] == 0

    def test_clear(self):
        fw = DataFlywheel(robot_id="test", joint_dof=6, storage_path=Path("/tmp/rosclaw_test"))
        state = RobotState(
            timestamp=0.0,
            joint_positions=np.zeros(6),
            joint_velocities=np.zeros(6),
            joint_torques=np.zeros(6),
        )
        fw.on_control_cycle(state)
        fw.clear()
        assert fw._buffers.size == 0
        assert len(fw._events) == 0
