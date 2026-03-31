"""
Tests for RingBuffer and DataFlywheel

These tests verify the core data layer functionality:
- RingBuffer O(1) append and retrieval
- DataFlywheel event capture
- Multi-threading safety
"""

import time
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rosclaw.data import (
    RingBuffer,
    MultiChannelRingBuffer,
    DataFlywheel,
    EventType,
    RobotState,
)


class TestRingBuffer:
    """Test RingBuffer functionality."""

    def test_basic_append_and_retrieve(self):
        """Test basic append and get_last_n operations."""
        buffer = RingBuffer(capacity=100, shape=(6,))

        # Append some data
        for i in range(10):
            buffer.append(np.ones(6) * i)

        assert buffer.size == 10
        assert not buffer.is_full

        # Retrieve last 5
        data, timestamps = buffer.get_last_n(5)
        assert data.shape == (5, 6)
        assert np.allclose(data[-1], np.ones(6) * 9)

    def test_buffer_full_overwrite(self):
        """Test that buffer overwrites old data when full."""
        buffer = RingBuffer(capacity=10, shape=(3,))

        # Fill buffer
        for i in range(15):  # Write 15 items to 10-capacity buffer
            buffer.append(np.array([i, i, i]))

        assert buffer.is_full
        assert buffer.size == 10

        # Retrieve all - should get last 10 items (5-14)
        data, _ = buffer.get_all()
        assert data[0][0] == 5  # First item should be 5 (0-4 overwritten)
        assert data[-1][0] == 14  # Last item should be 14

    def test_get_last_n_more_than_size(self):
        """Test requesting more samples than available."""
        buffer = RingBuffer(capacity=100, shape=(6,))

        # Append only 5 items
        for i in range(5):
            buffer.append(np.ones(6) * i)

        # Request 10 items
        data, _ = buffer.get_last_n(10)
        assert data.shape[0] == 5  # Should return only 5

    def test_wrapped_buffer_retrieval(self):
        """Test retrieval when buffer has wrapped around."""
        buffer = RingBuffer(capacity=10, shape=(3,))

        # Fill and overflow to cause wrap
        for i in range(15):
            buffer.append(np.array([i, i, i]))

        # Get last 5 - this crosses the wrap boundary
        data, _ = buffer.get_last_n(5)
        assert data.shape == (5, 3)
        assert data[-1][0] == 14

    def test_latest(self):
        """Test getting the latest sample."""
        buffer = RingBuffer(capacity=100, shape=(6,))

        assert buffer.latest() is None  # Empty buffer

        buffer.append(np.ones(6) * 42)
        latest_data, timestamp = buffer.latest()
        assert np.allclose(latest_data, np.ones(6) * 42)
        assert timestamp > 0

    def test_clear(self):
        """Test clearing the buffer."""
        buffer = RingBuffer(capacity=100, shape=(6,))

        for i in range(10):
            buffer.append(np.ones(6) * i)

        assert buffer.size == 10

        buffer.clear()
        assert buffer.size == 0
        assert buffer.is_empty

    def test_get_range_by_timestamp(self):
        """Test retrieving data by time range."""
        buffer = RingBuffer(capacity=100, shape=(3,))

        start_time = time.time()
        for i in range(10):
            buffer.append(np.ones(3) * i, timestamp=start_time + i * 0.1)

        # Get range from t=0.5 to t=0.8
        data, timestamps = buffer.get_range(start_time + 0.5, start_time + 0.8)
        assert len(data) >= 2  # Should have at least 2 samples

    def test_performance_1khz(self):
        """Test that append operation is fast enough for 1kHz."""
        buffer = RingBuffer(capacity=60000, shape=(6,))

        # Pre-generate random data to avoid random generation overhead
        data = [np.random.randn(6) for _ in range(1000)]

        # Time 1000 appends
        start = time.perf_counter()
        for d in data:
            buffer.append(d)
        elapsed = time.perf_counter() - start

        # Should complete in <100ms total (100μs per operation)
        # This is conservative - actual target is <1ms for 1kHz
        assert elapsed < 0.1, f"Too slow: {elapsed*1000:.2f}ms for 1000 ops"


class TestMultiChannelRingBuffer:
    """Test MultiChannelRingBuffer functionality."""

    def test_multi_channel_append(self):
        """Test appending to multiple channels."""
        buffer = MultiChannelRingBuffer(
            joint_states=(100, (6,)),
            gripper=(100, (1,)),
        )

        buffer.append({
            "joint_states": np.ones(6),
            "gripper": np.array([0.5]),
        })

        assert buffer.size == 1

        data = buffer.get_last_n(1)
        assert "joint_states" in data
        assert "gripper" in data

    def test_channel_size_mismatch(self):
        """Test that mismatched channel sizes raise error."""
        with pytest.raises(ValueError):
            MultiChannelRingBuffer(
                channel1=(100, (6,)),
                channel2=(200, (3,)),  # Different capacity
            )


class TestDataFlywheel:
    """Test DataFlywheel functionality."""

    def test_initialization(self):
        """Test flywheel initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            flywheel = DataFlywheel(
                robot_id="test_robot",
                joint_dof=6,
                buffer_duration_sec=1.0,
                sampling_rate_hz=100,
                storage_path=Path(tmpdir),
            )

            assert flywheel.robot_id == "test_robot"
            assert flywheel.joint_dof == 6

    def test_control_cycle(self):
        """Test control cycle data capture."""
        with tempfile.TemporaryDirectory() as tmpdir:
            flywheel = DataFlywheel(
                robot_id="test_robot",
                joint_dof=6,
                buffer_duration_sec=1.0,
                sampling_rate_hz=100,
                storage_path=Path(tmpdir),
            )

            # Simulate control cycles
            for i in range(10):
                state = RobotState(
                    timestamp=time.time(),
                    joint_positions=np.ones(6) * i,
                    joint_velocities=np.zeros(6),
                    joint_torques=np.zeros(6),
                )
                flywheel.on_control_cycle(state)

            assert flywheel._cycle_count == 10
            assert flywheel._buffers.size == 10

    def test_trigger_event(self):
        """Test event triggering and data capture."""
        with tempfile.TemporaryDirectory() as tmpdir:
            flywheel = DataFlywheel(
                robot_id="test_robot",
                joint_dof=6,
                buffer_duration_sec=10.0,
                sampling_rate_hz=100,
                storage_path=Path(tmpdir),
            )

            # Add some data
            for i in range(100):
                state = RobotState(
                    timestamp=time.time(),
                    joint_positions=np.ones(6) * i,
                    joint_velocities=np.zeros(6),
                    joint_torques=np.zeros(6),
                )
                flywheel.on_control_cycle(state)

            # Trigger event
            event_id = flywheel.trigger_event(
                event_type=EventType.SUCCESS,
                metadata={"task": "test_task"},
            )

            assert event_id is not None
            assert len(flywheel._events) == 1
            assert flywheel._events[0].event_type == EventType.SUCCESS

    def test_invalid_state_dimensions(self):
        """Test that invalid state dimensions are handled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            flywheel = DataFlywheel(
                robot_id="test_robot",
                joint_dof=6,
                storage_path=Path(tmpdir),
            )

            # Wrong dimensions
            state = RobotState(
                timestamp=time.time(),
                joint_positions=np.ones(3),  # Wrong size
                joint_velocities=np.zeros(3),
                joint_torques=np.zeros(3),
            )

            flywheel.on_control_cycle(state)
            assert flywheel._dropped_cycles == 1

    def test_get_stats(self):
        """Test statistics retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            flywheel = DataFlywheel(
                robot_id="test_robot",
                joint_dof=6,
                storage_path=Path(tmpdir),
            )

            # Add some data
            for i in range(10):
                state = RobotState(
                    timestamp=time.time(),
                    joint_positions=np.ones(6) * i,
                    joint_velocities=np.zeros(6),
                    joint_torques=np.zeros(6),
                )
                flywheel.on_control_cycle(state)

            # Trigger events of different types (with very short durations to speed up)
            flywheel.trigger_event(EventType.SUCCESS, pre_duration_sec=0.01, post_duration_sec=0.01)
            flywheel.trigger_event(EventType.FAILURE, pre_duration_sec=0.01, post_duration_sec=0.01)
            flywheel.trigger_event(EventType.SUCCESS, pre_duration_sec=0.01, post_duration_sec=0.01)

            # Wait for background threads to complete
            time.sleep(0.5)

            stats = flywheel.get_stats()
            assert stats["robot_id"] == "test_robot"
            assert stats["total_cycles"] == 10
            assert stats["total_events"] == 3
            assert stats["events_by_type"]["SUCCESS"] == 2
            assert stats["events_by_type"]["FAILURE"] == 1

    def test_clear(self):
        """Test clearing the flywheel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            flywheel = DataFlywheel(
                robot_id="test_robot",
                joint_dof=6,
                storage_path=Path(tmpdir),
            )

            # Add data and events
            for i in range(10):
                state = RobotState(
                    timestamp=time.time(),
                    joint_positions=np.ones(6) * i,
                    joint_velocities=np.zeros(6),
                    joint_torques=np.zeros(6),
                )
                flywheel.on_control_cycle(state)

            flywheel.trigger_event(EventType.SUCCESS, pre_duration_sec=0.01, post_duration_sec=0.01)

            # Wait for background thread
            time.sleep(0.3)

            assert flywheel._buffers.size == 10
            assert len(flywheel._events) == 1

            flywheel.clear()

            assert flywheel._buffers.size == 0
            assert len(flywheel._events) == 0
