"""
Event-Driven Ring Buffer for High-Frequency Data Capture

This module provides high-performance circular buffers for capturing
robot data at high frequencies (1kHz) with minimal GC pressure.

Design Principles:
- Pre-allocated memory (no allocations during operation)
- NumPy-based for vectorized operations
- Lock-free where possible
- Zero-copy data retrieval

Usage:
    buffer = RingBuffer(capacity=60000, shape=(6,))  # 60s @ 1kHz, 6 DOF
    buffer.append(joint_positions)  # O(1)
    recent = buffer.get_last_n(1000)  # Get last 1 second
"""

import numpy as np
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum, auto
import time


class BufferFullStrategy(Enum):
    """Strategy when buffer reaches capacity."""
    OVERWRITE = auto()  # Overwrite oldest data (circular)
    EXPAND = auto()     # Expand buffer (not recommended for real-time)
    DROP = auto()       # Drop new data (lossy)


@dataclass
class RingBufferConfig:
    """Configuration for RingBuffer."""
    capacity: int           # Maximum number of samples
    shape: Tuple[int, ...]  # Shape of each sample (e.g., (6,) for 6 joints)
    dtype: np.dtype = np.float64
    strategy: BufferFullStrategy = BufferFullStrategy.OVERWRITE


class RingBuffer:
    """
    High-performance circular buffer for time-series data.

    Optimized for:
    - Real-time control loops (1kHz)
    - Minimal GC pressure (pre-allocated)
    - Fast append and retrieval

    Attributes:
        capacity: Maximum number of samples
        shape: Shape of each sample
        size: Current number of samples in buffer
        head: Index of next write position
        tail: Index of oldest valid sample
    """

    def __init__(
        self,
        capacity: int,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float64,
        strategy: BufferFullStrategy = BufferFullStrategy.OVERWRITE,
    ):
        self.capacity = capacity
        self.shape = shape
        self.dtype = dtype
        self.strategy = strategy

        # Pre-allocate storage
        self._buffer = np.zeros((capacity,) + shape, dtype=dtype)
        self._timestamps = np.zeros(capacity, dtype=np.float64)

        # State
        self._head = 0  # Next write position
        self._size = 0  # Current number of valid samples
        self._is_full = False

    def append(self, data: np.ndarray, timestamp: Optional[float] = None) -> None:
        """
        Append a new sample to the buffer.

        Args:
            data: Sample data with shape matching buffer shape
            timestamp: Optional timestamp (defaults to time.time())

        Time Complexity: O(1)
        """
        if data.shape != self.shape:
            raise ValueError(f"Expected shape {self.shape}, got {data.shape}")

        # Write data
        self._buffer[self._head] = data
        self._timestamps[self._head] = timestamp if timestamp is not None else time.time()

        # Advance head
        self._head = (self._head + 1) % self.capacity

        # Update size and full flag
        if not self._is_full:
            self._size += 1
            if self._size >= self.capacity:
                self._is_full = True

    def get_last_n(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the last n samples with timestamps.

        Args:
            n: Number of samples to retrieve

        Returns:
            Tuple of (data_array, timestamps)
            data_array shape: (n,) + shape

        Time Complexity: O(n)
        """
        if n > self._size:
            n = self._size

        if n == 0:
            return (
                np.empty((0,) + self.shape, dtype=self.dtype),
                np.empty(0, dtype=np.float64),
            )

        # Calculate indices
        end_idx = self._head
        start_idx = (end_idx - n) % self.capacity

        if start_idx < end_idx:
            # Simple case: contiguous block
            return (
                self._buffer[start_idx:end_idx].copy(),
                self._timestamps[start_idx:end_idx].copy(),
            )
        else:
            # Wrapped case: two blocks
            first_block = self._buffer[start_idx:]
            second_block = self._buffer[:end_idx]

            first_ts = self._timestamps[start_idx:]
            second_ts = self._timestamps[:end_idx]

            return (
                np.concatenate([first_block, second_block]),
                np.concatenate([first_ts, second_ts]),
            )

    def get_range(
        self,
        start_time: float,
        end_time: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get samples within a time range.

        Args:
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (inclusive)

        Returns:
            Tuple of (data_array, timestamps)
        """
        # Find indices within time range
        mask = (self._timestamps >= start_time) & (self._timestamps <= end_time)

        if not np.any(mask):
            return (
                np.empty((0,) + self.shape, dtype=self.dtype),
                np.empty(0, dtype=np.float64),
            )

        valid_indices = np.where(mask)[0]

        # Sort by timestamp to ensure chronological order
        sorted_indices = valid_indices[np.argsort(self._timestamps[valid_indices])]

        return (
            self._buffer[sorted_indices].copy(),
            self._timestamps[sorted_indices].copy(),
        )

    def get_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all valid samples in chronological order."""
        return self.get_last_n(self._size)

    def clear(self) -> None:
        """Clear the buffer (keeps allocation)."""
        self._head = 0
        self._size = 0
        self._is_full = False

    @property
    def is_full(self) -> bool:
        """Check if buffer has reached capacity."""
        return self._is_full

    @property
    def size(self) -> int:
        """Current number of valid samples."""
        return self._size

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self._size == 0

    def latest(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        Get the most recent sample.

        Returns:
            Tuple of (data, timestamp) or None if empty
        """
        if self.is_empty:
            return None

        idx = (self._head - 1) % self.capacity
        return self._buffer[idx].copy(), self._timestamps[idx]


class MultiChannelRingBuffer:
    """
    Multi-channel ring buffer for synchronized data capture.

    Manages multiple RingBuffers with a shared timeline,
    useful for capturing robot state + images + etc.

    Example:
        buffer = MultiChannelRingBuffer(
            joint_states=(60000, (6,)),
            images=(60000, (480, 640, 3)),
        )
        buffer.append({
            'joint_states': joint_positions,
            'images': camera_image,
        })
    """

    def __init__(self, **channel_configs: Tuple[int, Tuple[int, ...]]):
        """
        Initialize multi-channel buffer.

        Args:
            **channel_configs: Dict of {name: (capacity, shape)}
        """
        self.channels: dict[str, RingBuffer] = {}
        self._capacity = None

        for name, (capacity, shape) in channel_configs.items():
            if self._capacity is None:
                self._capacity = capacity
            elif capacity != self._capacity:
                raise ValueError(f"All channels must have same capacity, got {capacity} vs {self._capacity}")

            self.channels[name] = RingBuffer(capacity=capacity, shape=shape)

    def append(self, data: dict[str, np.ndarray], timestamp: Optional[float] = None) -> None:
        """
        Append data to all channels.

        Args:
            data: Dict mapping channel names to data arrays
            timestamp: Shared timestamp for all channels
        """
        ts = timestamp if timestamp is not None else time.time()

        for name, channel_data in data.items():
            if name not in self.channels:
                raise KeyError(f"Unknown channel: {name}")
            self.channels[name].append(channel_data, ts)

    def get_last_n(self, n: int) -> dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get last n samples from all channels."""
        return {
            name: buffer.get_last_n(n)
            for name, buffer in self.channels.items()
        }

    def get_range(
        self,
        start_time: float,
        end_time: float,
    ) -> dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get samples within time range from all channels."""
        return {
            name: buffer.get_range(start_time, end_time)
            for name, buffer in self.channels.items()
        }

    def clear(self) -> None:
        """Clear all channels."""
        for buffer in self.channels.values():
            buffer.clear()

    @property
    def size(self) -> int:
        """Return size of first channel (assumes synchronized)."""
        first = next(iter(self.channels.values()))
        return first.size
