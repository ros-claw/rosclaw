"""
Event-Driven Ring Buffer for ROSClaw V4

Implements Layer 2: Physical Data with event-driven persistence.
60-second hot buffer with event-triggered persistence.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from collections import deque
import time
import numpy as np


@dataclass
class EventTrigger:
    """Event trigger configuration"""
    name: str
    description: str = ""
    save_window_before: float = 10.0  # seconds before event
    save_window_after: float = 5.0    # seconds after event


@dataclass
class Episode:
    """Single episode of robot interaction"""
    session_id: str
    task: str
    actions: list[dict]
    result: dict
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class RingBuffer:
    """
    Circular buffer for high-frequency robot data

    Maintains 60 seconds of history at 30Hz (1800 frames)
    Event-driven persistence to avoid TB/day storage
    """

    def __init__(
        self,
        capacity: int = 1800,
        feature_dim: int = 10,
        sample_rate_hz: float = 30.0
    ):
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.sample_rate_hz = sample_rate_hz
        self.buffer_duration_sec = capacity / sample_rate_hz

        # Circular buffers
        self._states = np.zeros((capacity, feature_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, feature_dim), dtype=np.float32)
        self._timestamps = np.zeros(capacity, dtype=np.float64)
        self._metadata: list[dict] = [dict() for _ in range(capacity)]

        self._index = 0
        self._is_full = False
        self._episodes: list[Episode] = []

    def add_frame(
        self,
        state: np.ndarray,
        action: np.ndarray,
        timestamp: Optional[float] = None,
        metadata: Optional[dict] = None
    ):
        """Add a single frame to the buffer"""
        if timestamp is None:
            timestamp = time.time()

        idx = self._index
        self._states[idx] = state
        self._actions[idx] = action
        self._timestamps[idx] = timestamp
        if metadata:
            self._metadata[idx] = metadata

        self._index = (self._index + 1) % self.capacity
        if self._index == 0:
            self._is_full = True

    def add_episode(self, episode_data: dict):
        """Add episode metadata"""
        episode = Episode(
            session_id=episode_data.get('session_id', ''),
            task=episode_data.get('task', ''),
            actions=episode_data.get('actions', []),
            result=episode_data.get('result', {}),
            timestamp=episode_data.get('timestamp', time.time()),
            metadata=episode_data.get('metadata', {})
        )
        self._episodes.append(episode)

    def trigger_event(self, event_type: str, data: Any):
        """
        Trigger event-based data persistence

        Args:
            event_type: Type of event (success, failure, collision, etc.)
            data: Associated data to save
        """
        # In real implementation, this would:
        # 1. Extract [T-before, T+after] window from buffer
        # 2. Save to persistent storage (SSD/Cloud)
        # 3. Clear from hot buffer

        episode = Episode(
            session_id=data.get('session_id', str(time.time())),
            task=data.get('task', ''),
            actions=data.get('actions', []),
            result=data.get('result', {}),
            metadata={'event_type': event_type, 'trigger_time': time.time()}
        )
        self._episodes.append(episode)

    def get_recent_frames(self, n_frames: int) -> dict[str, np.ndarray]:
        """Get n most recent frames"""
        if not self._is_full:
            start = max(0, self._index - n_frames)
            return {
                'states': self._states[start:self._index],
                'actions': self._actions[start:self._index],
                'timestamps': self._timestamps[start:self._index]
            }
        else:
            # Handle wrap-around
            if n_frames > self._index:
                start_idx = self.capacity - (n_frames - self._index)
                states = np.concatenate([
                    self._states[start_idx:],
                    self._states[:self._index]
                ])
                actions = np.concatenate([
                    self._actions[start_idx:],
                    self._actions[:self._index]
                ])
                timestamps = np.concatenate([
                    self._timestamps[start_idx:],
                    self._timestamps[:self._index]
                ])
            else:
                start_idx = self._index - n_frames
                states = self._states[start_idx:self._index]
                actions = self._actions[start_idx:self._index]
                timestamps = self._timestamps[start_idx:self._index]

            return {
                'states': states,
                'actions': actions,
                'timestamps': timestamps
            }

    def get_episodes(self, n_recent: Optional[int] = None) -> list[Episode]:
        """Get episodes, optionally limited to n most recent"""
        if n_recent is None:
            return self._episodes
        return self._episodes[-n_recent:]

    def clear(self):
        """Clear all buffers"""
        self._index = 0
        self._is_full = False
        self._states.fill(0)
        self._actions.fill(0)
        self._timestamps.fill(0)
        self._metadata = [dict() for _ in range(self.capacity)]
        self._episodes.clear()

    @property
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return self._is_full

    @property
    def size(self) -> int:
        """Current number of frames in buffer"""
        return self.capacity if self._is_full else self._index


class DataCollector:
    """
    High-level data collection interface

    Connects to RuntimeManager to collect trajectories
    """

    def __init__(self, ring_buffer: Optional[RingBuffer] = None):
        self.buffer = ring_buffer or RingBuffer()
        self._active = False
        self._session_id: Optional[str] = None

    def start_session(self, session_id: str):
        """Start a new collection session"""
        self._session_id = session_id
        self._active = True

    def stop_session(self):
        """Stop current session"""
        self._active = False
        self._session_id = None

    def collect_frame(
        self,
        state: np.ndarray,
        action: np.ndarray,
        metadata: Optional[dict] = None
    ):
        """Collect a single frame (if session active)"""
        if self._active:
            meta = metadata or {}
            meta['session_id'] = self._session_id
            self.buffer.add_frame(state, action, metadata=meta)

    def mark_success(self, episode_data: dict):
        """Mark episode as success"""
        self.buffer.trigger_event('success', episode_data)

    def mark_failure(self, episode_data: dict):
        """Mark episode as failure"""
        self.buffer.trigger_event('failure', episode_data)

    @property
    def is_active(self) -> bool:
        """Check if collection is active"""
        return self._active
