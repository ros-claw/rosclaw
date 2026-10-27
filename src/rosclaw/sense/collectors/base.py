"""Base class for BodyState collectors."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from rosclaw.sense.schemas import BodyState


class BodyStateCollector(ABC):
    """Abstract base class for sources of BodyState snapshots.

    Collectors are responsible for reading robot telemetry, DDS/ROS2 topics,
    log files, or synthetic data and returning a normalized ``BodyState``.
    """

    name: str = "abstract"

    @abstractmethod
    def collect(self) -> BodyState:
        """Return the latest BodyState snapshot."""
        raise NotImplementedError

    def start(self) -> None:  # noqa: B027
        """Start any background threads or connections."""
        pass

    def stop(self) -> None:  # noqa: B027
        """Stop background threads or connections gracefully."""
        pass


class UnconfiguredCollector(BodyStateCollector):
    """No configured telemetry source; never synthesize a healthy robot."""

    name = "unconfigured"

    def __init__(self, robot_id: str):
        self.robot_id = robot_id

    def collect(self) -> BodyState:
        return BodyState(
            robot_id=self.robot_id,
            timestamp=time.time(),
            source=self.name,
            raw={"collector_configured": False},
        )
