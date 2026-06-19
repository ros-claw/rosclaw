"""Base class for BodyState collectors."""

from __future__ import annotations

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
