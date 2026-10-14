"""Daemon-owned supplementary SIM observation guard, never authorization."""

from abc import ABC, abstractmethod

from rosclaw.connectors.ros.mission.navigation_binding import SimulationNavigationBinding


class NavigationExecutionSupervision(ABC):
    """Implemented by an SDK fixture, configured only by the daemon operator.

    Implementations independently collect current ROS/source/daemon facts and
    latch on loss. An action intent cannot supply this object or health claims.
    Passing the guard does not replace canonical Session/Lease/Permit checks.
    """

    def __init__(self, binding: SimulationNavigationBinding):
        if not isinstance(binding, SimulationNavigationBinding):
            raise ValueError("immutable SIM navigation binding required")
        self.binding = binding

    @abstractmethod
    def begin(self, action_id: str, *, duration_sec: float) -> None:
        """Check admission and conservative remaining source capacity before enable."""

    @abstractmethod
    def check(self, action_id: str) -> None:
        """Refresh independent observations/proof or raise permanently on loss."""
