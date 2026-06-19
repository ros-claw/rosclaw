"""Configuration for the rosclaw.sense module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SenseConfig:
    """Configuration for a SenseRuntime instance.

    Attributes:
        robot_id: Robot identifier for state/sense attribution.
        collector: Collector backend to use. Supported: ``mock``, ``file_replay``,
            ``ros2``, ``dds``.
        robot_profile: Optional robot family/profile used to load thresholds and
            capability requirements (e.g. ``unitree_g1``).
        thresholds_path: Optional path to a YAML file with threshold overrides.
        replay_path: Optional path to a JSONL file for ``file_replay`` collector.
        update_hz: Sense tick frequency in Hz.
        publish_events: Whether to publish EventBus events on each tick.
        max_event_history: Maximum number of BodyEvents to retain in memory.
        extra: Catch-all for collector-specific or future settings.
    """

    robot_id: str = "rosclaw_default"
    collector: str = "mock"
    robot_profile: str | None = None
    thresholds_path: str | None = None
    replay_path: str | None = None
    update_hz: float = 1.0
    publish_events: bool = True
    max_event_history: int = 1000
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary."""
        return {
            "robot_id": self.robot_id,
            "collector": self.collector,
            "robot_profile": self.robot_profile,
            "thresholds_path": self.thresholds_path,
            "replay_path": self.replay_path,
            "update_hz": self.update_hz,
            "publish_events": self.publish_events,
            "max_event_history": self.max_event_history,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SenseConfig:
        """Deserialize config from a dictionary."""
        return cls(
            robot_id=data.get("robot_id", "rosclaw_default"),
            collector=data.get("collector", "mock"),
            robot_profile=data.get("robot_profile"),
            thresholds_path=data.get("thresholds_path"),
            replay_path=data.get("replay_path"),
            update_hz=data.get("update_hz", 1.0),
            publish_events=data.get("publish_events", True),
            max_event_history=data.get("max_event_history", 1000),
            extra=data.get("extra", {}),
        )
