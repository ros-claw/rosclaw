"""
ROSClaw Practice - Physical Data Flywheel Runtime

MCAP-based black box recording and replay.
Captures robot execution traces for analysis and learning.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rosclaw.practice.config import PracticeConfig, PracticeSession, PracticeSummary
    from rosclaw.practice.coordinator import PracticeCoordinator
    from rosclaw.practice.episode_recorder import EpisodeRecorder
    from rosclaw.practice.recorder import PracticeRecorder
    from rosclaw.practice.timeline import TimelineChannel, TimelineEntry, UnifiedTimeline

# Import only the defining module of an explicitly requested public object.
_EXPORT_MODULES = {
    "PracticeConfig": "rosclaw.practice.config",
    "PracticeCoordinator": "rosclaw.practice.coordinator",
    "PracticeRecorder": "rosclaw.practice.recorder",
    "PracticeSession": "rosclaw.practice.config",
    "PracticeSummary": "rosclaw.practice.config",
    "UnifiedTimeline": "rosclaw.practice.timeline",
    "TimelineChannel": "rosclaw.practice.timeline",
    "TimelineEntry": "rosclaw.practice.timeline",
    "EpisodeRecorder": "rosclaw.practice.episode_recorder",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "PracticeConfig",
    "PracticeCoordinator",
    "PracticeRecorder",
    "PracticeSession",
    "PracticeSummary",
    "UnifiedTimeline",
    "TimelineChannel",
    "TimelineEntry",
    "EpisodeRecorder",
]
