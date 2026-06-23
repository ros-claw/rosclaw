"""SourceAdapter protocol for practice data sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol

from rosclaw.practice.schemas import PracticeEventEnvelope


@dataclass
class SourceHealth:
    """Health status reported by a source adapter."""

    source: str
    healthy: bool
    message: str = ""
    dropped_frames: int = 0


class SourceAdapter(Protocol):
    """Protocol for a practice data source."""

    source_name: str

    def start(self, session: Any) -> None:
        """Called when the practice session starts."""
        ...

    def stop(self) -> None:
        """Called when the practice session stops."""
        ...

    def health(self) -> SourceHealth:
        """Return current health status."""
        ...

    def poll(self) -> Iterable[PracticeEventEnvelope]:
        """Return zero or more events since the last poll.

        Implementations may alternatively use ``on_event`` callbacks for
        push-based sources.
        """
        ...

    def on_event(self, callback: Callable[[PracticeEventEnvelope], None]) -> None:
        """Optional callback registration for push-based sources."""
        ...
