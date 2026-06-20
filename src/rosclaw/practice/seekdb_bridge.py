"""Bridge from ROSClaw core PraxisEvent to rosclaw_practice SeekDB committer.

This module lets the main ROSClaw runtime persist its native
`rosclaw.core.types.PraxisEvent` records to SeekDB using the well-tested
`rosclaw_practice` package.  It is an optional integration: importing it
requires ``rosclaw-practice`` to be installed (``pip install rosclaw[practice]``).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rosclaw.core.types import PraxisEvent as RosclawPraxisEvent

try:
    from rosclaw_practice.committer import ExperienceCommitter
    from rosclaw_practice.schemas import (
        CognitiveContext,
        DataPointers,
        PhysicalFeedback,
    )
    from rosclaw_practice.schemas import (
        PraxisEvent as PracticePraxisEvent,
    )
except ImportError as exc:  # pragma: no cover - exercised by optional-deps tests
    raise ImportError(
        "rosclaw-practice is required for SeekDB integration. "
        "Install it with: pip install rosclaw[practice]"
    ) from exc


class SeekDBBridge:
    """Commits ROSClaw ``PraxisEvent`` records to SeekDB via ``rosclaw_practice``."""

    def __init__(
        self,
        seekdb_url: str = "http://localhost:2881",
        fallback_dir: str = "/data/rosclaw/fallback",
    ) -> None:
        """Initialize the bridge.

        :param seekdb_url: Base URL of the SeekDB instance.
        :param fallback_dir: Directory for offline JSON fallback files.
        """
        self._committer = ExperienceCommitter(
            seekdb_url=seekdb_url,
            fallback_dir=fallback_dir,
        )

    def commit(self, event: RosclawPraxisEvent) -> None:
        """Send *event* to SeekDB, falling back to local JSON on failure.

        This method is synchronous because ``ExperienceCommitter`` performs a
        blocking HTTP request.  Use :meth:`commit_async` from async code to avoid
        blocking the event loop.
        """
        practice_event = self._convert(event)
        self._committer.save_to_seekdb(practice_event.model_dump())

    async def commit_async(self, event: RosclawPraxisEvent) -> None:
        """Async wrapper around :meth:`commit`."""
        await asyncio.to_thread(self.commit, event)

    def _convert(self, event: RosclawPraxisEvent) -> PracticePraxisEvent:
        """Convert a ROSClaw ``PraxisEvent`` to a ``rosclaw_practice`` event."""
        return PracticePraxisEvent(
            practice_id=event.event_id,
            timestamp=_to_iso_utc(event.timestamp),
            robot_id=event.robot_id,
            cognitive_context=CognitiveContext(
                semantic_intent=event.agent_instruction,
                llm_cot="\n".join(event.cot_trace),
            ),
            physical_feedback=PhysicalFeedback(
                status=event.event_type.upper(),
                reward=float(event.metadata.get("reward", 0.0)),
                error_log=event.error_details or "",
            ),
            data_pointers=DataPointers(
                mcap_path=event.mcap_path or "",
            ),
        )


def _to_iso_utc(timestamp: float) -> str:
    """Format a Unix timestamp as an ISO 8601 UTC string."""
    dt = datetime.fromtimestamp(timestamp, tz=UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
