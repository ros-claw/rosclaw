"""Bridge from ROSClaw core PraxisEvent to rosclaw_practice SeekDB committer.

This module lets the main ROSClaw runtime persist its native
`rosclaw.core.types.PraxisEvent` records to SeekDB using the well-tested
`rosclaw_practice` package.  It is an optional integration: importing it
requires ``rosclaw-practice`` to be installed (``pip install rosclaw[practice]``).
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from rosclaw.practice.config import get_default_data_root

if TYPE_CHECKING:
    from rosclaw.core.types import PraxisEvent as RosclawPraxisEvent
    from rosclaw.storage.outbox import OutboxStore, OutboxWorker

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

logger = logging.getLogger("rosclaw.practice.seekdb_bridge")


class SeekDBBridge:
    """Commits ROSClaw ``PraxisEvent`` records to SeekDB via ``rosclaw_practice``."""

    def __init__(
        self,
        seekdb_url: str | None = None,
        fallback_dir: str | None = None,
        outbox: OutboxStore | None = None,
        outbox_worker: OutboxWorker | None = None,
        outbox_interval_sec: float = 5.0,
        outbox_batch_size: int = 100,
    ) -> None:
        """Initialize the bridge.

        :param seekdb_url: Base URL of the rosclaw_practice HTTP adapter.
            Defaults to ROSCLAW_PRACTICE_HTTP_ADAPTER_URL or
            http://localhost:2882 to avoid colliding with the SeekDB SQL
            protocol on port 2881.
        :param fallback_dir: Directory for offline JSON fallback files.
        :param outbox: Optional durable outbox. When provided, ``commit()``
            writes to the local outbox instead of performing a blocking HTTP
            request. A background worker must drain the outbox.
        :param outbox_worker: Optional worker that drains *outbox*. If omitted
            but *outbox* is provided, the bridge creates and starts its own
            worker using the internal committer.
        :param outbox_interval_sec: Drain interval for a bridge-owned worker.
        :param outbox_batch_size: Batch size for a bridge-owned worker.
        """
        seekdb_url = seekdb_url or os.environ.get(
            "ROSCLAW_PRACTICE_HTTP_ADAPTER_URL", "http://localhost:2882"
        )
        fallback_dir = fallback_dir or os.environ.get(
            "ROSCLAW_SEEKDB_FALLBACK_DIR", str(get_default_data_root() / "fallback")
        )
        self._committer = ExperienceCommitter(
            seekdb_url=seekdb_url,
            fallback_dir=fallback_dir,
        )
        self._outbox = outbox
        self._owned_worker: OutboxWorker | None = None
        self._outbox_worker = outbox_worker

        if self._outbox is not None and self._outbox_worker is None:
            from rosclaw.storage.outbox import OutboxWorker

            self._owned_worker = OutboxWorker(
                self._outbox,
                self._committer,
                interval_sec=outbox_interval_sec,
                batch_size=outbox_batch_size,
            )
            self._owned_worker.start()
            self._outbox_worker = self._owned_worker

    def commit(self, event: RosclawPraxisEvent) -> None:
        """Send *event* to SeekDB, falling back to local JSON on failure.

        When an outbox is configured, this method only writes to the local
        outbox and returns immediately; the background worker handles the
        upstream HTTP call.  Without an outbox, ``ExperienceCommitter`` performs
        a blocking HTTP request.  Use :meth:`commit_async` from async code to
        avoid blocking the event loop.
        """
        practice_event = self._convert(event)
        payload = practice_event.model_dump()
        if self._outbox is not None:
            self._outbox.enqueue(
                "seekdb_http",
                payload,
                idempotency_key=f"praxis_event:{event.event_id}",
                entity_type="praxis_event",
                entity_id=event.event_id,
            )
            return
        self._committer.save_to_seekdb(payload)

    async def commit_async(self, event: RosclawPraxisEvent) -> None:
        """Async wrapper around :meth:`commit`."""
        await asyncio.to_thread(self.commit, event)

    def close(self) -> None:
        """Flush and stop any outbox worker owned by this bridge."""
        if self._owned_worker is not None:
            self._owned_worker.flush()
            self._owned_worker.stop()
            self._owned_worker = None
            self._outbox_worker = None

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
