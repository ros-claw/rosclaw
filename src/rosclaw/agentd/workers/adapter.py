"""Worker adapter SPI (总纲 §9.9) and the native in-process adapter (PR-WF-053).

The adapter translates between WorkOrder/WorkResult contracts and a concrete
execution mechanism. Third-party product concepts (sessions, subagents,
channels) stay in adapter-private fields and never leak into core contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rosclaw.contracts.worker.order import WorkOrderV1, WorkResultV1


class AdapterError(Exception):
    """Adapter-level failure (start/poll/cancel)."""


@dataclass(frozen=True)
class WorkerProbeResult:
    ready: bool
    detail: str = ""
    card_digest: str | None = None


@dataclass
class RunHandle:
    work_order_id: str
    lease_id: str
    worker_id: str
    progress_seq: int = 0
    last_checkpoint: str | None = None
    status: str = "RUNNING"  # RUNNING | SUBMITTED | FAILED | CANCELLED
    private: dict = field(default_factory=dict)  # adapter-private state


class WorkerAdapter(Protocol):
    async def probe(self) -> WorkerProbeResult: ...

    async def start(self, order: WorkOrderV1, credential_refs: dict) -> RunHandle: ...

    async def poll(self, handle: RunHandle) -> RunHandle | WorkResultV1: ...

    async def cancel(self, handle: RunHandle, reason: str) -> None: ...

    async def reconcile(self, idempotency_key: str) -> str:
        """Return 'completed' | 'running' | 'not_found' for a side-effect key."""
        ...
