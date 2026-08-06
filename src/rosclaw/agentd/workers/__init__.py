"""Worker Fabric (ADR-0003): registry, scheduler, manager, adapters.

Cognitive workers only — hardware adapter subprocesses belong to
``rosclaw.daemon.worker_manager``; different schemas, logs and CLI.
"""

from rosclaw.agentd.workers.adapter import (
    AdapterError,
    RunHandle,
    WorkerAdapter,
    WorkerProbeResult,
)
from rosclaw.agentd.workers.manager import WorkerManager
from rosclaw.agentd.workers.native import NativeWorkerAdapter
from rosclaw.agentd.workers.registry import CardValidationError, WorkerRegistry
from rosclaw.agentd.workers.scheduler import Scheduler, SchedulingError
from rosclaw.agentd.workers.verify import VerificationReport, verify_result

__all__ = [
    "AdapterError",
    "CardValidationError",
    "NativeWorkerAdapter",
    "RunHandle",
    "Scheduler",
    "SchedulingError",
    "VerificationReport",
    "WorkerAdapter",
    "WorkerManager",
    "WorkerProbeResult",
    "WorkerRegistry",
    "verify_result",
]
