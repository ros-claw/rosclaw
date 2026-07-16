"""In-process resource leases for serialized physical action ownership."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from rosclaw.kernel.contracts import utc_now


@dataclass
class ResourceLease:
    """Exclusive ownership of one named physical resource."""

    lease_id: str
    resource_id: str
    action_id: str
    acquired_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "resource_id": self.resource_id,
            "action_id": self.action_id,
            "acquired_at": self.acquired_at.isoformat().replace("+00:00", "Z"),
            "exclusive": True,
        }


class ResourceLeaseHandle:
    """Context manager that releases its lease exactly once."""

    def __init__(self, manager: ResourceManager, lease: ResourceLease, lock: threading.Lock):
        self.manager = manager
        self.lease = lease
        self._lock = lock
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self.manager._release(self.lease, self._lock)

    def __enter__(self) -> ResourceLeaseHandle:
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


class ResourceManager:
    """Own exclusive, bounded leases for body-level actions."""

    def __init__(self) -> None:
        self._guard = threading.RLock()
        self._locks: dict[str, threading.Lock] = {}
        self._active: dict[str, ResourceLease] = {}

    def acquire(
        self,
        resource_id: str,
        action_id: str,
        *,
        timeout_sec: float,
    ) -> ResourceLeaseHandle | None:
        with self._guard:
            lock = self._locks.setdefault(resource_id, threading.Lock())
        acquired = lock.acquire(timeout=max(0.0, timeout_sec))
        if not acquired:
            return None
        lease = ResourceLease(
            lease_id=f"lease_{uuid.uuid4().hex}",
            resource_id=resource_id,
            action_id=action_id,
            acquired_at=utc_now(),
        )
        with self._guard:
            self._active[resource_id] = lease
        return ResourceLeaseHandle(self, lease, lock)

    def _release(self, lease: ResourceLease, lock: threading.Lock) -> None:
        with self._guard:
            active = self._active.get(lease.resource_id)
            if active is not None and active.lease_id == lease.lease_id:
                self._active.pop(lease.resource_id, None)
        lock.release()

    def active_lease(self, resource_id: str) -> ResourceLease | None:
        with self._guard:
            return self._active.get(resource_id)


__all__ = ["ResourceLease", "ResourceLeaseHandle", "ResourceManager"]
