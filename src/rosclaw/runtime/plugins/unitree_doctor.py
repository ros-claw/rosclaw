"""Runtime doctor plugin for Unitree robots (stub)."""

from __future__ import annotations

from collections.abc import Iterable

from rosclaw.runtime.doctor import DoctorCheck, RuntimeDoctorPlugin


class UnitreeDoctor(RuntimeDoctorPlugin):
    """Stub health checks for Unitree robot runtime integration."""

    name = "unitree"

    def check(self) -> Iterable[DoctorCheck]:
        yield DoctorCheck(
            plugin=self.name,
            check="unitree_sdk",
            status="warn",
            message="Unitree doctor is a stub; no runtime integration available",
        )
