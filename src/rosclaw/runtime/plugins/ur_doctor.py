"""Runtime doctor plugin for Universal Robots arms (stub)."""

from __future__ import annotations

from collections.abc import Iterable

from rosclaw.runtime.doctor import DoctorCheck, RuntimeDoctorPlugin


class URDoctor(RuntimeDoctorPlugin):
    """Stub health checks for UR robot runtime integration."""

    name = "ur"

    def check(self) -> Iterable[DoctorCheck]:
        yield DoctorCheck(
            plugin=self.name,
            check="ur_driver",
            status="warn",
            message="UR doctor is a stub; no runtime integration available",
        )
