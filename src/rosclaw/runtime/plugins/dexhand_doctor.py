"""Runtime doctor plugin for dexterous hands (stub)."""

from __future__ import annotations

from collections.abc import Iterable

from rosclaw.runtime.doctor import DoctorCheck, RuntimeDoctorPlugin


class DexHandDoctor(RuntimeDoctorPlugin):
    """Stub health checks for dexterous hand runtime integration."""

    name = "dexhand"

    def check(self) -> Iterable[DoctorCheck]:
        yield DoctorCheck(
            plugin=self.name,
            check="dexhand_driver",
            status="warn",
            message="DexHand doctor is a stub; no runtime integration available",
        )
