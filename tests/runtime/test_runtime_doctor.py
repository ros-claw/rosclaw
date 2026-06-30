"""Tests for the Runtime Doctor (Milestone 8)."""

from __future__ import annotations

from collections.abc import Iterable

from rosclaw.runtime.doctor import DoctorCheck, RuntimeDoctor, RuntimeDoctorPlugin
from rosclaw.runtime.plugins import (
    DexHandDoctor,
    RealSenseDoctor,
    UnitreeDoctor,
    URDoctor,
)


class _FailingPlugin(RuntimeDoctorPlugin):
    name = "failing"

    def check(self) -> Iterable[DoctorCheck]:
        raise RuntimeError("plugin failure")


class _CustomPlugin(RuntimeDoctorPlugin):
    name = "custom"

    def check(self) -> Iterable[DoctorCheck]:
        yield DoctorCheck(
            plugin=self.name,
            check="custom_check",
            status="ok",
            message="custom is fine",
        )


def test_doctor_runs_all_default_plugins() -> None:
    doctor = RuntimeDoctor()
    doctor.register_plugin(RealSenseDoctor())
    doctor.register_plugin(UnitreeDoctor())
    doctor.register_plugin(URDoctor())
    doctor.register_plugin(DexHandDoctor())

    summary = doctor.summary()
    assert summary["total"] >= 4
    assert summary["ok"] + summary["warn"] + summary["error"] == summary["total"]


def test_doctor_summary_counts_statuses() -> None:
    doctor = RuntimeDoctor()
    doctor.register_plugin(_CustomPlugin())
    summary = doctor.summary()
    assert summary["ok"] == 1
    assert summary["warn"] == 0
    assert summary["error"] == 0
    assert summary["total"] == 1


def test_doctor_catches_plugin_exceptions() -> None:
    doctor = RuntimeDoctor()
    doctor.register_plugin(_FailingPlugin())
    checks = doctor.check_all()
    assert len(checks) == 1
    assert checks[0].status == "error"
    assert "plugin failure" in checks[0].message


def test_realsense_doctor_status_includes_import_check() -> None:
    doctor = RuntimeDoctor()
    doctor.register_plugin(RealSenseDoctor())
    checks = doctor.check_all()
    import_check = next(c for c in checks if c.check == "pyrealsense2_import")
    # The import may or may not succeed in the test environment; either is fine.
    assert import_check.status in ("ok", "warn")
