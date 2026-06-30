"""Runtime doctor - health checks for runtime components and devices."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DoctorCheck:
    """A single runtime health check result."""

    plugin: str
    check: str
    status: str  # ok, warn, error
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin": self.plugin,
            "check": self.check,
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }


class RuntimeDoctorPlugin:
    """Base class for runtime doctor plugins."""

    name: str = ""

    def check(self) -> Iterable[DoctorCheck]:
        """Yield health checks for this plugin."""
        return []
        yield


class RuntimeDoctor:
    """Aggregates health checks from registered plugins."""

    def __init__(self) -> None:
        self._plugins: list[RuntimeDoctorPlugin] = []

    def register_plugin(self, plugin: RuntimeDoctorPlugin) -> None:
        """Register a doctor plugin."""
        self._plugins.append(plugin)

    def check_all(self) -> list[DoctorCheck]:
        """Run all registered checks and return results."""
        results: list[DoctorCheck] = []
        for plugin in self._plugins:
            try:
                results.extend(plugin.check())
            except Exception as exc:
                results.append(
                    DoctorCheck(
                        plugin=plugin.name or plugin.__class__.__name__,
                        check="plugin_exception",
                        status="error",
                        message=f"Plugin raised an exception: {exc}",
                    )
                )
        return results

    def summary(self) -> dict[str, Any]:
        """Return a summary of the latest check results."""
        checks = self.check_all()
        statuses = [c.status for c in checks]
        return {
            "ok": statuses.count("ok"),
            "warn": statuses.count("warn"),
            "error": statuses.count("error"),
            "total": len(checks),
            "checks": [c.to_dict() for c in checks],
        }
