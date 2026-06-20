"""Tests for rosclaw.runtime.health"""

from __future__ import annotations

from unittest.mock import patch

from rosclaw.runtime.health import overall_status, subsystem_health


class TestSubsystemHealth:
    def test_returns_all_sprint1_subsystems(self):
        health = subsystem_health()
        expected = {
            "runtime",
            "event_bus",
            "seekdb",
            "registry",
            "sandbox",
            "provider",
            "memory",
            "practice",
        }
        assert set(health.keys()) == expected

    def test_status_values_are_valid(self):
        health = subsystem_health()
        valid = {"healthy", "loaded", "disabled", "degraded"}
        for name, info in health.items():
            assert info.get("status") in valid, f"{name} has invalid status"

    def test_registry_status_is_loaded_when_available(self):
        health = subsystem_health()
        assert health["registry"]["status"] in ("loaded", "degraded")

    def test_overall_healthy_when_no_degraded(self):
        health = subsystem_health()
        assert overall_status(health) in ("HEALTHY", "DEGRADED")

    def test_overall_degraded_when_any_degraded(self):
        health = {"runtime": {"status": "degraded"}, "event_bus": {"status": "healthy"}}
        assert overall_status(health) == "DEGRADED"

    def test_overall_healthy_when_all_healthy(self):
        health = {
            "runtime": {"status": "healthy"},
            "event_bus": {"status": "healthy"},
            "registry": {"status": "loaded"},
        }
        assert overall_status(health) == "HEALTHY"


class TestSubsystemHealthFailures:
    @patch("importlib.import_module", side_effect=ImportError("no module"))
    def test_all_subsystems_degraded_on_import_failure(self, _mock_import):
        health = subsystem_health()
        for name, info in health.items():
            assert info["status"] == "degraded", f"{name} should be degraded"
        assert overall_status(health) == "DEGRADED"
