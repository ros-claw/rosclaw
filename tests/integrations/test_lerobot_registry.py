"""Test integration registry registration."""

from __future__ import annotations

from rosclaw.integrations import GLOBAL_INTEGRATION_REGISTRY
from rosclaw.integrations.lerobot import register_lerobot_capabilities
from rosclaw.integrations.lerobot.capabilities import get_lerobot_capabilities
from rosclaw.integrations.lerobot.dataset_exporter import LeRobotDatasetExporter
from rosclaw.integrations.lerobot.provider import LeRobotPolicyProvider


def test_register_lerobot_capabilities():
    """LeRobot provider type and exporter should be registered."""
    registry = type(GLOBAL_INTEGRATION_REGISTRY)()
    register_lerobot_capabilities(registry)

    assert registry.get_provider_factory("lerobot_policy") is LeRobotPolicyProvider
    assert registry.get_exporter_factory("lerobot") is LeRobotDatasetExporter

    integration = registry.get_integration("lerobot")
    assert integration.name == "lerobot"
    assert integration.status in ("not_installed", "degraded", "installed")


def test_get_lerobot_capabilities():
    """Static capability list should contain provider and exporter."""
    caps = get_lerobot_capabilities()
    names = {c.name for c in caps}
    assert "provider_type_lerobot_policy" in names
    assert "dataset_export_lerobot" in names


def test_register_integration_after_factory_preserves_factory():
    """Registration order must not leave a factory-only placeholder record."""
    registry = type(GLOBAL_INTEGRATION_REGISTRY)()
    registry.register_provider_type("lerobot_policy", LeRobotPolicyProvider)

    class Integration:
        @staticmethod
        def report():
            return "registered"

    registry.register_integration("lerobot", Integration)

    assert registry.get_provider_factory("lerobot_policy") is LeRobotPolicyProvider
    assert registry.get_integration("lerobot") == "registered"
