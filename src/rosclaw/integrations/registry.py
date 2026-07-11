"""ROSClaw integration registry.

Provides a lightweight, dependency-free registry for optional integrations
such as LeRobot. The registry itself never imports integration-specific
libraries, so rosclaw-core remains installable without them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

IntegrationStatus = Literal["not_installed", "installed", "degraded", "error"]


@dataclass
class IntegrationCapability:
    """A capability exposed by an integration."""

    name: str
    kind: str  # provider, exporter, eval_backend, reward_backend, rollout_backend
    enabled: bool
    experimental: bool = True
    description: str = ""


@dataclass
class IntegrationReport:
    """Status report for an integration."""

    name: str
    status: IntegrationStatus
    version: str | None
    capabilities: list[IntegrationCapability]
    message: str = ""


@dataclass
class RegisteredIntegration:
    """Internal record of a registered integration."""

    name: str
    integration_class: type | None
    provider_types: dict[str, type] = field(default_factory=dict)
    exporters: dict[str, type] = field(default_factory=dict)


class IntegrationRegistry:
    """Central registry for optional ROSClaw integrations.

    Responsibilities:
    - Register provider types and exporter formats contributed by integrations.
    - Produce integration reports on demand (status is dynamic).
    - Allow CLI and other modules to look up factories without hard dependencies.
    """

    def __init__(self) -> None:
        self._integrations: dict[str, RegisteredIntegration] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_integration(self, name: str, integration_class: type) -> None:
        """Register an integration class."""
        registered = self._integrations.get(name)
        if registered is None:
            registered = RegisteredIntegration(
                name=name,
                integration_class=integration_class,
            )
            self._integrations[name] = registered
        else:
            registered.integration_class = integration_class

    def register_provider_type(self, name: str, factory: type) -> None:
        """Register a provider type factory under the owning integration."""
        integration_name = self._guess_integration_name(factory)
        integration = self._integrations.setdefault(
            integration_name,
            RegisteredIntegration(name=integration_name, integration_class=None),
        )
        integration.provider_types[name] = factory

    def register_practice_exporter(self, name: str, factory: type) -> None:
        """Register a practice exporter format factory."""
        integration_name = self._guess_integration_name(factory)
        integration = self._integrations.setdefault(
            integration_name,
            RegisteredIntegration(name=integration_name, integration_class=None),
        )
        integration.exporters[name] = factory

    @staticmethod
    def _guess_integration_name(factory: type) -> str:
        """Infer integration name from the module path of the factory."""
        module = getattr(factory, "__module__", "")
        parts = module.split(".")
        if "integrations" in parts:
            idx = parts.index("integrations")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        return "unknown"

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    def get_integration(self, name: str) -> IntegrationReport:
        """Return the current report for a registered integration."""
        registered = self._integrations.get(name)
        if registered is None or registered.integration_class is None:
            return IntegrationReport(
                name=name,
                status="not_installed",
                version=None,
                capabilities=[],
                message=f"Integration '{name}' is not registered.",
            )
        return registered.integration_class.report()

    def list_integrations(self) -> list[IntegrationReport]:
        """Return reports for all registered integrations."""
        return [
            self.get_integration(name)
            for name in sorted(self._integrations.keys())
            if self._integrations[name].integration_class is not None
        ]

    def get_provider_factory(self, provider_type: str) -> type | None:
        """Return the provider factory for ``provider_type`` if registered."""
        for integration in self._integrations.values():
            factory = integration.provider_types.get(provider_type)
            if factory is not None:
                return factory
        return None

    def get_exporter_factory(self, exporter_name: str) -> type | None:
        """Return the exporter factory for ``exporter_name`` if registered."""
        for integration in self._integrations.values():
            factory = integration.exporters.get(exporter_name)
            if factory is not None:
                return factory
        return None

    def list_provider_types(self) -> dict[str, type]:
        """Return all registered provider types."""
        result: dict[str, type] = {}
        for integration in self._integrations.values():
            result.update(integration.provider_types)
        return result

    def list_exporters(self) -> dict[str, type]:
        """Return all registered exporter formats."""
        result: dict[str, type] = {}
        for integration in self._integrations.values():
            result.update(integration.exporters)
        return result


# Module-level singleton used by CLI and other consumers.
GLOBAL_INTEGRATION_REGISTRY = IntegrationRegistry()
