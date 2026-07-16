"""Provider readiness model that separates discovery from execution trust."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rosclaw.provider.core.manifest import ProviderManifest


@dataclass
class ProviderReadiness:
    """Independent lifecycle and trust facts for one provider."""

    provider: str
    implementation_kind: str
    execution_mode: str
    discovered: bool = True
    registered: bool = True
    loadable: bool = True
    loaded: bool = False
    healthy: bool = False
    executable: bool = False
    authorized: bool = False
    verified: bool = False
    requires_credentials: bool = False
    verified_environment: bool = False
    error: str = ""

    @classmethod
    def from_manifest(
        cls,
        manifest: ProviderManifest,
        *,
        implementation_kind: str | None = None,
    ) -> ProviderReadiness:
        extra = getattr(manifest, "extra", {})
        if not isinstance(extra, dict):
            extra = {}
        kind = implementation_kind or str(extra.get("implementation_kind", ""))
        if not kind:
            kind = "mock" if manifest.name.startswith("mock_") else "native"
        execution_mode = str(extra.get("execution_mode", ""))
        if not execution_mode:
            execution_mode = "FIXTURE" if kind in {"mock", "fixture"} else "DRY_RUN"
        return cls(
            provider=manifest.name,
            implementation_kind=kind,
            execution_mode=execution_mode.upper(),
            requires_credentials=bool(extra.get("requires_credentials", False)),
        )

    def refresh(self, manifest: ProviderManifest) -> None:
        """Recompute derived execution and verification states."""

        synthetic = self.implementation_kind.lower() in {"mock", "fixture", "stub"}
        mode_supports_execution = self.execution_mode in {"SIMULATION", "SHADOW", "REAL"}
        self.executable = bool(
            manifest.safety.executable
            and self.loaded
            and self.healthy
            and self.verified_environment
            and mode_supports_execution
            and not synthetic
        )
        self.verified = self.executable and self.authorized

    @property
    def stage(self) -> str:
        """Return the highest achieved readiness stage."""

        if self.verified:
            return "VERIFIED"
        if self.authorized:
            return "AUTHORIZED"
        if self.executable:
            return "EXECUTABLE"
        if self.healthy:
            return "HEALTHY"
        if self.loaded:
            return "LOADED"
        if self.loadable:
            return "LOADABLE"
        if self.registered:
            return "REGISTERED"
        return "DISCOVERED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "stage": self.stage,
            "implementation_kind": self.implementation_kind,
            "execution_mode": self.execution_mode,
            "discovered": self.discovered,
            "registered": self.registered,
            "loadable": self.loadable,
            "loaded": self.loaded,
            "healthy": self.healthy,
            "executable": self.executable,
            "authorized": self.authorized,
            "verified": self.verified,
            "requires_credentials": self.requires_credentials,
            "verified_environment": self.verified_environment,
            "error": self.error,
        }


__all__ = ["ProviderReadiness"]
