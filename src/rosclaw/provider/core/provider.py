"""ROSClaw Provider - Provider ABC.

All capability providers (LLM, VLM, VLA, Skill, etc.) implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Any

from rosclaw.provider.core.errors import CapabilityNotSupportedError
from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse


class Provider(ABC):
    """Abstract base for all ROSClaw capability providers.

    A Provider is NOT a model wrapper. It is a capability executor.
    The same underlying model may serve multiple capabilities.
    The same capability may be served by multiple providers.

    Lifecycle (managed by ProviderRegistry):
        1. load(manifest) -> classmethod factory
        2. health() -> checked periodically
        3. infer(request) -> per-request invocation
        4. unload() -> cleanup

    Example:
        class QwenVLProvider(Provider):
            name = "qwen_vl_provider"
            version = "0.1.0"
            capabilities = ["vlm.object_grounding", "vlm.scene_understanding"]

            async def infer(self, request) -> ProviderResponse:
                ...
    """

    # Override in subclass
    name: str = ""
    version: str = ""
    capabilities: list[str] = []

    def __init__(self, manifest: ProviderManifest):
        if not self.name:
            self.name = manifest.name
        if not self.version:
            self.version = manifest.version
        if not self.capabilities:
            self.capabilities = list(manifest.capabilities)
        self.manifest = manifest
        self._healthy: bool = False
        self._load_error: str = ""

    # ------------------------------------------------------------------
    # Lifecycle hooks (optional overrides)
    # ------------------------------------------------------------------
    async def load(self) -> None:
        """Load model weights, connect to runtime, warm up."""
        pass

    async def unload(self) -> None:
        """Release resources, disconnect."""
        pass

    # ------------------------------------------------------------------
    # Required implementations
    # ------------------------------------------------------------------
    @abstractmethod
    async def infer(self, request: ProviderRequest) -> ProviderResponse:
        """Execute the capability requested in `request`.

        The provider MUST validate that it supports `request.capability`
        before execution. If not supported, raise CapabilityNotSupportedError.
        """
        ...

    async def health(self) -> dict[str, Any]:
        """Return health status.

        Default implementation returns basic metadata.
        Subclasses SHOULD override with a lightweight probe
        (e.g., a minimal inference or ping to the runtime endpoint).
        """
        return {
            "ok": self._healthy,
            "provider": self.name,
            "version": self.version,
            "capabilities": self.capabilities,
            "load_error": self._load_error,
        }

    async def describe(self) -> dict[str, Any]:
        """Return detailed provider description for introspection."""
        health = await self.health()
        return {
            **health,
            "manifest": self.manifest.to_dict(),
        }

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------
    def _ensure_capability_supported(self, capability: str) -> None:
        if capability not in self.capabilities:
            raise CapabilityNotSupportedError(
                f"Provider '{self.name}' does not support capability '{capability}'",
                provider=self.name,
            )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @classmethod
    def from_manifest(cls, manifest: ProviderManifest) -> "Provider":
        """Factory: create from manifest. Subclasses may override."""
        return cls(manifest)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} capabilities={self.capabilities}>"
