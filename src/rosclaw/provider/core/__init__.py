"""ROSClaw Provider Core.

Public API for provider infrastructure.
"""

from rosclaw.provider.core.capability import (
    CAPABILITY_CATALOG,
    Capability,
    CapabilityDomain,
    is_valid_capability,
    list_capabilities,
)
from rosclaw.provider.core.errors import (
    CapabilityNotSupportedError,
    GuardBlockedError,
    ManifestValidationError,
    ProviderError,
    ProviderNotFoundError,
    ProviderUnavailableError,
    RuntimeAdapterError,
)
from rosclaw.provider.core.manifest import (
    EmbodimentSpec,
    ModelSpec,
    ObservabilitySpec,
    ProviderManifest,
    RuntimeSpec,
    SafetySpec,
)
from rosclaw.provider.core.provider import Provider
from rosclaw.provider.core.readiness import ProviderReadiness
from rosclaw.provider.core.registry import ProviderRegistry
from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse
from rosclaw.provider.core.router import CapabilityRouter, RouterDecision
from rosclaw.provider.core.trace import ProviderTrace, TraceStep

__all__ = [
    "Capability",
    "CapabilityDomain",
    "CAPABILITY_CATALOG",
    "is_valid_capability",
    "list_capabilities",
    "ProviderError",
    "ProviderNotFoundError",
    "ProviderUnavailableError",
    "CapabilityNotSupportedError",
    "GuardBlockedError",
    "ManifestValidationError",
    "RuntimeAdapterError",
    "ProviderManifest",
    "RuntimeSpec",
    "ModelSpec",
    "EmbodimentSpec",
    "SafetySpec",
    "ObservabilitySpec",
    "Provider",
    "ProviderReadiness",
    "ProviderRegistry",
    "ProviderRequest",
    "ProviderResponse",
    "CapabilityRouter",
    "RouterDecision",
    "ProviderTrace",
    "TraceStep",
]
