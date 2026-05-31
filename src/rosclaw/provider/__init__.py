"""ROSClaw Provider - Capability Provider Layer for Physical AI Agents.

rosclaw-provider turns heterogeneous models and robot skills into
standardized, routable, observable, and safety-guarded capabilities.

Public API:
    Provider              - ABC for all capability providers
    ProviderRegistry      - Register / discover / health-check providers
    CapabilityRouter      - Route requests to the best provider
    ProviderRequest       - Canonical request envelope
    ProviderResponse      - Canonical response envelope
    ProviderManifest      - Load provider.yaml manifests
    RuntimeAdapter        - Bridge to inference backends
    Guard                 - Output safety validation

Example:
    from rosclaw_provider import ProviderRegistry, CapabilityRouter, ProviderRequest

    registry = ProviderRegistry()
    router = CapabilityRouter(registry)

    request = ProviderRequest(
        request_id="req_001",
        capability="vlm.object_grounding",
        inputs={"image": {"type": "file", "value": "tabletop.jpg"}, "query": "red cup"},
        context={"robot": "ur5e", "scene": "tabletop"},
        constraints={"latency_ms": 500, "safety_level": "high"},
    )

    response = await router.invoke(request)
    import logging
    logging.getLogger("rosclaw.provider").info("%s", response.result)
"""

from rosclaw.provider.core import (
    Capability,
    CapabilityDomain,
    CapabilityRouter,
    GuardBlockedError,
    Provider,
    ProviderManifest,
    ProviderRegistry,
    ProviderRequest,
    ProviderResponse,
    ProviderTrace,
    RouterDecision,
    RuntimeAdapterError,
)

__version__ = "0.1.0"

__all__ = [
    "Capability",
    "CapabilityDomain",
    "CapabilityRouter",
    "GuardBlockedError",
    "Provider",
    "ProviderManifest",
    "ProviderRegistry",
    "ProviderRequest",
    "ProviderResponse",
    "ProviderTrace",
    "RouterDecision",
    "RuntimeAdapterError",
]
