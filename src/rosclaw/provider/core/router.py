"""ROSClaw Provider - CapabilityRouter.

Selects the best provider for a given request based on capability match,
embodiment compatibility, latency budget, safety level, health, and fallback chains.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from rosclaw.provider.core.errors import ProviderNotFoundError, ProviderUnavailableError
from rosclaw.provider.core.provider import Provider
from rosclaw.provider.core.registry import ProviderRegistry
from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse
from rosclaw.provider.core.trace import ProviderTrace


@dataclass
class RouterDecision:
    """Result of a router selection."""

    selected_provider: str
    reason: str
    fallbacks: list[str] = field(default_factory=list)
    estimated_latency_ms: int | None = None
    score: float = 0.0


class CapabilityRouter:
    """Routes ProviderRequests to the most suitable Provider.

    Routing dimensions (in priority order):
    1. capability match
    2. input modality support
    3. robot embodiment support
    4. latency budget
    5. provider health
    6. safety level compatibility
    7. historical success rate (future)
    8. cost / energy (future)
    9. fallback priority
    """

    def __init__(self, registry: ProviderRegistry):
        self.registry = registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def route(self, request: ProviderRequest) -> RouterDecision:
        """Select the best provider for `request`.

        Raises:
            ProviderNotFoundError: No provider supports this capability.
            ProviderUnavailableError: All matching providers are unhealthy.
        """
        candidates = self.registry.find_by_capability(
            request.capability, healthy_only=False
        )
        if not candidates:
            raise ProviderNotFoundError(
                f"No provider supports capability '{request.capability}'",
                request_id=request.request_id,
            )

        # Filter pass
        scored = []
        for provider in candidates:
            score, reason = self._score_provider(provider, request)
            if score > 0:
                scored.append((score, provider, reason))

        if not scored:
            # No provider passed filters — try without health filter for diagnostics
            healthy = [p for p in candidates if self.registry.is_healthy(p.name)]
            if not healthy:
                raise ProviderUnavailableError(
                    f"All providers for '{request.capability}' are unhealthy",
                    request_id=request.request_id,
                )
            raise ProviderNotFoundError(
                f"No provider passed constraints for '{request.capability}'",
                request_id=request.request_id,
            )

        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0]
        fallbacks = [p.name for _, p, _ in scored[1:3]]

        return RouterDecision(
            selected_provider=best[1].name,
            reason=best[2],
            fallbacks=fallbacks,
            estimated_latency_ms=best[1].manifest.runtime.endpoint and 100 or None,
            score=best[0],
        )

    async def invoke(
        self,
        request: ProviderRequest,
        trace: ProviderTrace | None = None,
    ) -> ProviderResponse:
        """Full invoke pipeline: route -> infer -> fallback if needed.

        Args:
            request: The capability request.
            trace: Optional trace collector.

        Returns:
            ProviderResponse from the selected (or fallback) provider.
        """
        decision = await self.route(request)
        provider = self.registry.get(decision.selected_provider)

        if trace:
            trace.add_step(
                name="route",
                provider=provider.name,
                capability=request.capability,
                latency_ms=0,
                status="success",
                metadata={"reason": decision.reason, "score": decision.score},
            )

        # Try primary
        response = await self._try_infer(provider, request, trace)
        if response.is_ok:
            return response

        # Fallback chain
        for fallback_name in decision.fallbacks:
            fb_provider = self.registry.get(fallback_name)
            fb_response = await self._try_infer(fb_provider, request, trace)
            if fb_response.is_ok:
                fb_response.trace["fallback_used"] = True
                fb_response.trace["primary_failed"] = provider.name
                return fb_response

        # All failed — return primary response (with errors)
        response.trace["fallbacks_exhausted"] = True
        return response

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _score_provider(
        self,
        provider: Provider,
        request: ProviderRequest,
    ) -> tuple[float, str]:
        """Score a provider for a request. Returns (score, reason).

        Score > 0 means the provider is viable.
        Higher is better.
        """
        manifest = provider.manifest
        score = 1.0
        reasons: list[str] = []

        # 1. Capability match (implicit — registry already filtered)
        # 2. Modality support
        input_modality = self._infer_input_modality(request)
        if input_modality and not manifest.supports_input_modality(input_modality):
            return 0.0, f"does not support input modality '{input_modality}'"

        # 3. Robot embodiment
        robot = request.robot_id
        if robot and not manifest.supports_robot(robot):
            return 0.0, f"does not support robot '{robot}'"

        # 4. Latency budget
        latency_budget = request.latency_budget_ms
        if latency_budget and manifest.runtime.endpoint:
            # Heuristic: HTTP endpoints are assumed ~100-500ms depending on model size
            # In production, this should come from historical p99 latency
            estimated = 200 if manifest.runtime.backend in {"ollama", "http"} else 500
            if estimated > latency_budget:
                return 0.0, f"estimated latency {estimated}ms > budget {latency_budget}ms"
            score += (latency_budget - estimated) / latency_budget

        # 5. Health
        if not self.registry.is_healthy(provider.name):
            return 0.0, "unhealthy"

        # 6. Safety level
        safety = request.safety_level
        if safety == "STRICT":
            # Prefer providers with requires_guard=True and executable=False
            if manifest.safety.requires_guard:
                score += 0.5
            if not manifest.safety.executable:
                score += 0.3

        # 7. Cost / energy heuristic (placeholder)
        if manifest.runtime.device == "cpu":
            score += 0.1  # Slight preference for CPU to save GPU

        reasons.append(f"score={score:.2f}")
        return score, "; ".join(reasons)

    @staticmethod
    def _infer_input_modality(request: ProviderRequest) -> str:
        """Infer primary input modality from request inputs."""
        inputs = request.inputs
        if "image" in inputs or "camera_topic" in inputs:
            return "image"
        if "video" in inputs:
            return "video"
        if "text" in inputs or "query" in inputs:
            return "text"
        if "trajectory" in inputs:
            return "trajectory"
        return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _try_infer(
        self,
        provider: Provider,
        request: ProviderRequest,
        trace: ProviderTrace | None,
    ) -> ProviderResponse:
        """Invoke a single provider with timing and error handling."""
        t0 = time.monotonic()
        try:
            response = await provider.infer(request)
        except Exception as e:
            response = ProviderResponse(
                request_id=request.request_id,
                provider=provider.name,
                capability=request.capability,
                status="failed",
                errors=[str(e)],
            )

        response.latency_ms = int((time.monotonic() - t0) * 1000)

        if trace:
            trace.add_step(
                name="infer",
                provider=provider.name,
                capability=request.capability,
                latency_ms=response.latency_ms,
                status="success" if response.is_ok else "failed",
                metadata={"warnings": response.warnings},
            )

        return response
