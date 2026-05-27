"""ROSClaw Provider - Canonical response envelope."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderResponse:
    """Standard response envelope for all ROSClaw provider invocations.

    Attributes:
        request_id: Mirrors the request_id.
        provider: Name of the provider that produced this response.
        capability: Capability name that was invoked.
        result: Structured result dict (modality-specific).
        confidence: Model confidence (0.0 - 1.0).
        evidence: Supporting evidence (attention maps, reasoning traces, etc.).
        latency_ms: End-to-end latency in milliseconds.
        model_info: Model name, version, runtime backend, etc.
        trace: Router decision, fallback usage, etc.
        warnings: Non-fatal warnings.
        errors: Fatal errors (if status is not ok).
        status: "ok" | "degraded" | "failed" | "blocked".
    """

    request_id: str
    provider: str
    capability: str
    result: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: int | None = None
    model_info: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    status: str = "ok"

    @property
    def is_ok(self) -> bool:
        return self.status == "ok" and not self.errors

    @property
    def is_degraded(self) -> bool:
        return self.status == "degraded" or bool(self.warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "provider": self.provider,
            "capability": self.capability,
            "result": self.result,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "warnings": self.warnings,
            "errors": self.errors,
        }
