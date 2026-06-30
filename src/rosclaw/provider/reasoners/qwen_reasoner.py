"""Alibaba Qwen physical reasoner stub."""

from __future__ import annotations

from typing import Any

from rosclaw.provider.core.response import ProviderResponse
from rosclaw.provider.reasoners.base import PhysicalReasoner


class QwenReasoner(PhysicalReasoner):
    """Stub for Alibaba Qwen-VL physical reasoning backend."""

    def __init__(self, endpoint: str | None = None, name: str = "qwen") -> None:
        self.name = name
        self.endpoint = endpoint

    def reason(
        self,
        question: str,
        image: str | bytes | None = None,
        image_mime: str = "image/png",
        capability: str = "vlm.risk_assessment",
    ) -> ProviderResponse:
        return self._stub_response("reason")

    def plan(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        capability: str = "reasoning.physical",
    ) -> ProviderResponse:
        return self._stub_response("plan")

    def analyze(
        self,
        observations: list[dict[str, Any]],
        capability: str = "reasoning.risk_explain",
    ) -> ProviderResponse:
        return self._stub_response("analyze")

    def _stub_response(self, method: str) -> ProviderResponse:
        return ProviderResponse(
            request_id="",
            provider=self.name,
            capability="stub",
            result={"status": "stub", "endpoint": self.endpoint},
            status="failed",
            errors=[f"{self.name} reasoner is not yet implemented (method={method})"],
        )
