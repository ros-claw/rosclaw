"""PhysicalReasoner base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rosclaw.provider.core.response import ProviderResponse
from rosclaw.provider.normalizer import ProviderResultNormalizer


class PhysicalReasoner(ABC):
    """Abstract reasoning interface for physical-AI tasks.

    Implementations may talk to Cosmos, Gemini, Qwen, or any other hosted
    reasoning endpoint. Callers only need the three semantic methods below.
    """

    name: str = ""

    @abstractmethod
    def reason(
        self,
        question: str,
        image: str | bytes | None = None,
        image_mime: str = "image/png",
        capability: str = "vlm.risk_assessment",
    ) -> ProviderResponse:
        """Answer ``question`` using optional image input."""
        ...

    @abstractmethod
    def plan(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        capability: str = "reasoning.physical",
    ) -> ProviderResponse:
        """Produce a plan for ``task`` given runtime ``context``."""
        ...

    @abstractmethod
    def analyze(
        self,
        observations: list[dict[str, Any]],
        capability: str = "reasoning.risk_explain",
    ) -> ProviderResponse:
        """Analyze a list of observations and return a decision/explanation."""
        ...

    def _normalized_result(self, raw_text: str, capability: str) -> dict[str, Any]:
        """Normalize raw provider text into the canonical risk schema."""
        normalized = ProviderResultNormalizer.normalize(raw_text, capability=capability)
        return normalized.to_dict()
