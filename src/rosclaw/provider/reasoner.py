"""PhysicalReasoner abstraction — skills reason about the world without
knowing which VLM/reasoning model is configured.

The reasoner hides provider transport (HTTP, registry, MCP) behind three
high-level operations:
    * reason(image, question, capability) -> visual/question answering
    * plan(task, context) -> task planning from natural language + state
    * analyze(observations) -> aggregate observations into a decision
"""

from __future__ import annotations

import os

from rosclaw.provider.reasoners.base import PhysicalReasoner
from rosclaw.provider.reasoners.cosmos_reasoner import CosmosReasoner
from rosclaw.provider.reasoners.gemini_reasoner import GeminiReasoner
from rosclaw.provider.reasoners.qwen_reasoner import QwenReasoner

__all__ = [
    "PhysicalReasoner",
    "CosmosReasoner",
    "GeminiReasoner",
    "QwenReasoner",
    "get_reasoner",
]


def get_reasoner(provider_id: str | None = None) -> PhysicalReasoner:
    """Factory: return a PhysicalReasoner for the configured provider.

    Resolution order:
        1. ``provider_id`` if explicitly given (cosmos, gemini, qwen, ...).
        2. ``ROSCLAW_DEFAULT_REASONER`` environment variable.
        3. ``COSMOS_ENDPOINT`` / ``GEMINI_ENDPOINT`` / ``QWEN_ENDPOINT`` env vars.
        4. Cosmos stub with a localhost endpoint.
    """
    pid = (provider_id or os.environ.get("ROSCLAW_DEFAULT_REASONER", "cosmos")).lower()

    if pid in ("cosmos", "gpu_cosmos", "cosmos-reason2-lan"):
        endpoint = os.environ.get("COSMOS_ENDPOINT", "http://localhost:8004")
        return CosmosReasoner(endpoint=endpoint)
    if pid in ("gemini", "gemini-pro"):
        endpoint = os.environ.get("GEMINI_ENDPOINT")
        return GeminiReasoner(endpoint=endpoint)
    if pid in ("qwen", "qwen-vl", "qwen2-vl"):
        endpoint = os.environ.get("QWEN_ENDPOINT")
        return QwenReasoner(endpoint=endpoint)

    # Default fallback: assume cosmos-compatible HTTP endpoint.
    endpoint = os.environ.get("COSMOS_ENDPOINT", "http://localhost:8004")
    return CosmosReasoner(endpoint=endpoint, name=pid)
