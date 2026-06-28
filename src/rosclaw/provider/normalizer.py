"""Provider result normalization for perception and risk outputs.

Turns raw provider text (e.g. from a Cosmos-style VLM/reasoning endpoint) into a
stable structured result. When the response already matches the expected schema,
it is validated and returned. Otherwise a safe fallback structure is produced
with ``schema_valid=False`` and ``requires_guard=True``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NormalizedProviderResult:
    """Canonical normalized provider result."""

    schema_valid: bool = False
    fallback_parse: bool = False
    capability: str = ""
    scene: str = ""
    objects: list[dict[str, Any]] = field(default_factory=list)
    physical_risks: list[dict[str, Any]] = field(default_factory=list)
    risk_score: float | None = None
    executable: bool = False
    requires_guard: bool = True
    reasoning: str = ""
    raw: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_valid": self.schema_valid,
            "fallback_parse": self.fallback_parse,
            "capability": self.capability,
            "scene": self.scene,
            "objects": self.objects,
            "physical_risks": self.physical_risks,
            "risk_score": self.risk_score,
            "executable": self.executable,
            "requires_guard": self.requires_guard,
            "reasoning": self.reasoning,
            "raw": self.raw,
            "reason": self.reason,
        }


class ProviderResultNormalizer:
    """Normalize raw provider outputs into a stable risk/ perception schema."""

    # Heuristic patterns for extracting a 0-1 risk score from unstructured text.
    _RISK_SCORE_PATTERNS = [
        re.compile(r"risk[_\s]?score[:\s=]*(\d+(?:\.\d+)?)", re.IGNORECASE),
        re.compile(r"risk[:\s=]*(\d+(?:\.\d+)?)\s*/\s*10", re.IGNORECASE),
        re.compile(r"overall risk[:\s=]*(high|medium|low)", re.IGNORECASE),
        re.compile(r"\b(high|medium|low)\s+risk\b", re.IGNORECASE),
    ]

    @classmethod
    def normalize(
        cls,
        raw_text: str,
        capability: str = "vlm.risk_assessment",
    ) -> NormalizedProviderResult:
        """Normalize ``raw_text`` into a ``NormalizedProviderResult``."""
        # 1. Try to parse the text as JSON.
        parsed = cls._extract_json(raw_text)
        if parsed is not None:
            return cls._from_dict(parsed, raw_text, capability)

        # 2. Fallback: produce a guarded result from heuristic extraction.
        return cls._fallback(raw_text, capability)

    @classmethod
    def _extract_json(cls, text: str) -> dict[str, Any] | None:
        """Extract the first JSON object from a string, if any."""
        text = text.strip()
        if not text:
            return None
        # Direct JSON.
        if text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
        # JSON inside markdown fences.
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1))
            except json.JSONDecodeError:
                pass
        # First curly-brace object.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None

    @classmethod
    def _from_dict(
        cls,
        data: dict[str, Any],
        raw_text: str,
        capability: str,
    ) -> NormalizedProviderResult:
        """Populate a normalized result from a parsed dict."""
        result = NormalizedProviderResult(raw=raw_text, capability=capability)

        # Accept either nested "normalized" or top-level schema.
        if "normalized" in data and isinstance(data["normalized"], dict):
            data = data["normalized"]

        required_keys = {"physical_risks", "risk_score", "requires_guard"}
        present_keys = set(data.keys())
        if not required_keys.issubset(present_keys):
            result.fallback_parse = True
            result.reason = f"Parsed JSON missing required keys: {sorted(required_keys - present_keys)}"
            return cls._merge_fallback(data, result)

        result.schema_valid = True
        result.fallback_parse = False
        result.scene = str(data.get("scene", ""))
        result.objects = cls._as_object_list(data.get("objects"))
        result.physical_risks = cls._as_risk_list(data.get("physical_risks"))
        result.risk_score = cls._as_float(data.get("risk_score"))
        result.executable = bool(data.get("executable", False))
        result.requires_guard = bool(data.get("requires_guard", True))
        result.reasoning = str(data.get("reasoning", data.get("explanation", "")))
        result.reason = "Provider output matched expected risk schema"
        return result

    @classmethod
    def _merge_fallback(
        cls,
        data: dict[str, Any],
        base: NormalizedProviderResult,
    ) -> NormalizedProviderResult:
        """Use any partial structured data that was available."""
        base.scene = str(data.get("scene", base.scene))
        base.objects = cls._as_object_list(data.get("objects", base.objects))
        base.physical_risks = cls._as_risk_list(data.get("physical_risks", base.physical_risks))
        if "risk_score" in data:
            base.risk_score = cls._as_float(data.get("risk_score"))
        if "requires_guard" in data:
            base.requires_guard = bool(data.get("requires_guard"))
        if "executable" in data:
            base.executable = bool(data.get("executable"))
        base.fallback_parse = True
        base.schema_valid = False
        base.requires_guard = True
        if not base.reason:
            base.reason = "Partial JSON; enforcing guard"
        return base

    @classmethod
    def _fallback(cls, raw_text: str, capability: str) -> NormalizedProviderResult:
        result = NormalizedProviderResult(raw=raw_text, capability=capability)
        result.fallback_parse = True
        result.schema_valid = False
        result.requires_guard = True
        result.risk_score = cls._heuristic_risk_score(raw_text)
        result.reason = "Provider output did not match expected schema; heuristic fallback used"
        # Try to extract a short scene summary from the first sentence.
        first_sentence = re.split(r"(?<=[.!?])\s+", raw_text.strip())
        result.scene = first_sentence[0][:240] if first_sentence else ""
        return result

    @classmethod
    def _heuristic_risk_score(cls, text: str) -> float | None:
        for pattern in cls._RISK_SCORE_PATTERNS:
            match = pattern.search(text)
            if match:
                value = match.group(1)
                if value.lower() == "high":
                    return 0.8
                if value.lower() == "medium":
                    return 0.5
                if value.lower() == "low":
                    return 0.2
                try:
                    score = float(value)
                    # Normalize 0-10 scale to 0-1.
                    if score > 1.0:
                        score = score / 10.0
                    return round(max(0.0, min(1.0, score)), 2)
                except ValueError:
                    continue
        return None

    @staticmethod
    def _as_object_list(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [dict(o) for o in value if isinstance(o, dict)]
        return []

    @staticmethod
    def _as_risk_list(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [dict(o) for o in value if isinstance(o, dict)]
        return []

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
