"""Prompt abstraction: keep only summary/hash metadata, never raw text."""

from __future__ import annotations

import hashlib
import re


class PromptRedactor:
    """Convert a raw prompt into a privacy-safe summary."""

    LENGTH_BUCKETS = [
        (0, "empty"),
        (100, "short"),
        (1000, "medium"),
        (5000, "long"),
        (20000, "very_long"),
    ]

    def redact(self, prompt: str | None) -> dict[str, object]:
        if not prompt:
            return {
                "prompt_hash": None,
                "prompt_length_bucket": "empty",
                "language": "unknown",
                "intent_summary": None,
                "tool_names": [],
                "safety_tags": [],
            }

        text = str(prompt)
        length = len(text)
        bucket = "very_long"
        for limit, name in self.LENGTH_BUCKETS:
            if length <= limit:
                bucket = name
                break

        return {
            "prompt_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            "prompt_length_bucket": bucket,
            "language": self._detect_language(text),
            "intent_summary": self._intent_summary(text),
            "tool_names": self._extract_tool_names(text),
            "safety_tags": self._extract_safety_tags(text),
        }

    @staticmethod
    def _detect_language(text: str) -> str:
        # Very rough heuristic for telemetry only.
        if re.search(r"[\u4e00-\u9fff]", text):
            return "zh"
        if re.search(r"[\u3040-\u309f\u30a0-\u30ff]", text):
            return "ja"
        return "en"

    @staticmethod
    def _intent_summary(text: str) -> str | None:
        lowered = text.lower()
        intents = []
        for keyword in ("move", "grasp", "navigate", "stop", "run", "start", "doctor", "status"):
            if keyword in lowered:
                intents.append(keyword)
        return ",".join(intents) if intents else None

    @staticmethod
    def _extract_tool_names(text: str) -> list[str]:
        return sorted(set(re.findall(r"`([\w_]+)`", text)))

    @staticmethod
    def _extract_safety_tags(text: str) -> list[str]:
        tags = []
        lowered = text.lower()
        if any(w in lowered for w in ("emergency", "stop", "halt")):
            tags.append("safety")
        if any(w in lowered for w in ("real robot", "execute", "motion")):
            tags.append("motion")
        return tags
