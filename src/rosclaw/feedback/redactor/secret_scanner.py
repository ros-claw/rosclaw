"""Detect likely secrets in text before upload."""

from __future__ import annotations

import re


class SecretScanner:
    """Quick heuristic scan for API keys and tokens."""

    PATTERNS = {
        "anthropic_key": re.compile(r"sk-ant-[\w-]{32,}", re.IGNORECASE),
        "openai_key": re.compile(r"sk-[\w-]{32,}", re.IGNORECASE),
        "aws_key": re.compile(r"AKIA[\w]{16}", re.IGNORECASE),
        "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\b"),
        "generic_secret": re.compile("(?:api[_-]?key|secret|token|password)\\s*[:=]\\s*['\"]?[\\w-]{16,}['\"]?", re.IGNORECASE),
    }

    def find_secrets(self, text: str) -> list[dict[str, str]]:
        results = []
        for name, pattern in self.PATTERNS.items():
            for match in pattern.finditer(text):
                results.append({"type": name, "position": str(match.start())})
        return results

    def has_secrets(self, text: str) -> bool:
        return bool(self.find_secrets(text))
