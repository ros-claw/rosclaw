"""Regex-based text redactor for PII, secrets, paths, and robot serials."""

from __future__ import annotations

import re


class TextRedactor:
    """Redact sensitive substrings from free text."""

    PATTERNS = {
        "email": (re.compile(r"[\w.-]+@[\w.-]+\.[\w]{2,}"), "[REDACTED_EMAIL]"),
        "url": (re.compile(r"https?://[^\s'\"<>]+"), "[REDACTED_URL]"),
        "jwt": (re.compile(r"\beyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\b"), "[REDACTED_TOKEN]"),
        "api_key": (re.compile(r"\b(?:api[_-]?key|token|secret|password|passwd|pwd)\s*[:=]\s*['\"]?[\w-]{8,}['\"]?", re.IGNORECASE), "[REDACTED_SECRET]"),
        "phone": (re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}"), "[REDACTED_PHONE]"),
        "ipv4": (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[REDACTED_IP]"),
        "ipv6": (re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b"), "[REDACTED_IP]"),
        "path": (re.compile(r"(?:^|\s)([\w.-]+)?(/(?:[^\s\":;|<>]*))+"), "[REDACTED_PATH]"),
        "username": (re.compile(r"\b(?:user|username)\s*[:=]\s*\S+", re.IGNORECASE), "[REDACTED_USERNAME]"),
        "hostname": (re.compile(r"\b(?:host|hostname)\s*[:=]\s*\S+", re.IGNORECASE), "[REDACTED_HOSTNAME]"),
        "robot_serial": (re.compile(r"\b(?:serial|sn)\s*[:=]\s*[A-Za-z0-9-]{6,}", re.IGNORECASE), "[REDACTED_SERIAL]"),
    }

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def redact(self, text: str | None) -> str:
        if not text or not self.enabled:
            return text or ""
        result = text
        for _name, (pattern, replacement) in self.PATTERNS.items():
            result = pattern.sub(replacement, result)
        return result
