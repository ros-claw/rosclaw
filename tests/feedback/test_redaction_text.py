"""Tests for TextRedactor."""

from __future__ import annotations

from rosclaw.feedback.redactor.text_redactor import TextRedactor


class TestTextRedactor:
    def test_redacts_email(self) -> None:
        r = TextRedactor()
        assert r.redact("contact me at foo@example.com please") == "contact me at [REDACTED_EMAIL] please"

    def test_redacts_ipv4(self) -> None:
        r = TextRedactor()
        assert "[REDACTED_IP]" in r.redact("server at 192.168.1.1")

    def test_redacts_url(self) -> None:
        r = TextRedactor()
        assert r.redact("see https://example.com/path") == "see [REDACTED_URL]"

    def test_redacts_secret(self) -> None:
        r = TextRedactor()
        assert "[REDACTED_SECRET]" in r.redact("api_key=abc1234567890abcdef")

    def test_redacts_serial(self) -> None:
        r = TextRedactor()
        assert "[REDACTED_SERIAL]" in r.redact("serial=SN123456789")

    def test_disabled_returns_original(self) -> None:
        r = TextRedactor(enabled=False)
        text = "foo@example.com"
        assert r.redact(text) == text
