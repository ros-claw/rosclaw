"""Startup must not silently enable one recovery after an explicit zero."""

import json

import pytest

from rosclaw.agentd.onboarding import _write_retry_budget


@pytest.mark.parametrize("enabled", [True, False])
def test_explicit_zero_survives_startup_byte_for_byte(tmp_path, enabled):
    path = tmp_path / "agent/settings.json"
    path.parent.mkdir()
    settings = {
        "defaultThinkingLevel": "low",
        "retry": {
            "enabled": enabled,
            "maxRetries": 0,
            "provider": {"maxRetries": 0},
            "baseDelayMs": 1500,
        },
    }
    path.write_text(json.dumps(settings))
    before = path.read_bytes()
    _write_retry_budget(tmp_path)
    _write_retry_budget(tmp_path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("value", [3, -1, False, True, 0.0, "0", None])
def test_malformed_or_excessive_retry_budget_still_defaults_to_one(tmp_path, value):
    path = tmp_path / "agent/settings.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"retry": {"maxRetries": value, "enabled": False}}))
    _write_retry_budget(tmp_path)
    actual = json.loads(path.read_text())["retry"]
    assert type(actual["maxRetries"]) is int and actual["maxRetries"] == 1
    assert actual["enabled"] is False
