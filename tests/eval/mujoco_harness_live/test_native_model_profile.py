"""Ensure model qualification uses the selected PI provider contract."""

import json

from benchmarks.harnessbench.runner import (
    MODEL_PROFILES,
    _prepare_a_leg_env,
    _prepare_home_with_profile,
    has_model_key,
)


def test_native_kimi_requires_real_key(monkeypatch):
    monkeypatch.delenv("ROSCLAW_KIMI_API_KEY", raising=False)
    assert not has_model_key("kimi-coding")
    monkeypatch.setenv("ROSCLAW_KIMI_API_KEY", "test-secret-not-written")
    assert has_model_key("kimi-coding")


def test_native_provider_contract_identical_across_legs(tmp_path, monkeypatch):
    import benchmarks.harnessbench.runner as runner

    monkeypatch.setattr(runner, "_a_leg_python", lambda: "/unused/python")
    profile = MODEL_PROFILES["kimi-coding"]
    work = tmp_path / "a"
    work.mkdir()
    _prepare_a_leg_env(work, profile)
    home, _ = _prepare_home_with_profile(tmp_path / "b", profile)
    a = json.loads((work / ".pi-agent/models.json").read_text())["providers"]["kimi-coding"]
    b = json.loads((home / "agent/models.json").read_text())["providers"]["kimi-coding"]
    assert a == b
    assert a["api"] == "anthropic-messages"
    assert a["apiKey"] == "$ROSCLAW_KIMI_API_KEY"
    assert a["models"][0]["id"] == "kimi-for-coding"
    assert a["models"][0]["reasoning"] is True


def test_local_profile_still_uses_completions_without_remote_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ROSCLAW_KIMI_API_KEY", raising=False)
    assert has_model_key("deepseekv4")
    home, _ = _prepare_home_with_profile(tmp_path, MODEL_PROFILES["deepseekv4"])
    provider = json.loads((home / "agent/models.json").read_text())["providers"]["local-vllm"]
    assert provider["api"] == "openai-completions"
