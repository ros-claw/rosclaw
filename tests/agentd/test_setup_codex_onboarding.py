"""Source-only built-in OAuth onboarding; all homes are owned fixtures."""

import argparse
import json

import pytest

from rosclaw.agentd import onboarding
from rosclaw.agentd.cli import build_parser, cmd_init
from rosclaw.setup_cli import _model_status, dispatch_setup_argv


def test_builtin_defaults(tmp_path):
    out = onboarding.configure_model(tmp_path, "openai-codex")
    cfg = onboarding.read_pi_model_config(tmp_path)
    assert cfg.provider == out["provider"] == "openai-codex"
    assert cfg.model == "gpt-5.4"
    assert out["api_key_ref"] == ""
    assert "/login" in out["key_hint"]
    assert not (tmp_path / "agent/models.json").exists()
    assert not (tmp_path / "agent/auth.json").exists()


def test_explicit_model_and_settings(tmp_path):
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "settings.json").write_text(
        json.dumps({"unrelated": [1, 2], "retry": {"maxRetries": 0, "other": True}})
    )
    onboarding.configure_model(tmp_path, "openai-codex", model="synthetic-model")
    cfg = json.loads((agent / "settings.json").read_text())
    assert cfg["defaultModel"] == "synthetic-model"
    assert cfg["unrelated"] == [1, 2]
    assert cfg["retry"] == {"maxRetries": 0, "other": True}


def test_preserves_models_auth_and_recorded_session(tmp_path):
    agent = tmp_path / "agent"
    agent.mkdir()
    files = {
        "models.json": b'{"providers":{}}\n',
        "auth.json": b'{"synthetic":true}\n',
        "recorded.jsonl": b'{"provider":"kimi-coding","model":"kimi-for-coding"}\n',
    }
    for name, raw in files.items():
        (agent / name).write_bytes(raw)
    onboarding.configure_model(tmp_path, "openai-codex")
    for name, raw in files.items():
        assert (agent / name).read_bytes() == raw


@pytest.mark.parametrize(
    "option",
    [
        {"base_url": "https://synthetic.invalid"},
        {"api_key_ref": "env:SYNTHETIC"},
        {"base_url": ""},
        {"api_key_ref": ""},
    ],
)
def test_rejects_custom_options_without_writes(tmp_path, option):
    with pytest.raises(ValueError, match="built-in OAuth"):
        onboarding.configure_model(tmp_path, "openai-codex", **option)
    assert list(tmp_path.iterdir()) == []


def test_init_parser_accepts_builtin():
    args = build_parser().parse_args(["init", "--provider", "openai-codex"])
    assert args.provider == "openai-codex"


def test_setup_routes_shared_init(monkeypatch):
    seen = []
    monkeypatch.setattr("rosclaw.agentd.cli.cmd_init", lambda args: seen.append(args) or 0)
    assert dispatch_setup_argv(["setup", "model", "--provider", "openai-codex"]) == 0
    assert seen[0].provider == "openai-codex"


def test_local_readiness_no_auth_or_network(tmp_path, monkeypatch):
    onboarding.configure_model(tmp_path, "openai-codex")

    def forbidden(*args, **kwargs):
        pytest.fail("local onboarding must not inspect auth or probe")

    monkeypatch.setattr("rosclaw.agentd.pi_config.credential_source_report", forbidden)
    monkeypatch.setattr(onboarding, "pi_probe_home", forbidden)
    monkeypatch.setattr(onboarding, "_component_report", forbidden)
    assert onboarding.doctor(tmp_path)["status"] == "NEEDS_LOGIN"
    assert _model_status(tmp_path)["state"] == "NEEDS_LOGIN"


def test_init_reports_needs_login(tmp_path, monkeypatch, capsys):
    async def forbidden(*args, **kwargs):
        pytest.fail("init must not probe")

    monkeypatch.setattr(onboarding, "pi_probe_home", forbidden)
    args = argparse.Namespace(
        home=str(tmp_path),
        provider="openai-codex",
        model=None,
        base_url=None,
        api_key_ref=None,
        json=True,
    )
    assert cmd_init(args) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["doctor"]["status"] == "NEEDS_LOGIN"


def test_skip_and_old_builtin_unchanged(tmp_path):
    assert not onboarding.configure_model(tmp_path, "skip")["configured"]
    out = onboarding.configure_model(tmp_path, "kimi-code")
    assert out["provider"] == "kimi-coding"
    assert out["model"] == "kimi-for-coding"
