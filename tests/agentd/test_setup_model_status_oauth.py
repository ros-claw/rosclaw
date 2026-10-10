"""ROSClaw Native local OAuth status tests; only synthetic, owned homes."""

import json
import time
from pathlib import Path

import pytest

from rosclaw.setup_cli import _model_status

NOW_SECONDS = 1_900_000_000.0
NOW_MS = 1_900_000_000_000
ACCESS = "synthetic-access-DO-NOT-DISCLOSE"
REFRESH = "synthetic-refresh-DO-NOT-DISCLOSE"


def oauth(**changes):
    """Construct PI 1.1 fixture data, not an implementation/classifier."""
    return {
        "type": "oauth",
        "access": ACCESS,
        "refresh": REFRESH,
        "expires": NOW_MS + 60_000,
        **changes,
    }


@pytest.fixture
def local_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "unused-user-home"))
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
    monkeypatch.setattr(time, "time", lambda: NOW_SECONDS)
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "settings.json").write_text(
        json.dumps(
            {
                "defaultProvider": "openai-codex",
                "defaultModel": "synthetic-model",
                "unrelated": {"keep": True},
                "retry": {"maxRetries": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (agent / "models.json").write_bytes(b'{"providers":{}}\n')
    (agent / "sessions").mkdir()
    (agent / "sessions" / "recorded.jsonl").write_bytes(b'{"synthetic":"keep"}\n')
    (tmp_path / "config.yaml").write_bytes(b"synthetic: unchanged\n")
    return tmp_path


def write_auth(home, data):
    (home / "agent" / "auth.json").write_text(json.dumps(data) + "\n", encoding="utf-8")


def snapshot(home):
    return {p.relative_to(home): p.read_bytes() for p in home.rglob("*") if p.is_file()}


class LocalOnly:
    """Tripwires around the original entry: no writes, probes or external reads."""

    def __init__(self, home, monkeypatch):
        self.home = home
        self.context = monkeypatch.context()

    def __enter__(self):
        self.before = snapshot(self.home)
        allowed = {
            self.home / "agent" / name for name in ("settings.json", "models.json", "auth.json")
        }
        real_open = open

        def forbidden(*args, **kwargs):
            pytest.fail("local status attempted a forbidden effect")

        def read_only_open(file, mode="r", *args, **kwargs):
            assert Path(file) in allowed, "status read outside its local metadata files"
            assert not any(flag in mode for flag in "wax+"), "status opened a file for mutation"
            return real_open(file, mode, *args, **kwargs)

        guard = self.context.__enter__()
        guard.setattr(Path, "home", forbidden)
        for name in ("write_text", "write_bytes", "mkdir", "unlink", "rename", "replace", "touch"):
            guard.setattr(Path, name, forbidden)
        for name in ("system", "popen", "remove", "unlink", "rename", "replace", "chmod"):
            guard.setattr(f"os.{name}", forbidden)
        for name in ("Popen", "run", "call", "check_call", "check_output"):
            guard.setattr(f"subprocess.{name}", forbidden)
        for name in (
            "doctor",
            "probe_home",
            "pi_probe_home",
            "_component_report",
            "configure_model",
        ):
            guard.setattr(f"rosclaw.agentd.onboarding.{name}", forbidden)
        guard.setattr("rosclaw.agentd.pi_config.credential_source_report", forbidden)
        guard.setattr("rosclaw.agentd.pi_config.write_pi_model_config", forbidden)
        guard.setattr("socket.socket", forbidden)
        guard.setattr("socket.create_connection", forbidden)
        guard.setattr("builtins.open", read_only_open)
        guard.setattr("io.open", read_only_open)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.context.__exit__(exc_type, exc_value, traceback)
        assert snapshot(self.home) == self.before


def local_only(home, monkeypatch):
    return LocalOnly(home, monkeypatch)


def assert_local(result, state, credential_status):
    assert result["state"] == state
    assert result["provider"] == "openai-codex"
    assert result["model"] == "synthetic-model"
    assert result["credential_status"] == credential_status
    assert result["network_verified"] is False
    assert "本地" in result["detail"]
    assert "未联网验证" in result["detail"]
    assert "不证明认证成功或模型权益" in result["detail"]
    rendered = json.dumps(result, ensure_ascii=False)
    for secret in (ACCESS, REFRESH):
        assert secret not in rendered
    assert "fingerprint" not in rendered


@pytest.mark.parametrize(
    "expires,state,credential_status",
    [
        (NOW_MS + 60_000, "READY", "UNEXPIRED_LOCAL"),
        (NOW_MS + 1, "READY", "UNEXPIRED_LOCAL"),
        (NOW_MS + 0.5, "READY", "UNEXPIRED_LOCAL"),
        (NOW_MS - 1, "NEEDS_SETUP", "EXPIRED"),
        (NOW_MS, "NEEDS_SETUP", "EXPIRED"),
        (NOW_SECONDS + 60, "NEEDS_SETUP", "EXPIRED"),
    ],
    ids=["future-ms", "one-ms-future", "fractional-ms", "expired", "boundary", "seconds-not-ms"],
)
def test_expiry_original_entry(local_home, monkeypatch, capsys, expires, state, credential_status):
    write_auth(local_home, {"openai-codex": oauth(expires=expires), "unrelated": {"keep": [1, 2]}})
    with local_only(local_home, monkeypatch):
        first = _model_status(local_home, probe=False)
        second = _model_status(local_home)
    assert first == second
    assert_local(first, state, credential_status)
    if credential_status == "EXPIRED":
        assert "可能刷新" in first["detail"]
        assert "无需一律重新登录" in first["detail"]
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize(
    "changes",
    [
        {"type": "api_key"},
        {"type": None},
        {"expires": True},
        {"expires": False},
        {"expires": str(NOW_MS + 60_000)},
        {"expires": float("nan")},
        {"expires": float("inf")},
        {"expires": float("-inf")},
        {"expires": None},
        {"access": ""},
        {"refresh": ""},
        {"access": "   "},
        {"refresh": "\t"},
        {"access": 123},
        {"refresh": [REFRESH]},
    ],
    ids=[
        "wrong-type",
        "null-type",
        "bool-true",
        "bool-false",
        "string-ms",
        "nan",
        "infinity",
        "negative-infinity",
        "null-expiry",
        "empty-access",
        "empty-refresh",
        "blank-access",
        "blank-refresh",
        "numeric-access",
        "list-refresh",
    ],
)
def test_invalid_fields(local_home, monkeypatch, changes):
    write_auth(local_home, {"openai-codex": oauth(**changes)})
    with local_only(local_home, monkeypatch):
        result = _model_status(local_home)
    assert_local(result, "NEEDS_LOGIN", "INVALID")


@pytest.mark.parametrize("field", ["type", "access", "refresh", "expires"])
def test_missing_required_field(local_home, monkeypatch, field):
    entry = oauth()
    del entry[field]
    write_auth(local_home, {"openai-codex": entry})
    with local_only(local_home, monkeypatch):
        result = _model_status(local_home)
    assert_local(result, "NEEDS_LOGIN", "INVALID")


@pytest.mark.parametrize("entry", [None, [], "synthetic-string", 42, True])
def test_nonmapping_entry(local_home, monkeypatch, entry):
    write_auth(local_home, {"openai-codex": entry})
    with local_only(local_home, monkeypatch):
        result = _model_status(local_home)
    assert_local(result, "NEEDS_LOGIN", "INVALID")


@pytest.mark.parametrize("document", [None, [], [oauth()], ACCESS, 7, True])
def test_nonmapping_document(local_home, monkeypatch, document):
    write_auth(local_home, document)
    with local_only(local_home, monkeypatch):
        result = _model_status(local_home)
    assert_local(result, "NEEDS_LOGIN", "INVALID")


@pytest.mark.parametrize("document", [None, {}, {"openai": oauth()}, {"unrelated": oauth()}])
def test_absent_or_unrelated_provider(local_home, monkeypatch, document):
    if document is not None:
        write_auth(local_home, document)
    with local_only(local_home, monkeypatch):
        result = _model_status(local_home)
        assert _model_status(local_home) == result
    assert_local(result, "NEEDS_LOGIN", "MISSING")


@pytest.mark.parametrize("raw", [b'{"secret":"synthetic-access-DO-NOT-DISCLOSE",', b"\xff\xfe"])
def test_malformed_auth_secret_free(local_home, monkeypatch, capsys, raw):
    (local_home / "agent" / "auth.json").write_bytes(raw)
    with local_only(local_home, monkeypatch):
        result = _model_status(local_home)
    assert_local(result, "NEEDS_LOGIN", "INVALID")
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("error_type", [PermissionError, OSError, ValueError])
def test_read_failure_never_returns_exception(local_home, monkeypatch, capsys, error_type):
    write_auth(local_home, {"openai-codex": oauth()})
    original = Path.read_text
    auth_path = local_home / "agent" / "auth.json"

    def unreadable(path, *args, **kwargs):
        if path == auth_path:
            raise error_type(f"attacker-controlled-error {ACCESS} {REFRESH}")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)
    with local_only(local_home, monkeypatch):
        result = _model_status(local_home)
    assert_local(result, "NEEDS_LOGIN", "INVALID")
    assert "attacker-controlled-error" not in json.dumps(result)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_directory_instead_of_auth_is_invalid(local_home, monkeypatch):
    (local_home / "agent" / "auth.json").mkdir()
    with local_only(local_home, monkeypatch):
        result = _model_status(local_home)
    assert_local(result, "NEEDS_LOGIN", "INVALID")


def test_credentials_are_data_never_commands(local_home, monkeypatch):
    command = "!touch synthetic-should-never-exist"
    write_auth(local_home, {"openai-codex": oauth(access=command, refresh=command)})
    with local_only(local_home, monkeypatch):
        result = _model_status(local_home)
    assert_local(result, "READY", "UNEXPIRED_LOCAL")
    assert command not in json.dumps(result)


def test_unconfigured_still_needs_setup(tmp_path, monkeypatch):
    with local_only(tmp_path, monkeypatch):
        assert _model_status(tmp_path) == {
            "state": "NEEDS_SETUP",
            "detail": "未配置模型——`rosclaw setup model`",
        }


@pytest.mark.parametrize(
    "credential_provider,source,state",
    [
        ("kimi-coding", "pi-auth-file", "READY"),
        ("kimi-code", "env", "READY"),
        ("openai-codex", "pi-auth-file", "NEEDS_SETUP"),
        ("anthropic", "env", "NEEDS_SETUP"),
        ("kimi-coding", "none", "NEEDS_SETUP"),
    ],
)
def test_other_provider_keeps_existing_rules(
    local_home, monkeypatch, credential_provider, source, state
):
    settings = local_home / "agent" / "settings.json"
    settings.write_text(
        json.dumps({"defaultProvider": "kimi-coding", "defaultModel": "kimi-for-coding"})
    )
    calls = []

    def credentials(home):
        calls.append(home)
        return [{"provider": credential_provider, "source": source}]

    monkeypatch.setattr("rosclaw.agentd.pi_config.credential_source_report", credentials)
    before = snapshot(local_home)
    result = _model_status(local_home, probe=False)
    assert calls == [local_home]
    assert result["state"] == state
    assert result["provider"] == "kimi-coding"
    assert "credential_status" not in result
    assert "network_verified" not in result
    assert snapshot(local_home) == before


@pytest.mark.parametrize("status", ["TOOL_READY", "CHAT_READY", "DEGRADED", "NEEDS_LOGIN"])
def test_explicit_probe_remains_doctor_path(local_home, monkeypatch, status):
    calls = []

    def doctor(home):
        calls.append(home)
        return {"status": status, "model": {"provider": "openai-codex", "model": "probe-model"}}

    def no_local_config(*args, **kwargs):
        pytest.fail("probe=True must not take the local metadata path")

    monkeypatch.setattr("rosclaw.agentd.onboarding.doctor", doctor)
    monkeypatch.setattr("rosclaw.agentd.onboarding.read_pi_model_config", no_local_config)
    before = snapshot(local_home)
    result = _model_status(local_home, probe=True)
    assert calls == [local_home]
    assert result == {
        "state": "NEEDS_SETUP" if status == "NEEDS_LOGIN" else "READY",
        "provider": "openai-codex",
        "model": "probe-model",
        "detail": status,
    }
    assert snapshot(local_home) == before


def test_positive_overflow_expiry_invalid(local_home, monkeypatch, capsys):
    write_auth(local_home, {"openai-codex": oauth(expires=10**400)})
    with local_only(local_home, monkeypatch):
        result = _model_status(local_home)
    assert_local(result, "NEEDS_LOGIN", "INVALID")
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_negative_overflow_expiry_invalid(local_home, monkeypatch, capsys):
    write_auth(local_home, {"openai-codex": oauth(expires=-(10**400))})
    with local_only(local_home, monkeypatch):
        result = _model_status(local_home)
    assert_local(result, "NEEDS_LOGIN", "INVALID")
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
