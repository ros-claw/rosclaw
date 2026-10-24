"""Ensure model qualification uses the selected PI provider contract."""

import json

import pytest

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


def test_chatgpt_builtin_selection_is_explicit_and_identical_across_legs(tmp_path, monkeypatch):
    import benchmarks.harnessbench.runner as runner
    from rosclaw.agentd.pi_config import read_pi_model_config

    monkeypatch.setattr(runner, "_a_leg_python", lambda: "/unused/python")
    profile = runner.resolve_model_profile("chatgpt", "operator-selected-model")
    work = tmp_path / "a"
    work.mkdir()
    # An opaque sentinel: builders must neither read nor overwrite credentials.
    auth = work / ".pi-agent" / "auth.json"
    auth.parent.mkdir()
    auth.write_bytes(b"opaque-test-fixture-not-a-real-credential")
    original = auth.read_bytes()
    env = _prepare_a_leg_env(work, profile)
    home, _ = _prepare_home_with_profile(tmp_path / "b", profile)
    a = json.loads((work / ".pi-agent/settings.json").read_text())
    b = json.loads((home / "agent/settings.json").read_text())
    assert (
        a
        == b
        == {
            "defaultProvider": "openai-codex",
            "defaultModel": "operator-selected-model",
        }
    )
    assert env["PI_CODING_AGENT_DIR"] == str(work / ".pi-agent")
    assert not (work / ".pi-agent/models.json").exists()
    assert not (home / "agent/models.json").exists()
    assert "models:" not in (home / "config.yaml").read_text()
    assert auth.read_bytes() == original
    configured = read_pi_model_config(home)
    assert configured.provider == "openai-codex"
    assert configured.api == "openai-codex-responses"
    assert configured.api_key_ref == ""


@pytest.mark.parametrize("model_id", [None, "", " ", " padded-model "])
def test_chatgpt_does_not_guess_a_model(model_id):
    from benchmarks.harnessbench.runner import resolve_model_profile

    with pytest.raises(ValueError, match="BENCH_MODEL_ID_REQUIRED"):
        resolve_model_profile("chatgpt", model_id)


def test_unknown_profile_never_falls_back_to_kimi_credentials(monkeypatch):
    from benchmarks.harnessbench.runner import resolve_model_profile

    monkeypatch.setenv("ROSCLAW_KIMI_API_KEY", "test-only")
    with pytest.raises(ValueError, match="BENCH_MODEL_UNKNOWN"):
        has_model_key("unknown")
    with pytest.raises(ValueError, match="BENCH_MODEL_UNKNOWN"):
        resolve_model_profile("unknown")
    with pytest.raises(ValueError, match="BENCH_MODEL_ID_UNEXPECTED"):
        resolve_model_profile("kimi-coding", "different-model")


def test_chatgpt_not_run_is_not_satisfied_by_api_keys(tmp_path, monkeypatch):
    import benchmarks.harnessbench.runner as runner

    monkeypatch.setenv("ROSCLAW_KIMI_API_KEY", "test-only")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    assert not has_model_key("chatgpt")
    assert "--readonly-oauth-path" in runner.model_credential_issue("chatgpt")
    monkeypatch.setattr(runner, "stage_workspace", lambda *a, **k: pytest.fail("staged a run"))
    with pytest.raises(ValueError, match="BENCH_OAUTH_SOURCE_UNAVAILABLE"):
        runner.run_leg("A", "U01", tmp_path, 1, model="chatgpt", model_id="selected")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("leg", ["A", "B"])
def test_builtin_builder_rejects_existing_shadow_provider_before_writing(
    tmp_path, monkeypatch, leg
):
    import benchmarks.harnessbench.runner as runner

    monkeypatch.setattr(runner, "_a_leg_python", lambda: pytest.fail("prepared environment"))
    agent = tmp_path / (".pi-agent" if leg == "A" else "agent")
    agent.mkdir()
    models = agent / "models.json"
    models.write_text('{"providers":{"openai-codex":{"api":"wrong"}}}')
    before = models.read_bytes()
    profile = runner.resolve_model_profile("chatgpt", "selected")
    with pytest.raises(ValueError, match="BENCH_BUILTIN_PROVIDER_CONFIG_CONFLICT"):
        if leg == "A":
            _prepare_a_leg_env(tmp_path, profile)
        else:
            _prepare_home_with_profile(tmp_path, profile)
    assert models.read_bytes() == before
    assert not (agent / "settings.json").exists()


def test_cli_chatgpt_reports_not_run_without_launch_or_auth_read(tmp_path, monkeypatch, capsys):
    from pathlib import Path

    from scripts import harnessbench_run

    def no_auth_read(path, *args, **kwargs):
        pytest.fail("credential file read attempted")

    monkeypatch.setattr(Path, "read_text", no_auth_read)
    monkeypatch.setattr(harnessbench_run, "run_leg", lambda *a, **k: pytest.fail("model launch"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "harnessbench_run",
            "--model",
            "chatgpt",
            "--model-id",
            "selected",
            "--out",
            str(tmp_path / "out"),
        ],
    )
    assert harnessbench_run.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "NOT_RUN"
    assert result["provider"] == "openai-codex"
    assert result["model_id"] == "selected"
    assert "OAuth" in result["reason"]
    assert "KIMI" not in result["reason"]
    assert not (tmp_path / "out").exists()


def test_cli_missing_kimi_key_names_only_selected_credential(monkeypatch, capsys):
    from scripts import harnessbench_run

    monkeypatch.delenv("ROSCLAW_KIMI_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["harnessbench_run", "--model", "kimi-coding"])
    assert harnessbench_run.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "NOT_RUN"
    assert result["reason"] == "no model credential in environment (ROSCLAW_KIMI_API_KEY)"


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--model", "unknown"], "BENCH_MODEL_UNKNOWN"),
        (["--model", "chatgpt"], "BENCH_MODEL_ID_REQUIRED"),
    ],
)
def test_cli_invalid_selection_is_explicit_error_without_launch(monkeypatch, capsys, argv, message):
    from scripts import harnessbench_run

    monkeypatch.setenv("ROSCLAW_KIMI_API_KEY", "test-only")
    monkeypatch.setattr(harnessbench_run, "run_leg", lambda *a, **k: pytest.fail("model launch"))
    monkeypatch.setattr("sys.argv", ["harnessbench_run", *argv])
    with pytest.raises(SystemExit) as info:
        harnessbench_run.main()
    assert info.value.code == 2
    assert message in capsys.readouterr().err


def test_readonly_preflight_uses_safe_pi_protocol_and_full_trial_validity(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace

    import benchmarks.harnessbench.runner as runner

    entry = tmp_path / "bridge.js"
    entry.touch()
    auth = tmp_path / "operator-dedicated-auth.json"  # never read by Python
    observed = []
    monkeypatch.setattr(runner, "_builtin_bridge_entry", lambda: entry)
    monkeypatch.setattr("shutil.which", lambda _: "/node")

    def check(command, **kwargs):
        observed.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "status": "READY",
                    "provider": "openai-codex",
                    "model_id": "selected",
                    "credential_mode": "EXPLICIT_READONLY_NO_REFRESH",
                    "remote_auth_verified": False,
                }
            ),
        )

    monkeypatch.setattr(subprocess, "run", check)
    assert (
        runner.model_credential_issue(
            "chatgpt", model_id="selected", readonly_oauth_path=auth, settle_timeout=1200
        )
        is None
    )
    command, kwargs = observed[0]
    assert command == ["/node", str(entry), "--check", str(auth), "selected", "1631000"]
    assert kwargs["timeout"] == 30
    assert kwargs["env"]["PI_SKIP_VERSION_CHECK"] == "1"
    assert not auth.exists()


@pytest.mark.parametrize(
    "output,code",
    [
        (
            {"status": "NOT_RUN", "code": "BENCH_OAUTH_RENEW_EXTERNALLY_REQUIRED"},
            "BENCH_OAUTH_RENEW_EXTERNALLY_REQUIRED",
        ),
        (
            {"status": "NOT_RUN", "code": "untrusted-secret-error"},
            "BENCH_READONLY_OAUTH_PREFLIGHT_FAILED",
        ),
        (
            {"status": "READY", "provider": "kimi-coding", "model_id": "selected"},
            "BENCH_READONLY_OAUTH_PREFLIGHT_FAILED",
        ),
        ([], "BENCH_READONLY_OAUTH_PREFLIGHT_FAILED"),
    ],
)
def test_readonly_preflight_cannot_turn_failure_or_provider_substitution_into_ready(
    tmp_path, monkeypatch, output, code
):
    import subprocess
    from types import SimpleNamespace

    import benchmarks.harnessbench.runner as runner

    entry = tmp_path / "bridge.js"
    entry.touch()
    monkeypatch.setattr(runner, "_builtin_bridge_entry", lambda: entry)
    monkeypatch.setattr("shutil.which", lambda _: "/node")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=2, stdout=json.dumps(output))
    )
    assert (
        runner.model_credential_issue(
            "chatgpt", model_id="selected", readonly_oauth_path=tmp_path / "auth.json"
        )
        == code
    )


def test_readonly_source_is_chatgpt_only(monkeypatch, tmp_path):
    import benchmarks.harnessbench.runner as runner

    for model in ["kimi-k3", "kimi-coding", "deepseekv4"]:
        with pytest.raises(ValueError, match="BENCH_OAUTH_PATH_UNEXPECTED"):
            runner.model_credential_issue(model, readonly_oauth_path=tmp_path / "auth.json")


def test_builtin_run_leg_uses_sdk_launcher_and_one_source_with_isolated_trial_sessions(
    tmp_path, monkeypatch
):
    import benchmarks.harnessbench.runner as runner

    observed = []

    class Session:
        def __init__(self, argv, env, cwd=None, log_path=None):
            observed.append((argv, env, cwd))

        def expect(self, marker, timeout):
            observed[-1] += (marker,)

        def send(self, text):
            pass

        def stop(self):
            pass

    source = tmp_path / "dedicated-auth.json"
    source.write_bytes(b"opaque-fixture-no-runtime-token")
    before = source.read_bytes()
    monkeypatch.setattr(runner, "model_credential_issue", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_a_leg_python", lambda: "/unused/python")
    monkeypatch.setattr("tests.agentd.test_product_journey.PtySession", Session)
    monkeypatch.setattr(runner, "_wait_settled", lambda *a, **k: 0)
    monkeypatch.setattr(runner.oracle, "judge", lambda *a, **k: {"task_success": False})
    monkeypatch.setenv("PI_CODING_AGENT_DIR", "/global-must-not-be-used")
    monkeypatch.setenv("ROSCLAW_PI_READONLY_OAUTH_PATH", "/unrequested-global-source")
    records = [
        runner.run_leg(
            leg,
            "U01",
            tmp_path / "runs",
            1,
            model="chatgpt",
            model_id="selected",
            readonly_oauth_path=source,
        )
        for leg in ["A", "B"]
    ]
    a, b = observed
    assert str(runner._builtin_bridge_entry()) in a[0]
    assert a[0][-2:] == [str(source), "selected"]
    assert a[3] == b"HARNESSBENCH_NATIVE_PI_READY"
    assert b[3] == b"ROSClaw Native Agent"
    assert a[1]["PI_CODING_AGENT_DIR"] != b[1]["PI_CODING_AGENT_DIR"]
    assert b[1]["ROSCLAW_PI_READONLY_OAUTH_PATH"] == str(source)
    assert "ROSCLAW_PI_READONLY_OAUTH_PATH" not in a[1]
    for _, env, _, _ in observed:
        agent = __import__("pathlib").Path(env["PI_CODING_AGENT_DIR"])
        assert json.loads((agent / "settings.json").read_text()) == {
            "defaultProvider": "openai-codex",
            "defaultModel": "selected",
        }
        assert not (agent / "auth.json").exists()
        assert not (agent / "models.json").exists()
    assert records[0]["launcher"] == "PI_SDK_INTERACTIVE_MODE"
    assert records[1]["launcher"] == "ROSCLAW_NATIVE_AGENT"
    assert all(
        r["model_id"] == "selected" and r["credential_mode"] == "EXPLICIT_READONLY_NO_REFRESH"
        for r in records
    )
    assert source.read_bytes() == before


@pytest.mark.parametrize("model", ["kimi-k3", "kimi-coding", "deepseekv4"])
def test_existing_b_profiles_do_not_inherit_optin_readonly_chatgpt_source(
    tmp_path, monkeypatch, model
):
    import os

    import benchmarks.harnessbench.runner as runner
    from tests.eval.agent_tier import driver

    captured = []
    sentinel = "/must-not-be-opened/or-used/as-a-legacy-model-source"
    monkeypatch.setenv("ROSCLAW_PI_READONLY_OAUTH_PATH", sentinel)

    class Run:
        def __init__(self, work, **kwargs):
            self.home = work / "rh"
            self.env = dict(os.environ)

    class Session:
        def __init__(self, argv, env, **kwargs):
            captured.append((argv, env))

        def expect(self, *args, **kwargs):
            pass

        def send(self, text):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(driver, "AgentRun", Run)
    monkeypatch.setattr("tests.agentd.test_product_journey.PtySession", Session)
    monkeypatch.setattr(runner, "_wait_settled", lambda *a, **k: 0)
    monkeypatch.setattr(runner.oracle, "judge", lambda *a, **k: {"task_success": False})
    record = runner.run_leg("B", "U01", tmp_path, 1, model=model)
    assert record["model"] == model
    assert len(captured) == 1
    argv, env = captured[0]
    assert argv[1:4] == ["-m", "rosclaw.entrypoint", "chat"]
    assert "ROSCLAW_PI_READONLY_OAUTH_PATH" not in env
    assert os.environ["ROSCLAW_PI_READONLY_OAUTH_PATH"] == sentinel


@pytest.mark.parametrize(
    "relative_source",
    [
        "a_u01_chatgpt_run1/.pi-agent/settings.json",
        "b_u01_chatgpt_run1/rh/agent/settings.json",
        "a_u01_chatgpt_run1/input.json",
        "results.json",
    ],
)
def test_readonly_source_inside_run_output_is_rejected_before_any_stage_or_preflight(
    tmp_path, monkeypatch, relative_source
):
    import benchmarks.harnessbench.runner as runner

    output = tmp_path / "runs"
    source = output / relative_source
    source.parent.mkdir(parents=True)
    source.write_bytes(
        b'{"openai-codex":{"type":"oauth","access":"offline","refresh":"offline","expires":9999999999999}}'
    )
    source.chmod(0o600)
    before = source.read_bytes()
    monkeypatch.setattr(
        runner,
        "model_credential_issue",
        lambda *a, **k: pytest.fail("preflight should not read a colliding source"),
    )
    monkeypatch.setattr(
        runner, "stage_workspace", lambda *a, **k: pytest.fail("stage must not write")
    )
    with pytest.raises(ValueError, match="BENCH_OAUTH_SOURCE_OVERLAPS_WRITABLE_SCOPE"):
        runner.run_leg(
            "A", "U01", output, 1, model="chatgpt", model_id="selected", readonly_oauth_path=source
        )
    assert source.read_bytes() == before


def test_readonly_source_output_alias_is_rejected(tmp_path):
    from benchmarks.harnessbench.runner import validate_readonly_oauth_namespace

    actual = tmp_path / "actual"
    actual.mkdir()
    source = actual / "auth.json"
    source.write_bytes(b"opaque-fixture")
    alias = tmp_path / "alias-output"
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match="BENCH_OAUTH_SOURCE_OVERLAPS_WRITABLE_SCOPE"):
        validate_readonly_oauth_namespace(source, alias)
    assert source.read_bytes() == b"opaque-fixture"


def test_cli_rejects_source_within_output_before_mkdir_or_preflight(tmp_path, monkeypatch, capsys):
    from scripts import harnessbench_run

    output = tmp_path / "out"
    source = output / "auth.json"
    source.parent.mkdir()
    source.write_bytes(b"opaque-fixture")
    before = source.read_bytes()
    monkeypatch.setattr(
        harnessbench_run,
        "model_credential_issue",
        lambda *a, **k: pytest.fail("preflight should not run"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "harnessbench_run",
            "--model",
            "chatgpt",
            "--model-id",
            "selected",
            "--readonly-oauth-path",
            str(source),
            "--out",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as info:
        harnessbench_run.main()
    assert info.value.code == 2
    assert "BENCH_OAUTH_SOURCE_OVERLAPS_WRITABLE_SCOPE" in capsys.readouterr().err
    assert source.read_bytes() == before
    assert list(output.iterdir()) == [source]


def test_builtin_trial_namespace_must_be_fresh_before_preflight(tmp_path, monkeypatch):
    import benchmarks.harnessbench.runner as runner

    source = tmp_path / "separate-auth.json"
    source.write_bytes(b"opaque-fixture")
    output = tmp_path / "out"
    prior_work = output / "a_u01_chatgpt_run1"
    prior_work.mkdir(parents=True)
    (prior_work / "input.json").symlink_to(source)
    monkeypatch.setattr(
        runner, "model_credential_issue", lambda *a, **k: pytest.fail("preflight should not run")
    )
    with pytest.raises(ValueError, match="BENCH_BUILTIN_TRIAL_NAMESPACE_NOT_FRESH"):
        runner.run_leg(
            "A", "U01", output, 1, model="chatgpt", model_id="selected", readonly_oauth_path=source
        )
    assert source.read_bytes() == b"opaque-fixture"


@pytest.mark.parametrize("returncode", [0, 1, -2])
def test_settled_exited_pty_fails_without_waiting_or_retry(tmp_path, monkeypatch, returncode):
    import threading
    from types import SimpleNamespace

    import benchmarks.harnessbench.runner as runner

    session = SimpleNamespace(
        _lock=threading.Lock(),
        output=bytearray(),
        proc=SimpleNamespace(poll=lambda: returncode),
    )
    monkeypatch.setattr(runner, "_session_turn_stopped", lambda *args: False)
    monkeypatch.setattr(runner.time, "sleep", lambda _: pytest.fail("exited process must not wait"))
    with pytest.raises(
        AssertionError, match=f"INFRA_PROCESS_EXITED_BEFORE_SETTLED: exit={returncode}"
    ):
        runner._wait_settled(session, tmp_path, 1200, prompt="unused", session_dir=tmp_path)


def test_settled_persisted_success_precedes_process_exit(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace

    import benchmarks.harnessbench.runner as runner

    session = SimpleNamespace(
        _lock=threading.Lock(), output=bytearray(), proc=SimpleNamespace(poll=lambda: 0)
    )
    monkeypatch.setattr(runner, "_session_turn_stopped", lambda *args: True)
    assert runner._wait_settled(session, tmp_path, 1200, session_dir=tmp_path) == 0


def test_settled_real_pty_child_exit_is_observed(tmp_path):
    import os
    import sys
    import time

    import benchmarks.harnessbench.runner as runner
    from tests.agentd.test_product_journey import PtySession

    session = PtySession([sys.executable, "-c", "raise SystemExit(7)"], dict(os.environ))
    try:
        assert session.proc.wait(timeout=10) == 7
        started = time.monotonic()
        with pytest.raises(AssertionError, match="INFRA_PROCESS_EXITED_BEFORE_SETTLED: exit=7"):
            runner._wait_settled(session, tmp_path, 1200)
        assert time.monotonic() - started < 2
    finally:
        session.stop()
