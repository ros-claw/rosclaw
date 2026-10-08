"""Public parser/launcher tests. Synthetic state only; all processes/network mocked."""

import argparse
import json
import subprocess
from unittest.mock import Mock

import pytest

from rosclaw.agentd import cli


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    import socket

    monkeypatch.setattr(socket, "create_connection", Mock(side_effect=AssertionError("network")))
    monkeypatch.setattr(subprocess, "call", Mock(side_effect=AssertionError("paid launcher")))
    monkeypatch.setattr(subprocess, "Popen", Mock(side_effect=AssertionError("process")))


def parser():
    root = argparse.ArgumentParser()
    cli.add_agent_subparsers(root.add_subparsers(dest="command"))
    return root


@pytest.mark.parametrize(
    "surface", [["chat"], ["chat", "--continue"], ["continue"], ["resume", "approved-title"]]
)
def test_public_surfaces_forward_exact_pair(surface, monkeypatch):
    args = parser().parse_args(surface + ["--provider", "openai-codex", "--model", "gpt-6.1-sol"])
    seen = []
    monkeypatch.setattr(
        cli, "_chat_pi", lambda home, a: seen.append(cli._model_override_argv(a)) or 0
    )
    assert args.func(args) == 0
    assert seen == [["--provider", "openai-codex", "--model", "gpt-6.1-sol"]]
    if surface[0] == "continue" or "--continue" in surface:
        assert args.continue_last
    if surface[0] == "resume":
        assert args.resume == "approved-title"


@pytest.mark.parametrize(
    "flags",
    [
        ["--provider", "openai-codex"],
        ["--model", "gpt-6.1-sol"],
        ["--provider", "", "--model", "x"],
    ],
)
def test_partial_pair_rejected_before_home_or_service(flags, monkeypatch):
    args = parser().parse_args(["chat"] + flags)
    monkeypatch.setattr(cli, "_home", Mock(side_effect=AssertionError("home accessed")))
    monkeypatch.setattr(cli, "AgentService", Mock(side_effect=AssertionError("kernel")))
    assert cli.cmd_chat(args) == 2


def test_ordinary_continue_does_not_imply_global_override(monkeypatch):
    args = parser().parse_args(["continue"])
    monkeypatch.setattr(cli, "cmd_chat", lambda a: 17)
    assert cli.cmd_continue(args) == 17
    assert cli._model_override_argv(args) == []
    assert args.continue_last is True


@pytest.mark.parametrize(
    "error", ["UNKNOWN_MODEL_PROVIDER", "UNKNOWN_MODEL_TARGET", "MODEL_AUTH_UNAVAILABLE"]
)
def test_native_authority_errors_before_kernel(tmp_path, monkeypatch, error):
    args = parser().parse_args(["chat", "--provider", "synthetic", "--model", "target"])
    monkeypatch.setattr(cli, "_find_pi_agent_entry", lambda **kw: ("node", "compiled-main"))
    monkeypatch.setattr(cli, "AgentService", Mock(side_effect=AssertionError("kernel constructed")))
    monkeypatch.setattr(
        cli, "_route_internal_diagnostics_to_log", Mock(side_effect=AssertionError("logs written"))
    )
    called = []

    def native(cmd, **kw):
        called.append(cmd)
        assert kw["env"]["PI_OFFLINE"] == "1"
        return subprocess.CompletedProcess(cmd, 2, "", error)

    monkeypatch.setattr(subprocess, "run", native)
    assert cli._chat_pi(tmp_path, args) == 2
    assert called == [
        [
            "node",
            "compiled-main",
            "--validate-model-selection",
            "--provider",
            "synthetic",
            "--model",
            "target",
        ]
    ]
    assert not (tmp_path / "agentd").exists()


def test_native_response_must_match_exact_target(tmp_path, monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(
            a[0], 0, json.dumps({"provider": "other", "model": "x"}), ""
        ),
    )
    with pytest.raises(ValueError, match="MODEL_SELECTION_MISMATCH"):
        cli._validate_native_model_selection(
            "node", "main", tmp_path, ["--provider", "p", "--model", "m"]
        )


def test_no_python_provider_catalog_or_auto_configuration(monkeypatch):
    args = parser().parse_args(["chat", "--provider", "custom-provider", "--model", "custom-model"])
    monkeypatch.setattr(cli, "_chat_pi", lambda h, a: 9)
    import rosclaw.agentd.pi_config as config

    monkeypatch.setattr(
        config, "pi_model_configured", Mock(side_effect=AssertionError("global defaults"))
    )
    assert cli._cmd_chat_impl(args, __import__("pathlib").Path("/nonexistent/synthetic")) == 9
