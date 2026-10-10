"""Operator public boundary oracle calls production Python CLI; no product implementation."""

import json

import pytest

from rosclaw.agentd import cli

BASE = {
    "version": 1,
    "lifetime": "runtime",
    "inclusiveInputLimit": 1000,
    "outputLimit": 100,
    "deliveryReserveInput": 100,
    "deliveryReserveOutput": 10,
}
INVALID = [
    None,
    False,
    [],
    {},
    dict(BASE, extra=1),
    dict(BASE, version=2),
    dict(BASE, lifetime="session"),
    dict(BASE, inclusiveInputLimit=True),
    dict(BASE, inclusiveInputLimit=0),
    dict(BASE, inclusiveInputLimit=1.5),
    dict(BASE, outputLimit=-1),
    dict(BASE, outputLimit=9007199254740992),
    dict(BASE, deliveryReserveInput=1001),
    dict(BASE, deliveryReserveOutput=101),
]


@pytest.mark.parametrize("value", INVALID)
def test_operator_usage_bad_policy_prehome_auth_node(tmp_path, monkeypatch, value):
    p = tmp_path / "policy.json"
    p.write_text(json.dumps({"allowedTools": [], "modelUsageAwareness": value}))
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("EARLY_VALIDATION_FAILED")

    monkeypatch.setattr(cli, "_home", forbidden)
    monkeypatch.setattr(cli, "_cmd_chat_impl", forbidden)
    args = cli.build_parser().parse_args(["chat", "--tool-call-policy", str(p)])
    assert cli.cmd_chat(args) == 2
    assert not calls
    assert list(tmp_path.iterdir()) == [p]


@pytest.mark.parametrize("amount", [1000, 1000.0])
def test_operator_usage_valid_cli_canonical_samefile(tmp_path, monkeypatch, amount):
    p = tmp_path / "policy.json"
    config = dict(BASE, inclusiveInputLimit=amount)
    p.write_text(json.dumps({"allowedTools": [], "modelUsageAwareness": config}))
    calls = []

    def got(args, home):
        calls.append(args.tool_call_policy)
        return 23

    monkeypatch.setattr(cli, "_cmd_chat_impl", got)
    args = cli.build_parser().parse_args(["chat", "--tool-call-policy", str(p)])
    assert cli.cmd_chat(args) == 23
    assert calls == [str(p.resolve())]
    assert cli._validate_tool_call_policy(str(p)) == p.resolve()
