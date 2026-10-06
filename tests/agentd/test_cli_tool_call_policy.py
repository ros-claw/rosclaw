"""Stage B：--tool-call-policy JSON_FILE CLI plumbing 回归测试。

覆盖：合法 policy（含 allowedTools=[] block-all、空可选 map、JSON 1.0
整数值浮点）通过；全部畸形类别在 home/auth/Node 启动前被
_validate_tool_call_policy 拒绝；argparse 接受该 flag 且默认 None。
Node 侧 policy 行为属 Stage A + A+B 集成范围，不在此 B-only 证明内。
"""

from __future__ import annotations

import json

import pytest

from rosclaw.agentd.cli import (
    _validate_tool_call_policy,
    build_parser,
)


def _write(tmp_path, payload) -> str:
    p = tmp_path / "policy.json"
    p.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    return str(p)


VALID = [
    {"allowedTools": []},
    {"allowedTools": ["bash", "read"]},
    {"allowedTools": [], "maxCalls": {}, "exactCommands": {}, "visibleBudget": False},
    {"allowedTools": ["bash"], "maxCalls": {"bash": 3}, "maxTotalCalls": 10},
    {"allowedTools": ["bash"], "exactCommands": {"bash": ["ls", "pwd"]}},
    # JSON 1.0 整数值有限浮点与 JS Number.isSafeInteger 一致地接受。
    '{"allowedTools": ["bash"], "maxTotalCalls": 1.0}',
    {"allowedTools": [], "visibleBudget": True},
    {"allowedTools": ["__proto__", "constructor"]},  # 保留名为普通 own key
]

INVALID = [
    "{",  # malformed JSON
    "[]",  # root 非 object
    "{}",  # 缺 allowedTools
    {"allowedTools": [], "unknown": 1},  # 未知顶层键
    {"allowedTools": ["bash", "bash"]},  # 重复名
    {"allowedTools": [" "]},  # 空白名
    {"allowedTools": [" bash"]},  # 未 trim
    {"allowedTools": [""]},  # 空名
    {"allowedTools": [1]},  # 非字符串名
    {"allowedTools": ["bash"], "maxTotalCalls": True},  # bool
    {"allowedTools": ["bash"], "maxTotalCalls": 0.5},  # 小数
    {"allowedTools": ["bash"], "maxTotalCalls": -1},  # 负数
    {"allowedTools": ["bash"], "maxTotalCalls": 9007199254740992},  # 非安全整数
    '{"allowedTools": [], "maxTotalCalls": 1e999}',  # 非有限
    '{"allowedTools": [], "maxTotalCalls": NaN}',  # 非标准常量
    '{"allowedTools": [], "maxTotalCalls": Infinity}',
    {"allowedTools": [], "maxCalls": {"bash": 1}},  # 未声明的 maxCalls 键
    {"allowedTools": [], "exactCommands": {"bash": ["ok"]}},  # 未声明的 exactCommands 键
    {"allowedTools": ["bash"], "exactCommands": {"bash": []}},  # 空命令数组
    {"allowedTools": ["bash"], "exactCommands": {"bash": ["ls", "ls"]}},  # 重复命令
    {"allowedTools": ["bash"], "exactCommands": {"bash": [""]}},  # 空命令
    {"allowedTools": [], "visibleBudget": 1},  # visibleBudget 非 bool
    {"allowedTools": "bash"},  # allowedTools 非数组
    {"allowedTools": ["bash"], "maxCalls": [1]},  # maxCalls 非 object
    {"allowedTools": ["bash"], "maxCalls": {"bash": False}},  # maxCalls 值 bool
    # 显式 JSON null 非法（缺失才合法）；必须在任何副作用之前拒绝。
    '{"allowedTools": ["bash"], "maxCalls": null}',
    '{"allowedTools": ["bash"], "exactCommands": null}',
    '{"allowedTools": ["bash"], "visibleBudget": null}',
    '{"allowedTools": ["bash"], "maxTotalCalls": null}',
    # 保留名为 JSON 文本真 own key（非 object-literal __proto__ 假测试）。
    '{"allowedTools": ["__proto__"], "maxCalls": {"__proto__": 1.5}}',
]


@pytest.mark.parametrize("payload", VALID)
def test_valid_policies_accepted(tmp_path, payload):
    assert _validate_tool_call_policy(_write(tmp_path, payload)).is_file()


@pytest.mark.parametrize("payload", INVALID)
def test_invalid_policies_rejected(tmp_path, payload):
    with pytest.raises(ValueError):
        _validate_tool_call_policy(_write(tmp_path, payload))


def test_missing_file_rejected(tmp_path):
    with pytest.raises(ValueError):
        _validate_tool_call_policy(str(tmp_path / "nope.json"))


def test_returns_canonical_absolute_resolved_path(tmp_path, monkeypatch):
    from pathlib import Path

    payload = {"allowedTools": ["bash"], "maxCalls": {"bash": 3}}
    # 相对路径：相对调用方 cwd 解析，调用方 cwd 不变。
    d = tmp_path / "sub dir"  # 含空格
    d.mkdir()
    rel_target = d / "policy.json"
    rel_target.write_text(json.dumps(payload))
    monkeypatch.chdir(tmp_path)
    before = Path.cwd()
    got = _validate_tool_call_policy("sub dir/policy.json")
    assert got == rel_target.resolve() and got.is_absolute()
    assert Path.cwd() == before
    # 绝对路径：同一文件解析到同一路径。
    assert _validate_tool_call_policy(str(rel_target)) == rel_target.resolve()
    # ~ 相对调用方 HOME 展开。
    home_target = tmp_path / "home" / "policy.json"
    home_target.parent.mkdir()
    home_target.write_text(json.dumps(payload))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert _validate_tool_call_policy("~/policy.json") == home_target.resolve()
    # 符号链接解析到真实文件。
    link = tmp_path / "link.json"
    link.symlink_to(rel_target)
    assert _validate_tool_call_policy(str(link)) == rel_target.resolve()


def test_argparse_accepts_flag_and_defaults_none():
    parser = build_parser()
    args = parser.parse_args(["chat"])
    assert args.tool_call_policy is None
    args = parser.parse_args(["chat", "--tool-call-policy", "policy.json"])
    assert args.tool_call_policy == "policy.json"
    with pytest.raises(SystemExit):
        parser.parse_args(["chat", "--tool-call-policy"])  # 缺 flag 参数


def test_cmd_chat_rejects_malformed_before_any_side_effect(tmp_path, monkeypatch, capsys):
    import rosclaw.agentd.cli as cli

    def _boom(*a, **k):
        raise AssertionError("home/store/auth/Node 不得在 policy 拒绝前启动")

    monkeypatch.setattr(cli, "_home", _boom)
    monkeypatch.setattr(cli, "_cmd_chat_impl", _boom)
    for payload in (
        '{"allowedTools": ["bash"], "maxCalls": null}',
        '{"allowedTools": ["bash"], "exactCommands": null}',
        '{"allowedTools": ["bash"], "visibleBudget": null}',
        "{",
    ):
        p = tmp_path / "bad.json"
        p.write_text(payload)
        parser = build_parser()
        args = parser.parse_args(["chat", "--tool-call-policy", str(p)])
        assert cli.cmd_chat(args) == 2
        assert "--tool-call-policy 无效" in capsys.readouterr().err


def test_unknown_tilde_username_rejected_friendly(tmp_path, monkeypatch, capsys):
    import rosclaw.agentd.cli as cli

    def _boom(*a, **k):
        raise AssertionError("home/store/auth/Node 不得在 policy 拒绝前启动")

    monkeypatch.setattr(cli, "_home", _boom)
    monkeypatch.setattr(cli, "_cmd_chat_impl", _boom)
    # 未知 ~username：expanduser 内部 RuntimeError 必须归一为 ValueError，
    # cmd_chat 友好 invalid exit 2，不泄 traceback、零副作用。
    raw = "~rosclaw_no_such_user_zz/policy.json"
    with pytest.raises(ValueError):
        _validate_tool_call_policy(raw)
    parser = build_parser()
    args = parser.parse_args(["chat", "--tool-call-policy", raw])
    assert cli.cmd_chat(args) == 2
    err = capsys.readouterr().err
    assert "--tool-call-policy 无效" in err
    assert "Traceback" not in err


def test_cmd_chat_forwards_canonical_resolved_path(tmp_path, monkeypatch):
    from pathlib import Path

    import rosclaw.agentd.cli as cli

    target = tmp_path / "policy dir" / "policy.json"
    target.parent.mkdir()
    target.write_text(json.dumps({"allowedTools": []}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_home", lambda args: tmp_path / "home")
    monkeypatch.setattr(cli, "_ensure_home_env", lambda home: None)
    monkeypatch.setattr(cli, "_restore_home_env", lambda prev: None)
    seen = {}

    def _impl(args, home):
        seen["policy"] = args.tool_call_policy
        return 0

    monkeypatch.setattr(cli, "_cmd_chat_impl", _impl)
    parser = build_parser()
    args = parser.parse_args(["chat", "--tool-call-policy", "policy dir/policy.json"])
    assert cli.cmd_chat(args) == 0
    assert seen["policy"] == str(target.resolve())
    assert Path.cwd() == tmp_path  # 调用方 cwd 不变
