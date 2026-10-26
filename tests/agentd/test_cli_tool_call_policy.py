"""Stage B：--tool-call-policy JSON_FILE CLI plumbing 回归测试。

覆盖：合法 policy（含 allowedTools=[] block-all、空可选 map、JSON 1.0
整数值浮点）通过；全部畸形类别在 home/auth/Node 启动前被
_validate_tool_call_policy 拒绝；argparse 接受该 flag 且默认 None。
深层 JSON 解码失败与 --basic Python REPL 退役必须在启动前友好拒绝；
覆盖公开 parser/dispatch 与统一入口 subprocess，默认 Native 路径不变。
Node 侧 policy 行为属 Stage A + A+B 集成范围，不在本测试证明内。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rosclaw.agentd.cli import (
    _validate_tool_call_policy,
    add_agent_subparsers,
    build_parser,
)

REPO = Path(__file__).resolve().parents[2]


def _public_parser():
    parser = argparse.ArgumentParser(prog="rosclaw")
    add_agent_subparsers(parser.add_subparsers(dest="command", required=True))
    return parser


@pytest.fixture
def deep_policy(tmp_path):
    # 完整实证 fixture：1100 层数组，无空格/换行；不依赖作者的外部文件。
    payload = b'{"allowedTools":' + b"[" * 1100 + b"]" * 1100 + b"}"
    assert len(payload) == 2217
    assert hashlib.sha256(payload).hexdigest() == (
        "4cbfc8c507728c815d7940bf76d67d6c0c1caae6fbed06c453b982542edf2acc"
    )
    path = tmp_path / "deep-policy.json"
    path.write_bytes(payload)
    return path


@pytest.fixture
def no_chat_setup(monkeypatch):
    import rosclaw.agentd.cli as cli
    import rosclaw.agentd.pi_config as pi_config

    def _boom(*a, **k):
        raise AssertionError("拒绝前不得进入 home/store/auth/runtime 启动")

    for name in (
        "_home",
        "_ensure_home_env",
        "_cmd_chat_impl",
        "load_agent_config",
        "AgentService",
        "create_app",
        "_find_pi_agent_entry",
        "_chat_pi",
        "_route_internal_diagnostics_to_log",
    ):
        monkeypatch.setattr(cli, name, _boom)
    monkeypatch.setattr(pi_config, "pi_model_configured", _boom)
    return cli


def _run_public_chat(tmp_path, *argv):
    home = tmp_path / "absent-rosclaw-home"
    assert not home.exists()
    env = dict(
        os.environ,
        ROSCLAW_HOME=str(home),
        HOME=str(tmp_path / "absent-user-home"),
        PI_OFFLINE="1",
        PYTHONIOENCODING="utf-8",
        PYTHONDONTWRITEBYTECODE="1",
    )
    # 在 Root 的独立 worktree 中测试该树源码，而非旧 editable install。
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(REPO / "src"), env.get("PYTHONPATH"))))
    result = subprocess.run(
        [sys.executable, "-m", "rosclaw.entrypoint", "chat", *argv],
        cwd=REPO,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert not home.exists(), "早拒绝/帮助不得创建 ROSCLAW_HOME"
    return result


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


def test_deep_policy_rejected_as_value_error(deep_policy):
    with pytest.raises(ValueError, match="policy JSON 嵌套过深") as caught:
        _validate_tool_call_policy(str(deep_policy))
    assert isinstance(caught.value.__cause__, RecursionError)


def test_cmd_chat_rejects_deep_policy_before_any_setup(deep_policy, no_chat_setup, capsys):
    args = build_parser().parse_args(["chat", "--tool-call-policy", str(deep_policy)])
    assert no_chat_setup.cmd_chat(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "--tool-call-policy 无效：policy JSON 嵌套过深\n"
    assert "Traceback" not in captured.err
    assert "RecursionError" not in captured.err


def test_public_chat_rejects_exact_deep_policy_without_home(tmp_path, deep_policy):
    result = _run_public_chat(tmp_path, "--tool-call-policy", str(deep_policy))
    assert result.returncode == 2, result.stderr
    assert result.stdout == ""
    assert "--tool-call-policy 无效：policy JSON 嵌套过深" in result.stderr
    assert "Traceback" not in result.stderr
    assert "RecursionError" not in result.stderr
    assert deep_policy.read_text(encoding="utf-8") not in result.stderr
    assert len(result.stderr.encode("utf-8")) < 1024


@pytest.mark.parametrize("error_type", [RuntimeError, RecursionError])
def test_policy_boundary_does_not_hide_unrelated_validator_errors(
    tmp_path,
    monkeypatch,
    error_type,
):
    import rosclaw.agentd.cli as cli

    error = error_type("unrelated validator failure")

    def _boom(name):
        raise error

    monkeypatch.setattr(cli, "_policy_tool_name_ok", _boom)
    with pytest.raises(error_type) as caught:
        _validate_tool_call_policy(_write(tmp_path, {"allowedTools": ["read"]}))
    assert caught.value is error  # 仅 JSON 解码边界归一，不 blanket catch。


def test_public_parser_basic_defaults_and_retirement_help(capsys):
    parser = _public_parser()
    default = parser.parse_args(["chat"])
    assert default.basic is False
    assert default.tool_call_policy is None
    assert parser.parse_args(["chat", "--basic"]).basic is True
    with pytest.raises(SystemExit) as caught:
        parser.parse_args(["chat", "--help"])
    assert caught.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "--basic" in help_text
    assert "Python REPL" in help_text
    assert "已退役" in help_text
    assert "使用时报错退出" in help_text
    assert "rosclaw chat" in help_text
    assert "input()" not in help_text


@pytest.mark.parametrize(
    "extra_args",
    [[], ["--debug"], ["--tool-call-policy", "must-not-read.json"]],
    ids=["basic", "basic-debug", "basic-with-policy"],
)
def test_dispatch_rejects_basic_before_any_setup(
    no_chat_setup,
    monkeypatch,
    capsys,
    extra_args,
):
    def _boom(*a, **k):
        raise AssertionError("退役 --basic 不得读取 policy 或启动 Native Agent")

    monkeypatch.setattr(no_chat_setup, "_validate_tool_call_policy", _boom)
    args = _public_parser().parse_args(["chat", "--basic", *extra_args])
    assert no_chat_setup.dispatch_agent_command(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "--basic 的 Python REPL 已退役；请使用 rosclaw chat（不带 --basic）。\n"
    )
    assert "Traceback" not in captured.err


def test_public_chat_basic_retirement_without_home(tmp_path):
    result = _run_public_chat(tmp_path, "--basic")
    assert result.returncode == 2, result.stderr
    assert result.stdout == ""
    assert "--basic 的 Python REPL 已退役" in result.stderr
    assert "请使用 rosclaw chat（不带 --basic）" in result.stderr
    assert "Traceback" not in result.stderr


def test_public_chat_help_reports_basic_retirement_without_home(tmp_path):
    result = _run_public_chat(tmp_path, "--help")
    assert result.returncode == 0, result.stderr
    help_text = " ".join(result.stdout.split())
    assert "--basic" in help_text
    assert "Python REPL" in help_text
    assert "已退役" in help_text
    assert "使用时报错退出" in help_text
    assert "rosclaw chat" in help_text
    assert "input()" not in help_text
    assert "Traceback" not in result.stderr


def test_default_public_chat_still_enters_native_dispatch(tmp_path, monkeypatch):
    import rosclaw.agentd.cli as cli

    home = tmp_path / "home"
    monkeypatch.delenv("ROSCLAW_HOME", raising=False)
    monkeypatch.setattr(cli, "_home", lambda args: home)
    seen = []

    def _impl(args, got_home):
        assert args.basic is False
        assert args.tool_call_policy is None
        assert os.environ["ROSCLAW_HOME"] == str(home)
        seen.append(got_home)
        return 17

    def _boom(*a, **k):
        raise AssertionError("默认 chat 不得触发 policy 校验")

    monkeypatch.setattr(cli, "_validate_tool_call_policy", _boom)
    monkeypatch.setattr(cli, "_cmd_chat_impl", _impl)
    args = _public_parser().parse_args(["chat"])
    assert cli.dispatch_agent_command(args) == 17
    assert seen == [home]
    assert "ROSCLAW_HOME" not in os.environ  # 原有配对恢复行为不变。
    assert not home.exists()
