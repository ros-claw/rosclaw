"""WP-P0-1 红测试（总纲 §5.1）：会话可发现性紧急修复。

红测试先行——当前断点（总纲 §2.2 表）：

1. `chat --resume` 必须带 SESSION_ID，而系统从不展示 ID；
2. main.ts 自己 readdirSync+mtime 扫描——绕过 Pi SessionManager；
3. 没有 rosclaw sessions/resume/continue 顶层命令；
4. 退出提示暴露内部 session id（应给 rosclaw continue/sessions）。
"""

from __future__ import annotations

import json
from pathlib import Path


class TestCliSurface:
    def test_resume_bare_is_picker(self) -> None:
        """`rosclaw chat --resume` 无参数 → picker 标记（不再缺参报错）。"""
        import argparse

        from rosclaw.agentd.cli import add_agent_subparsers

        parser = argparse.ArgumentParser(prog="rosclaw")
        add_agent_subparsers(parser.add_subparsers(dest="command"))
        args = parser.parse_args(["chat", "--resume"])
        assert args.resume == "__picker__", (
            f"裸 --resume 应为 picker 标记，实际 {args.resume!r}"
        )
        # 带参数仍按原样解析。
        args2 = parser.parse_args(["chat", "--resume", "abc123"])
        assert args2.resume == "abc123"

    def test_top_level_commands_exist(self) -> None:
        """rosclaw sessions / resume / continue 在 root registry。"""
        from rosclaw.root_cli import COMMAND_REGISTRY

        for name in ("sessions", "resume", "continue"):
            assert name in COMMAND_REGISTRY, f"顶层命令 {name} 不存在"


class TestSessionListing:
    def _write_session(self, sessions_dir: Path, session_id: str, name: str,
                       first_message: str, when: str) -> Path:
        sessions_dir.mkdir(parents=True, exist_ok=True)
        path = sessions_dir / f"2026-08-10T10-00-00_{session_id}.jsonl"
        lines = [
            json.dumps({"type": "session", "id": session_id, "timestamp": when}),
            json.dumps({"type": "session_info", "name": name}),
            json.dumps({"type": "message", "message": {"role": "user", "content": first_message}}),
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_list_sessions_reads_pi_jsonl(self, tmp_path: Path) -> None:
        """sessions 列表读 Pi JSONL（时间前缀文件名/名称/首条消息/
        损坏文件不拖垮列表）。"""
        from rosclaw.agentd.session_list import list_sessions

        sessions_dir = tmp_path / "agent" / "sessions"
        self._write_session(sessions_dir, "abc123", "五角星轨迹仿真", "画五角星", "2026-08-10T10:00:00Z")
        self._write_session(sessions_dir, "def456", "", "巡检", "2026-08-11T09:00:00Z")
        (sessions_dir / "broken.jsonl").write_text("{not json", encoding="utf-8")
        sessions = list_sessions(tmp_path)
        assert len(sessions) == 2, sessions
        by_id = {s["session_id"]: s for s in sessions}
        assert by_id["abc123"]["display_name"] == "五角星轨迹仿真"
        assert by_id["def456"]["first_message"] == "巡检"
        # 最近活动排序：def456 更新。
        assert sessions[0]["session_id"] == "def456"

    def test_resolve_query_id_prefix_title(self, tmp_path: Path) -> None:
        """精确 ID / 唯一前缀 / 标题解析；歧义报候选不猜。"""
        from rosclaw.agentd.session_list import list_sessions, resolve_session_query

        sessions_dir = tmp_path / "agent" / "sessions"
        self._write_session(sessions_dir, "abc123", "五角星轨迹仿真", "画五角星", "2026-08-10T10:00:00Z")
        self._write_session(sessions_dir, "abd999", "五角星复测", "再画", "2026-08-11T10:00:00Z")
        # 精确 ID
        hit = resolve_session_query(list_sessions(tmp_path), "abc123")
        assert hit.get("path", "").endswith(".jsonl")
        # 标题
        hit = resolve_session_query(list_sessions(tmp_path), "五角星轨迹仿真")
        assert "abc123" in hit.get("path", "")
        # 歧义前缀
        miss = resolve_session_query(list_sessions(tmp_path), "ab")
        assert miss.get("error") == "AMBIGUOUS"
        assert len(miss.get("candidates", [])) == 2
        # 不存在
        none = resolve_session_query(list_sessions(tmp_path), "zzz")
        assert none.get("error") == "NOT_FOUND"


class TestExitHintIsProductCommand:
    def test_resume_hint_uses_rosclaw_continue(self) -> None:
        """退出提示必须是 rosclaw continue/sessions——不暴露内部
        session id（patch-01 替换文本断言）。"""
        patch = Path(
            "packages/rosclaw-agent/patches/apply-upstream-patches.mjs"
        ).read_text(encoding="utf-8")
        assert "rosclaw continue" in patch, "退出提示未给 rosclaw continue"
        assert "rosclaw sessions" in patch
        # 01b 清理补丁的 anchor 合法含旧串（它是删除目标）——剥离后
        # 不得再出现（patch-01 的 replacement 不得含内部 id）。
        stripped = patch.replace(
            '"    return `rosclaw chat --resume ${sessionManager.getSessionId()}`;\\n",', ""
        )
        assert "chat --resume ${sessionManager.getSessionId()}" not in stripped, (
            "退出提示仍暴露内部 session id"
        )

    def test_main_ts_has_no_handrolled_scan(self) -> None:
        """main.ts 不得再自己 readdirSync+mtime 扫描会话目录。"""
        source = Path("packages/rosclaw-agent/src/main.ts").read_text(encoding="utf-8")
        assert "readdirSync" not in source, "手写目录扫描仍在"
        assert "SessionManager" in source, "应复用 Pi SessionManager"
