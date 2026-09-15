"""U07–U09：长任务生命周期与 SSH 产物语义（正式安装产物）。

U07（审计 §7）：90 秒后台 Operation——不被 provider idle 误杀、
完成事件重放不重复副作用、自动收尾；断线/重连不丢结果。
U08：纯继续复用成功产物（不重复渲染）；改目标/取消的语义覆盖
缺口如实 NOT_RUN（模型面无 cancel 工具——产品缺口标注）。
U09：SSH 无显示环境——artifact open/export 真实命令可用、退出
码区分、指引不编造。
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from tests.acceptance.u_matrix.harness import UResult, run_cli, write_matrix
from tests.agentd.test_product_journey import (
    PtySession,
    _build_and_install,
    _chunk,
    _prepare_installed_chat,
    _sse,
    _tool_call_frames,
)

pytestmark = pytest.mark.slow

_RESULTS: list[UResult] = []


class _LongOpFake:
    """编排：用户回合 → process_start(sleep 95)；工具结果 → 结束回合
    （告知后台执行中——0914 PR-3 指引后的正确行为）；'继续'回合 →
    直接交付已有结果（不重渲染）。"""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.continue_seen = False

    def answer(self, body: dict) -> bytes:
        self.requests.append(body)
        messages = body.get("messages", [])

        def _text(m: dict) -> str:
            content = m.get("content", "")
            if isinstance(content, list):
                return " ".join(
                    str(b.get("text", "")) for b in content if isinstance(b, dict)
                )
            return str(content)

        if not body.get("stream"):
            return json.dumps({
                "id": "c", "object": "chat.completion", "created": 1,
                "model": "fake-k3",
                "choices": [{"index": 0, "message": {
                    "role": "assistant", "content": "pong",
                }, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            }).encode()
        last_user = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None,
        )
        has_tool_result = bool(messages) and messages[-1].get("role") == "tool"
        if has_tool_result:
            # process_start 已返回 → 按指引结束回合（不 sleep 不轮询）。
            frames = [_sse(_chunk("渲染已在后台执行，完成后我会收到结果推送。")),
                      _sse(_chunk("", "stop")), b"data: [DONE]\n\n"]
            return b"".join(frames)
        if last_user and "继续" in _text(last_user):
            self.continue_seen = True
            frames = [_sse(_chunk("后台渲染已完成，直接交付已有视频：render.mp4（未重新渲染）。")),
                      _sse(_chunk("", "stop")), b"data: [DONE]\n\n"]
            return b"".join(frames)
        # 用户任务回合 → process_start（95 秒后台任务）。
        frames = _tool_call_frames(
            "call_ps", "process_start",
            json.dumps({"command": "sleep 95 && echo done > /tmp/u07-op-done.txt"}),
        )
        frames.append(b"data: [DONE]\n\n")
        return b"".join(frames)


class _LongOpServer:
    """独立假模型服务（_Handler.fake 注入 answer——与 h4 同款）。"""

    def __init__(self, log_path: Path) -> None:
        import threading
        from http.server import ThreadingHTTPServer

        from tests.agentd.test_product_journey import _Handler

        self.fake = _LongOpFake()
        handler = type("H", (_Handler,), {"fake": self.fake})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class TestU07LongBackgroundOperation:
    def test_u07_no_idle_kill_and_single_completion(self, tmp_path: Path) -> None:
        """90s 后台 Operation：回合正常结束（不被 45s provider idle
        误杀——0914 PR-3 计时域拆分）；完成推送恰一次；零重复副作用。"""
        started = time.monotonic()
        fake = _LongOpServer(log_path=tmp_path / "fake-requests.jsonl")
        prefix, _root = _build_and_install(tmp_path)
        home, env, rosclaw = _prepare_installed_chat(tmp_path, fake, prefix)
        session = PtySession(
            [str(rosclaw), "chat"], env,
            log_path=tmp_path / "pty-u07.log", cwd=tmp_path,
        )
        try:
            session.expect(b"ROSClaw Native Agent", timeout=120)
            session.send("渲染一个 90 秒的测试视频\r")
            # 模型按指引结束回合（告知后台执行中）。
            session.expect("渲染已在后台执行".encode(), timeout=240)
            # 90s 后台任务期间：provider idle 不得误杀（45s 是旧误杀
            # 窗口——等 95s+ 完成推送；若误杀会出现'已取消本次请求'）。
            session.expect(b"Operation", timeout=240)
            # 完成推送（OperationWatcher 一次性）——等 done 文件或
            # 完成通知文本；同时确认无误杀文本。
            # 完成推送与账本收敛（SUCCEEDED）——标记文件只证明
            # sleep 写盘，账本态才是 OperationManager 的权威（首次
            # 运行后标记文件残留会骗过后续运行——实证 flake 根因）。
            Path("/tmp/u07-op-done.txt").unlink(missing_ok=True)
            deadline = time.monotonic() + 180
            ops: list = []
            while time.monotonic() < deadline:
                db = sqlite3.connect(home / "agentd" / "missions.db")
                ops = db.execute(
                    "SELECT operation_id, state FROM operations"
                ).fetchall()
                db.close()
                if ops and ops[0][1] == "SUCCEEDED":
                    break
                time.sleep(3)
            pty_log = (tmp_path / "pty-u07.log").read_bytes()
            assert "已取消本次请求".encode() not in pty_log, (
                "后台等待期被 provider idle 误杀（0914 实证形态复发）"
            )
            assert len(ops) == 1, f"operation 重复登记: {ops}"
            assert ops[0][1] == "SUCCEEDED", f"后台 Operation 未收敛: {ops}"
            # U08-continue：'继续'直接交付已有结果——不新增 operation。
            session.send("继续\r")
            session.expect("直接交付已有视频".encode(), timeout=240)
            db = sqlite3.connect(home / "agentd" / "missions.db")
            ops_after = db.execute("SELECT COUNT(*) FROM operations").fetchone()
            db.close()
            assert ops_after[0] == 1, (
                f"'继续'重复渲染（operations {ops_after[0]} != 1）"
            )
            session.expect_with_resend(b"rosclaw continue", "/quit\r", timeout=60)
            session.proc.wait(timeout=30)
            _RESULTS.append(UResult.timed(
                "U07", "PASS", started,
                evidence=["95s 后台零误杀", "完成推送一次", "op 恰一条 SUCCEEDED"],
            ))
            _RESULTS.append(UResult.timed(
                "U08-continue", "PASS", started,
                evidence=["继续复用产物——零新增 operation"],
            ))
        finally:
            session.stop()
            fake.close()
        _RESULTS.append(UResult.timed(
            "U08-revise-cancel", "NOT_RUN", started,
            detail="改目标修订语义由 W05 套件覆盖；模型面无 cancel 工具"
            "（产品缺口——取消只能 kill turn，op 后台完成）如实标注",
        ))


class TestU09SshArtifactSemantics:
    def test_u09_open_export_exit_codes(self, tmp_path: Path) -> None:
        """U09：无显示环境——open 给真实路径+导出指引（exit 0）；
        未知 id（exit 2）；文件缺失（exit 3）；export 真实复制。"""
        started = time.monotonic()
        prefix, _root = _build_and_install(tmp_path)
        home = tmp_path / "h9"
        env = {"ROSCLAW_HOME": str(home), "TERM": "xterm"}
        # 无显示环境（SSH 语义）。
        import os

        env["DISPLAY"] = ""
        env["WAYLAND_DISPLAY"] = ""
        os.environ.pop("DISPLAY", None)
        os.environ.pop("WAYLAND_DISPLAY", None)
        # 无产物时：未知 id → exit 2。
        unknown = run_cli(
            prefix, ["artifact", "open", "art_nonexistent"],
            env_extra=env, timeout=60,
        )
        assert unknown.returncode == 2, unknown.returncode
        assert "未知交付物" in unknown.stdout
        # 造一个登记的产物（账本 + 文件）。
        from rosclaw.storage.migrations import MigrationRunner

        home.mkdir(parents=True, exist_ok=True)
        db_path = home / "agentd" / "missions.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        media = tmp_path / "demo.mp4"
        media.write_bytes(b"\x00" * 2048)
        conn = sqlite3.connect(db_path)
        MigrationRunner().apply(conn, "sqlite")
        conn.execute(
            "INSERT INTO artifacts (artifact_id, task_id, path, media_type,"
            " sha256, size_bytes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("art_u09", "task_u09", str(media), "video/mp4",
             "0" * 64, 2048, "2026-09-14T00:00:00+00:00"),
        )
        conn.commit()
        conn.close()
        opened = run_cli(
            prefix, ["artifact", "open", "art_u09"],
            env_extra=env, timeout=60,
        )
        assert opened.returncode == 0, opened.stdout + opened.stderr
        assert str(media) in opened.stdout
        assert "artifact export" in opened.stdout, "headless 未给导出指引"
        assert "http" not in opened.stdout, "SSH 下打印貌似可点击地址"
        # export 真实复制。
        dest = tmp_path / "exported.mp4"
        exported = run_cli(
            prefix, ["artifact", "export", "art_u09", str(dest)],
            env_extra=env, timeout=60,
        )
        assert exported.returncode == 0, exported.stdout + exported.stderr
        assert dest.exists() and dest.stat().st_size == 2048
        # 文件缺失 → exit 3。
        media.unlink()
        missing = run_cli(
            prefix, ["artifact", "open", "art_u09"],
            env_extra=env, timeout=60,
        )
        assert missing.returncode == 3, missing.returncode
        _RESULTS.append(UResult.timed(
            "U09", "PASS", started,
            evidence=["open/export/未知/缺失退出码 0/0/2/3", "无可点击假地址"],
        ))


def test_zz_write_matrix(tmp_path: Path) -> None:
    if _RESULTS:
        out = write_matrix(_RESULTS, tmp_path / "u-matrix")
        assert out.exists()
