"""T-PRODUCT/T-REASONING/T-IDENTITY（二次审计 Gate E 核心）：

安装产物黑盒旅程——从签名 bundle 离线安装，经 PTY 驱动完整旅程：

clean install → rosclaw chat --engine pi → 品牌 header → 普通对话
→ reasoning 不泄露（SECRET_RAW_REASONING_123 永不出现）
→ /status → delegate Worker → request SIM action → 精确卡片 Y →
结构化回执 → /compact → /quit（rosclaw resume 命令，无 pi --session）
→ --continue 恢复。

假模型：进程内 OpenAI 兼容服务器（SSE 流式 + tool_calls + reasoning
标记注入），绝不依赖真实 provider。
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import socket
import subprocess
import tarfile
import termios
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BUILD = REPO / "scripts" / "build_release.sh"

REASONING_MARKER = "SECRET_RAW_REASONING_123"


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


def _chunk(content: str = "", finish: str | None = None) -> dict:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "fake-k3",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish,
            }
        ],
    }


def _role_chunk() -> dict:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "fake-k3",
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }


def _tool_head_chunk(call_id: str, name: str) -> dict:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "fake-k3",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": ""},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
    }


def _tool_args_chunk(arguments: str) -> dict:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "fake-k3",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [{"index": 0, "function": {"arguments": arguments}}]
                },
                "finish_reason": None,
            }
        ],
    }


def _tool_call_frames(call_id: str, name: str, arguments: str) -> list[bytes]:
    return [
        _sse(_role_chunk()),
        _sse(_tool_head_chunk(call_id, name)),
        _sse(_tool_args_chunk(arguments)),
        _sse(_chunk("", "tool_calls")),
    ]


class _FakeModel:
    """按消息内容编排回答/tool_calls/reasoning 的假模型。"""

    def __init__(self, log_path: Path | None = None) -> None:
        self.requests: list[dict] = []
        self.log_path = log_path

    def answer(self, body: dict) -> bytes:
        self.requests.append(body)
        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as handle:
                # 系统提示占头部——保留尾部（用户消息/工具结果/广告工具在那边）。
                handle.write(json.dumps(body, ensure_ascii=False)[-4000:] + "\n")
        messages = body.get("messages", [])
        # before_agent_start 会把 ROSCLAW_TRUSTED_CONTEXT 作为额外 user 消息
        # 追加在真实用户输入之后——取最后一条非注入的 user 消息。
        def _text(message: dict) -> str:
            content = message.get("content", "")
            if isinstance(content, list):
                return " ".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
            return str(content)

        user_texts = [
            _text(m)
            for m in messages
            if m.get("role") == "user" and not _text(m).startswith("<ROSCLAW_TRUSTED_CONTEXT>")
        ]
        text = user_texts[-1] if user_texts else ""
        # 仅当最新一条消息是工具结果时才进入"工具已回→最终回答"分支——
        # 历史消息里的旧 tool 结果不能劫持后续 turn。
        has_tool_result = bool(messages) and messages[-1].get("role") == "tool"
        # native worker 走 legacy 非流式 complete：返回普通 JSON 而非 SSE。
        if not body.get("stream"):
            answer_text = "Worker 分析结果：日志要点已归纳（fake worker）。"
            return json.dumps(
                {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "fake-k3",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": answer_text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10},
                },
                ensure_ascii=False,
            ).encode()
        frames: list[bytes] = []
        if has_tool_result:
            # tool result 已回 → 最终回答。按工具区分文本，避免旅程里
            # delegate 步骤的旧输出误匹配 action 步骤的等待。
            last_tool = messages[-1]
            tool_content = str(last_tool.get("content", ""))
            if "receipt" in tool_content or "grant" in tool_content or "已批准" in tool_content:
                answer = "动作已执行，结构化回执已确认。"
            else:
                answer = "Worker 结果已收到并验证。"
            frames.append(_sse(_chunk(answer)))
            frames.append(_sse(_chunk("", "stop")))
        elif "SECRET_PROBE" in text:
            # reasoning 标记注入（openai-compat reasoning_content 形状）。
            marker_frame = {
                "id": "chatcmpl-fake",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "fake-k3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": REASONING_MARKER},
                        "finish_reason": None,
                    }
                ],
            }
            frames.append(_sse(marker_frame))
            frames.append(_sse(_chunk("这是最终回答（无推理泄露）。")))
            frames.append(_sse(_chunk("", "stop")))
        elif "读取系统状态" in text:
            frames.extend(_tool_call_frames("call_status", "rosclaw_status", "{}"))
        elif "委派" in text:
            frames.extend(
                _tool_call_frames(
                    "call_delegate",
                    "rosclaw_delegate",
                    json.dumps({"goal": "总结这段日志", "worker_id": "auto"}),
                )
            )
        elif "播放提示音" in text or "初始位姿" in text:
            frames.extend(
                _tool_call_frames(
                    "call_action",
                    "rosclaw_request_action",
                    json.dumps(
                        {
                            "capability_id": "sim_ground_truth",
                            "arguments": {},
                            "expected_effect": "旅程验收动作",
                            "risk_tier": "LOW",
                        }
                    ),
                )
            )
        else:
            frames.append(_sse(_chunk("你好，我是 ROSClaw Native Agent。")))
            frames.append(_sse(_chunk("", "stop")))
        frames.append(b"data: [DONE]\n\n")
        return b"".join(frames)


class _Handler(BaseHTTPRequestHandler):
    fake: _FakeModel  # class attr injected by factory

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        payload = self.fake.answer(body)
        self.send_response(200)
        ctype = "text/event-stream" if body.get("stream") else "application/json"
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:  # quiet
        return


class FakeModelServer:
    def __init__(self, log_path: Path | None = None) -> None:
        self.fake = _FakeModel(log_path)
        handler = type("H", (_Handler,), {"fake": self.fake})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def close(self) -> None:
        self.server.shutdown()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _build_and_install(tmp_path: Path) -> tuple[Path, Path]:
    env = dict(os.environ, ROSCLAW_SIGNING_HOME=str(tmp_path / "signing"))
    result = subprocess.run(
        ["bash", str(BUILD)], cwd=REPO, capture_output=True, text=True, timeout=1800, env=env
    )
    assert result.returncode == 0, result.stderr[-1500:]
    bundle = sorted((REPO / "dist").glob("rosclaw-*-linux-*.tar.gz"))[-1]
    stage = tmp_path / "stage"
    stage.mkdir()
    with tarfile.open(bundle) as tf:
        tf.extractall(stage)
    root = next(stage.iterdir())
    install = subprocess.run(
        [
            "bash", str(root / "install.sh"), "--offline",
            "--trusted-key", str(tmp_path / "signing" / "dev-signing-public.pem"),
        ],
        capture_output=True, text=True, timeout=900,
        env=dict(os.environ, ROSCLAW_PREFIX=str(tmp_path / "prefix")),
    )
    assert install.returncode == 0, install.stderr[-1500:]
    return tmp_path / "prefix", root


class PtySession:
    """最小 PTY 驱动：expect/send。"""

    def __init__(self, argv: list[str], env: dict[str, str]) -> None:
        import pty as _pty

        self.master, slave = _pty.openpty()

        def _make_controlling_tty() -> None:
            # 新 session + 控制终端——否则 TIOCSWINSZ 的 SIGWINCH 没有
            # 前台进程组可投递，TUI 永远收不到 resize 事件。
            os.setsid()
            fcntl.ioctl(slave, termios.TIOCSCTTY, 0)

        self.proc = subprocess.Popen(
            argv, stdin=slave, stdout=slave, stderr=slave, env=env, close_fds=True,
            preexec_fn=_make_controlling_tty,
        )
        os.close(slave)
        self.output = b""
        self.last_at = time.monotonic()
        # 后台持续 drain——PTY 缓冲满会阻塞子进程写（没有它测试会假死）。
        self._lock = threading.Lock()
        self._draining = True

        def _drain() -> None:
            import select as _select

            while self._draining:
                try:
                    ready, _, _ = _select.select([self.master], [], [], 0.2)
                    if ready:
                        chunk = os.read(self.master, 4096)
                        if not chunk:
                            break
                        with self._lock:
                            self.output += chunk
                            self.last_at = time.monotonic()
                except OSError:
                    break

        self._drain_thread = threading.Thread(target=_drain, daemon=True)
        self._drain_thread.start()

    def expect(self, marker: bytes, timeout: float = 60.0) -> bytes:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if marker in self.output:
                    return self.output
            if self.proc.poll() is not None:
                with self._lock:
                    if marker in self.output:
                        return self.output
                break
            time.sleep(0.1)
        with self._lock:
            tail = self.output[-3000:]
        raise AssertionError(f"PTY 超时未等到 {marker!r}；已收输出尾部: {tail!r}")

    def send(self, text: str) -> None:
        os.write(self.master, text.encode())

    def stop(self) -> None:
        self._draining = False
        with contextlib.suppress(OSError):
            os.close(self.master)
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


@pytest.mark.slow
class TestProductJourney:
    def test_full_journey_pty(self, tmp_path: Path) -> None:
        fake = FakeModelServer(log_path=tmp_path / "fake-requests.jsonl")
        prefix, _root = _build_and_install(tmp_path)
        home = tmp_path / "rh"
        # kernel（python agentd）配置：fake base_url；Pi 侧 models.json。
        (home / "run").mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(
            "agent:\n  enabled: true\n  default_profile: embodied_default\n"
            "models:\n  backend: legacy\n  profiles:\n    embodied_default:\n"
            "      provider: kimi_code\n      model: fake-k3\n"
            f"      base_url: {fake.base_url}\n"
            "      api_key_ref: env:FAKE_JOURNEY_KEY\n"
            "      capabilities: [llm.chat, llm.structured_decision, llm.tool_use]\n",
            encoding="utf-8",
        )
        (home / "agent").mkdir(parents=True, exist_ok=True)
        (home / "agent" / "settings.json").write_text(
            json.dumps({"defaultProvider": "journey-fake", "defaultModel": "fake-k3"}),
            encoding="utf-8",
        )
        (home / "agent" / "models.json").write_text(
            json.dumps(
                {
                    "providers": {
                        "journey-fake": {
                            "name": "Journey Fake",
                            "baseUrl": fake.base_url,
                            "api": "openai-completions",
                            "apiKey": "$FAKE_JOURNEY_KEY",
                            "models": [
                                {
                                    "id": "fake-k3",
                                    "name": "Fake K3",
                                    "contextWindow": 8192,
                                    "maxTokens": 4096,
                                }
                            ],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        rosclaw = prefix / "bin" / "rosclaw"
        env = dict(
            os.environ,
            ROSCLAW_HOME=str(home),
            TERM="xterm",
            FAKE_JOURNEY_KEY="sk-fake-journey",
            KIMI_API_KEY="sk-fake-journey",
            PATH=f"{prefix / 'bin'}:{os.environ['PATH']}",
        )
        # P0-01 架构：授权决定只能经独立 rosclaw-operatord——旅程必须
        # enroll + start 一个真实 operatord（SIM 卡走 Ed25519 签名
        # apply_decision；agentd 自身拒绝 decide）。
        enroll = subprocess.run(
            [str(rosclaw), "operatord", "enroll"],
            env=env, capture_output=True, text=True, timeout=60,
        )
        assert enroll.returncode == 0, enroll.stderr[-500:]
        operatord = subprocess.Popen(
            [str(rosclaw), "operatord", "start", "--no-human-presence-check"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not (home / "run" / "operatord.sock").exists():
                assert operatord.poll() is None, "operatord 启动失败"
                time.sleep(0.2)
            assert (home / "run" / "operatord.sock").exists(), "operatord.sock 未出现"
            self._run_journey(rosclaw, env, home)
        finally:
            operatord.terminate()
            try:
                operatord.wait(timeout=10)
            except subprocess.TimeoutExpired:
                operatord.kill()
            fake.close()

    def _assert_grant_consumed(self, home: Path) -> None:
        """授权→执行闭环：grant 必须被精确消费（单次），不许悬置。"""
        import sqlite3

        db = sqlite3.connect(home / "agentd" / "missions.db")
        try:
            rows = db.execute(
                "SELECT grant_id, consumed, revoked FROM mission_grants"
            ).fetchall()
        finally:
            db.close()
        assert rows, "批准应产生 grant"
        assert all(consumed == 1 and revoked == 0 for _, consumed, revoked in rows), (
            f"grant 未被消费或被撤销: {rows}"
        )

    def _run_journey(self, rosclaw: Path, env: dict[str, str], home: Path) -> None:
        # NA-FIX-9 后默认引擎即 Native Agent——旅程显式验证无 --engine 的默认路径。
        session = PtySession([str(rosclaw), "chat"], env)
        try:
            # 1. 品牌 header（T-IDENTITY：无 engine/pi 字样）。
            session.expect(b"ROSClaw Native Agent", timeout=60)
            assert b"engine=pi" not in session.output
            # 2. 普通对话。
            session.send("你好\r")
            session.expect("你好，我是 ROSClaw".encode(), timeout=90)
            # 3. T-REASONING：推理标记绝不出现。
            marker_at = len(session.output)
            session.send("SECRET_PROBE 测试\r")
            session.expect("这是最终回答".encode(), timeout=90)
            assert REASONING_MARKER.encode() not in session.output[marker_at:]
            time.sleep(3.0)  # 等回合完全 settled 再发命令
            # 4. /status。
            session.send("/status\r")
            session.expect(b"agentd=READY", timeout=30)
            time.sleep(2.0)
            # 5. delegate worker。
            session.send("请委派 worker 总结这段日志\r")
            session.expect("Worker 结果已收到并验证".encode(), timeout=180)
            # 6. request SIM action → 卡片 → Y。
            session.send("请播放提示音\r")
            session.expect("等待 Operator 决定".encode(), timeout=120)
            session.expect("ROSCLAW 授权请求".encode(), timeout=60)
            session.send("y")
            # 已批准 → 执行（结构化回执或诚实失败，但不许是未决状态）。
            session.expect(b"\xe5\xb7\xb2\xe6\x89\xb9\xe5\x87\x86", timeout=120)
            # 等工具结果回到模型并产出最终回答——证明 execute 阶段真正完成
            # （grant 被消费），而不是只停在批准通知。
            session.expect("动作已执行，结构化回执已确认".encode(), timeout=120)
            self._assert_grant_consumed(home)
            # 7. /compact。
            session.send("/compact\r")
            time.sleep(2.0)
            # 8. /quit → resume 提示必须是 ROSClaw 命令（T-IDENTITY）。
            session.send("/quit\r")
            session.expect(b"rosclaw chat --resume", timeout=30)
            assert b"pi --session" not in session.output
            assert b"--session-dir" not in session.output
            session.proc.wait(timeout=30)
            assert session.proc.returncode == 0, session.output[-400:]
        finally:
            session.stop()
        # 9. --continue 恢复（同 session/binding）。
        resumed = PtySession([str(rosclaw), "chat", "--continue"], env)
        try:
            resumed.expect(b"ROSClaw Native Agent", timeout=60)
            resumed.send("/quit\r")
            resumed.expect(b"rosclaw chat --resume", timeout=30)
            resumed.proc.wait(timeout=30)
        finally:
            resumed.stop()
