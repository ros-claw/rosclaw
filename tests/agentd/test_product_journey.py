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
import sys
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
            tool_call_id = str(last_tool.get("tool_call_id", ""))
            if tool_call_id == "call_action_lie":
                # 对抗场景（P0-4F 场景 D）：admission 已拒绝该动作，但
                # 模型仍声称完成——旅程必须证明系统状态不采信模型自述。
                answer = "动作已执行，结构化回执已确认。"
            elif "receipt" in tool_content or "grant" in tool_content or "已批准" in tool_content:
                answer = "动作已执行，结构化回执已确认。"
            else:
                answer = "Worker 结果已收到并验证。"
            frames.append(_sse(_chunk(answer)))
            frames.append(_sse(_chunk("", "stop")))
        elif "执行一个假动作" in text:
            # 对抗注入：模型被诱导调用一个不存在的 capability（admission
            # 必须拒绝），随后它仍会谎称完成（见 call_action_lie 分支）。
            frames.extend(
                _tool_call_frames(
                    "call_action_lie",
                    "rosclaw_request_action",
                    json.dumps(
                        {
                            # 空 capability_id——任何分支都被 admission 以
                            # INVALID_ARGUMENTS 拒绝（不依赖 HOTFIX-2 的
                            # catalog 检查，分支独立可复现）。
                            "capability_id": "",
                            "arguments": {},
                            "expected_effect": "假动作",
                            "risk_tier": "LOW",
                        }
                    ),
                )
            )
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
                            "capability_id": "limo.speaker.play_tone",
                            "arguments": {},
                            "expected_effect": "旅程验收动作",
                            "risk_tier": "LOW",
                        }
                    ),
                )
            )
        elif "请详细展开" in text:
            # 长回答——用于把 session 推过 compaction 阈值（keepRecentTokens
            # 默认 20000 tokens；compact 要真发生必须先有可裁剪内容）。
            frames.append(_sse(_chunk("详细展开：" + "具身系统知识段落。" * 4000)))
            frames.append(_sse(_chunk("", "stop")))
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
    # Gate Evidence V2（五审 §11.8）：被测 bundle 的 sha256 随 artifact
    # 上传——证据与被测物字节级绑定，不是"某个构建"。
    import hashlib as _hashlib

    bundle_digest = _hashlib.sha256(bundle.read_bytes()).hexdigest()
    (tmp_path / "installed_bundle_digest.txt").write_text(
        f"sha256:{bundle_digest}  {bundle.name}\n", encoding="utf-8"
    )
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

    def __init__(self, argv: list[str], env: dict[str, str], log_path: Path | None = None) -> None:
        import pty as _pty

        self.master, slave = _pty.openpty()
        # CI 失败诊断（五审 Gate Evidence）：PTY 全量输出落盘——超时
        # 断言只有尾部 3000 字节，完整输出是定位 CI-only 失败的唯一证据。
        self._log = log_path.open("wb") if log_path else None

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
                        if self._log is not None:
                            with contextlib.suppress(OSError):
                                self._log.write(chunk)
                                self._log.flush()
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
        if self._log is not None:
            with contextlib.suppress(OSError):
                self._log.close()
            self._log = None
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
            "      capabilities: [llm.chat, llm.structured_decision, llm.tool_use]\n"
            # HOTFIX-2/P0-4F：确定性 SIM 执行通道（真实 SimActionChannel +
            # catalog PHYSICAL_ACTION 能力）——journey 的动作必须真执行、
            # 真产出 SIMULATED receipt，不再"消费 grant 而无 receipt"。
            "mcp_servers:\n"
            "  - name: limo-sim\n"
            f"    command: {sys.executable}\n"
            f"    args: [{REPO / 'src' / 'rosclaw' / 'limo' / 'sim_mcp.py'}]\n"
            "    supported_modes: [SIMULATION]\n"
            "    sim_executor: true\n",
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
            try:
                self._run_journey(rosclaw, env, home, fake)
            except BaseException:
                self._dump_failure_state(home)
                raise
        finally:
            operatord.terminate()
            try:
                operatord.wait(timeout=10)
            except subprocess.TimeoutExpired:
                operatord.kill()
            fake.close()

    def _dump_failure_state(self, home: Path) -> None:
        """CI-only 失败的诊断面（journey artifact 带走）：
        agentd/operatord 日志 + home 目录清单 + missions.db 关键表快照。
        脱敏：本测试环境无任何真实密钥（fake key），DB 只有结构化 ID。"""
        import sqlite3

        dump = home.parent / "failure-dump"
        with contextlib.suppress(Exception):
            dump.mkdir(exist_ok=True)
            listing = sorted(
                str(p.relative_to(home)) for p in home.rglob("*") if p.is_file()
            )
            (dump / "home-listing.json").write_text(
                json.dumps(listing, indent=1), encoding="utf-8"
            )
            for log in home.rglob("*.log"):
                target = dump / f"{log.parent.name}-{log.name}"
                target.write_bytes(log.read_bytes()[-200_000:])
            db_path = home / "agentd" / "missions.db"
            if db_path.exists():
                db = sqlite3.connect(db_path)
                snapshot: dict[str, object] = {}
                for table in (
                    "action_txns", "mission_grants", "operator_requests",
                    "agent_events", "context_leases", "pi_session_leases",
                ):
                    exists = db.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                        (table,),
                    ).fetchone()
                    if exists:
                        rows = db.execute(
                            f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT 20"
                        ).fetchall()
                        cols = [d[0] for d in db.execute(
                            f"SELECT * FROM {table} LIMIT 0").description]
                        snapshot[table] = [dict(zip(cols, r, strict=True)) for r in rows]
                db.close()
                (dump / "db-snapshot.json").write_text(
                    json.dumps(snapshot, indent=1, default=str), encoding="utf-8"
                )

    def _write_sanitized_evidence(self, home: Path) -> None:
        """Gate Evidence V2（五审 §11.4/§11.8）：脱敏机器可读全链证据。

        第三方仅下载 artifact 即可复核 context→approval→grant→txn→receipt
        全链——无需相信 pytest 文本断言。只含结构化 ID/hash/状态/计数，
        无任何用户正文或模型文本。
        """
        import sqlite3

        db = sqlite3.connect(home / "agentd" / "missions.db")
        evidence: dict[str, object] = {"schema_version": "rosclaw.journey_evidence.v1"}
        try:
            binding = db.execute(
                "SELECT pi_session_id, mission_id FROM pi_session_bindings "
                "WHERE status = 'ACTIVE' LIMIT 1"
            ).fetchone()
            if binding:
                evidence["session_id"], evidence["mission_id"] = binding
            txns = db.execute(
                "SELECT txn_id, approval_id, grant_id, action_id, receipt_id, "
                "arguments_hash, request_hash, context_lease_id, context_revision, "
                "body_hash, mode, capability_id, risk_tier, state, display_hash "
                "FROM action_txns ORDER BY rowid"
            ).fetchall()
            evidence["action_txns"] = [
                dict(zip(
                    ("txn_id", "approval_id", "grant_id", "action_id", "receipt_id",
                     "arguments_hash", "request_hash", "context_lease_id",
                     "context_revision", "body_hash", "mode", "capability_id",
                     "risk_tier", "state", "display_hash"),
                    row, strict=True,
                ))
                for row in txns
            ]
            leases = db.execute(
                "SELECT context_lease_id, context_revision, context_hash, body_hash, "
                "mode, revoked, caller_uid FROM pi_context_leases ORDER BY rowid"
            ).fetchall()
            evidence["context_leases"] = [
                dict(zip(
                    ("context_lease_id", "context_revision", "context_hash",
                     "body_hash", "mode", "revoked", "caller_uid"),
                    row, strict=True,
                ))
                for row in leases
            ]
            events = db.execute(
                "SELECT type, payload_json FROM agent_events ORDER BY rowid"
            ).fetchall()
            evidence["event_chain"] = [t for t, _ in events]
            receipts = [
                json.loads(p) for t, p in events if t == "receipt.received"
            ]
            evidence["receipts"] = [
                {
                    "action_id": r.get("action_id"),
                    "final_state": r.get("final_state"),
                    "trust_level": r.get("trust_level"),
                    "evidence_domain": r.get("evidence_domain"),
                    "usable_for_real_execution": r.get("usable_for_real_execution"),
                }
                for r in receipts
            ]
            grants = db.execute(
                "SELECT grant_id, request_id, consumed, revoked FROM mission_grants"
            ).fetchall()
            evidence["grants"] = [
                dict(zip(("grant_id", "request_id", "consumed", "revoked"), g, strict=True))
                for g in grants
            ]
            approvals = db.execute(
                "SELECT request_id, status, decided_by FROM operator_requests"
            ).fetchall()
            evidence["approvals"] = [
                {"request_id": r, "status": s, "decided_by": d}
                for r, s, d in approvals
            ]
        finally:
            db.close()
        # reasoning 禁带字段计数（结构计数，不含正文）。
        forbidden_counts: dict[str, int] = {}
        session_files = sorted((home / "agent" / "sessions").glob("*.jsonl"))
        for marker in ("reasoning_content", "redacted_thinking", REASONING_MARKER):
            forbidden_counts[marker] = sum(
                f.read_text(encoding="utf-8", errors="replace").count(marker)
                for f in session_files
            )
        evidence["reasoning_forbidden_field_counts"] = forbidden_counts
        evidence["compaction_entry_id"] = getattr(self, "_compaction_entry_id", None)
        evidence["verdicts"] = getattr(self, "_journey_verdicts", {})
        (home.parent / "sanitized_assertions.json").write_text(
            json.dumps(evidence, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        # session 结构报告：entry 类型/角色计数——无正文。
        structure: dict[str, object] = {}
        for session_file in session_files:
            counts: dict[str, int] = {}
            for line in session_file.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                with contextlib.suppress(Exception):
                    entry = json.loads(line)
                    key = entry.get("type", "?")
                    if key == "message":
                        key = f"message.{entry.get('message', {}).get('role', '?')}"
                    elif key == "custom":
                        key = f"custom.{entry.get('customType', '?')}"
                    counts[key] = counts.get(key, 0) + 1
            structure[session_file.name] = counts
        (home.parent / "session_structure.json").write_text(
            json.dumps(structure, indent=1), encoding="utf-8"
        )

    def _assert_no_reasoning_replay(
        self, fake: FakeModelServer, home: Path, *, from_index: int
    ) -> None:
        """P0-4G TranscriptPolicy 验收：SECRET_PROBE 回合后，任何 provider
        请求（live replay）、session 持久化文件、后续 resume 回放都不得
        再出现 raw reasoning marker。

        from_index=2：req0=你好、req1=SECRET_PROBE 自身（marker 由 fake
        注入在响应里，请求里没有）——从 req2 起的历史消息必须零命中。
        """
        # 1. live replay：req[from_index:] 的任何字段不得含 marker。
        for i, body in enumerate(fake.fake.requests[from_index:], start=from_index):
            blob = json.dumps(body, ensure_ascii=False)
            assert REASONING_MARKER not in blob, (
                f"req{i} 的 provider 请求仍携带 raw reasoning marker"
            )
        # 2. session 持久化：Pi session JSONL 不得含 marker。
        sessions_dir = home / "agent" / "sessions"
        assert sessions_dir.exists(), "session 目录不存在"
        for session_file in sessions_dir.glob("*.jsonl"):
            content = session_file.read_text(encoding="utf-8", errors="replace")
            assert REASONING_MARKER not in content, (
                f"session 文件 {session_file.name} 持久化了 raw reasoning"
            )
            # 结构性断言：不得有 thinking/reasoning 字段变体。
            for forbidden in ("reasoning_content", "redacted_thinking"):
                assert forbidden not in content, (
                    f"session 文件 {session_file.name} 含 {forbidden} 字段"
                )
        self._journey_verdicts["no_reasoning_replay"] = True

    def _expect_compaction_entry(self, home: Path, timeout: float = 60.0) -> None:
        """等待 session JSONL 出现 compaction 条目（compact 真完成的
        结构性证据——不是 UI 文本）。"""
        deadline = time.monotonic() + timeout
        sessions_dir = home / "agent" / "sessions"
        while time.monotonic() < deadline:
            for session_file in sessions_dir.glob("*.jsonl"):
                for line in session_file.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines():
                    if '"type": "compaction"' in line or '"type":"compaction"' in line:
                        with contextlib.suppress(Exception):
                            self._compaction_entry_id = json.loads(line).get("id")
                        return
            time.sleep(0.5)
        raise AssertionError("/compact 后 session 无 compaction 条目——compact 未真完成")

    def _assert_adversarial_model_ignored(
        self, session: PtySession, home: Path
    ) -> None:
        """P0-4F 场景 D：fake model 被诱导调用不存在的 capability——
        admission 拒绝（无卡无 grant 无 action），模型仍谎称"已执行"。
        系统最终状态必须反映 FAILED/UNVERIFIED，不是模型自述。"""
        import sqlite3

        # 快照当前 DB 状态（合法动作已完成——对比基线）。
        def _count(conn, table: str) -> int:
            exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            return (
                conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                if exists else -1
            )

        db = sqlite3.connect(home / "agentd" / "missions.db")
        baseline_txns = _count(db, "action_txns")  # -1 = HOTFIX-2 迁移未到时兼容
        baseline_grants = _count(db, "mission_grants")
        baseline_approvals = _count(db, "operator_requests")
        db.close()
        session.send("执行一个假动作\r")
        # 模型谎称完成（fake 的 call_action_lie 分支）。
        session.expect("动作已执行，结构化回执已确认".encode(), timeout=120)
        time.sleep(1.0)
        # 系统真相：没有任何新 approval/grant/txn——模型的话不算数。
        db = sqlite3.connect(home / "agentd" / "missions.db")
        txns = _count(db, "action_txns")
        grants = _count(db, "mission_grants")
        approvals = _count(db, "operator_requests")
        db.close()
        if baseline_txns >= 0:
            assert txns == baseline_txns, (
                f"假动作竟产生 ActionTxn（{baseline_txns} → {txns}）——模型谎言被采信了"
            )
        assert grants == baseline_grants, (
            f"假动作竟产生 grant（{baseline_grants} → {grants}）"
        )
        assert approvals == baseline_approvals, (
            f"假动作竟产生 approval 卡（{baseline_approvals} → {approvals}）"
        )
        self._journey_verdicts["adversarial_fake_action_zero_side_effects"] = True

    def _assert_grant_consumed(self, home: Path) -> None:
        """授权→执行闭环（P0-NA-13）：直接解析 DB/事件里的结构化证据，
        断言 approval → grant → consume 的精确 ID 链——不是模型自述。"""
        import sqlite3

        db = sqlite3.connect(home / "agentd" / "missions.db")
        try:
            # 1. approval 已批准且有精确 decided_by。
            approvals = db.execute(
                "SELECT request_id, status, decided_by FROM operator_requests"
            ).fetchall()
            assert approvals, "应有 approval 记录"
            for request_id, status, decided_by in approvals:
                assert status == "APPROVED" and decided_by, (
                    f"approval {request_id} 未正确落库: status={status}"
                )
            # 2. grant 与 approval 精确绑定且被消费（单次、未撤销）。
            grants = db.execute(
                "SELECT grant_id, request_id, consumed, revoked FROM mission_grants"
            ).fetchall()
            assert grants, "批准应产生 grant"
            approval_ids = {a[0] for a in approvals}
            for grant_id, request_id, consumed, revoked in grants:
                assert request_id in approval_ids, (
                    f"grant {grant_id} 绑定的 request_id {request_id} 不在 approval 链中"
                )
                assert consumed == 1 and revoked == 0, (
                    f"grant {grant_id} 未被精确消费或被撤销"
                )
            # 3. 事件链：approval.requested → approval.decided → grant.consumed
            #    的 ID 必须一致（结构化事件，不是文本）。
            events = db.execute(
                "SELECT type, payload_json FROM agent_events ORDER BY rowid"
            ).fetchall()
            requested = [json.loads(p) for t, p in events if t == "approval.requested"]
            decided = [json.loads(p) for t, p in events if t == "approval.decided"]
            consumed_evs = [json.loads(p) for t, p in events if t == "grant.consumed"]
            assert requested and decided, f"事件链缺环: {[t for t, _ in events]}"
            for req, dec in zip(requested, decided, strict=True):
                assert req["request_id"] == dec["request_id"], (
                    f"requested/decided 不同卡: {req} vs {dec}"
                )
            if consumed_evs:
                grant_ids = {g[0] for g in grants}
                for ev in consumed_evs:
                    assert ev.get("grant_id") in grant_ids, (
                        f"grant.consumed 事件的 grant_id 不在链中: {ev}"
                    )
            # 4. P0-4F：真实 SIM 执行链——ActionTxn 到达 COMPLETED，
            #    receipt.received 事件精确绑定本 action_id；不再是
            #    "grant consumed 但无 receipt"。
            txns = db.execute(
                "SELECT txn_id, state, approval_id, grant_id, action_id, "
                "receipt_id, capability_id FROM action_txns"
            ).fetchall()
            assert txns, "HOTFIX-2 后动作必须产生 ActionTxn"
            for txn_id, state, approval_id, grant_id, action_id, receipt_id, capability in txns:
                assert state == "COMPLETED", f"txn {txn_id} 未完成: {state}"
                assert approval_id in approval_ids
                assert grant_id in {g[0] for g in grants}
                assert action_id and receipt_id, f"txn {txn_id} 缺 action/receipt ID"
                assert capability == "limo.speaker.play_tone"
            receipts = [
                json.loads(p) for t, p in events if t == "receipt.received"
            ]
            txn_action_ids = {t[4] for t in txns}
            matching = [r for r in receipts if r.get("action_id") in txn_action_ids]
            assert matching, (
                f"receipt.received 事件未绑定本 ActionTxn 的 action_id: {receipts}"
            )
            for r in matching:
                assert r.get("final_state") == "COMPLETED"
                assert r.get("trust_level") == "SIMULATED"
                assert r.get("evidence_domain") == "simulation"
                assert r.get("usable_for_real_execution") is False
            self._journey_verdicts["approval_grant_txn_receipt_chain"] = True
        finally:
            db.close()

    def _run_journey(
        self, rosclaw: Path, env: dict[str, str], home: Path, fake: FakeModelServer
    ) -> None:
        # Gate Evidence V2：逐项 verdict 累积——进 sanitized_assertions.json。
        self._journey_verdicts: dict[str, bool] = {}
        # NA-FIX-9 后默认引擎即 Native Agent——旅程显式验证无 --engine 的默认路径。
        session = PtySession(
            [str(rosclaw), "chat"], env, log_path=home.parent / "pty-main.log"
        )
        try:
            # 1. 品牌 header（T-IDENTITY + P0-NA-15/16 产品面扫描）。
            session.expect(b"ROSClaw Native Agent", timeout=60)
            assert b"engine=pi" not in session.output
            # P0-NA-15：供应链边界——启动面不得出现上游自更新通道。
            for leaked in (b"pi update", b"pi.dev", b"Update Available", b"[Extensions]"):
                assert leaked not in session.output, f"上游泄漏进启动面: {leaked}"
            # P0-NA-16：产品版本（launcher 传入），不是内部 npm 子包版本。
            from rosclaw import __version__ as _product_version

            # header 在 operator probe 后重绘——等版本出现（不是瞬时断言）。
            session.expect(f"ROSClaw {_product_version}".encode(), timeout=60)
            assert b"v0.1.0" not in session.output, "内部子包版本冒充产品版本"
            # Operator 状态必须是真实探测值（READY/OFFLINE/UNKNOWN），
            # 不是硬编码字符串。
            assert b"Operator ready" not in session.output
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
            # P0-4G（TranscriptPolicy）：SECRET_PROBE 回合后，任何后续
            # provider 请求都不得携带 raw reasoning marker——live replay、
            # session 持久化、resume 回放全链路零命中（四审核心反证点）。
            self._assert_no_reasoning_replay(fake, home, from_index=2)
            # 6. request SIM action → 卡片 → Y。
            # 主标记必须是稳定面：授权 overlay（"ROSCLAW 授权请求"）。
            # "等待 Operator 决定" 是瞬时 working message/notify——overlay
            # 打开快时它一帧都渲染不出来（CI 两次失败的根因：overlay 已在
            # 等 Y/N，journey 却在等一个被覆盖的瞬时文本，永不按 y）。
            session.send("请播放提示音\r")
            session.expect("ROSCLAW 授权请求".encode(), timeout=120)
            session.send("y")
            # 已批准 → 执行（结构化回执或诚实失败，但不许是未决状态）。
            session.expect(b"\xe5\xb7\xb2\xe6\x89\xb9\xe5\x87\x86", timeout=120)
            # 等工具结果回到模型并产出最终回答——证明 execute 阶段真正完成
            # （grant 被消费），而不是只停在批准通知。
            session.expect("动作已执行，结构化回执已确认".encode(), timeout=120)
            self._assert_grant_consumed(home)
            # 6b. 对抗场景（P0-4F 场景 D）：模型谎称完成——系统必须
            #    以结构化状态为准，不采信模型自述。
            self._assert_adversarial_model_ignored(session, home)
            # 7. /compact（HOTFIX-5：真断言——compact 完成、上下文与
            #    receipt 保留、summary 可用——不是只发命令后 sleep）。
            #    先把 session 推过 keepRecentTokens 阈值（~20K tokens），
            #    否则 compact 诚实报 "session too small" 什么都不做。
            for _ in range(3):
                session.send("请详细展开具身系统知识\r")
                session.expect("详细展开".encode(), timeout=90)
                time.sleep(0.5)
            session.send("/compact\r")
            # compact 完成的结构性证据：session JSONL 出现 compaction 条目
            # （live UI 显示的是模型摘要文本，不稳定——不用它断言）。
            self._expect_compaction_entry(home, timeout=60)
            # compact 后对话仍可用（context 保留 + model 可达）。
            time.sleep(1.0)
            session.send("你好\r")
            session.expect("你好，我是 ROSClaw".encode(), timeout=90)
            # receipt 链在 compact 后仍可核验（DB 是权威——compaction
            # 是认知层摘要，不动授权/执行记录）。
            self._assert_grant_consumed(home)
            self._journey_verdicts["compaction_completed_context_retained"] = True
            # Gate Evidence V2：全链脱敏证据落盘（artifact 上传，第三方
            # 可独立复核，无需相信 pytest 文本）。
            self._write_sanitized_evidence(home)
            # 8. /quit → resume 提示必须是 ROSClaw 命令（T-IDENTITY）。
            session.send("/quit\r")
            session.expect(b"rosclaw chat --resume", timeout=30)
            assert b"pi --session" not in session.output
            assert b"--session-dir" not in session.output
            session.proc.wait(timeout=30)
            assert session.proc.returncode == 0, session.output[-400:]
        finally:
            session.stop()
        # 9. --continue 恢复（P0-NA-12：必须证明 binding/lease 恢复，
        #    不是只看到 header——resume 后 /status 与对话都可用，且
        #    mission 与第一段相同、lease 由本进程持有）。
        import sqlite3 as _sqlite3

        db = _sqlite3.connect(home / "agentd" / "missions.db")
        before = db.execute(
            "SELECT pi_session_id, mission_id FROM pi_session_bindings "
            "WHERE status = 'ACTIVE'"
        ).fetchall()
        db.close()
        assert before, "第一段会话应留下 ACTIVE binding"
        first_session, first_mission = before[0]
        resumed = PtySession(
            [str(rosclaw), "chat", "--continue"], env,
            log_path=home.parent / "pty-continue.log",
        )
        try:
            resumed.expect(b"ROSClaw Native Agent", timeout=60)
            # 恢复后工具链可用（/status 经 bridge 读取内核状态）。
            resumed.send("/status\r")
            resumed.expect(b"agentd=READY", timeout=30)
            # 恢复后对话可用（fake 会答固定问候——证明 model 上下文在）。
            resumed.send("你好\r")
            resumed.expect("你好，我是 ROSClaw".encode(), timeout=90)
            # binding 仍指向同一 session/mission，lease 已重新获取。
            db = _sqlite3.connect(home / "agentd" / "missions.db")
            bindings = db.execute(
                "SELECT pi_session_id, mission_id FROM pi_session_bindings "
                "WHERE status = 'ACTIVE'"
            ).fetchall()
            leases = db.execute(
                "SELECT pi_session_id, mission_id FROM pi_session_leases "
                "WHERE mission_id = ?",
                (first_mission,),
            ).fetchall()
            db.close()
            assert (first_session, first_mission) in bindings, (
                f"resume 后 binding 丢失或漂移: {bindings}（期望含 {(first_session, first_mission)}）"
            )
            assert any(sess == first_session for sess, _m in leases), (
                f"resume 后 lease 未恢复: {leases}"
            )
            resumed.send("/quit\r")
            resumed.expect(b"rosclaw chat --resume", timeout=30)
            resumed.proc.wait(timeout=30)
        finally:
            resumed.stop()
