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
import re
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
            tool_call_id = str(last_tool.get("tool_call_id", ""))
            if tool_call_id == "call_action_lie":
                # 对抗场景（P0-4F 场景 D）：admission 已拒绝该动作，但
                # 模型仍声称完成——旅程必须证明系统状态不采信模型自述。
                answer = "动作已执行，结构化回执已确认。"
                frames.append(_sse(_chunk(answer)))
                frames.append(_sse(_chunk("", "stop")))
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
            if tool_call_id == "call_status":
                # 六审 §2.2.5：自然语言 status 的专属回答（与 delegate 的
                # 通用回答区分，避免 expect 误匹配历史文本）。
                answer = "内核状态已读取。"
                frames.append(_sse(_chunk(answer)))
                frames.append(_sse(_chunk("", "stop")))
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
            if tool_call_id == "call_star_action":
                # Journey B 批准腿：admission COMPLETED → 完成回答。
                if "COMPLETED" in tool_content:
                    answer = "位姿移动测试完成，结构化回执已确认。"
                else:
                    answer = f"动作未完成（{tool_content[:80]}）。"
                frames.append(_sse(_chunk(answer)))
                frames.append(_sse(_chunk("", "stop")))
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
            if tool_call_id == "call_task":
                # 八审 P0-5：任务结果即最终回答依据（state VERIFIED +
                # verification PASS 才说完成）。tool 内容可能是包装
                # JSON 或直接内层 JSON——两种都解析。
                state = ""
                verdict = ""
                inner: dict = {}
                with contextlib.suppress(Exception):
                    wrapper = json.loads(tool_content)
                    if isinstance(wrapper, dict) and "content" in wrapper:
                        inner = json.loads(wrapper["content"][0])
                    elif isinstance(wrapper, dict):
                        inner = wrapper
                    state = str(inner.get("state", ""))
                    verdict = str((inner.get("verification") or {}).get("verdict", ""))
                if state == "VERIFIED" and verdict == "PASS":
                    answer = "五角星已绘制完成，几何验证通过。"
                else:
                    answer = f"任务未完成（{state or 'UNKNOWN'}）。"
                frames.append(_sse(_chunk(answer)))
                frames.append(_sse(_chunk("", "stop")))
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
            # 七审 §6 PR-SEVEN-4：五角星任务链——capabilities →
            # plan(COMPUTE) → execute（整条轨迹一个 ExactAction）→
            # trace → verify(COMPUTE) → 基于 verifier 的回答。
            if tool_call_id == "call_caps":
                frames.extend(
                    _tool_call_frames(
                        "call_plan",
                        "rosclaw_compute",
                        json.dumps(
                            {
                                "capability_id": "ur5e.plan_cartesian_path",
                                "arguments": {
                                    "shape": "star5",
                                    "center_x": 0.35,
                                    "center_y": 0.25,
                                    "z": 0.30,
                                    "outer_radius": 0.10,
                                },
                            }
                        ),
                    )
                )
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
            if tool_call_id == "call_plan":
                # 八审 P0-3：不透明句柄——plan 结果只含 plan_id/digest/
                # 摘要；动作只带 plan_id（模型不搬运轨迹/hash）。
                plan_id = ""
                with contextlib.suppress(Exception):
                    wrapper = json.loads(tool_content)
                    inner = json.loads(wrapper["content"][0])
                    plan_id = inner["plan_id"]
                frames.extend(
                    _tool_call_frames(
                        "call_action",
                        "rosclaw_request_action",
                        json.dumps(
                            {
                                "capability_id": "ur5e.execute_plan",
                                "arguments": {"plan_id": plan_id},
                                "expected_effect": "绘制五角星轨迹",
                                "risk_tier": "LOW",
                            }
                        ),
                    )
                )
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
            if tool_call_id == "call_action":
                frames.extend(
                    _tool_call_frames(
                        "call_trace",
                        "rosclaw_observe",
                        json.dumps({"capability_id": "ur5e.get_cartesian_trace"}),
                    )
                )
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
            if tool_call_id == "call_trace":
                # 八审 P0-3：verify 默认验证最近执行的 plan——模型不传 hash。
                frames.extend(
                    _tool_call_frames(
                        "call_verify",
                        "rosclaw_compute",
                        json.dumps(
                            {
                                "capability_id": "ur5e.verify_drawing",
                                "arguments": {},
                            }
                        ),
                    )
                )
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
            if tool_call_id == "call_verify":
                frames.append(_sse(_chunk("五角星已绘制完成，几何验证通过。")))
                frames.append(_sse(_chunk("", "stop")))
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
            # 大道至简 R0-1：Pi 驱动的通用物理工具链（plan → simulate
            # → verify → render）。工具结果可能是 rosclaw_* 包装
            # （{"content": [...]}）或物化工具裸 JSON——两种都解析。
            def _payload() -> dict:
                with contextlib.suppress(Exception):
                    wrapper = json.loads(tool_content)
                    if isinstance(wrapper, dict) and "content" in wrapper:
                        wrapper = json.loads(wrapper["content"][0])
                    if isinstance(wrapper, dict):
                        # rosclaw_execute/compute 投影：{status, value}
                        value = wrapper.get("value")
                        if isinstance(value, dict):
                            return value
                        return wrapper
                return {}

            if tool_call_id == "call_plan2":
                plan_id = str(_payload().get("plan_id", ""))
                assert plan_id, f"plan 结果缺 plan_id：{tool_content[:200]}"
                frames.extend(
                    _tool_call_frames(
                        "call_sim",
                        "ur5e_simulate_cartesian_trajectory",
                        json.dumps({"plan_id": plan_id}),
                    )
                )
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
            if tool_call_id == "call_sim":
                trace_id = str(_payload().get("trace_id", ""))
                assert trace_id, f"simulate 结果缺 trace_id：{tool_content[:200]}"
                frames.extend(
                    _tool_call_frames(
                        "call_vtrack",
                        "simulation_verify_tracking",
                        json.dumps(
                            {"trace_id": trace_id, "max_tracking_error_m": 0.05}
                        ),
                    )
                )
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
            if tool_call_id == "call_vtrack":
                payload = _payload()
                trace_id = ""
                with contextlib.suppress(Exception):
                    trace_id = str(payload.get("trace_id", ""))
                if not trace_id:
                    # verify 结果不回带 trace_id 时从仿真结果上下文取——
                    # 假模型没有记忆，从历史消息里翻最近的 trace_id。
                    for m in reversed(messages):
                        c = str(m.get("content", ""))
                        mm = re.search(r"trace_[0-9a-f]+", c)
                        if mm:
                            trace_id = mm.group(0)
                            break
                frames.extend(
                    _tool_call_frames(
                        "call_render",
                        "simulation_render_trace",
                        json.dumps({"trace_id": trace_id, "format": "gif"}),
                    )
                )
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
            if tool_call_id == "call_render":
                frames.append(_sse(_chunk("五角星已绘制完成，几何验证通过。")))
                frames.append(_sse(_chunk("", "stop")))
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
            if tool_call_id == "call_limo_tone":
                # 交叉本体拒绝的诚实回答（六审 §6.3.10）。
                answer = "本体不兼容，未执行。"
                frames.append(_sse(_chunk(answer)))
                frames.append(_sse(_chunk("", "stop")))
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
            if tool_call_id == "call_home_action":
                # 七审 PR-SEVEN-7 Journey B deny 腿：按工具结果诚实
                # 报告（拒绝≠执行）。
                if "DECLINED" in tool_content or "拒绝" in tool_content:
                    answer = "动作已被操作员拒绝，未执行。"
                else:
                    answer = "已回到零点。"
                frames.append(_sse(_chunk(answer)))
                frames.append(_sse(_chunk("", "stop")))
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
            if tool_call_id == "call_direct":
                # PR-H1：主会话直接工作的最终回答（无 Worker）。
                answer = "已直接完成日志总结（同一 session，未委派 Worker）。"
                frames.append(_sse(_chunk(answer)))
                frames.append(_sse(_chunk("", "stop")))
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
            if "receipt" in tool_content or "grant" in tool_content or "已批准" in tool_content:
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
        elif (
            "后台 Worker 已完成" in text
            or "后台 Worker 已完成" in _text(messages[-1] if messages else {})
        ):
            # 十审 W2：完成推送回合——custom message 注入（triggerTurn）
            # 后模型做最终综合。独立文本，旅程可精确等待（不与 delegate
            # 步骤的"Worker 结果已收到并验证"混淆）。
            frames.append(_sse(_chunk("Worker 结果已综合给用户。")))
            frames.append(_sse(_chunk("", "stop")))
        elif "委派" in text:
            # PR-H1（ADR-0012）：模型面不再有 task_submit/delegate——
            # Native Agent 用自己的工作工具在同一 session 直接完成。
            frames.extend(
                _tool_call_frames(
                    "call_direct",
                    "bash",
                    json.dumps({"command": "echo 日志要点已归纳"}),
                )
            )
        elif "画五角星" in text:
            # 大道至简 R0-1：画五角星不再被确定性链接管——Pi（假模型）
            # 自己驱动通用物理工具链（0905 真实会话同款四步）：
            # plan → simulate → verify → render → 基于结果的回答。
            frames.extend(
                _tool_call_frames(
                    "call_plan2",
                    "trajectory_generate_planar_path",
                    json.dumps(
                        {
                            "shape": "star5",
                            "center_m": [0.35, 0.25, 0.30],
                            "scale_m": 0.10,
                            "plane": "xy",
                            "max_segment_m": 0.02,
                        }
                    ),
                )
            )
        elif "你这是啥" in text:
            # 大道至简 R0-1：解释性追问也进 Pi——Pi 读账本自己回答
            # （EXPLAIN_HANDLER 退役）。
            frames.append(
                _sse(_chunk("刚才的任务是让机械臂画五角星，仿真已完成。"))
            )
            frames.append(_sse(_chunk("", "stop")))
        elif "位姿移动测试" in text:
            # Journey B（ask 策略，PR-H9）：人工卡走唯一授权面
            # rosclaw_request_action（admission）。0827 P0-1 后已知
            # recipe 由 Input Arbiter 认领（suppress 模型回合）——
            # 授权卡旅程改用非 recipe 指令保持模型腿可测。
            frames.extend(
                _tool_call_frames(
                    "call_star_action",
                    "rosclaw_request_action",
                    json.dumps(
                        {
                            "capability_id": "ur5e.move_to_pose",
                            "arguments": {
                                "x": 0.35, "y": 0.25, "z": 0.40,
                            },
                            "expected_effect": "小幅位姿移动测试",
                            "risk_tier": "LOW",
                        }
                    ),
                )
            )
        elif "回到零点" in text:
            # 七审 PR-SEVEN-7 Journey B：单独动作（deny 腿）——一次
            # 人工拒绝必须 fail closed（无 txn、无 grant、诚实回答）。
            frames.extend(
                _tool_call_frames(
                    "call_home_action",
                    "rosclaw_request_action",
                    json.dumps(
                        {
                            "capability_id": "ur5e.move_to_pose",
                            "arguments": {
                                "x": 0.35, "y": 0.25, "z": 0.40,
                            },
                            "expected_effect": "机械臂回到零点",
                            "risk_tier": "LOW",
                        }
                    ),
                )
            )
        elif "播放提示音" in text or "初始位姿" in text:
            # 六审 §6.3.10：LIMO 动作在 UR5e body 上——建卡前必须
            # BODY_CAPABILITY_MISMATCH。
            frames.extend(
                _tool_call_frames(
                    "call_limo_tone",
                    "rosclaw_request_action",
                    json.dumps(
                        {
                            "capability_id": "limo.speaker.play_tone",
                            "arguments": {},
                            "expected_effect": "交叉本体动作（应被拒）",
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
        # 0902 R1-b：provider 停滞故障注入——挂起的请求永不写回
        # （watchdog 必须在窗口内取消，不能让回合静默死等）。
        if getattr(self.fake, "stall", False):
            import threading as _th

            _th.Event().wait(120)
            return
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


def _bridge_call(home: Path, method: str, params: dict) -> dict:
    """pi-bridge UDS JSONL 调用（与 packages/rosclaw-agent bridge-client
    同一 wire 格式）——Journey C 的 REAL 边界探测用。"""
    token = (home / "run" / "agentd-control.token").read_text(encoding="utf-8").strip()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(10)
        sock.connect(str(home / "run" / "pi-bridge.sock"))
        sock.sendall(
            (json.dumps({"method": method, "params": {"token": token, **params}}) + "\n").encode()
        )
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                break
            buf += chunk
    finally:
        sock.close()
    return json.loads(buf.split(b"\n", 1)[0])


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


# 七审 PR-SEVEN-7：版本在 import 期解析——journey 运行期间源码
# checkout 被改名隐藏（editable 安装路径同时失效），函数内 import
# rosclaw 会炸。
from rosclaw import __version__ as PRODUCT_VERSION  # noqa: E402,N812
from rosclaw.agentd.operator_socket import display_hash_for  # noqa: E402
from rosclaw.contracts.operator.approval import ApprovalRequestV2  # noqa: E402
from rosclaw.firstboot.os_isolation import probe_os_isolation  # noqa: E402


@contextlib.contextmanager
def _hidden_source_checkout():
    """七审 PR-SEVEN-7：journey 运行期间源码 checkout 不可达。

    "clean-install 闭环"的证据不能被仓库源码路径喂绿——安装产物若
     secretly 引用 REPO/src 下的 executor/包，改名后旅程必然失败。
    rename 是最强的可恢复隔离（chmod 撤权对 root 无效）。finally
    恢复；进程崩溃时 checkout 留在 .journey-hidden——CI job 即失败，
    本地手动改回即可。
    """
    hidden = REPO.with_name(REPO.name + ".journey-hidden")
    os.rename(REPO, hidden)
    try:
        yield hidden
    finally:
        os.rename(hidden, REPO)


def _journey_scope(prefix: Path, journey: str, checkout_accessible: bool) -> dict:
    """七审 PR-SEVEN-7：journey 证据 scope——独立 verifier 据此确认
    证据确实是 clean-install 产物，而不是夹具/源码路径喂绿。"""
    import hashlib as _hashlib

    kit_manifests = sorted(prefix.glob("**/rosclaw/sim/kits/ur5e_sim.json"))
    kit_digest = ""
    if kit_manifests:
        kit_digest = _hashlib.sha256(kit_manifests[0].read_bytes()).hexdigest()
    return {
        "journey": journey,
        "install_origin": "release_tarball",
        "config_origin": "generated_no_server_fixtures",
        "robot_kit_digest": f"sha256:{kit_digest}" if kit_digest else "",
        "source_checkout_accessible": checkout_accessible,
    }


_ANSI_RE = None


def _strip_ansi(data: bytes) -> bytes:
    """去 ANSI 转义/控制序列（PTY 文本断言用——换行重绘会把控制码
    插进文本中间）。"""
    global _ANSI_RE
    if _ANSI_RE is None:
        import re as _re

        # 只去 CSI/字符集序列——OSC（窗口标题 \x1b]0;...\x07）保留：
        # 旅程的品牌标题断言匹配的就是 OSC 内容。
        _ANSI_RE = _re.compile(rb"\x1b\[[0-9;?]*[a-zA-Z]|\x1b[()][0-9A-B]")
    return _ANSI_RE.sub(b"", data)


class PtySession:
    """最小 PTY 驱动：expect/send。"""

    def __init__(self, argv: list[str], env: dict[str, str], log_path: Path | None = None,
                 cwd: Path | None = None) -> None:
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
            cwd=str(cwd) if cwd else None,
        )
        os.close(slave)
        self.output = b""
        # 六审 §7：长行在窄 PTY 换行后 ANSI 控制码会把文本切碎——
        # expect 一律匹配去 ANSI 的缓冲（raw output 保留给日志/诊断）。
        self.clean = b""
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
                            self.clean += _strip_ansi(chunk)
                            self.last_at = time.monotonic()
                        if self._log is not None:
                            with contextlib.suppress(OSError):
                                self._log.write(chunk)
                                self._log.flush()
                except OSError:
                    break

        self._drain_thread = threading.Thread(target=_drain, daemon=True)
        self._drain_thread.start()

    def expect(self, marker: bytes, timeout: float = 60.0, *, after: int = 0) -> bytes:
        # 七审 PR-SEVEN-7：after 偏移——同一标记（如授权卡）在一次
        # 会话里出现多次时，只匹配 after 之后的新内容。
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if marker in self.clean[after:]:
                    return self.clean
            if self.proc.poll() is not None:
                with self._lock:
                    if marker in self.clean[after:]:
                        return self.clean
                break
            time.sleep(0.1)
        with self._lock:
            tail = self.output[-3000:]
        raise AssertionError(f"PTY 超时未等到 {marker!r}；已收输出尾部: {tail!r}")

    def expect_with_resend(
        self,
        marker: bytes,
        payload: str,
        timeout: float = 60.0,
        *,
        after: int = 0,
        interval: float = 2.0,
    ) -> bytes:
        """发送 payload 直到 marker 出现（七审 PR-SEVEN-7：overlay 聚焦/
        输入时机竞态的确定性重试——授权卡 decided 后 handleInput 忽略
        后续按键，/quit 重复发送无害）。"""
        deadline = time.monotonic() + timeout
        last_sent = 0.0
        while time.monotonic() < deadline:
            with self._lock:
                if marker in self.clean[after:]:
                    return self.clean
            if self.proc.poll() is not None:
                with self._lock:
                    if marker in self.clean[after:]:
                        return self.clean
                break
            now = time.monotonic()
            if now - last_sent >= interval:
                self.send(payload)
                last_sent = now
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


def _prepare_installed_chat(
    tmp_path: Path, fake: FakeModelServer, prefix: Path, *, sim_policy: str | None = None
) -> tuple[Path, dict[str, str], Path]:
    """安装产物 + 声明式最小配置（kernel fake base_url + Pi 侧
    models.json）——七审 PR-SEVEN-1.8：禁止手写 MCP server 配置/引用
    仓库源码路径，UR5e 能力必须来自发行包第一方 Robot Kit 自动激活。
    sim_policy=ask 时预写 safety.json（Journey B）。"""
    home = tmp_path / "rh"
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
    if sim_policy is not None:
        (home / "agent" / "safety.json").write_text(
            json.dumps({"sim_policy": sim_policy}), encoding="utf-8"
        )
    rosclaw = prefix / "bin" / "rosclaw"
    env = dict(
        os.environ,
        ROSCLAW_HOME=str(home),
        TERM="xterm",
        FAKE_JOURNEY_KEY="sk-fake-journey",
        KIMI_API_KEY="sk-fake-journey",
        # 六审 §8：chrome 走 i18n catalog——旅程固定 en-US（断言
        # 与 locale 无关的稳定性由 catalog parity 测试保证）。
        ROSCLAW_UI_LOCALE="en-US",
        # 0902 R1-a：全局环境变量授权已从正式路径删除——无 bwrap
        # 主机走会话内确认卡（委派腿回答），有 bwrap 主机沙箱直跑。
        PATH=f"{prefix / 'bin'}:{os.environ['PATH']}",
    )
    return home, env, rosclaw


@pytest.mark.slow
class TestProductJourney:
    def test_full_journey_pty(self, tmp_path: Path) -> None:
        fake = FakeModelServer(log_path=tmp_path / "fake-requests.jsonl")
        prefix, _root = _build_and_install(tmp_path)
        home, env, rosclaw = _prepare_installed_chat(tmp_path, fake, prefix)
        # 七审 PR-SEVEN-7：journey scope 随证据落盘（install_origin/
        # config_origin/robot_kit_digest/source_checkout_accessible）。
        self._journey_scope = _journey_scope(prefix, "A", checkout_accessible=False)
        # 六审 §7（黑盒验收）：不再手工 enroll/start operatord——真实
        # 用户路径就是直接 chat；Operator 初始化必须在 TUI 内单键完成。
        # 七审 PR-SEVEN-7：journey 运行期间源码 checkout 改名隐藏——
        # 安装产物若偷偷引用仓库路径，旅程必然失败。
        try:
            with _hidden_source_checkout():
                assert not REPO.exists(), "源码 checkout 隐藏失败"
                self._run_journey(rosclaw, env, home, fake)
        except BaseException:
            self._dump_failure_state(home)
            raise
        finally:
            fake.close()
        (tmp_path / "journey-scope.json").write_text(
            json.dumps(self._journey_scope, indent=1), encoding="utf-8"
        )

    def test_journey_b_ask_every_time(self, tmp_path: Path) -> None:
        """七审 PR-SEVEN-7 Journey B：SIM ask-every-time——

        用户启用每次确认策略；TUI 内一键 Operator 初始化；exact card
        一次批准整条轨迹（不是每个插值点一张卡）；deny fail closed
        （无 txn、无 grant、模型诚实报告拒绝）。
        """
        fake = FakeModelServer(log_path=tmp_path / "fake-requests.jsonl")
        prefix, _root = _build_and_install(tmp_path)
        home, env, rosclaw = _prepare_installed_chat(
            tmp_path, fake, prefix, sim_policy="ask"
        )
        self._journey_scope = _journey_scope(prefix, "B", checkout_accessible=False)
        try:
            with _hidden_source_checkout():
                assert not REPO.exists(), "源码 checkout 隐藏失败"
                self._run_journey_b(rosclaw, env, home, fake)
        except BaseException:
            self._dump_failure_state(home)
            raise
        finally:
            fake.close()
        (tmp_path / "journey-scope.json").write_text(
            json.dumps(self._journey_scope, indent=1), encoding="utf-8"
        )

    def _run_journey_b(
        self, rosclaw: Path, env: dict[str, str], home: Path, fake: FakeModelServer
    ) -> None:
        fake.fake._ask_policy = True
        self._journey_verdicts = {}
        session = PtySession(
            [str(rosclaw), "chat"], env, log_path=home.parent / "pty-main.log"
        )
        try:
            session.expect(b"ROSClaw Native Agent", timeout=60)
            session.expect(f"ROSClaw {PRODUCT_VERSION}".encode(), timeout=60)
            # ask 策略 + Operator 未就绪 → 启动面给一键初始化提示。
            session.expect(b"Press Shift+Ctrl+B", timeout=90)
            # TUI 内一键初始化（不离开终端、不另开命令行）。
            session.send("/operator-init\r")
            session.expect(b"Operator initialized", timeout=90)
            # operator 探测复探后 Header 显示真实 READY。
            session.expect(b"Operator Ready", timeout=60)
            self._journey_verdicts["operator_bootstrapped_in_tui"] = True
            # -- 批准腿：一次人工卡覆盖整条轨迹 --------------------------
            # 0827 P0-1：已知 recipe（画五角星）由 Input Arbiter 认领后
            # suppress 模型回合——授权卡旅程改用非 recipe 指令（位姿
            # 移动）保持模型腿/人工卡链路真实可测。
            approve_start = len(session.clean)
            session.send("让机械臂做一次小幅位姿移动测试\r")
            overlay_at = len(session.clean)
            session.expect("ROSCLAW 授权请求".encode(), timeout=240)
            # overlay 聚焦竞态：渲染出现 ≠ 键盘路由就绪——重试发 y
            # 直到任务完成文本出现（decided 后卡片立即关闭，决定行
            # 不一定渲染——不能用它当 marker）。
            session.expect_with_resend(
                "位姿移动测试完成，结构化回执已确认。".encode(), "y", timeout=240
            )
            segment = session.clean[approve_start:]
            assert b"POLICY_AUTO" not in segment, "ask 策略竟走政策自动授权"
            import sqlite3

            db = sqlite3.connect(home / "agentd" / "missions.db")
            approved = db.execute(
                "SELECT decided_by FROM operator_requests "
                "WHERE status = 'APPROVED' ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            assert approved and "POLICY_AUTO" not in str(approved[0]), (
                f"ask 策略的批准必须来自人工决定: {approved}"
            )
            cards = db.execute("SELECT COUNT(*) FROM operator_requests").fetchone()[0]
            assert cards == 1, f"整条轨迹应只有一张人工卡: {cards}"
            db.close()
            self._journey_verdicts["single_card_covers_whole_trajectory"] = True
            # PR-H9：批准腿的 admission 证据——txn COMPLETED + receipt
            # 事件（task_records 已删；卡/txn/事件是权威链）。
            import sqlite3 as _sq2

            _db2 = _sq2.connect(home / "agentd" / "missions.db")
            _txn = _db2.execute(
                "SELECT state FROM action_txns ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            _rcpt = _db2.execute(
                "SELECT COUNT(*) FROM agent_events "
                "WHERE type = 'receipt.received'"
            ).fetchone()[0]
            _db2.close()
            assert _txn and _txn[0] == "COMPLETED", f"txn 未完成: {_txn}"
            assert _rcpt >= 1, "缺 receipt.received 事件"
            self._journey_verdicts["star_trajectory_verified"] = True
            # -- deny 腿：人工拒绝 fail closed ----------------------------
            deny_start = len(session.clean)
            session.send("让机械臂回到零点\r")
            session.expect("ROSCLAW 授权请求".encode(), timeout=240, after=overlay_at)
            session.expect_with_resend(
                "动作已被操作员拒绝，未执行。".encode(), "n", timeout=180, after=deny_start
            )
            db = sqlite3.connect(home / "agentd" / "missions.db")
            denied = db.execute(
                "SELECT COUNT(*) FROM operator_requests WHERE status = 'DENIED'"
            ).fetchone()[0]
            # txn 在 propose 时建行（AWAITING_OPERATOR），拒绝后转
            # DECLINED——fail closed 的断言是"无执行态事务"，不是无行。
            txn_states = db.execute(
                "SELECT state, COUNT(*) FROM action_txns GROUP BY state"
            ).fetchall()
            grants = db.execute("SELECT COUNT(*) FROM mission_grants").fetchone()[0]
            db.close()
            assert denied == 1, f"拒绝未记录: {denied}"
            states = dict(txn_states)
            assert states.get("COMPLETED", 0) == 1 and states.get("DECLINED", 0) == 1, (
                f"txn 状态不符（期望 1 COMPLETED + 1 DECLINED）: {states}"
            )
            assert grants == 1, f"被拒绝的动作竟产生 grant: {grants}"
            self._journey_verdicts["deny_fail_closed"] = True
            self._write_sanitized_evidence(home)
            session.expect_with_resend(b"rosclaw continue", "/quit\r", timeout=60)
            session.proc.wait(timeout=30)
            assert session.proc.returncode == 0, session.clean[-400:]
        finally:
            session.stop()

    def test_journey_c_real_boundary(self, tmp_path: Path) -> None:
        """七审 PR-SEVEN-7 Journey C：REAL hard boundary——

        安装产物上 REAL/SHADOW mission 显式 MODE_FORBIDDEN（不是静默
        降级）；零 REAL 工件（mission/grant/txn）；SIM 授权链不跨
        REAL（签名链/no-presence 拒签由单测覆盖——见 seven-6）。
        """
        fake = FakeModelServer(log_path=tmp_path / "fake-requests.jsonl")
        prefix, _root = _build_and_install(tmp_path)
        home, env, rosclaw = _prepare_installed_chat(tmp_path, fake, prefix)
        self._journey_scope = _journey_scope(prefix, "C", checkout_accessible=False)
        try:
            with _hidden_source_checkout():
                assert not REPO.exists(), "源码 checkout 隐藏失败"
                self._run_journey_c(rosclaw, env, home)
        except BaseException:
            self._dump_failure_state(home)
            raise
        finally:
            fake.close()
        (tmp_path / "journey-scope.json").write_text(
            json.dumps(self._journey_scope, indent=1), encoding="utf-8"
        )

    def _run_journey_c(self, rosclaw: Path, env: dict[str, str], home: Path) -> None:
        self._journey_verdicts = {}
        session = PtySession(
            [str(rosclaw), "chat"], env, log_path=home.parent / "pty-main.log"
        )
        try:
            session.expect(b"ROSClaw Native Agent", timeout=60)
            session.expect(f"ROSClaw {PRODUCT_VERSION}".encode(), timeout=60)
            # 先证明输入管道活着（一条普通对话回合）——/quit 石沉大海
            # 时能把"输入未路由"和"quit 本身坏了"区分开。
            session.send("你好\r")
            session.expect("你好，我是 ROSClaw".encode(), timeout=90)
            # 等 bridge token 落盘（agentd 就绪）。
            token_file = home / "run" / "agentd-control.token"
            deadline = time.monotonic() + 60
            while not token_file.exists() and time.monotonic() < deadline:
                time.sleep(0.2)
            assert token_file.exists(), "agentd control token 未落盘"
            # REAL/SHADOW mission 显式拒绝（MODE_FORBIDDEN，不是静默降级）。
            for mode in ("REAL", "SHADOW"):
                result = _bridge_call(
                    home, "pi.mission.create", {"goal": "boundary probe", "mode": mode}
                )
                assert not result.get("ok"), f"{mode} mission 竟被创建: {result}"
                assert result.get("code") == "MODE_FORBIDDEN", result
            self._journey_verdicts["real_shadow_mission_refused"] = True
            # 对照：默认 SIM 会话本身活着（header 已断言）。
            # 零 REAL 工件：mission/grant/txn 全部不存在非 SIMULATION 行。
            import sqlite3

            db = sqlite3.connect(home / "agentd" / "missions.db")
            tables = {
                r[0]
                for r in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "missions" in tables:
                non_sim = db.execute(
                    "SELECT COUNT(*) FROM missions WHERE mode != 'SIMULATION'"
                ).fetchone()[0]
                assert non_sim == 0, f"存在非 SIM mission: {non_sim}"
            for table in ("mission_grants", "action_txns", "operator_requests"):
                if table in tables:
                    count = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    assert count == 0, f"边界探测不应产生 {table} 行: {count}"
            db.close()
            self._journey_verdicts["zero_real_artifacts"] = True
            session.expect_with_resend(b"rosclaw continue", "/quit\r", timeout=60)
            session.proc.wait(timeout=30)
            assert session.proc.returncode == 0, session.clean[-400:]
        finally:
            session.stop()

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
        evidence: dict[str, object] = {"schema_version": "rosclaw.journey_evidence.v2"}
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
                    # 六审 §9.2.1：receipt_id 独立副本——第三方可直接比
                    # 对 txn.receipt_id ↔ receipt event。
                    "receipt_id": r.get("receipt_id"),
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
            kernel_tasks = db.execute(
                "SELECT task_id, state, active_revision FROM tasks ORDER BY rowid"
            ).fetchall()
            evidence["kernel_tasks"] = [
                dict(zip(("task_id", "state", "active_revision"), r, strict=True))
                for r in kernel_tasks
            ]
            kernel_artifacts = db.execute(
                "SELECT artifact_id, task_id, path, media_type, sha256, size_bytes "
                "FROM artifacts ORDER BY rowid"
            ).fetchall()
            evidence["kernel_artifacts"] = [
                dict(zip(
                    ("artifact_id", "task_id", "path", "media_type",
                     "sha256", "size_bytes"),
                    r, strict=True,
                ))
                for r in kernel_artifacts
            ]
            approvals = db.execute(
                "SELECT request_id, status, decided_by, request_json FROM operator_requests"
            ).fetchall()
            evidence["approvals"] = []
            for r, s, d, request_json in approvals:
                # 六审 §9.2.2：display/action intent hash 独立副本——
                # approval↔txn 的 hash 关系可离线复核。
                intent_hash = ""
                with contextlib.suppress(Exception):
                    exact = json.loads(
                        json.loads(request_json).get("exact_action_json") or "{}"
                    )
                    intent_hash = str(exact.get("action_intent_hash") or "")
                req_obj = ApprovalRequestV2.model_validate_json(request_json)
                evidence["approvals"].append(
                    {
                        "request_id": r,
                        "status": s,
                        "decided_by": d,
                        "display_hash": display_hash_for(req_obj),
                        "action_intent_hash": intent_hash,
                    }
                )
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
        # 七审 PR-SEVEN-7：journey scope（install_origin/config_origin/
        # robot_kit_digest/source_checkout_accessible）——独立 verifier
        # 据此确认证据是 clean-install 产物而非夹具喂绿。
        scope = getattr(self, "_journey_scope", None)
        if scope is not None:
            evidence["journey_scope"] = scope
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

    def _assert_star_verified(self, home: Path) -> None:
        """PR-H9：五角星任务级证据——kernel 任务 + 产物账本（gif/
        trace 带 sha256 + 非空）+ 模型上下文无完整插值点数组。
        （task_records/ActionTxn 已删——确定性链的验收证据归
        SimTrajectoryService 产物 + kernel 账本。）"""
        import sqlite3

        db = sqlite3.connect(home / "agentd" / "missions.db")
        tasks = db.execute("SELECT task_id, state FROM tasks").fetchall()
        artifacts = db.execute(
            "SELECT path, media_type, sha256, size_bytes FROM artifacts"
        ).fetchall()
        db.close()
        assert tasks, "kernel 无任务——仿真未走 rosclaw_task 闭环"
        gif = [a for a in artifacts if str(a[0]).endswith(".gif")]
        trace = [a for a in artifacts if str(a[0]).endswith("trace.json")]
        assert gif and gif[0][2] and gif[0][3] > 1000, (
            f"GIF 产物未登记（带 sha+非空）: {artifacts}"
        )
        assert trace and trace[0][2] and trace[0][3] > 0, (
            f"trace 产物未登记: {artifacts}"
        )
        self._journey_verdicts["star_trajectory_verified"] = True
        # 八审 P0-3 验收：模型上下文（session JSONL）不得出现完整
        # 插值点数组。
        sessions_dir = home / "agent" / "sessions"
        for session_file in sessions_dir.glob("*.jsonl"):
            content = session_file.read_text(encoding="utf-8", errors="replace")
            assert '"points": [{' not in content, (
                "模型上下文泄漏完整插值点数组（plan 必须是句柄视图）"
            )
        self._journey_verdicts["no_payload_in_model_context"] = True

    def _assert_sim_task_evidence(self, home: Path) -> None:
        """PR-H9：SIM 自动执行的证据面——kernel 任务存在、GIF 产物
        登记、零 Operator 卡（确定性任务链无需政策/人工卡）。"""
        import sqlite3

        db = sqlite3.connect(home / "agentd" / "missions.db")
        tasks = db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        gifs = db.execute(
            "SELECT COUNT(*) FROM artifacts WHERE path LIKE '%.gif'"
        ).fetchone()[0]
        cards = db.execute("SELECT COUNT(*) FROM operator_requests").fetchone()[0]
        db.close()
        assert tasks >= 1, "kernel 无任务记录"
        assert gifs >= 1, "GIF 未登记进产物账本"
        assert cards == 0, f"默认安全 SIM 竟产生审批卡: {cards}"


    def _assert_cross_body_rejected(self, session: PtySession, home: Path) -> None:
        """六审 §6.3.10 + 七审 kit 化：LIMO 动作不属于 UR5e 机器人——
        建卡前拒绝，零副作用。kit 时代 LIMO capability 不在本机目录
        （CAPABILITY_UNKNOWN）或显式 body 不兼容（BODY_CAPABILITY_
        MISMATCH，单测覆盖）——两者都是 fail closed。"""
        import sqlite3

        db = sqlite3.connect(home / "agentd" / "missions.db")

        def _count(table: str) -> int:
            return db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

        baseline = (_count("action_txns"), _count("mission_grants"), _count("operator_requests"))
        db.close()
        session.send("请播放提示音\r")
        session.expect("本体不兼容，未执行".encode(), timeout=120)
        time.sleep(1.0)
        db = sqlite3.connect(home / "agentd" / "missions.db")
        after = (_count("action_txns"), _count("mission_grants"), _count("operator_requests"))
        db.close()
        assert after == baseline, (
            f"交叉本体动作竟产生副作用: {baseline} → {after}"
        )
        # 拒绝码必须出现在发给模型的 tool 结果里（不是模型自编理由）。
        assert (
            b"BODY_CAPABILITY_MISMATCH" in session.clean
            or b"CAPABILITY_UNKNOWN" in session.clean
        ), "缺交叉本体拒绝证据（BODY_CAPABILITY_MISMATCH 或 CAPABILITY_UNKNOWN）"
        self._journey_verdicts["cross_body_action_rejected_before_card"] = True

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
                assert capability == "ur5e.execute_plan"
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
            assert b"engine=pi" not in session.clean
            # P0-NA-15：供应链边界——启动面不得出现上游自更新通道。
            for leaked in (b"pi update", b"pi.dev", b"Update Available", b"[Extensions]"):
                assert leaked not in session.clean, f"上游泄漏进启动面: {leaked}"
            # P0-NA-16：产品版本（launcher 传入），不是内部 npm 子包版本。
            # header 在 operator probe 后重绘——等版本出现（不是瞬时断言）。
            session.expect(f"ROSClaw {PRODUCT_VERSION}".encode(), timeout=60)
            assert b"v0.1.0" not in session.clean, "内部子包版本冒充产品版本"
            # Operator 状态必须是真实探测值（READY/OFFLINE/UNKNOWN），
            # 不是硬编码字符串。
            assert b"Operator ready" not in session.clean
            # 七审 §2.5：默认安全 SIM 自动执行（POLICY_AUTO）——不需要
            # Operator 初始化，不弹逐动作人工卡（ask-every-time 旅程在
            # SEVEN-7 单独覆盖 bootstrap 路径）。
            self._journey_verdicts["operator_not_required_for_safe_sim"] = True
            # 0902 R1-c（§5.3）：无可用 OS 沙箱（缺失或 present-but-
            # broken——userns 受限）时，会话开始即一次性提示沙箱缺失 +
            # doctor 修复入口（任务开始前），而不是执行到一半才甩卡。
            # 探测单源 = os_isolation.probe_os_isolation（功能 smoke，
            # 不是 which——本机 bwrap 装了但 RTM_NEWADDR 被拒）。
            # （模块级 import——journey 运行期间源码 checkout 隐藏。）
            if not probe_os_isolation()["isolation_ready"]:
                session.expect("本机无 OS 沙箱".encode(), timeout=30)
                assert b"rosclaw doctor" in session.clean, (
                    "沙箱提示必须带 doctor 修复入口"
                )
                self._journey_verdicts["sandbox_notice_at_session_start"] = True
            # 2. 普通对话。
            session.send("你好\r")
            session.expect("你好，我是 ROSClaw".encode(), timeout=90)
            # 3. T-REASONING：推理标记绝不出现。
            marker_at = len(session.clean)
            session.send("SECRET_PROBE 测试\r")
            session.expect("这是最终回答".encode(), timeout=90)
            assert REASONING_MARKER.encode() not in session.clean[marker_at:]
            time.sleep(3.0)  # 等回合完全 settled 再发命令
            # 4. /status。
            session.send("/status\r")
            session.expect(b"agentd=READY", timeout=30)
            time.sleep(2.0)
            # 5. delegate worker。
            session.send("请委派 worker 总结这段日志\r")
            # 0902 R1-a：无 bwrap 主机上 bash 降级走确认卡（允许一次
            # = 第一项，Enter 选定）——不再是全局环境变量。有可用 OS
            # 沙箱的主机直跑（无卡）。守卫必须功能探测（bwrap 装了但
            # userns 受限 = 不可用；which 误判曾让本机从未走过批准路径
            # ——卡 120s 超时默认拒绝蒙混过关，批准腿零覆盖）。
            if not probe_os_isolation()["isolation_ready"]:
                session.expect(
                    "本机无 OS 沙箱".encode(), timeout=120,
                )
                # 允许一次（第一项）。Enter 竞态实证：对话框首帧
                # 渲染与键盘接管之间存在窗口——标题仍在可见期内
                # 每 1s 重试 Enter，直到对话框消失。
                for _ in range(15):
                    time.sleep(1.0)
                    session.send("\r")
                    time.sleep(0.5)
                    if "本机无 OS 沙箱".encode() not in session.clean[-4000:]:
                        break
                # 批准路径必须真实走到（不是 120s 超时默认拒绝蒙混）。
                session.expect("已批准——原操作继续".encode(), timeout=30)
                self._journey_verdicts["shell_approval_card_approved"] = True
            # PR-H1：Native 直接完成（不委派）——同一 session 的工具执行。
            session.expect("已直接完成日志总结".encode(), timeout=180)
            # P0-4G（TranscriptPolicy）：SECRET_PROBE 回合后，任何后续
            # provider 请求都不得携带 raw reasoning marker——live replay、
            # session 持久化、resume 回放全链路零命中（四审核心反证点）。
            self._assert_no_reasoning_replay(fake, home, from_index=2)
            # 5b. 自然语言 status（六审 §2.2.5）：模型调用的 rosclaw_status
            #     必须与 /status 同一 UDS 快照——此前它访问旧 HTTP
            #     127.0.0.1:8765（chat 的 agentd 用 port=0，必然误报
            #     UNREACHABLE）。PTY 输出与发给模型的 tool 结果双重断言。
            session.send("读取系统状态\r")
            status_start = len(session.clean)
            session.expect("内核状态已读取".encode(), timeout=120)
            assert b"127.0.0.1:8765" not in session.clean, (
                "Native Agent 输出仍引用旧 HTTP 面 8765"
            )
            # 0902 R3-c（§6.1 三层界面）：status 的用户面是单行摘要——
            # 整段原始 JSON 不得进 scrollback（0902 实录：16 行 JSON
            # 直接上屏）。模型面仍是完整 JSON（下面的 tool 结果断言
            # 不受影响 = 诚实性不降级）。
            status_segment = session.clean[status_start:]
            assert b'"agentd": "READY"' not in status_segment, (
                "status 原始 JSON 整段上屏——renderResult 折叠未生效"
            )
            assert "内核状态".encode() in status_segment, (
                "紧凑摘要（✓ 内核状态 …）未出现"
            )
            # 发给模型的 tool 结果（fake 请求的 tool 消息）必须是 READY——
            # 与 /status 同一内核视图。
            status_tool_results = [
                str(m.get("content", ""))
                for body in fake.fake.requests
                for m in body.get("messages", [])
                if m.get("role") == "tool"
                and '"agentd"' in str(m.get("content", ""))
            ]
            assert status_tool_results, "rosclaw_status 未被调用或无结果"
            assert any('"agentd": "READY"' in r for r in status_tool_results), (
                f"rosclaw_status 未报 READY: {status_tool_results[-1][:200]}"
            )
            assert not any("UNREACHABLE" in r for r in status_tool_results), (
                "UDS 可用时 rosclaw_status 误报 UNREACHABLE"
            )
            # 5d. PR-H1（ADR-0012）：直接工作零 Worker——work_orders
            #     写入为 0，且不存在任何 Worker 完成推送。
            import sqlite3 as _sq3

            _db3 = _sq3.connect(home / "agentd" / "missions.db")
            _orders = _db3.execute("SELECT COUNT(*) FROM work_orders").fetchone()[0]
            _execs = _db3.execute("SELECT COUNT(*) FROM task_executions").fetchone()[0]
            _db3.close()
            assert _orders == 0, f"默认旅程不得创建 WorkOrder: {_orders}"
            assert _execs == 0, f"直接工作不得创建 execution: {_execs}"
            time.sleep(3.0)
            assert "后台 Worker 已完成".encode() not in session.clean, (
                "零 Worker 旅程竟出现 Worker 完成推送"
            )
            self._journey_verdicts["direct_work_zero_workers"] = True
            # 5c. 动作准入前置（六审 §3.4）：动作发起前 Header 必须是真实
            #     READY——此前显式 mission 路径 leaseState 不写回，
            #     "Action LOCKED" 假锁与成功执行同时存在。
            session.expect(b"Action Ready", timeout=60)
            # 6. UR5e 机械臂 SIM 闭环（六审 §6.3）：能力面 → 初始观测 →
            #    exact action 卡 → Y → 执行 → 后置观测验证。
            # 主标记必须是稳定面：授权 overlay（"ROSCLAW 授权请求"）。
            # "等待 Operator 决定" 是瞬时 working message/notify——overlay
            # 打开快时它一帧都渲染不出来（CI 两次失败的根因：overlay 已在
            # 等 Y/N，journey 却在等一个被覆盖的瞬时文本，永不按 y）。
            # 6. UR5e 机械臂 SIM 闭环（大道至简 R0-1：Pi 唯一大脑）——
            #    画五角星无条件进模型回合；Pi 自己驱动通用物理工具链
            #    （capabilities → plan → execute → trace → verify），
            #    最终回答基于 verify 结果。SIM POLICY_AUTO：无人工卡。
            action_start = len(session.clean)
            model_turns_before = len(fake.fake.requests)
            session.send("我想跑一个机械臂仿真，让机械臂画五角星\r")
            # Pi 驱动硬证据①：模型基于 verify 结果回答（工具链真实
            # 跑完——plan/execute/trace/verify 都是真工具调用）。
            session.expect("五角星已绘制完成，几何验证通过。".encode(), timeout=180)
            # Pi 驱动硬证据②：模型回合真实发生且多次（工具链每个
            # 环节一个回合）——不再有任何确定性链抢答。
            assert len(fake.fake.requests) > model_turns_before, (
                "画五角星竟没有模型回合——确定性链抢答未根治"
            )
            self._journey_verdicts["pi_drives_sim_chain"] = True
            action_segment = session.clean[action_start:]
            import re as _re

            # 大道至简 R0-1：不再出现确定性链终态卡/接管回声卡——
            # 用户看到的是 Pi 的回答，不是 Kernel 的 PASS 宣称。
            assert "任务完成：验收 PASS".encode() not in action_segment, (
                "Kernel 竟自动宣布任务完成（用户目标判定未退役）"
            )
            assert "确定性链接管".encode() not in action_segment, (
                "确定性链接管回声卡未退役"
            )
            assert not _re.search(rb"/[\w./-]*sim/traces/", action_segment), (
                "屏幕裸打内部 trace 路径（0902 §6.1 违规）"
            )
            assert b"[rosclaw." not in action_segment, (
                "内部 customType 标签漏到屏幕——渲染卡未注册"
            )
            assert "ROSCLAW 授权请求".encode() not in action_segment, (
                "默认安全 SIM 竟弹人工审批卡"
            )
            # R0-8（0826 体验审计 §2.6/§5.R0-8）：TUI 默认层是任务卡
            # ——屏幕上不得出现整段 payload raw JSON（真实终端屏幕
            # 断言，不是 DB 事件扫描）。
            for raw_key in (b'"artifact_refs"', b'"plan":', b'"verification":'):
                assert raw_key not in action_segment, (
                    f"TUI 仍在刷 raw JSON（{raw_key!r}）——任务卡未生效"
                )
            # 证据面不变：kernel 任务 + GIF 产物登记 + 零 Operator 卡
            # （SIM 工具链自动执行不需要人工/政策卡）。
            self._assert_sim_task_evidence(home)
            self._journey_verdicts["safe_sim_auto_executed_with_audit"] = True
            # 七审 §6 PR-SEVEN-4：轨迹级证据——verify PASS（端点/RMSE/
            # 闭合误差全过）+ 单 ExactAction 覆盖整条轨迹。
            self._assert_star_verified(home)
            # 大道至简 R0-1：解释性追问"你这是啥？"也进 Pi——Pi 读
            # 账本自己回答（EXPLAIN_HANDLER 退役）；0 新 task、0 次
            # 新仿真（0901 实证：解释追问曾重跑整个任务制造第二套
            # artifact）。
            import sqlite3 as _sq3

            tasks_before = _sq3.connect(
                home / "agentd" / "missions.db"
            ).execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            model_turns_before_explain = len(fake.fake.requests)
            session.send("你这是啥？\r")
            # Pi 自己回答（读账本/上下文）——回答出现在屏幕上。
            session.expect("刚才的任务".encode(), timeout=60)
            assert b"[rosclaw." not in session.clean, (
                "内部 customType 标签漏到屏幕——渲染卡未注册"
            )
            # 大道至简 R0-1：解释追问**必须**触发模型回合（Pi 是唯一
            # 大脑——不再有 EXPLAIN_HANDLER 确定性回答），但不得
            # 建新任务、不得重跑仿真（重跑会有新 task）。
            assert len(fake.fake.requests) > model_turns_before_explain, (
                "解释追问竟没进模型——唯一大脑未生效"
            )
            tasks_after = _sq3.connect(
                home / "agentd" / "missions.db"
            ).execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            assert tasks_after == tasks_before, (
                f"解释追问竟建新任务：{tasks_before} → {tasks_after}"
            )
            self._journey_verdicts["explain_followup_via_pi"] = True
            # 6c. LIMO 交叉（六审 §6.3.10）：LIMO 动作在 UR5e body 上必须
            #     建卡前 BODY_CAPABILITY_MISMATCH，零 approval/grant/txn。
            self._assert_cross_body_rejected(session, home)
            # 6b. 对抗场景（P0-4F 场景 D）：模型谎称完成——系统必须
            #    以结构化状态为准，不采信模型自述。
            self._assert_adversarial_model_ignored(session, home)
            # 0902 R0（反假成功 P0 事故复归）：用户对已完成任务新增
            # 材料性要求（红笔 + 3D 实际轨迹 + 不要 2D）→ 覆盖率门禁
            # 不得让旧 recipe 认领重跑，旧 revision 的视频不得满足新
            # revision——绝不出现第二个"任务完成"卡；输入落到模型
            # 路径（覆盖率不足 → 交 Native Agent，不猜）。
            seg_before_rev = len(session.clean)
            model_turns_before_rev = len(fake.fake.requests)
            session.send(
                "不对——加红色圆柱笔，在 3D 画面里显示本次实际轨迹，"
                "不要 2D\r"
            )
            import time as _time

            _time.sleep(8)  # 路由/模型回合窗口
            rev_seg = session.clean[seg_before_rev:]
            assert "任务完成：验收 PASS".encode() not in rev_seg, (
                "0902 假成功复现：新要求竟被旧产物满足并宣布 PASS"
            )
            assert len(fake.fake.requests) > model_turns_before_rev, (
                "未覆盖要求竟没落到模型路径（被旧 recipe 吞了）"
            )
            self._journey_verdicts["revision_no_fake_success"] = True
            # 0902 R1-b（审计 §7）：provider 停滞——分阶段看门狗提示
            # +取消，不再静默 300s（fake server 挂起不应答的故障注入）。
            fake.fake.stall = True
            session.send("再聊聊这个任务\r")
            session.expect("响应迟滞".encode(), timeout=40)  # 10s 提示
            session.expect("已取消本次请求".encode(), timeout=60)  # 30s 取消
            fake.fake.stall = False
            self._journey_verdicts["provider_stall_watchdog"] = True
            # 0901 P0-1（硬 Gate B）：安装产物的 `rosclaw artifact` 一族
            # 真实可达——list 不回落顶层帮助、path 给绝对路径（0901
            # 实证：open_command 给了 `rosclaw artifact open` 但入口
            # 没有 dispatch——承诺无实现）。
            import subprocess as _subp

            art_list = _subp.run(
                [str(rosclaw), "artifact", "list"], env=env,
                capture_output=True, text=True, timeout=60,
            )
            assert art_list.returncode == 0, art_list.stderr
            assert "usage: rosclaw" not in art_list.stdout, (
                f"安装产物的 artifact list 回落顶层帮助：{art_list.stdout[:300]}"
            )
            assert ".mp4" in art_list.stdout or ".gif" in art_list.stdout, (
                f"交付物未列出：{art_list.stdout[:300]}"
            )
            self._journey_verdicts["artifact_cli_reachable_in_install"] = True
            # 0902 R3-a（§6.2）：`rosclaw open <id>` 最短主入口在安装
            # 产物上真实可达（重写为 artifact open，同一 handler）。
            import re as _re

            id_match = _re.search(r"art_[0-9a-f]{24}", art_list.stdout)
            assert id_match, f"artifact list 无 artifact id：{art_list.stdout[:300]}"
            art_path = _subp.run(
                [str(rosclaw), "open", id_match.group(0)], env=env,
                capture_output=True, text=True, timeout=60,
            )
            # open 在无图形环境下可能打不开查看器——但 dispatch 必须
            # 到达 artifact handler（不回落顶层帮助/未知命令）。
            combined = art_path.stdout + art_path.stderr
            assert "usage: rosclaw" not in combined, (
                f"rosclaw open 回落顶层帮助：{combined[:300]}"
            )
            self._journey_verdicts["open_shortcut_reachable_in_install"] = True
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
            # 产物账本在 compact 后仍可核验（DB 是权威——compaction
            # 是认知层摘要，不动 kernel/产物记录）。
            self._assert_sim_task_evidence(home)
            self._journey_verdicts["compaction_completed_context_retained"] = True
            # Gate Evidence V2：全链脱敏证据落盘（artifact 上传，第三方
            # 可独立复核，无需相信 pytest 文本）。
            self._write_sanitized_evidence(home)
            # 8. /quit → resume 提示必须是 ROSClaw 命令（T-IDENTITY）。
            session.send("/quit\r")
            session.expect(b"rosclaw continue", timeout=30)
            assert b"pi --session" not in session.clean
            assert b"--session-dir" not in session.clean
            # WP-P0-1：退出提示不暴露内部 session id。
            assert b"chat --resume" not in session.clean
            session.proc.wait(timeout=30)
            assert session.proc.returncode == 0, session.clean[-400:]
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
            resumed.expect(b"rosclaw continue", timeout=30)
            resumed.proc.wait(timeout=30)
        finally:
            resumed.stop()
