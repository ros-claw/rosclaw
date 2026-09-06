"""大道至简 R1-1 产品级实证（PTY 黑盒）：「画 hello」不再画成
五角星——Pi 自己把字母表达成任意路径点，驱动通用物理工具链
完成仿真+验证+渲染。

0905 体验实证（rosclaw体验0905.txt）：用户要「画个hello」→ 模型
第一次 plan_cartesian_path 直接 ValidationError（工具只认 star5/
circle）→ 无路可走拿五角星冒充交付。本测试钉死根治后的完整
闭环：字母 → 自定义 waypoints（多笔画+抬笔）→ 动力学 rollout →
跟踪验证 → GIF 交付。

假模型驱动（与 journey 同一 FakeModelServer 设施）：用户文本含
「画个HI」→ 四步工具链，全部真执行。
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path

import pytest

from tests.agentd.test_product_journey import (
    FakeModelServer,
    PtySession,
    _build_and_install,
    _chunk,
    _hidden_source_checkout,
    _prepare_installed_chat,
    _sse,
    _tool_call_frames,
)

#: "HI" 两字母五笔画（h 竖/横/竖，抬笔，i 竖）——安全工作空间内。
_HI_WAYPOINTS = [
    {"x": 0.35, "y": 0.18, "z": 0.30},
    {"x": 0.35, "y": 0.18, "z": 0.36},
    {"x": 0.35, "y": 0.18, "z": 0.33},
    {"x": 0.35, "y": 0.24, "z": 0.33},
    {"x": 0.35, "y": 0.24, "z": 0.30},
    {"x": 0.35, "y": 0.24, "z": 0.36},
    {"x": 0.35, "y": 0.32, "z": 0.30, "contact": False},
    {"x": 0.35, "y": 0.32, "z": 0.34},
]


def _payload_of(tool_content: str) -> dict:
    with contextlib.suppress(Exception):
        wrapper = json.loads(tool_content)
        if isinstance(wrapper, dict) and "content" in wrapper:
            wrapper = json.loads(wrapper["content"][0])
        if isinstance(wrapper, dict):
            value = wrapper.get("value")
            if isinstance(value, dict):
                return value
            return wrapper
    return {}


class _HiModel:
    """最小假模型：「画个HI」→ 自定义 waypoints 四步链。"""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    def answer(self, body: dict) -> bytes:
        self.requests.append(body)
        messages = body.get("messages", [])
        if not body.get("stream"):
            return json.dumps(
                {
                    "id": "chatcmpl-fake", "object": "chat.completion",
                    "created": 1, "model": "fake-k3",
                    "choices": [{"index": 0, "message": {
                        "role": "assistant", "content": "好的。"},
                        "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10},
                },
                ensure_ascii=False,
            ).encode()
        frames: list[bytes] = []
        if messages and messages[-1].get("role") == "tool":
            last = messages[-1]
            content = str(last.get("content", ""))
            call_id = str(last.get("tool_call_id", ""))
            if call_id == "hi_plan":
                plan_id = str(_payload_of(content).get("plan_id", ""))
                assert plan_id, f"plan 缺 plan_id：{content[:200]}"
                frames.extend(_tool_call_frames(
                    "hi_sim", "ur5e_simulate_cartesian_trajectory",
                    json.dumps({"plan_id": plan_id})))
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
            if call_id == "hi_sim":
                trace_id = str(_payload_of(content).get("trace_id", ""))
                assert trace_id, f"simulate 缺 trace_id：{content[:200]}"
                frames.extend(_tool_call_frames(
                    "hi_verify", "simulation_verify_tracking",
                    json.dumps({"trace_id": trace_id,
                                "max_tracking_error_m": 0.05})))
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
            if call_id == "hi_verify":
                verdict = str(_payload_of(content).get("verdict", ""))
                assert verdict == "PASS", f"跟踪验证未过：{content[:200]}"
                import re

                trace_id = ""
                for m in reversed(messages):
                    mm = re.search(r"trace_[0-9a-f]+", str(m.get("content", "")))
                    if mm:
                        trace_id = mm.group(0)
                        break
                frames.extend(_tool_call_frames(
                    "hi_render", "simulation_render_trace",
                    json.dumps({"trace_id": trace_id, "format": "gif"})))
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
            # hi_render 已回 → 最终回答（基于 verify PASS + GIF 交付）。
            frames.append(_sse(_chunk(
                "HI 字母已绘制完成：自定义路径仿真跟踪验证 PASS，"
                "轨迹 GIF 已交付。")))
            frames.append(_sse(_chunk("", "stop")))
            frames.append(b"data: [DONE]\n\n")
            return b"".join(frames)
        text = ""
        for m in messages:
            if m.get("role") == "user":
                c = m.get("content", "")
                if isinstance(c, list):
                    c = " ".join(str(b.get("text", ""))
                                 for b in c if isinstance(b, dict))
                if not str(c).startswith("<ROSCLAW_TRUSTED_CONTEXT>"):
                    text = str(c)
        if "画个HI" in text:
            frames.extend(_tool_call_frames(
                "hi_plan", "trajectory_generate_planar_path",
                json.dumps({"waypoints": _HI_WAYPOINTS,
                            "max_segment_m": 0.02})))
        else:
            frames.append(_sse(_chunk("你好，我是 ROSClaw。")))
            frames.append(_sse(_chunk("", "stop")))
        frames.append(b"data: [DONE]\n\n")
        return b"".join(frames)


@pytest.mark.slow
class TestPiDrawsLettersViaWaypoints:
    def test_hello_letters_not_a_star(self, tmp_path: Path) -> None:
        fake = FakeModelServer(log_path=tmp_path / "fake-requests.jsonl")
        fake.fake = _HiModel()  # type: ignore[assignment]
        fake.server.RequestHandlerClass.fake = fake.fake  # type: ignore[attr-defined]
        prefix, _root = _build_and_install(tmp_path)
        home, env, rosclaw = _prepare_installed_chat(tmp_path, fake, prefix)
        try:
            with _hidden_source_checkout():
                session = PtySession(
                    [str(rosclaw), "chat"], env,
                    log_path=tmp_path / "pty-hi.log",
                )
                try:
                    session.expect(b"ROSClaw Native Agent", timeout=60)
                    session.send("你好\r")
                    session.expect("你好，我是 ROSClaw".encode(), timeout=90)
                    session.send(
                        "请用机械臂画个HI字母，我要看仿真轨迹视频\r"
                    )
                    # Pi 驱动闭环：回答基于 verify PASS + GIF 交付。
                    session.expect(
                        "HI 字母已绘制完成".encode(), timeout=180,
                    )
                    # 0905 事故对面：五角星冒充绝不再出现。
                    assert "五角星".encode() not in session.clean, (
                        "画HI竟出现五角星——0905 假成功未根治"
                    )
                    session.send("/quit\r")
                    session.expect(b"rosclaw continue", timeout=30)
                    session.proc.wait(timeout=30)
                finally:
                    session.stop()
        finally:
            fake.close()
        # 证据面：plan 记录是 custom（不是形状枚举）+ GIF 产物登记。
        plans = list((home / "sim" / "plans").glob("*.json"))
        assert plans, "无 plan 记录"
        custom = [
            p for p in plans
            if json.loads(p.read_text()).get("shape") == "custom"
        ]
        assert custom, f"HI 路径竟不是 custom 形状记录: {plans}"
        waypoints = json.loads(custom[0].read_text())["waypoints"]
        assert any(not w.get("contact", True) for w in waypoints), (
            "抬笔段标记丢失"
        )
        db = sqlite3.connect(home / "agentd" / "missions.db")
        gifs = db.execute(
            "SELECT COUNT(*) FROM artifacts WHERE path LIKE '%.gif'"
        ).fetchone()[0]
        db.close()
        assert gifs >= 1, "GIF 未登记进产物账本"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
