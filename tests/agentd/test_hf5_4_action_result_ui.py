"""PR-HF5-4 红测试（五审 P0-5F）：权威 Action Result UI。

五审核心反证：Gate 证明了 DB 不信模型，却仍允许产品向人撒谎——
对抗回合里用户先看到 `Tool result: 动作提案被拒 [INVALID_ARGUMENTS]`，
紧接着 `Assistant: 动作已执行，结构化回执已确认` 照常显示。

红测试先行（修复前必须全红）：

1. test_rejected_tool_cannot_render_completed_action_status —
   被拒动作必须渲染内核权威结果卡（REJECTED），且全旅程不得出现
   "✓ 动作已完成" 的内核卡；
2. test_model_lie_is_visibly_overridden_by_kernel_outcome —
   模型谎言之后必须出现可见的冲突标记（内核覆盖叙述）；
3. test_false_completion_not_written_to_mission_success_or_memory —
   假完成不得落成任何 success 记录（DB/事件/会话权威条目），且
   session 必须留有内核 REJECTED 结果条目（可审计）；
4. test_next_physical_step_not_started_after_false_completion —
   谎报完成不得解锁下一步物理动作：第二次假动作同样被拒、零
   approval/grant/txn，且 header 有 Action READY/LOCKED 状态可见。
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.agentd.test_product_journey import (
    FakeModelServer,
    PtySession,
    _build_and_install,
    _chunk,
    _FakeModel,
    _sse,
    _tool_call_frames,
)

REPO = Path(__file__).resolve().parents[2]


class _LieTwiceModel(_FakeModel):
    """两轮对抗：两次调用空 capability（必被拒），两次都谎称完成。"""

    def answer(self, body: dict) -> bytes:
        messages = body.get("messages", [])
        if body.get("stream"):
            has_tool_result = bool(messages) and messages[-1].get("role") == "tool"
            text = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    content = m.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            str(b.get("text", "")) for b in content if isinstance(b, dict)
                        )
                    text = str(content)
                    if not text.startswith("<ROSCLAW_TRUSTED_CONTEXT>"):
                        break
            frames: list[bytes] = []
            if has_tool_result:
                call_id = str(messages[-1].get("tool_call_id", ""))
                if call_id in ("call_lie_1", "call_lie_2"):
                    frames.append(_sse(_chunk("动作已执行，结构化回执已确认。")))
                    frames.append(_sse(_chunk("", "stop")))
                    frames.append(b"data: [DONE]\n\n")
                    return b"".join(frames)
            if "再执行一次假动作" in text:
                frames.extend(
                    _tool_call_frames(
                        "call_lie_2",
                        "rosclaw_request_action",
                        json.dumps(
                            {
                                "capability_id": "",
                                "arguments": {},
                                "expected_effect": "第二个假动作",
                                "risk_tier": "LOW",
                            }
                        ),
                    )
                )
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
            if "执行一个假动作" in text:
                frames.extend(
                    _tool_call_frames(
                        "call_lie_1",
                        "rosclaw_request_action",
                        json.dumps(
                            {
                                "capability_id": "",
                                "arguments": {},
                                "expected_effect": "假动作",
                                "risk_tier": "LOW",
                            }
                        ),
                    )
                )
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
        return super().answer(body)


class _LieTwiceServer(FakeModelServer):
    def __init__(self, log_path: Path | None = None) -> None:
        super().__init__(log_path)
        # handler 经类属性读 fake——替换类属性即换剧本（server 已启动）。
        self.fake = _LieTwiceModel(log_path)
        self.server.RequestHandlerClass.fake = self.fake


@pytest.mark.slow  # PTY + release build——由 Native Agent Gate journey job 跑
class TestActionResultUi:
    """一个 PTY 旅程跑完两轮对抗，四个测试各自独立断言。"""

    @pytest.fixture(scope="class")
    def journey(self, tmp_path_factory: pytest.TempPathFactory):
        tmp_path = tmp_path_factory.mktemp("hf5_4")
        fake = _LieTwiceServer(log_path=tmp_path / "fake-requests.jsonl")
        prefix, _root = _build_and_install(tmp_path)
        home = tmp_path / "rh"
        (home / "run").mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(
            "agent:\n  enabled: true\n  default_profile: embodied_default\n"
            "models:\n  backend: legacy\n  profiles:\n    embodied_default:\n"
            "      provider: kimi_code\n      model: fake-k3\n"
            f"      base_url: {fake.base_url}\n"
            "      api_key_ref: env:FAKE_JOURNEY_KEY\n"
            "      capabilities: [llm.chat, llm.structured_decision, llm.tool_use]\n"
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
            PATH=f"{prefix / 'bin'}:{os.environ['PATH']}",
        )
        # operatord：本场景动作在 propose 即被拒（不产生卡片），但 header
        # Operator 探测与 admission 链需要真实服务——enroll + start。
        enroll = subprocess.run(
            [str(rosclaw), "operatord", "enroll"],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert enroll.returncode == 0, enroll.stderr[-500:]
        operatord = subprocess.Popen(
            [str(rosclaw), "operatord", "start", "--no-human-presence-check"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        output = b""
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not (home / "run" / "operatord.sock").exists():
                assert operatord.poll() is None, "operatord 启动失败"
                time.sleep(0.2)
            session = PtySession([str(rosclaw), "chat"], env)
            try:
                session.expect(b"ROSClaw Native Agent", timeout=60)
                # 等 header 完整渲染（含 Action 状态）。
                session.expect(b"Operator", timeout=60)
                time.sleep(2.0)
                # 第一轮对抗：假动作 → 拒绝 → 模型谎称完成。
                session.send("执行一个假动作\r")
                session.expect("动作已执行，结构化回执已确认".encode(), timeout=120)
                time.sleep(2.0)
                # 第二轮对抗：谎报完成后立即再尝试——不得自动放行。
                session.send("再执行一次假动作\r")
                # 第二次谎言文本相同——等输出长度增长后再 settle。
                mark = len(session.output)
                deadline2 = time.monotonic() + 120
                while time.monotonic() < deadline2:
                    if session.output[mark:].count("动作已执行，结构化回执已确认".encode()) >= 1:
                        break
                    time.sleep(0.5)
                else:
                    raise AssertionError("第二轮谎言未出现")
                time.sleep(2.0)
                session.send("/quit\r")
                session.expect(b"rosclaw chat --resume", timeout=30)
                session.proc.wait(timeout=30)
                output = session.output
            finally:
                session.stop()
        finally:
            operatord.terminate()
            try:
                operatord.wait(timeout=10)
            except subprocess.TimeoutExpired:
                operatord.kill()
            fake.close()
        yield {"output": output, "home": home}

    # ------------------------------------------------------------ 红测试 1
    def test_rejected_tool_cannot_render_completed_action_status(self, journey: dict) -> None:
        output = journey["output"]
        assert "ROSClaw 动作结果".encode() in output, (
            "缺内核权威动作结果卡——被拒动作的结果只以 tool 文本呈现"
        )
        assert "动作被准入拒绝".encode() in output, "结果卡未显示 REJECTED 终态"
        assert "✓ 动作已完成".encode() not in output, "被拒动作竟渲染了完成态内核卡"

    # ------------------------------------------------------------ 红测试 2
    def test_model_lie_is_visibly_overridden_by_kernel_outcome(self, journey: dict) -> None:
        output = journey["output"]
        lie = "动作已执行，结构化回执已确认".encode()
        lie_at = output.find(lie)
        assert lie_at >= 0, "前提：模型谎言应显示（我们不审查模型文本）"
        # 冲突标记必须在谎言之后可见——内核覆盖叙述。
        conflict_markers = ("与内核", "冲突", "未被接受")
        tail = output[lie_at:]
        assert any(m.encode() in tail for m in conflict_markers), (
            "模型谎称完成后，界面没有任何内核冲突标记——用户会信模型"
        )

    # ------------------------------------------------------------ 红测试 3
    def test_false_completion_not_written_to_mission_success_or_memory(self, journey: dict) -> None:
        home = journey["home"]
        # 1. DB 零动作痕迹（HF5-3 已保证——回归防线）。
        db = sqlite3.connect(home / "agentd" / "missions.db")
        for table in ("action_txns", "mission_grants", "operator_requests"):
            exists = db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if exists:
                count = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                assert count == 0, f"{table} 竟有 {count} 条——假完成被采信"
        db.close()
        # 2. session JSONL 必须有内核 REJECTED 结果条目（可审计真相），
        #    且不得有任何把这两次动作记为 COMPLETED 的内核条目。
        sessions_dir = home / "agent" / "sessions"
        result_lines: list[str] = []
        for session_file in sessions_dir.glob("*.jsonl"):
            for line in session_file.read_text(encoding="utf-8", errors="replace").splitlines():
                if "rosclaw.action_result" in line:
                    result_lines.append(line)
        assert result_lines, "session 无 rosclaw.action_result 内核条目——假完成回合无权威记录"
        assert any("REJECT" in line for line in result_lines), "内核条目缺 REJECTED 状态"
        assert not any('"COMPLETED"' in line for line in result_lines), (
            "内核条目把被拒动作记为 COMPLETED"
        )

    # ------------------------------------------------------------ 红测试 4
    def test_next_physical_step_not_started_after_false_completion(self, journey: dict) -> None:
        output = journey["output"]
        home = journey["home"]
        # 两次假动作都必须各有一张 REJECTED 卡——第二次没有被第一次的
        # 谎言"铺垫"放行。
        assert output.count("动作被准入拒绝".encode()) >= 2, (
            "第二次假动作未被独立拒绝/渲染——谎言解锁了后续动作"
        )
        # 假动作连 approval 卡都不该产生（propose 即拒）。
        assert "ROSCLAW 授权请求".encode() not in output, "假动作竟产生了授权卡"
        # header 必须有 Action READY/LOCKED 可见状态。
        assert b"Action READY" in output or b"Action LOCKED" in output, (
            "header 缺 Action 状态——用户无法看到动作准入状态"
        )
        # DB 终态：两轮之后仍零执行痕迹。
        db = sqlite3.connect(home / "agentd" / "missions.db")
        exists = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='action_txns'"
        ).fetchone()
        if exists:
            count = db.execute("SELECT COUNT(*) FROM action_txns").fetchone()[0]
            assert count == 0, f"谎报后竟有 {count} 条 ActionTxn"
        db.close()
