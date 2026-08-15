"""十四审 PR-14.1 Gate 0：跨进程集成验收（真实 Node Worker + Python
supervisor + mock OpenAI provider——不许用只记录调用的 fake session）。

总纲 §9.2：先写会失败的跨进程集成测试。这里跑的是真实
`rosclaw-agent worker --headless` 子进程（真实 Pi AgentSession、真实
session.abort/resume 语义），provider 是本机回环 mock（确定性 SSE）。

无 Node ≥22.19 / dist 未构建 → 诚实 skip（CI 无 Node 环境不假装）。

验收（Gate 0）：
1. pause → control.ack PAUSED，进程 PID 存活、session file 保持；
2. resume → 同一 session 继续并完成（attempt_finished，无 exit 130）；
3. token 超 soft target 只 warning——进程不停、状态不变、正常完成；
4. SIGKILL（无 termination.json）→ INTERRUPTED_RESUMABLE，不是 FAILED。
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from rosclaw.agentd.pi_entry import find_pi_agent_entry
from rosclaw.agentd.workers.scheduler import CandidateView
from rosclaw.contracts.common import new_id
from rosclaw.contracts.worker.order import (
    BudgetEnvelope,
    ExpectedOutput,
    SideEffectPolicy,
    WorkOrderV1,
)
from tests.agentd.test_pi_tool_bridge import _setup


def _runtime() -> tuple[str, str] | None:
    try:
        return find_pi_agent_entry()
    except Exception:  # noqa: BLE001 - 无 node/dist 诚实 skip
        return None


class _MockProvider:
    """OpenAI chat-completions mock。usage 只在 turn 末块到达（worker 的
    usage 事件同理）；tool_call_first=True 时：call1 工具调用+usage
    （快），call2 慢速长流（告警观察窗）——"工作中超限"场景必需
    （turn 内 usage 不可见）。"""

    def __init__(self, *, tool_call_first: bool = False) -> None:
        self.calls = 0
        self.tool_call_first = tool_call_first
        self._lock = threading.Lock()

    def handler(self) -> type[BaseHTTPRequestHandler]:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # 静默
                return

            def do_POST(self) -> None:  # noqa: N802
                if not self.path.endswith("/chat/completions"):
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                with outer._lock:
                    outer.calls += 1
                    call = outer.calls
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                def chunk(content: str, usage: dict | None = None,
                          extra: dict | None = None,
                          finish: str = "stop") -> bytes:
                    delta = {"content": content} if content else {}
                    if extra:
                        delta.update(extra)
                    payload = {
                        "id": "chatcmpl-mock",
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": delta}],
                    }
                    if usage is not None:
                        payload["usage"] = usage
                        payload["choices"][0]["finish_reason"] = finish
                    return f"data: {json.dumps(payload)}\n\n".encode()

                import time

                try:
                    if call == 1 and outer.tool_call_first:
                        # turn 1：工具调用 + usage（turn 结束即产生 worker
                        # usage 事件——monitor 在 turn 2 运行中看到超限）。
                        args = json.dumps({"path": "nonexistent.txt"})
                        self.wfile.write(chunk("", None, {
                            "role": "assistant",
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_read1",
                                "type": "function",
                                "function": {"name": "read", "arguments": ""},
                            }],
                        }))
                        self.wfile.write(chunk("", None, {
                            "tool_calls": [{
                                "index": 0,
                                "function": {"arguments": args},
                            }],
                        }))
                        self.wfile.write(chunk("", {
                            "prompt_tokens": 500,
                            "completion_tokens": 600,
                            "total_tokens": 1100,
                        }, finish="tool_calls"))
                    elif call == 1 or (call == 2 and outer.tool_call_first):
                        # 慢速长流（约 18s 窗口）——pause/告警观察必须
                        # 在此期间到达（对 CI/慢机器留出竞态余量）。
                        self.wfile.write(chunk(""))
                        self.wfile.flush()
                        for i in range(120):
                            self.wfile.write(chunk(f"工作中-{i} "))
                            self.wfile.flush()
                            time.sleep(0.15)
                        self.wfile.write(chunk("", {
                            "prompt_tokens": 500,
                            "completion_tokens": 600,
                            "total_tokens": 1100,
                        }))
                    else:
                        self.wfile.write(chunk("继续并完成：最终报告 DONE-7"))
                        self.wfile.write(chunk("", {
                            "prompt_tokens": 800,
                            "completion_tokens": 50,
                            "total_tokens": 850,
                        }))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass  # abort 断流是预期路径

        return Handler

    def start(self) -> int:
        server = ThreadingHTTPServer(("127.0.0.1", 0), self.handler())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server.server_address[1]


def _write_mock_agent_dir(home: Path, port: int) -> None:
    agent_dir = home / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "models.json").write_text(
        json.dumps({
            "providers": {
                "mock": {
                    "baseUrl": f"http://127.0.0.1:{port}/v1",
                    "api": "openai-completions",
                    "apiKey": "mock-key",
                    "compat": {
                        "supportsDeveloperRole": False,
                        "supportsReasoningEffort": False,
                    },
                    "models": [{"id": "mock-1", "name": "Mock", "reasoning": False}],
                }
            }
        }),
        encoding="utf-8",
    )


async def _hire_real(service, mission, tmp_path, *, policy=None):
    from rosclaw.agentd.workers import pi_managed

    adapter = pi_managed.PiManagedAdapter(
        rosclaw_home=tmp_path, conn=service._store.connection
    )
    service._worker_manager._adapters["pi_managed"] = adapter
    adapter._manager_ref = service._worker_manager
    if service._registry.status_of("worker:rosclaw:pi") != "ENABLED":
        service._registry.set_status(
            "worker:rosclaw:pi", "ENABLED", actor_id="test", reason="real entry"
        )
    card = service._registry.get("worker:rosclaw:pi")
    order = WorkOrderV1(
        work_order_id=new_id("wo"),
        mission_id=mission.mission_id,
        issued_by="test",
        capability="analysis.text",
        goal="写一份简短报告",
        inputs={
            "instructions": "写一份简短报告",
            "worker_profile": "scout",
            "model_snapshot": {"provider": "mock", "model": "mock-1"},
            **({"execution_policy": policy} if policy else {}),
        },
        budgets=BudgetEnvelope(wall_time_sec=600, model_tokens=150_000),
        expected_output=ExpectedOutput(artifacts=["text/plain"]),
        side_effect_policy=SideEffectPolicy(**{"class": "none"}),
    )
    return service._worker_manager.hire(
        order,
        [CandidateView(card=card, registry_status="ENABLED", running_orders=0,
                       circuit_open=False)],
    )


@pytest.mark.skipif(_runtime() is None, reason="无 Node≥22.19/dist——诚实 skip")
class TestGate0CrossProcess:
    async def test_pause_resume_same_session_completes(self, tmp_path: Path) -> None:
        """Gate 0 核心：pause 后进程存活 + session 保持；resume 同会话
        完成；全程无 exit 130。"""
        provider = _MockProvider()
        port = provider.start()
        _write_mock_agent_dir(tmp_path, port)
        service, mission = await _setup(tmp_path)
        scheduled = await _hire_real(service, mission, tmp_path)
        adapter = service._worker_manager._adapters["pi_managed"]
        driver = asyncio.create_task(
            service._worker_manager.run_to_completion(scheduled)
        )
        # 等模型流真的开始（mock 慢流窗口内）。
        started = False
        for _ in range(600):
            events = adapter._events.tail(scheduled.work_order_id, limit=500)
            if any(e["kind"] == "model_started" for e in events):
                started = True
                break
            await asyncio.sleep(0.1)
        assert started, "worker 未进入模型流"
        # pause → ACK PAUSED；进程必须存活。
        assert await adapter.request_pause(scheduled.work_order_id, reason="user")
        proc = adapter._procs[scheduled.work_order_id]
        assert proc.returncode is None, "pause 后进程退出——exit130 缺陷回归"
        session_files = list(
            (tmp_path / "work" / scheduled.work_order_id / "session").glob("*.jsonl")
        )
        assert session_files, "session 未落盘"
        # resume → 同一 session 继续并完成。
        assert await adapter.request_resume(scheduled.work_order_id)
        result, _report = await asyncio.wait_for(driver, 180)
        assert result.status == "COMPLETED", result.summary
        assert "DONE-7" in result.summary
        assert provider.calls >= 2, "resume 未触发新的模型 turn"
        await service.close()

    async def test_soft_token_target_never_pauses_real(self, tmp_path: Path) -> None:
        """Gate 0：token 远超 soft target（1）只 warning——进程不停、
        状态不变、继续工作到完成（不再 exit130）。turn1 工具调用结束
        即产生 usage(1100>>1)，monitor 在 turn2 慢流期间必须只告警。"""
        provider = _MockProvider(tool_call_first=True)
        port = provider.start()
        _write_mock_agent_dir(tmp_path, port)
        service, mission = await _setup(tmp_path)
        scheduled = await _hire_real(
            service, mission, tmp_path, policy={"token_soft_limit": 1}
        )
        adapter = service._worker_manager._adapters["pi_managed"]
        driver = asyncio.create_task(
            service._worker_manager.run_to_completion(scheduled)
        )
        warned = False
        for _ in range(600):
            events = adapter._events.tail(scheduled.work_order_id, limit=500)
            kinds = [e["kind"] for e in events]
            if "budget_warning" in kinds:
                warned = True
                break
            if any(e["kind"] == "attempt_finished" for e in events):
                break
            await asyncio.sleep(0.1)
        assert warned, "token 超 soft target 未产生 budget_warning"
        assert "budget_paused" not in kinds
        # 告警后 worker 必须仍在工作（turn2 慢流中），没有被停。
        proc = adapter._procs.get(scheduled.work_order_id)
        if proc is not None:
            assert proc.returncode is None, "soft target 到限后进程被停——缺陷回归"
        result, _report = await asyncio.wait_for(driver, 180)
        assert result.status == "COMPLETED", result.summary
        order = service._worker_manager.order(scheduled.work_order_id)
        assert order.status != "BUDGET_PAUSED"
        await service.close()

    async def test_sigkill_is_interrupted_resumable(self, tmp_path: Path) -> None:
        """SIGKILL（无 termination.json）→ INTERRUPTED_RESUMABLE，
        不是 FAILED（exit code 不得直接当语义）。"""
        provider = _MockProvider()
        port = provider.start()
        _write_mock_agent_dir(tmp_path, port)
        service, mission = await _setup(tmp_path)
        scheduled = await _hire_real(service, mission, tmp_path)
        adapter = service._worker_manager._adapters["pi_managed"]
        driver = asyncio.create_task(
            service._worker_manager.run_to_completion(scheduled)
        )
        for _ in range(600):
            events = adapter._events.tail(scheduled.work_order_id, limit=500)
            if any(e["kind"] == "model_started" for e in events):
                break
            await asyncio.sleep(0.1)
        import signal as _sig

        proc = adapter._procs[scheduled.work_order_id]
        proc.send_signal(_sig.SIGKILL)
        result, _report = await asyncio.wait_for(driver, 120)
        assert result.status == "INTERRUPTED", result.summary
        order = service._worker_manager.order(scheduled.work_order_id)
        assert order.status == "INTERRUPTED_RESUMABLE", order.status
        await service.close()
