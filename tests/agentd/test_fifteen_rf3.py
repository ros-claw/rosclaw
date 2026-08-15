"""十五审 PR-RF-3 红测试：ACP Runtime MVP（真实 stdio JSON-RPC）。

红测试先行——修复前必须红：
1. ACP driver 完整生命周期（initialize/new/prompt/流式 update/终态）；
2. Harness 未安装 → 诚实 BLOCKED（preflight 失败不创建执行）；
3. 事件映射进 EventStore（message/tool/plan）——无 PTY/文本推断。
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from rosclaw.agentd.acp_driver import AcpHarnessDriver
from tests.agentd.test_pi_tool_bridge import _setup

#: fake ACP server（node 运行，协议级应答）。
_FAKE_SERVER = r"""
let buffer = "";
function send(msg) { process.stdout.write(JSON.stringify(msg) + "\n"); }
process.stdin.setEncoding("utf-8");
process.stdin.on("data", (chunk) => {
  buffer += chunk;
  const lines = buffer.split("\n");
  buffer = lines.pop() ?? "";
  for (const line of lines) {
    if (!line.trim()) continue;
    const msg = JSON.parse(line);
    if (msg.method === "initialize") {
      send({ jsonrpc: "2.0", id: msg.id, result: { protocolVersion: 1, agentCapabilities: {} } });
    } else if (msg.method === "session/new") {
      send({ jsonrpc: "2.0", id: msg.id, result: { sessionId: "sess_py_1" } });
    } else if (msg.method === "session/prompt") {
      const sid = msg.params.sessionId;
      setTimeout(() => send({ jsonrpc: "2.0", method: "session/update", params: {
        sessionId: sid, update: { sessionUpdate: "agent_message_chunk", content: { type: "text", text: "修复中…" } },
      }}), 10);
      setTimeout(() => send({ jsonrpc: "2.0", method: "session/update", params: {
        sessionId: sid, update: { sessionUpdate: "tool_call", toolCallId: "t1", title: "edit", status: "pending" },
      }}), 20);
      setTimeout(() => send({ jsonrpc: "2.0", method: "session/update", params: {
        sessionId: sid, update: { sessionUpdate: "tool_call_update", toolCallId: "t1", status: "completed" },
      }}), 30);
      setTimeout(() => send({ jsonrpc: "2.0", id: msg.id, result: { stopReason: "end_turn" } }), 40);
    }
  }
});
"""


def _node() -> str | None:
    return shutil.which("node")


@pytest.mark.skipif(_node() is None, reason="无 Node——诚实 skip")
class TestAcpDriver:
    async def test_full_lifecycle_with_events(self, tmp_path: Path) -> None:
        server = tmp_path / "fake-acp.mjs"
        server.write_text(_FAKE_SERVER)
        events: list[tuple[str, dict]] = []

        async def sink(kind: str, payload: dict) -> None:
            events.append((kind, payload))

        driver = AcpHarnessDriver(
            "harness:acp:claude-local", cwd=str(tmp_path), event_sink=sink
        )
        # 注入 fake server 路径（替代真实二进制）。
        import rosclaw.agentd.acp_driver as mod

        original = mod.acp_binary_for
        mod.acp_binary_for = lambda _runtime: _node()
        try:
            # driver 用 binary 直接 exec——fake server 是 .mjs，需要
            # node script 形式：binary=node + argv[0]=script。driver 当前
            # 只 exec(binary)——用 shim 包装。
            shim = tmp_path / "fake-acp-shim"
            shim.write_text(f'#!/bin/sh\nexec {_node()} "{server}"\n')
            shim.chmod(0o755)
            mod.acp_binary_for = lambda _runtime: str(shim)
            result = await driver.run("修复失败测试")
        finally:
            mod.acp_binary_for = original
        assert result["ok"], result
        assert result["stop_reason"] == "end_turn"
        kinds = [k for k, _ in events]
        assert "message_delta" in kinds
        assert "tool_started" in kinds
        assert "tool_finished" in kinds

    async def test_not_installed_honest(self, tmp_path: Path) -> None:
        async def sink(_kind: str, _payload: dict) -> None:
            return

        driver = AcpHarnessDriver(
            "harness:acp:pi-acp", cwd=str(tmp_path), event_sink=sink
        )
        result = await driver.run("x")
        # pi-acp 未安装（本机）→ not_installed，绝不假装。
        if shutil.which("pi-acp") is None:
            assert result["ok"] is False
            assert result["stop_reason"] == "not_installed"


class TestControlPlaneAcp:
    async def test_acp_runtime_blocked_when_missing(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """router 选中 ACP runtime 但二进制缺失 → BLOCKED（不是
        FAILED，不创建 worker 进程）。"""
        from rosclaw.agentd import control_plane as cp

        monkeypatch.setattr(
            cp.ExecutionRouter, "_preferred_harness",
            staticmethod(lambda: "harness:acp:pi-acp"),
        )
        service, mission = await _setup(tmp_path)
        plane = service._task_control_plane
        view = await plane.submit(
            mission.mission_id,
            {"goal": "写个脚本", "required_capabilities": [], "effects": ""},
            idem="rf3_acp_missing",
        )
        for _ in range(100):
            row = plane._get(view["execution_id"])
            if row["state"] in ("BLOCKED", "FAILED", "SUCCEEDED"):
                break
            await asyncio.sleep(0.05)
        row = plane._get(view["execution_id"])
        assert row["state"] == "BLOCKED", row["state"]
        assert "未安装" in (row["summary"] or "")
        await service.close()
