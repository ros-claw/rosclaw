"""十五审 PR-RF-5 红测试：Codex app-server 优化路径（协议级）。

红测试先行——修复前必须红：
1. codex app-server 生命周期：initialize → initialized → thread/start
   → turn/start → 流式 item/* → turn/completed（无墙钟超时）；
2. CODEX_HOME 限定 sandbox（不读宿主 codex 配置）；
3. codex 未安装 → 诚实 not_installed。
"""

from __future__ import annotations

import shutil
import stat
from pathlib import Path

import pytest

from rosclaw.agentd.codex_driver import CodexAppServerDriver


def _node() -> str | None:
    return shutil.which("node")


_FAKE_CODEX = r"""
function send(msg) { process.stdout.write(JSON.stringify(msg) + "\n"); }
let buffer = "";
process.stdin.setEncoding("utf-8");
process.stdin.on("data", (chunk) => {
  buffer += chunk;
  const lines = buffer.split("\n");
  buffer = lines.pop() ?? "";
  for (const line of lines) {
    if (!line.trim()) continue;
    const msg = JSON.parse(line);
    if (msg.method === "initialize") {
      send({ id: msg.id, result: { userAgent: "fake-codex", codexHome: process.env.CODEX_HOME || "" } });
    } else if (msg.method === "initialized") {
      // notification——无应答
    } else if (msg.method === "thread/start") {
      send({ id: msg.id, result: { thread: { id: "thr_fake", ephemeral: false } } });
      send({ method: "thread/started", params: { thread: { id: "thr_fake" } } });
    } else if (msg.method === "turn/start") {
      send({ id: msg.id, result: { turn: { id: "turn_1", status: "inProgress" } } });
      setTimeout(() => send({ method: "item/agentMessage/delta", params: { delta: "分析中" } }), 10);
      setTimeout(() => send({ method: "item/started", params: { item: { type: "commandExecution", id: "cmd1" } } }), 20);
      setTimeout(() => send({ method: "item/completed", params: { item: { type: "commandExecution", id: "cmd1", status: "completed" } } }), 30);
      setTimeout(() => send({ method: "turn/completed", params: { turn: { id: "turn_1", status: "completed" } } }), 40);
    }
  }
});
"""


@pytest.mark.skipif(_node() is None, reason="无 Node——诚实 skip")
class TestCodexAppServer:
    async def test_full_turn_lifecycle(self, tmp_path: Path) -> None:
        server = tmp_path / "fake-codex.mjs"
        server.write_text(_FAKE_CODEX)
        shim = tmp_path / "codex-shim"
        shim.write_text(f'#!/bin/sh\nexec {_node()} "{server}"\n')
        shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
        events: list[tuple[str, dict]] = []

        async def sink(kind: str, payload: dict) -> None:
            events.append((kind, payload))

        import rosclaw.agentd.codex_driver as mod

        original = mod.codex_binary
        mod.codex_binary = lambda: str(shim)
        try:
            driver = CodexAppServerDriver(
                cwd=str(tmp_path),
                event_sink=sink,
                sandbox_home=tmp_path / "work" / "exec_codex",
            )
            result = await driver.run("修复失败测试")
        finally:
            mod.codex_binary = original
        assert result["ok"], result
        assert result["stop_reason"] == "completed"
        kinds = [k for k, _ in events]
        assert "message_delta" in kinds
        assert "tool_started" in kinds
        assert "tool_finished" in kinds
        # CODEX_HOME 限定 sandbox（不读宿主配置）。
        assert not (tmp_path / "work" / "exec_codex" / "home" / ".rosclaw").exists()

    async def test_codex_missing_honest(self, tmp_path: Path) -> None:
        async def sink(_k: str, _p: dict) -> None:
            return

        import rosclaw.agentd.codex_driver as mod

        original = mod.codex_binary
        mod.codex_binary = lambda: None
        try:
            driver = CodexAppServerDriver(cwd=str(tmp_path), event_sink=sink)
            result = await driver.run("x")
        finally:
            mod.codex_binary = original
        assert result["ok"] is False
        assert result["stop_reason"] == "not_installed"
