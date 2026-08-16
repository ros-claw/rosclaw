"""十五审 PR-RF-4 红测试：Harness Sandbox（总纲 §9）。

红测试先行——修复前必须红：
1. ACP Harness 进程 env 不含 ROSCLAW_HOME/宿主 HOME/ROS/DDS 变量；
2. cwd 是隔离 workspace（不是 rosclaw home 也不是用户目录）；
3. 凭据只给目标 Harness（symlink 其自身配置，不暴露 rosclaw agent
   auth.json）；
4. Harness 侧 fs/terminal/permission 请求按策略拒绝或限制在
   workspace 内（fail-closed 默认）；
5. 进程树统一清理。
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path

import pytest

from rosclaw.agentd.acp_driver import AcpHarnessDriver


def _node() -> str | None:
    return shutil.which("node")


#: 红队 fake server：启动即把 env/cwd 实况经 session/update 回传。
_PROBE_SERVER = r"""
function send(msg) { process.stdout.write(JSON.stringify(msg) + "\n"); }
let buffer = "";
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
      send({ jsonrpc: "2.0", id: msg.id, result: { sessionId: "sess_probe" } });
    } else if (msg.method === "session/prompt") {
      const report = {
        home: process.env.HOME || "",
        rosclaw_home: process.env.ROSCLAW_HOME || "",
        has_ros: Boolean(process.env.ROS_DISTRO || process.env.DDS_VER),
        cwd: process.cwd(),
        tmpdir: process.env.TMPDIR || "",
      };
      send({ jsonrpc: "2.0", method: "session/update", params: {
        sessionId: msg.params.sessionId,
        update: { sessionUpdate: "agent_message_chunk",
                  content: { type: "text", text: JSON.stringify(report) } },
      }});
      setTimeout(() => send({ jsonrpc: "2.0", id: msg.id, result: { stopReason: "end_turn" } }), 20);
    }
  }
});
process.stdin.setEncoding("utf-8");
"""


@pytest.mark.skipif(_node() is None, reason="无 Node——诚实 skip")
class TestHarnessSandbox:
    async def test_sandbox_env_isolation(self, tmp_path: Path) -> None:
        """Harness 只能看到隔离环境——宿主 HOME/ROSCLAW_HOME/ROS 不泄漏。"""
        server = tmp_path / "probe.mjs"
        server.write_text(_PROBE_SERVER)
        shim = tmp_path / "probe-shim"
        shim.write_text(f'#!/bin/sh\nexec {_node()} "{server}"\n')
        shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
        seen: list[dict] = []

        async def sink(kind: str, payload: dict) -> None:
            # 驱动侧文本可能截断——解析失败不破坏 sink（事件读者健壮性
            # 也是本测试的一部分）。
            if kind == "message_delta" and payload.get("text"):
                import contextlib

                with contextlib.suppress(json.JSONDecodeError):
                    seen.append(json.loads(payload["text"]))

        import rosclaw.agentd.acp_driver as mod

        original = mod.acp_binary_for
        mod.acp_binary_for = lambda _runtime: str(shim)
        # 宿主环境放毒：如果这些变量泄漏给 harness 就是缺陷。
        # 必须恢复——污染会经进程 env 泄漏到后续测试（CI 实证）。
        _saved_home = os.environ.get("ROSCLAW_HOME")
        os.environ["ROSCLAW_HOME"] = str(tmp_path)
        try:
            driver = AcpHarnessDriver(
                "harness:acp:claude-local",
                cwd=str(tmp_path),
                event_sink=sink,
                sandbox_home=tmp_path / "work" / "exec_1",
            )
            result = await driver.run("probe")
        finally:
            mod.acp_binary_for = original
            if _saved_home is None:
                os.environ.pop("ROSCLAW_HOME", None)
            else:
                os.environ["ROSCLAW_HOME"] = _saved_home
        assert result["ok"], result
        assert seen, "probe 未回传环境"
        report = seen[-1]
        assert report["rosclaw_home"] == "", (
            f"ROSCLAW_HOME 泄漏给 harness: {report['rosclaw_home']}"
        )
        assert report["home"] != str(Path.home()), "宿主 HOME 泄漏"
        assert not report["has_ros"], "ROS/DDS 环境泄漏"
        assert "work" in report["cwd"] and "exec_1" in report["cwd"], (
            f"cwd 不是隔离 workspace: {report['cwd']}"
        )
        assert report["tmpdir"] and "exec_1" in report["tmpdir"], (
            f"TMPDIR 未隔离: {report['tmpdir']}"
        )

    async def test_harness_config_scoped(self, tmp_path: Path) -> None:
        """sandbox HOME 只含目标 harness 自己的配置（不暴露 rosclaw
        agent auth.json）。"""
        from rosclaw.agentd.acp_driver import prepare_sandbox_home

        sandbox = prepare_sandbox_home(
            tmp_path / "work" / "exec_2", harness_config_dirs=[".claude"]
        )
        assert (sandbox / "home").is_dir()
        # rosclaw 自身的 agent 凭据目录绝不出现在 sandbox。
        assert not (sandbox / "home" / ".rosclaw").exists()
