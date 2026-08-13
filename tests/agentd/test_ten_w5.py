"""十审 Gate W5 红测试：外部 Worker streaming conformance。

红测试先行——修复前必须红：
1. claude stream-json / codex JSONL 逐行事件驱动 progress（不再
   communicate() 到最后）；
2. startup/idle/wall 超时分离——挂死 Worker 诚实 FAILED；
3. cwd 来自 WorkOrder workspace（不再固定 ~/.rosclaw）；
4. read-only 权限档：claude 禁写/禁 bash/禁网络（不再全禁工具）、
   codex --sandbox read-only。
"""

from __future__ import annotations

import asyncio
import stat
import time
from pathlib import Path

from rosclaw.agentd.workers.external import ExternalHarnessAdapter
from rosclaw.agentd.workers.packs import WorkerPackManifest
from rosclaw.contracts.common import new_id
from rosclaw.contracts.worker.order import (
    BudgetEnvelope,
    ExpectedOutput,
    SideEffectPolicy,
    WorkOrderLease,
    WorkOrderV1,
)


def _pack(product: str, exe: str) -> WorkerPackManifest:
    return WorkerPackManifest(
        pack_id=f"fake-{product}",
        worker_id=f"worker:fake-{product}:local",
        product=product,
        display_name="Fake",
        executable=exe,
        min_version="0.0.0",
        install_hint="",
        license="MIT",
        capabilities=(("code.repository_analysis", "rosclaw://schemas/text-task.v1"),),
    )


def _script(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _order(tmp_path: Path, pack: WorkerPackManifest, workspace: Path | None = None) -> WorkOrderV1:
    return WorkOrderV1(
        work_order_id=new_id("wo"),
        mission_id="mis_x",
        issued_by="test",
        capability="code.repository_analysis",
        goal="分析",
        inputs={
            "instructions": "x",
            **({"workspace": str(workspace)} if workspace else {}),
        },
        budgets=BudgetEnvelope(wall_time_sec=60, model_tokens=1000),
        expected_output=ExpectedOutput(artifacts=["text/plain"]),
        side_effect_policy=SideEffectPolicy(**{"class": "none"}),
        lease=WorkOrderLease(lease_id="lease_1", issued_at="t", expires_at="t"),
        assigned_to=pack.worker_id,
    )


class TestStreamingProtocol:
    async def test_claude_stream_lines_drive_progress_and_result(
        self, tmp_path: Path
    ) -> None:
        fake = _script(
            tmp_path / "fake-claude",
            "#!/bin/sh\n"
            'echo \'{"type":"system","subtype":"init"}\'\n'
            "sleep 0.1\n"
            'echo \'{"type":"assistant","message":{"role":"assistant"}}\'\n'
            "sleep 0.1\n"
            'echo \'{"type":"result","subtype":"success","result":"根因是 X",'
            '"usage":{"input_tokens":120,"output_tokens":30},"total_cost_usd":0.0004}\'\n',
        )
        pack = _pack("claude-code", str(fake))
        adapter = ExternalHarnessAdapter(cwd=tmp_path)
        adapter._packs = {pack.worker_id: pack}
        order = _order(tmp_path, pack)
        handle = await adapter.start(order, {})
        result = None
        for _ in range(100):
            polled = await adapter.poll(handle)
            if not hasattr(polled, "progress_seq"):
                result = polled
                break
            await asyncio.sleep(0.05)
        assert result is not None
        assert result.status == "COMPLETED"
        assert "根因是 X" in result.summary
        assert result.usage.prompt_tokens == 120
        assert result.usage.cost_microunits == 400

    async def test_codex_jsonl_result(self, tmp_path: Path) -> None:
        fake = _script(
            tmp_path / "fake-codex",
            "#!/bin/sh\n"
            'echo \'{"type":"turn.started"}\'\n'
            'echo \'{"type":"item.completed","item":{"type":"agent_message","text":"codex 分析结果"}}\'\n'
            'echo \'{"type":"turn.completed","usage":{"input_tokens":50,"output_tokens":10}}\'\n',
        )
        pack = _pack("codex-cli", str(fake))
        adapter = ExternalHarnessAdapter(cwd=tmp_path)
        adapter._packs = {pack.worker_id: pack}
        order = _order(tmp_path, pack)
        handle = await adapter.start(order, {})
        result = None
        for _ in range(100):
            polled = await adapter.poll(handle)
            if not hasattr(polled, "progress_seq"):
                result = polled
                break
            await asyncio.sleep(0.05)
        assert result is not None
        assert "codex 分析结果" in result.summary
        assert result.usage.prompt_tokens == 50


class TestTimeouts:
    async def test_idle_worker_fails_honestly(self, tmp_path: Path, monkeypatch) -> None:
        from rosclaw.agentd.workers import external

        monkeypatch.setattr(external, "STARTUP_TIMEOUT_SEC", 1.0)
        monkeypatch.setattr(external, "IDLE_TIMEOUT_SEC", 1.0)
        fake = _script(
            tmp_path / "fake-hang",
            "#!/bin/sh\n"
            'echo \'{"type":"system","subtype":"init"}\'\n'
            "sleep 30\n",
        )
        pack = _pack("claude-code", str(fake))
        adapter = ExternalHarnessAdapter(cwd=tmp_path)
        adapter._packs = {pack.worker_id: pack}
        order = _order(tmp_path, pack)
        started = time.monotonic()
        handle = await adapter.start(order, {})
        result = None
        for _ in range(200):
            polled = await adapter.poll(handle)
            if not hasattr(polled, "progress_seq"):
                result = polled
                break
            await asyncio.sleep(0.05)
        elapsed = time.monotonic() - started
        assert result is not None
        assert result.status == "FAILED"
        assert "idle timeout" in result.summary
        assert elapsed < 15, f"idle 检测耗时 {elapsed:.1f}s"

    async def test_silent_worker_startup_timeout(self, tmp_path: Path, monkeypatch) -> None:
        from rosclaw.agentd.workers import external

        monkeypatch.setattr(external, "STARTUP_TIMEOUT_SEC", 1.0)
        fake = _script(tmp_path / "fake-silent", "#!/bin/sh\nsleep 30\n")
        pack = _pack("claude-code", str(fake))
        adapter = ExternalHarnessAdapter(cwd=tmp_path)
        adapter._packs = {pack.worker_id: pack}
        order = _order(tmp_path, pack)
        handle = await adapter.start(order, {})
        result = None
        for _ in range(200):
            polled = await adapter.poll(handle)
            if not hasattr(polled, "progress_seq"):
                result = polled
                break
            await asyncio.sleep(0.05)
        assert result is not None and result.status == "FAILED"
        assert "startup timeout" in result.summary


class TestWorkspaceAndPermissions:
    async def test_cwd_is_order_workspace(self, tmp_path: Path) -> None:
        ws = tmp_path / "target-repo"
        ws.mkdir()
        fake = _script(
            tmp_path / "fake-claude",
            "#!/bin/sh\n"
            'echo \'{"type":"system","subtype":"init"}\'\n'
            'printf \'{"type":"result","result":"cwd=%s","usage":{}}\\n\' "$(pwd)"\n',
        )
        pack = _pack("claude-code", str(fake))
        adapter = ExternalHarnessAdapter(cwd=tmp_path)  # adapter 默认 cwd 不同
        adapter._packs = {pack.worker_id: pack}
        order = _order(tmp_path, pack, workspace=ws)
        handle = await adapter.start(order, {})
        result = None
        for _ in range(100):
            polled = await adapter.poll(handle)
            if not hasattr(polled, "progress_seq"):
                result = polled
                break
            await asyncio.sleep(0.05)
        assert result is not None
        assert str(ws) in result.summary

    def test_command_permission_profiles(self) -> None:
        adapter = ExternalHarnessAdapter()
        claude = adapter._command(
            _pack("claude-code", "claude"), "p"
        )
        assert "stream-json" in claude
        joined = " ".join(claude)
        assert "--disallowedTools" in joined
        # read-only 档：禁写/禁 bash/禁网络，但不再全禁（Read/Grep/Glob 可用）。
        assert '"*"' not in claude and " *" not in joined.split("--disallowedTools")[-1]
        assert "Write" in joined and "Bash" in joined
        codex = adapter._command(_pack("codex-cli", "codex"), "p")
        assert "--sandbox" in codex
        assert "read-only" in codex
