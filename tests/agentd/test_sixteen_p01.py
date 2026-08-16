"""建议-0816 P0-1 红测试：删除默认硬超时（turn/bash 不杀）。

红测试先行——修复前必须红：
1. 模型 turn 超过默认阈值（未显式授权）→ 只告警（provider_slow），
   绝不 abort；任务继续完成；
2. 显式 ROSCLAW_WORKER_TURN_TIMEOUT_MS（操作员/benchmark 权威）→
   保留 abort（PROVIDER_TRANSIENT）；
3. workbench bash 无 timeout_sec 参数时无默认 120s SIGKILL——
   _bashTimeoutMs 返回 null（无定时器）；
4. bash 显式 timeout_sec 仍然生效（模型自选）。
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.agentd.test_fourteen_1_live import (
    _hire_real,
    _MockProvider,
    _runtime,
    _write_mock_agent_dir,
)
from tests.agentd.test_pi_tool_bridge import _setup

_NODE = shutil.which("node")
_WORKBENCH_JS = (
    Path(__file__).resolve().parents[2]
    / "packages" / "rosclaw-agent" / "dist" / "src" / "workers" / "workbench.js"
)


class _SlowTurnProvider(_MockProvider):
    """单 turn 流式超过阈值（告警在 turn 内到达）。"""

    def __init__(self) -> None:
        super().__init__(tool_call_first=False)


@pytest.mark.skipif(_runtime() is None, reason="无 Node≥22.19/dist——诚实 skip")
class TestNoDefaultTurnKill:
    async def test_slow_turn_warns_but_never_aborts(self, tmp_path: Path,
                                                    monkeypatch) -> None:
        """未设 TURN_TIMEOUT：turn 超过阈值 → provider_slow 告警事件，
         turn 继续到完成（不 abort、不 PROVIDER_TRANSIENT）。"""
        provider = _SlowTurnProvider()
        port = provider.start()
        _write_mock_agent_dir(tmp_path, port)
        service, mission = await _setup(tmp_path)
        # 告警阈值调到 2s（测试可观察）；不设 abort 阈值。
        monkeypatch.setenv("ROSCLAW_WORKER_TURN_WARN_MS", "2000")
        monkeypatch.delenv("ROSCLAW_WORKER_TURN_TIMEOUT_MS", raising=False)
        scheduled = await _hire_real(service, mission, tmp_path)
        adapter = service._worker_manager._adapters["pi_managed"]
        result, _report = await asyncio.wait_for(
            service._worker_manager.run_to_completion(scheduled), 120
        )
        assert result.status == "COMPLETED", result.summary
        events = adapter._events.tail(scheduled.work_order_id, limit=500)
        kinds = [e["kind"] for e in events]
        assert "provider_slow" in kinds, kinds
        assert not any(
            e.get("error_code") == "PROVIDER_TIMEOUT"
            for e in events
            if e["kind"] == "attempt_failed"
        )
        await service.close()


@pytest.mark.skipif(
    _NODE is None or not _WORKBENCH_JS.exists(),
    reason="无 Node/dist——诚实 skip",
)
class TestBashNoDefaultTimeout:
    def test_no_default_timeout(self) -> None:
        """bash 无 timeout_sec 且无配置 → None（不装 SIGKILL 定时器）。"""
        proc = subprocess.run(
            [
                _NODE,
                "--input-type=module",
                "-e",
                f"import {{ _bashTimeoutMs }} from '{_WORKBENCH_JS}';"
                "console.log(JSON.stringify(_bashTimeoutMs({}, {})));",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "null"

    def test_explicit_timeout_honored(self) -> None:
        proc = subprocess.run(
            [
                _NODE,
                "--input-type=module",
                "-e",
                f"import {{ _bashTimeoutMs }} from '{_WORKBENCH_JS}';"
                "console.log(JSON.stringify(_bashTimeoutMs({timeout_sec: 5}, {})));",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "5000"
