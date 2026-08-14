"""十四审 PR-14.3 live：跨进程 transcript 完整性（真实 Node Worker +
mock provider）——总纲 §1.6/§4.4：不再只有 160/4000 字预览。

验收：
1. assistant 完整全文进 transcript（不被 4000 字截断）；
2. tools channel：工具调用带 args + 完整输出（is_error/exit 信息）；
3. usage channel；control channel（pause/resume ACK）；
4. artifacts channel：完成时产物清单带 sha256；
5. tseq 单调递增；transcript 与主对话 compaction 无关（独立文件）。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tests.agentd.test_fourteen_1_live import (
    _hire_real,
    _MockProvider,
    _runtime,
    _write_mock_agent_dir,
)
from tests.agentd.test_pi_tool_bridge import _setup


def _read_transcript(home: Path, wo: str) -> list[dict]:
    path = home / "work" / wo / "transcript.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.skipif(_runtime() is None, reason="无 Node≥22.19/dist——诚实 skip")
class TestTranscriptCompleteness:
    async def test_full_transcript_channels(self, tmp_path: Path) -> None:
        provider = _MockProvider(tool_call_first=True)
        port = provider.start()
        _write_mock_agent_dir(tmp_path, port)
        service, mission = await _setup(tmp_path)
        scheduled = await _hire_real(service, mission, tmp_path)
        adapter = service._worker_manager._adapters["pi_managed"]
        driver = asyncio.create_task(
            service._worker_manager.run_to_completion(scheduled)
        )
        # pause/resume 一次——control channel 也要有记录。
        for _ in range(600):
            events = adapter._events.tail(scheduled.work_order_id, limit=500)
            if any(e["kind"] == "model_started" for e in events):
                break
            await asyncio.sleep(0.1)
        assert await adapter.request_pause(scheduled.work_order_id, reason="user")
        assert await adapter.request_resume(scheduled.work_order_id)
        result, _report = await asyncio.wait_for(driver, 180)
        assert result.status == "COMPLETED", result.summary

        records = _read_transcript(tmp_path, scheduled.work_order_id)
        assert records, "transcript 为空"
        # tseq 单调递增。
        tseqs = [r.get("tseq") for r in records]
        assert all(isinstance(t, int) for t in tseqs), "缺 tseq 字段"
        assert tseqs == sorted(tseqs), "tseq 非单调"
        # channel 覆盖：conversation/tools/usage/control/artifacts。
        channels = {r.get("channel") for r in records}
        assert "conversation" in channels
        assert "tools" in channels
        assert "usage" in channels
        assert "control" in channels
        assert "artifacts" in channels
        # 完整 assistant 全文（DONE-7 全文，非截断预览）。
        conv = [r for r in records if r.get("channel") == "conversation"]
        full = "\n".join(str(r.get("text", "")) for r in conv)
        assert "继续并完成：最终报告 DONE-7" in full
        # tools channel：read 调用带 args；结束带输出（错误输出也是完整证据）。
        tools = [r for r in records if r.get("channel") == "tools"]
        starts = [r for r in tools if r.get("phase") == "start"]
        ends = [r for r in tools if r.get("phase") == "end"]
        assert any(r.get("tool") == "read" for r in starts), tools
        assert any("args" in r and r["args"] for r in starts)
        assert any("output" in r for r in ends)
        # artifacts channel：产物清单记录（只读 scout 任务产物可为空；
        # 有文件时必须带 sha256）。
        artifacts = [r for r in records if r.get("channel") == "artifacts"]
        assert artifacts, "缺 artifacts channel 记录"
        files = artifacts[-1].get("files")
        assert isinstance(files, list)
        assert all("sha256" in f and "bytes" in f for f in files)
        # control channel：pause→PAUSED + resume→RUNNING 都有。
        control = [r for r in records if r.get("channel") == "control"]
        states = [r.get("state") for r in control]
        assert "PAUSED" in states and "RUNNING" in states
        await service.close()
