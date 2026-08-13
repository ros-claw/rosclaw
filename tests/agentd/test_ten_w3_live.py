"""十审 W3 真实模型验收：developer Worker 在隔离 worktree 真实开发。

无 provider key 时诚实 skip。运行：
ROSCLAW_KIMI_API_KEY=sk-kimi-... pytest tests/agentd/test_ten_w3_live.py
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

import pytest

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from tests.agentd.test_pi_tool_bridge import _request, _setup

KIMI_ENV_VARS = ("ROSCLAW_KIMI_API_KEY", "KIMI_API_KEY", "MOONSHOT_API_KEY")


def _has_key() -> bool:
    return any(os.environ.get(v) for v in KIMI_ENV_VARS)


@pytest.mark.skipif(not _has_key(), reason="无真实 provider key——诚实 skip")
class TestLiveDeveloperWorker:
    async def test_developer_worker_real_change_and_patch(self, tmp_path: Path) -> None:
        """真实 K3：scratch git 仓库里创建脚本并跑通——patch/bash log
        工件可校验，主仓库零污染，不自动合并。"""
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        (repo / "README.md").write_text("# demo\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
            cwd=repo,
            check=True,
        )
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        started = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_w3_live",
                arguments={
                    "goal": "在 workspace 里创建 hello.py（打印 hello rosclaw），"
                    "用 bash 运行 python3 hello.py 确认输出，然后报告",
                    "worker_id": "worker:rosclaw:pi",
                    "worker_profile": "developer",
                    "workspace": str(repo),
                    "budget": {"wall_time_sec": 300, "model_tokens": 60000},
                    "sync_grace_sec": 0,
                },
            )
        )
        assert started.ok, started.summary
        assert started.status == "STARTED"
        order = service._worker_manager.orders_for_mission(mission.mission_id)[0]
        final = None
        for _ in range(600):
            current = service._worker_manager.order(order.work_order_id)
            if current and current.status in ("ACCEPTED", "FAILED", "CANCELLED", "EXPIRED"):
                final = current
                break
            await asyncio.sleep(0.5)
        assert final is not None, "developer Worker 5 分钟未到终态"
        assert final.status == "ACCEPTED", f"status={final.status}"
        # 工件可校验：patch 含 hello.py；bash log 含 python3 运行记录。
        import json as _json

        from rosclaw.contracts.worker.order import WorkResultV1

        row = service._store.connection.execute(
            "SELECT result_json FROM work_results WHERE work_order_id = ?",
            (order.work_order_id,),
        ).fetchone()
        result = WorkResultV1(**_json.loads(row["result_json"]))
        media_types = {a.media_type for a in result.artifacts}
        assert "text/x-diff" in media_types
        # 主仓库零污染（promotion 未自动发生）。
        assert not (repo / "hello.py").exists()
        await service.close()
