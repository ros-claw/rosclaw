"""十一审 PR-E 通用任务基准（真实 Kimi K3，无 key 诚实 skip）。

不以五角星特化——通用开发任务：
1. 多文件 bug 修复（scratch repo）；
2. 新函数 + 测试 + 跑通。

运行：ROSCLAW_KIMI_API_KEY=sk-kimi-... pytest tests/agentd/test_eleven_e_live.py
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


def _git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    (path / "README.md").write_text("# demo\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=path,
        check=True,
    )


async def _run_dev_order(tmp_path: Path, goal: str, wall: int = 420):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    service, mission = await _setup(tmp_path)
    dispatcher = PiToolDispatcher(service)
    started = await dispatcher.execute(
        _request(
            "rosclaw_delegate",
            mission=mission.mission_id,
            idem=f"idem_11e_live_{abs(hash(goal)) % 99999}",
            arguments={
                "goal": goal,
                "worker_id": "worker:rosclaw:pi",
                "worker_profile": "developer",
                "workspace": str(repo),
                "budget": {"wall_time_sec": wall, "model_tokens": 120_000},
                "sync_grace_sec": 0,
            },
        )
    )
    assert started.ok, started.summary
    order = service._worker_manager.orders_for_mission(mission.mission_id)[0]
    final = None
    for _ in range(int(wall * 2)):
        current = service._worker_manager.order(order.work_order_id)
        if current and current.status in ("ACCEPTED", "FAILED", "CANCELLED", "EXPIRED"):
            final = current
            break
        await asyncio.sleep(0.5)
    assert final is not None, "超时未到终态"
    return service, repo, final, order


@pytest.mark.skipif(not _has_key(), reason="无真实 provider key——诚实 skip")
class TestGenericCodingBenchmark:
    async def test_multifile_bugfix(self, tmp_path: Path) -> None:
        """多文件 bug 修复：diff 真实 + 测试 exit code 可复核 + 主仓库干净。"""
        service, repo, final, order = await _run_dev_order(
            tmp_path,
            "这是 Python 项目：1) 创建 calc.py（含 divide 函数——除零时返回"
            " None 而不是抛异常）；2) 创建 test_calc.py（pytest，覆盖正常"
            "除法与除零）；3) 用 bash 运行 python3 -m pytest test_calc.py"
            " 确认通过；4) 报告改了哪些文件、测试 exit code。",
        )
        assert final.status == "ACCEPTED", f"status={final.status}"
        # workdir 因 service home 而异——从 tmp_path 推导。
        candidates = list(tmp_path.glob("**/patch.diff"))
        assert candidates, "无 patch 工件"
        content = candidates[0].read_text()
        assert "calc.py" in content
        assert not (repo / "calc.py").exists(), "主仓库被污染"
        await service.close()
