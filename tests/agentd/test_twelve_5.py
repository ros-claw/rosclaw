"""十二审 PR-12.5 红测试：优雅收尾与 partial 回收。

红测试先行——修复前必须红：
1. wall 超时被终止的 workbench 单：patch.diff（partial）已收集、
   checkpoint.json 存在且含 outcome/session 字段、state.json 是
   终态（不得 DB FAILED 而 state.json 仍 RUNNING）；
2. cancel 路径同样落 checkpoint。
"""

from __future__ import annotations

import json
import stat
import subprocess
from pathlib import Path

from rosclaw.agentd.workers.scheduler import CandidateView
from rosclaw.contracts.common import new_id
from rosclaw.contracts.worker.order import (
    BudgetEnvelope,
    ExpectedOutput,
    SideEffectPolicy,
    WorkOrderV1,
)
from tests.agentd.test_pi_tool_bridge import _setup


def _git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    (path / "seed.txt").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "seed"],
        cwd=path,
        check=True,
    )


class TestGracefulTermination:
    async def test_wall_timeout_collects_partial_and_checkpoint(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.workers import pi_managed

        monkeypatch.setattr(pi_managed, "WRAPUP_GRACE_SEC", 1.0)
        monkeypatch.setattr(pi_managed, "LIVENESS_TIMEOUT_SEC", 30.0)
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_repo(repo)
        # fake worker：写了一个文件但永不完工（liveness 持续）。
        fake = tmp_path / "fake-slow"
        fake.write_text(
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            "echo 'partial work' > partial.py\n"
            "while true; do\n"
            '  echo \'{"kind":"liveness","phase":"RUNNING_TOOL"}\'\n'
            "  sleep 0.3\n"
            "done\n"
        )
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        monkeypatch.setattr(pi_managed, "find_pi_agent_entry", lambda: ("/bin/sh", str(fake)))
        adapter = pi_managed.PiManagedAdapter(
            rosclaw_home=tmp_path / "rh", conn=service._store.connection
        )
        service._worker_manager._adapters["pi_managed"] = adapter
        adapter._manager_ref = service._worker_manager
        if service._registry.status_of("worker:rosclaw:pi") != "ENABLED":
            service._registry.set_status(
                "worker:rosclaw:pi", "ENABLED", actor_id="test", reason="fake entry"
            )
        card = service._registry.get("worker:rosclaw:pi")
        order = WorkOrderV1(
            work_order_id=new_id("wo"),
            mission_id=mission.mission_id,
            issued_by="test",
            capability="code.develop",
            goal="写不完的活",
            inputs={
                "instructions": "x",
                "worker_profile": "developer",
                "workspace": str(repo),
                # 十三审：硬截止必须显式带来源——benchmark。
                "execution_policy": {
                    "hard_deadline_sec": 2,
                    "hard_deadline_source": "benchmark",
                },
            },
            budgets=BudgetEnvelope(wall_time_sec=2, model_tokens=1000),
            expected_output=ExpectedOutput(artifacts=["text/plain", "text/x-diff"]),
            side_effect_policy=SideEffectPolicy(**{"class": "sandbox_process"}),
        )
        scheduled = service._worker_manager.hire(
            order,
            [CandidateView(card=card, registry_status="ENABLED", running_orders=0,
                           circuit_open=False)],
        )
        result, _report = await service._worker_manager.run_to_completion(scheduled)
        assert result.status == "FAILED"
        assert "hard deadline" in result.summary
        # partial 回收：patch 工件含 partial.py
        work_dir = tmp_path / "rh" / "work" / scheduled.work_order_id
        patch = work_dir / "artifacts" / "patch.diff"
        assert patch.exists(), "wall 超时未回收 partial patch"
        assert "partial.py" in patch.read_text()
        assert "partial" in result.summary
        # checkpoint + terminal state
        checkpoint = json.loads((work_dir / "checkpoint.json").read_text())
        assert checkpoint["outcome"] == "FAILED"
        state = json.loads((work_dir / "state.json").read_text())
        assert state["status"] == "FAILED"  # 不得仍是 RUNNING
        assert state["phase"] == "TERMINAL"
        await service.close()
