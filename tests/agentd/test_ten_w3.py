"""十审 Gate W3 红测试：Developer Workbench（Python 侧）。

红测试先行——修复前必须红：
1. developer profile 的 delegate 产出 code.develop + sandbox_process +
   期望 text/x-diff 工件（此前一律 analysis.text + none）。
2. git 目标 → 独立 worktree（主仓库不被污染）；Worker 改动收进
   patch.diff 工件（text/x-diff）；promotion 不自动发生。
3. 空改动诚实：patch.diff 带 EMPTY 标记 + summary 注明。
4. registry：worker:rosclaw:pi 声明 code.develop/sandbox_process。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from tests.agentd.test_pi_tool_bridge import _request, _setup


def _git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    (path / "seed.txt").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "seed"],
        cwd=path,
        check=True,
    )


def _fake_worker(tmp_path: Path, body: str) -> Path:
    fake = tmp_path / "fake-entry"
    fake.write_text(body)
    fake.chmod(0o755)
    return fake


async def _run_developer_order(service, mission, tmp_path, fake, repo, monkeypatch):
    from rosclaw.agentd.workers import pi_managed

    monkeypatch.setattr(pi_managed, "find_pi_agent_entry", lambda: ("/bin/sh", str(fake)))
    adapter = pi_managed.PiManagedAdapter(rosclaw_home=tmp_path / "rh")
    service._worker_manager._adapters["pi_managed"] = adapter

    from rosclaw.agentd.workers.scheduler import CandidateView
    from rosclaw.contracts.common import new_id
    from rosclaw.contracts.worker.order import (
        BudgetEnvelope,
        ExpectedOutput,
        SideEffectPolicy,
        WorkOrderV1,
    )

    card = service._registry.get("worker:rosclaw:pi")
    order = WorkOrderV1(
        work_order_id=new_id("wo"),
        mission_id=mission.mission_id,
        issued_by="test",
        capability="code.develop",
        goal="实现功能",
        inputs={
            "instructions": "x",
            "worker_profile": "developer",
            "workspace": str(repo),
        },
        budgets=BudgetEnvelope(wall_time_sec=120, model_tokens=1000),
        expected_output=ExpectedOutput(artifacts=["text/plain", "text/x-diff"]),
        side_effect_policy=SideEffectPolicy(**{"class": "sandbox_process"}),
    )
    scheduled = service._worker_manager.hire(
        order,
        [CandidateView(card=card, registry_status="ENABLED", running_orders=0,
                       circuit_open=False)],
    )
    return await service._worker_manager.run_to_completion(scheduled)


class TestDeveloperContract:
    async def test_delegate_developer_profile_produces_sandbox_order(
        self, tmp_path: Path
    ) -> None:
        service, mission = await _setup(tmp_path)
        if service._registry.status_of("worker:rosclaw:pi") != "ENABLED":
            # CI 无 node/dist：内置 Worker 按设计 DISABLED——注册 stub
            # 开发者卡验证合约语义（真实产品路径由 W3 live 覆盖）。
            from rosclaw.contracts.worker.card import (
                CapabilityDecl,
                WorkerCardV1,
                WorkerConstraints,
                WorkerHealth,
                WorkerImplementation,
                WorkerKind,
                WorkerProvenance,
                WorkerSecurity,
                WorkerTrust,
            )

            service._registry.register(
                WorkerCardV1(
                    worker_id="worker:stub:dev",
                    display_name="Stub Dev",
                    kind=WorkerKind.HARNESS,
                    adapter_type="process_stdio",
                    adapter_version="1.0.0",
                    implementation=WorkerImplementation(
                        product="stub", version="1.0.0", executable_ref="inproc:"
                    ),
                    capabilities=[
                        CapabilityDecl(name="code.develop", side_effect_class="sandbox_process")
                    ],
                    constraints=WorkerConstraints(
                        supported_platforms=["linux"], max_concurrency=4
                    ),
                    security=WorkerSecurity(isolation="process"),
                    health=WorkerHealth(
                        probe="adapter:ping", heartbeat_interval_sec=15, lease_ttl_sec=3600
                    ),
                    provenance=WorkerProvenance(source="test", license="MIT"),
                    trust=WorkerTrust(initial_level="T3", evidence_count=0),
                ),
                actor_id="test",
            )
            from tests.agentd.test_ten_w0 import _slow_adapter_module

            service._worker_manager._adapters["process_stdio"] = _slow_adapter_module()()
        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_w3_contract",
                arguments={
                    "goal": "改代码",
                    "worker_profile": "developer",
                    "workspace": str(tmp_path),
                    "sync_grace_sec": 0,
                },
            )
        )
        orders = service._worker_manager.orders_for_mission(mission.mission_id)
        assert orders, f"未产生订单: {result.summary}"
        order = orders[0]
        assert order.capability == "code.develop"
        assert order.side_effect_policy.class_ == "sandbox_process"
        assert "text/x-diff" in order.expected_output.artifacts
        await dispatcher.execute(
            _request(
                "rosclaw_cancel_work",
                mission=mission.mission_id,
                idem="idem_w3_contract_c",
                arguments={"work_order_id": order.work_order_id},
            )
        )
        await service.close()

    async def test_registry_declares_code_develop_sandbox(self, tmp_path: Path) -> None:
        service, _mission = await _setup(tmp_path)
        card = service._registry.get("worker:rosclaw:pi")
        dev = next((c for c in card.capabilities if c.name == "code.develop"), None)
        assert dev is not None
        assert dev.side_effect_class == "sandbox_process"
        await service.close()


class TestWorktreeAndPatch:
    async def test_worktree_patch_collected_main_untouched(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        service, mission = await _setup(tmp_path)
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_repo(repo)
        # fake worker：在 cwd（worktree）写新文件并报告。
        fake = _fake_worker(
            tmp_path,
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            "echo 'print(\"hello\")' > hello.py\n"
            'echo \'{"kind":"attempt_finished","report":"已创建 hello.py"}\'\n',
        )
        result, report = await _run_developer_order(
            service, mission, tmp_path, fake, repo, monkeypatch
        )
        assert result.status == "COMPLETED", result.summary
        media_types = {a.media_type for a in result.artifacts}
        assert "text/x-diff" in media_types, f"缺 patch 工件: {media_types}"
        # patch 内容真实含新文件。
        patch_path = tmp_path / "rh" / "work" / result.work_order_id / "artifacts" / "patch.diff"
        assert patch_path.exists()
        assert "hello.py" in patch_path.read_text()
        # promotion 不自动发生：主仓库干净、worktree/branch 保留。
        assert not (repo / "hello.py").exists()
        summary = result.summary
        assert "未合并" in summary
        assert report.accepted, report.reasons
        await service.close()

    async def test_empty_patch_honest(self, tmp_path: Path, monkeypatch) -> None:
        service, mission = await _setup(tmp_path)
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_repo(repo)
        fake = _fake_worker(
            tmp_path,
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            'echo \'{"kind":"attempt_finished","report":"分析完毕，无需改动"}\'\n',
        )
        result, _report = await _run_developer_order(
            service, mission, tmp_path, fake, repo, monkeypatch
        )
        patch_path = tmp_path / "rh" / "work" / result.work_order_id / "artifacts" / "patch.diff"
        assert patch_path.exists()
        assert "EMPTY DIFF" in patch_path.read_text()
        assert "未产生文件改动" in result.summary
        await service.close()

    async def test_non_git_workspace_honest_note(self, tmp_path: Path, monkeypatch) -> None:
        service, mission = await _setup(tmp_path)
        plain = tmp_path / "plain"
        plain.mkdir()
        fake = _fake_worker(
            tmp_path,
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            "echo 'x' > out.txt\n"
            'echo \'{"kind":"attempt_finished","report":"done"}\'\n',
        )
        result, _report = await _run_developer_order(
            service, mission, tmp_path, fake, plain, monkeypatch
        )
        assert "非 git" in result.summary or "无 VCS" in result.summary
        # 文件写在 scratch workspace，不污染原目录。
        assert not (plain / "out.txt").exists()
        await service.close()
