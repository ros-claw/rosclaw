"""十二审 PR-12.4 红测试：WorkSpecV2 + Deliverable Verifier + preflight。

红测试先行——修复前必须红：
1. delegate task_type=artifact_build 默认期望 gif/mp4（不再硬编码
   text/x-diff）；simulation_run 期望 trace json + 媒体；
2. 媒体魔数校验：GIF89a 通过、假 gif 拒绝、空文件拒绝、坏 JSON 拒绝；
3. 交付物未过 → DELIVERABLE_FAILED（不得宣布完成）；
4. 无编码器时 preflight 5 秒内 BLOCKED_PREFLIGHT（不起 Worker）。
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from rosclaw.contracts.worker.workspec import (
    DeliverableV1,
    WorkSpecV2,
    expected_media_types,
    validate_media_file,
)
from tests.agentd.test_pi_tool_bridge import _request, _setup


class TestWorkSpec:
    def test_task_type_drives_expected_artifacts(self) -> None:
        from rosclaw.agentd.pi_bridge.tool_dispatch import _expected_artifacts

        assert _expected_artifacts("analyze", [], False) == ["text/plain"]
        assert "text/x-diff" in _expected_artifacts("code_change", [], True)
        media = _expected_artifacts("artifact_build", [], True)
        assert "image/gif" in media and "video/mp4" in media
        assert "text/x-diff" not in media  # artifact_build diff 可选
        sim = _expected_artifacts("simulation_run", [], True)
        assert "application/json" in sim and "image/gif" in sim

    def test_explicit_deliverables_win(self) -> None:
        spec = WorkSpecV2(
            task_type="artifact_build",
            deliverables=[DeliverableV1(id="chart", media_types=["image/png"])],
        )
        assert expected_media_types(spec) == ["image/png"]


class TestMediaValidation:
    def test_magic_bytes(self, tmp_path: Path) -> None:
        gif = tmp_path / "a.gif"
        gif.write_bytes(b"GIF89a" + b"\x00" * 100)
        assert validate_media_file(gif, "image/gif") is None
        fake = tmp_path / "fake.gif"
        fake.write_bytes(b"not a gif at all............")
        assert validate_media_file(fake, "image/gif") is not None
        empty = tmp_path / "empty.gif"
        empty.write_bytes(b"")
        assert validate_media_file(empty, "image/gif") is not None
        assert validate_media_file(tmp_path / "missing.gif", "image/gif") is not None
        mp4 = tmp_path / "v.mp4"
        mp4.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 32)
        assert validate_media_file(mp4, "video/mp4") is None
        bad_json = tmp_path / "t.json"
        bad_json.write_text("{not json")
        assert validate_media_file(bad_json, "application/json") is not None
        good_json = tmp_path / "ok.json"
        good_json.write_text('{"trace": []}')
        assert validate_media_file(good_json, "application/json") is None


class TestDeliverableGate:
    async def test_missing_media_deliverable_fails_honestly(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """要求 GIF 而 Worker 只交文字 → DELIVERABLE_FAILED。"""
        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.workers import pi_managed

        fake = tmp_path / "fake-nomedia"
        fake.write_text(
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            "echo 'hello' > notes.txt\n"
            'echo \'{"kind":"attempt_finished","report":"写好了文档"}\'\n'
        )
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        monkeypatch.setattr(pi_managed, "find_pi_agent_entry", lambda: ("/bin/sh", str(fake)))
        adapter = pi_managed.PiManagedAdapter(
            rosclaw_home=tmp_path, conn=service._store.connection
        )
        service._worker_manager._adapters["pi_managed"] = adapter
        adapter._manager_ref = service._worker_manager
        if service._registry.status_of("worker:rosclaw:pi") != "ENABLED":
            service._registry.set_status(
                "worker:rosclaw:pi", "ENABLED", actor_id="test", reason="fake entry"
            )
        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_124_gate",
                arguments={
                    "goal": "生成动画",
                    "worker_id": "worker:rosclaw:pi",
                    "worker_profile": "sim-builder",
                    "task_type": "artifact_build",
                    "workspace": str(tmp_path),
                },
            )
        )
        assert not result.ok
        assert "DELIVERABLE_FAILED" in (result.summary + str(result.error_code or "")), (
            result.summary
        )
        await service.close()

    async def test_preflight_blocks_without_encoder(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """无 ffmpeg 且无 Pillow → BLOCKED_PREFLIGHT（不起 Worker 进程）。"""
        import shutil

        from rosclaw.agentd.workers import pi_managed

        monkeypatch.setattr(shutil, "which", lambda _x: None)
        # PIL 不可用场景
        import sys

        monkeypatch.setitem(sys.modules, "PIL", None)
        adapter = pi_managed.PiManagedAdapter(rosclaw_home=tmp_path)
        from rosclaw.contracts.common import new_id
        from rosclaw.contracts.worker.order import (
            BudgetEnvelope,
            ExpectedOutput,
            SideEffectPolicy,
            WorkOrderV1,
        )

        order = WorkOrderV1(
            work_order_id=new_id("wo"),
            mission_id="mis_x",
            issued_by="test",
            capability="code.develop",
            goal="渲染 GIF",
            inputs={"task_type": "artifact_build", "worker_profile": "sim-builder"},
            budgets=BudgetEnvelope(wall_time_sec=60, model_tokens=1000),
            expected_output=ExpectedOutput(artifacts=["image/gif"]),
            side_effect_policy=SideEffectPolicy(**{"class": "sandbox_process"}),
        )
        from rosclaw.agentd.workers.adapter import AdapterError

        with pytest.raises(AdapterError, match="BLOCKED_PREFLIGHT"):
            adapter._preflight(order)


class TestDelegateWorkSpecArgs:
    async def test_delegate_carries_task_type_and_deliverables(
        self, tmp_path: Path
    ) -> None:
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_124_args",
                arguments={
                    "goal": "渲染五角星动画",
                    "worker_profile": "sim-builder",
                    "task_type": "artifact_build",
                    "deliverables": [{"media_types": ["image/gif"], "required": True}],
                    "workspace": str(tmp_path),
                    "sync_grace_sec": 0,
                },
            )
        )
        orders = service._worker_manager.orders_for_mission(mission.mission_id)
        if orders:
            order = orders[0]
            assert order.inputs.get("task_type") == "artifact_build"
            assert "image/gif" in order.expected_output.artifacts
            await dispatcher.execute(
                _request(
                    "rosclaw_cancel_work",
                    mission=mission.mission_id,
                    idem="idem_124_args_c",
                    arguments={"work_order_id": order.work_order_id},
                )
            )
        else:
            # CI 无 node——调度拒绝也要携带正确错误（诚实）。
            assert not result.ok
        await service.close()


class TestNoFabricatedMedia:
    async def test_preexisting_repo_gif_does_not_satisfy_deliverable(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """自审发现的工件造假洞：worktree 里仓库既有的 GIF 不得满足
        deliverable——只有本 attempt 实际产出的才算。"""
        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.workers import pi_managed

        repo = tmp_path / "repo"
        repo.mkdir()
        import subprocess

        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        # 仓库自带一个合法 GIF（非 Worker 产出）。
        (repo / "existing.gif").write_bytes(b"GIF89a" + b"\x00" * 200)
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
            cwd=repo,
            check=True,
        )
        fake = tmp_path / "fake-noprod"
        fake.write_text(
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            'echo \'{"kind":"attempt_finished","report":"没做动画"}\'\n'
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
        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_124_nofab",
                arguments={
                    "goal": "生成动画",
                    "worker_id": "worker:rosclaw:pi",
                    "worker_profile": "sim-builder",
                    "task_type": "artifact_build",
                    "deliverables": [{"media_types": ["image/gif"], "required": True}],
                    "workspace": str(repo),
                    "sync_grace_sec": 5,
                },
            )
        )
        assert not result.ok, "仓库既有 GIF 被误算为 Worker 产出（造假洞未修）"
        await service.close()

    async def test_worker_created_gif_satisfies_deliverable(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """对照：Worker 真实新建的 GIF 必须满足 deliverable。"""
        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.workers import pi_managed

        repo = tmp_path / "repo"
        repo.mkdir()
        import subprocess

        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        (repo / "seed.txt").write_text("x")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
            cwd=repo,
            check=True,
        )
        fake = tmp_path / "fake-prod"
        fake.write_text(
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            "printf 'GIF89a' > out.gif; head -c 200 /dev/zero >> out.gif\n"
            'echo \'{"kind":"attempt_finished","report":"动画已生成"}\'\n'
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
        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_124_fabok",
                arguments={
                    "goal": "生成动画",
                    "worker_id": "worker:rosclaw:pi",
                    "worker_profile": "sim-builder",
                    "task_type": "artifact_build",
                    "deliverables": [{"media_types": ["image/gif"], "required": True}],
                    "workspace": str(repo),
                    "sync_grace_sec": 5,
                },
            )
        )
        assert result.ok, result.summary
        await service.close()
