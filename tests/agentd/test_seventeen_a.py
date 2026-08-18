"""十六审 PR-17.1 红测试：终态契约与假成功根治（P0-A）。

红测试先行——修复前必须红：
A1. WorkResultV1 必须接受 CANCELLED（用户取消崩成 FAILED 是契约缺陷）；
A2. checks==0 绝不 SUCCEEDED；acceptance 缺失时交付物→required_files、
    纯文本任务→报告非空检查；shell 字符串 tests_command 禁止（注入面）；
A3. Worker BLOCKED → Task BLOCKED（诚实映射）；FAILED 状态判定走结构化
    cause，禁止摘要字符串推断；execution 终态不可被后续驱动覆盖。
"""

from __future__ import annotations

import asyncio
import stat
from pathlib import Path

import pytest

from tests.agentd.test_pi_tool_bridge import _setup


def _enable_fake(service, tmp_path: Path, monkeypatch, body: str, name: str) -> None:
    from rosclaw.agentd.workers import pi_managed

    fake = tmp_path / name
    fake.write_text(body)
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(
        pi_managed, "find_pi_agent_entry", lambda: ("/bin/sh", str(fake))
    )
    adapter = pi_managed.PiManagedAdapter(
        rosclaw_home=tmp_path, conn=service._store.connection
    )
    service._worker_manager._adapters["pi_managed"] = adapter
    adapter._manager_ref = service._worker_manager
    if service._registry.status_of("worker:rosclaw:pi") != "ENABLED":
        service._registry.set_status(
            "worker:rosclaw:pi", "ENABLED", actor_id="test", reason="fake"
        )


async def _wait_terminal(plane, execution_id: str, timeout: float = 60.0):
    for _ in range(int(timeout * 20)):
        row = plane._get(execution_id)
        if row["state"] in ("SUCCEEDED", "FAILED", "BLOCKED", "CANCELLED"):
            return row
        await asyncio.sleep(0.05)
    return None


class TestCancelledContract:
    def test_work_result_accepts_cancelled(self) -> None:
        """A1：CANCELLED 是合法 WorkResult 终态（用户取消不得崩契约）。"""
        from rosclaw.contracts.worker.order import WorkResultV1

        result = WorkResultV1(
            work_order_id="wo_x", worker_id="w", lease_id="l",
            status="CANCELLED", summary="user cancelled",
        )
        assert result.status == "CANCELLED"

    async def test_manager_cancelled_result_constructs(
        self, tmp_path: Path
    ) -> None:
        """A1：manager 取消路径直接构造 CANCELLED 结果（不 ValidationError）。"""
        from rosclaw.contracts.worker.order import WorkOrderV1

        service, mission = await _setup(tmp_path)
        order = WorkOrderV1(
            work_order_id="wo_c1", mission_id=mission.mission_id,
            issued_by="t", capability="code.develop", goal="g",
        )
        result = service._worker_manager._cancelled_result(order)
        assert result.status == "CANCELLED"
        await service.close()


class TestBlockedMapping:
    async def test_worker_blocked_maps_task_blocked(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A3：Worker 结构化 BLOCKED（termination.json）→ Task BLOCKED。
        绝不进 VERIFYING/SUCCEEDED，也不FAILED化。"""
        service, mission = await _setup(tmp_path)
        _enable_fake(
            service, tmp_path, monkeypatch,
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            'echo \'{"kind":"attempt_finished","report":"只读 profile 无法安装软件"}\'\n'
            # 原子写 termination.json（work/<wo>/termination.json）。
            "TERM_DIR=$(dirname \"$0\")/work\n"
            "for d in \"$TERM_DIR\"/wo_*; do\n"
            '  echo \'{"cause":"BLOCKED","detail":"missing capability: '
            'process.exec"}\' > "$d/termination.json.tmp"\n'
            '  mv "$d/termination.json.tmp" "$d/termination.json"\n'
            "done\n",
            "fake-blocked",
        )
        plane = service._task_control_plane
        view = await plane.submit(
            mission.mission_id,
            {"goal": "安装 Pillow 并渲染 GIF",
             "required_capabilities": ["code.implement"],
             "effects": "workspace_only", "acceptance": {}},
            idem="17a_blocked",
        )
        row = await _wait_terminal(plane, view["execution_id"])
        assert row is not None
        assert row["state"] == "BLOCKED", (
            f"Worker BLOCKED 必须映射 Task BLOCKED，实际 {row['state']}: "
            f"{row['summary']}"
        )
        await service.close()


class TestZeroAcceptanceTruth:
    async def test_empty_acceptance_empty_report_not_succeeded(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A2：acceptance={} + 空报告 → 绝不 SUCCEEDED（checks==0 禁令）。"""
        service, mission = await _setup(tmp_path)
        _enable_fake(
            service, tmp_path, monkeypatch,
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            'echo \'{"kind":"attempt_finished","report":""}\'\n',
            "fake-empty",
        )
        plane = service._task_control_plane
        view = await plane.submit(
            mission.mission_id,
            {"goal": "回答问题", "required_capabilities": ["research.answer"],
             "effects": "workspace_only", "acceptance": {}},
            idem="17a_empty",
        )
        row = await _wait_terminal(plane, view["execution_id"])
        assert row is not None
        assert row["state"] != "SUCCEEDED", (
            f"零验收+空报告不得 SUCCEEDED: {row['summary']}"
        )
        await service.close()

    async def test_empty_acceptance_with_report_has_real_check(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A2：纯文本任务无验收定义 → 报告非空是真实检查（checks≥1，
        摘要不得再出现 PASS·0 项）。"""
        service, mission = await _setup(tmp_path)
        _enable_fake(
            service, tmp_path, monkeypatch,
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            'echo \'{"kind":"attempt_finished","report":"答案是 42"}\'\n',
            "fake-report",
        )
        plane = service._task_control_plane
        view = await plane.submit(
            mission.mission_id,
            {"goal": "回答问题", "required_capabilities": ["research.answer"],
             "effects": "workspace_only", "acceptance": {}},
            idem="17a_report",
        )
        row = await _wait_terminal(plane, view["execution_id"])
        assert row is not None
        assert row["state"] == "SUCCEEDED", row["summary"]
        assert "PASS·0" not in (row["summary"] or ""), "零检查成功是假成功"
        await service.close()


class TestVerifierSecurity:
    def test_zero_checks_never_pass(self, tmp_path: Path) -> None:
        """A2 契约：checks==0 时 verdict 绝不 pass。"""
        from rosclaw.agentd.control_plane import TaskControlPlane

        plane = TaskControlPlane.__new__(TaskControlPlane)
        verdict = asyncio.run(
            plane._verify_acceptance({}, tmp_path, report="")
        )
        assert not verdict["pass"], "零证据不得 pass"
        assert any("ACCEPTANCE_MISSING" in f for f in verdict["failures"])

    async def test_shell_tests_command_rejected_at_submit(
        self, tmp_path: Path
    ) -> None:
        """安全：模型提交的 shell 字符串 tests_command 在 submit 即拒绝
        （命令注入面——禁止 create_subprocess_shell 执行模型字符串）。"""
        service, mission = await _setup(tmp_path)
        plane = service._task_control_plane
        pwned = tmp_path / "pwned"
        with pytest.raises(ValueError, match="tests_command|shell"):
            await plane.submit(
                mission.mission_id,
                {"goal": "x", "required_capabilities": ["code.implement"],
                 "acceptance": {"tests_command": f"touch {pwned}"}},
                idem="17a_shell",
            )
        assert not pwned.exists()
        await service.close()

    def test_structured_argv_verifier(self, tmp_path: Path) -> None:
        """结构化验收：run.argv 不走 shell；argv[0] 白名单外拒绝执行。"""
        from rosclaw.agentd.control_plane import TaskControlPlane

        plane = TaskControlPlane.__new__(TaskControlPlane)
        ok = asyncio.run(
            plane._verify_acceptance(
                {"acceptance": {"run": {"argv": ["python3", "-c", "pass"]}}},
                tmp_path,
                report="",
            )
        )
        assert ok["pass"], ok["failures"]
        bad = asyncio.run(
            plane._verify_acceptance(
                {"acceptance": {"run": {"argv": ["sh", "-c", "touch /tmp/x"]}}},
                tmp_path,
                report="",
            )
        )
        assert not bad["pass"], "sh -c 必须被拒绝（shell 后门）"
        assert not Path("/tmp/x").exists()

    def test_verifier_env_carries_no_secrets(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """安全：验收子进程 env 不得继承 API key/HOME 等宿主凭据。"""
        from rosclaw.agentd.control_plane import TaskControlPlane

        monkeypatch.setenv("KIMI_API_KEY", "sk-should-not-leak")
        monkeypatch.setenv("MOONSHOT_API_KEY", "sk-should-not-leak")
        plane = TaskControlPlane.__new__(TaskControlPlane)
        verdict = asyncio.run(
            plane._verify_acceptance(
                {"acceptance": {"run": {"argv": [
                    "python3", "-c",
                    "import os,sys; "
                    "sys.exit(1 if os.environ.get('KIMI_API_KEY') else 0)",
                ]}}},
                tmp_path,
                report="",
            )
        )
        assert verdict["pass"], (
            f"验收子进程必须无凭据 env: {verdict['failures']}"
        )


class TestAuthoritativeMapping:
    async def test_failed_with_deliverable_word_not_repaired(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A3：FAILED 判定走结构化 cause——摘要含 DELIVERABLE 字样但
        cause=PROVIDER_FATAL → FAILED，不进 REPAIRING（禁止字符串推断）。"""
        service, mission = await _setup(tmp_path)
        _enable_fake(
            service, tmp_path, monkeypatch,
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            "TERM_DIR=$(dirname \"$0\")/work\n"
            "for d in \"$TERM_DIR\"/wo_*; do\n"
            '  echo \'{"cause":"PROVIDER_FATAL","detail":"user prose mentions '
            'DELIVERABLE casually"}\' > "$d/termination.json.tmp"\n'
            '  mv "$d/termination.json.tmp" "$d/termination.json"\n'
            "done\n"
            "exit 1\n",
            "fake-fatal",
        )
        plane = service._task_control_plane
        view = await plane.submit(
            mission.mission_id,
            {"goal": "实现功能", "required_capabilities": ["code.implement"],
             "effects": "workspace_only",
             "acceptance": {"required_files": ["x.txt"]}},
            idem="17a_fatal",
        )
        row = await _wait_terminal(plane, view["execution_id"], timeout=90)
        assert row is not None
        assert row["state"] == "FAILED", row["state"]
        attempts = service._store.connection.execute(
            "SELECT COUNT(*) AS c FROM work_orders WHERE mission_id = ?",
            (mission.mission_id,),
        ).fetchone()["c"]
        assert attempts == 1, f"PROVIDER_FATAL 不得触发 REPAIRING（{attempts} 单）"
        await service.close()

    async def test_terminal_state_not_overwritten(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A1/A3：用户取消后 execution CANCELLED 是终态——后续驱动收尾
        不得覆盖成 FAILED/SUCCEEDED。"""
        service, mission = await _setup(tmp_path)
        _enable_fake(
            service, tmp_path, monkeypatch,
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            "sleep 30\n"
            'echo \'{"kind":"attempt_finished","report":"late"}\'\n',
            "fake-slow",
        )
        plane = service._task_control_plane
        view = await plane.submit(
            mission.mission_id,
            {"goal": "长任务", "required_capabilities": ["code.implement"],
             "effects": "workspace_only",
             "acceptance": {"required_files": ["x.txt"]}},
            idem="17a_cancel",
        )
        execution_id = view["execution_id"]
        # 等 RUNNING（进程起来）再取消。
        for _ in range(600):
            row = plane._get(execution_id)
            if row["state"] == "RUNNING" and row.get("work_order_id"):
                break
            await asyncio.sleep(0.05)
        await service._worker_manager.cancel_order(
            row["work_order_id"], reason="user_task_cancel"
        )
        plane._update_state(execution_id, "CANCELLED", summary="用户取消")
        row = await _wait_terminal(plane, execution_id, timeout=60)
        assert row is not None
        assert row["state"] == "CANCELLED", (
            f"终态被覆盖成 {row['state']}: {row['summary']}"
        )
        await service.close()
