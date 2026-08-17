"""十六审 PR-17.2 红测试：Task 编译器 + 能力/副作用集合路由（P0-B）。

红测试先行——修复前必须红：
1. profile 由结构化需求编译（effects/capabilities/runtime_requirements），
   不按 capability 名称前缀猜——"装依赖跑脚本"（effects=workspace_only）
   必须是 developer，不是只读 scout；
2. 集合包含授权：required_effects ⊄ granted → 启动前 BLOCKED（零 Worker
   预算燃烧），不抱侥幸启动；
3. 运行中缺能力 → 结构化 CapabilityRequest → 升级 profile 恢复同一
   session（同一 execution，attempt 折叠）——不重新招聘新任务；
4. 指纹覆盖 kind/runtime_requirements（改依赖不错挂旧任务）。
"""

from __future__ import annotations

from pathlib import Path

from tests.agentd.test_pi_tool_bridge import _setup
from tests.agentd.test_seventeen_a import _enable_fake, _wait_terminal


class TestTaskCompiler:
    def test_workspace_effects_compile_developer(self) -> None:
        """effects=workspace_only（要 bash+写文件）→ developer profile。
        红：当前按 capability 前缀猜——research.* 落 scout。"""
        from rosclaw.agentd.task_compiler import compile_task

        plan = compile_task(
            {"goal": "安装 Pillow 并渲染 GIF",
             "required_capabilities": ["research.setup"],
             "effects": "workspace_only"}
        )
        assert plan.profile in ("developer", "sim-builder"), plan.profile
        assert {"process.exec", "workspace.write"} <= plan.granted_effects

    def test_readonly_task_compiles_scout(self) -> None:
        """无 effects + 只读 capability → scout（诚实只读，不过度授权）。"""
        from rosclaw.agentd.task_compiler import compile_task

        plan = compile_task(
            {"goal": "读日志找根因", "required_capabilities": ["repo.inspect"]}
        )
        assert plan.profile == "scout"
        assert plan.granted_effects == frozenset()

    def test_network_requirement_blocked_prestart(self) -> None:
        """network.write 不在任何 profile 授权内 → 编译期 blocked_reason
        （启动前 BLOCKED，不烧 Worker）。"""
        from rosclaw.agentd.task_compiler import compile_task

        plan = compile_task(
            {"goal": "上传结果到外部服务器",
             "effects": ["process.exec", "network.write"]}
        )
        assert plan.blocked_reason, "network.write 必须编译期拦截"
        assert "network" in plan.blocked_reason

    def test_runtime_requirements_in_plan(self) -> None:
        """runtime_requirements 进编译产物（Runtime Manager 预置依据）。"""
        from rosclaw.agentd.task_compiler import compile_task

        plan = compile_task(
            {"goal": "渲染", "effects": "workspace_only",
             "runtime_requirements": {"python_packages": ["Pillow>=10"]}}
        )
        assert plan.runtime_requirements.get("python_packages") == ["Pillow>=10"]

    def test_fingerprint_covers_runtime_requirements(self) -> None:
        """指纹覆盖 kind/runtime_requirements——改依赖版本不 attach 旧任务。"""
        from rosclaw.agentd.control_plane import _fingerprint

        base = {"goal": "g", "runtime_requirements": {"python_packages": ["Pillow>=10"]}}
        changed = {"goal": "g", "runtime_requirements": {"python_packages": ["Pillow>=11"]}}
        assert _fingerprint("m", base) != _fingerprint("m", changed)


class TestPreStartBlocking:
    async def test_ungrantable_effects_blocked_zero_workers(
        self, tmp_path: Path
    ) -> None:
        """集合包含失败 → 启动前 BLOCKED：不创建任何 WorkOrder。"""
        service, mission = await _setup(tmp_path)
        plane = service._task_control_plane
        view = await plane.submit(
            mission.mission_id,
            {"goal": "把结果传到外部 API",
             "effects": ["process.exec", "network.write"]},
            idem="17b_network",
        )
        row = await _wait_terminal(plane, view["execution_id"], timeout=30)
        assert row is not None
        assert row["state"] == "BLOCKED", row["state"]
        orders = service._store.connection.execute(
            "SELECT COUNT(*) AS c FROM work_orders WHERE mission_id = ?",
            (mission.mission_id,),
        ).fetchone()["c"]
        assert orders == 0, f"启动前 BLOCKED 不得烧 Worker（{orders} 单）"
        await service.close()

    async def test_workspace_effects_task_gets_developer_profile(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """端到端：effects=workspace_only 的任务，WorkOrder 必须带
        worker_profile=developer（不是 scout）。"""
        service, mission = await _setup(tmp_path)
        _enable_fake(
            service, tmp_path, monkeypatch,
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            'echo \'{"kind":"attempt_finished","report":"done"}\'\n',
            "fake-ok",
        )
        plane = service._task_control_plane
        view = await plane.submit(
            mission.mission_id,
            {"goal": "写个脚本跑一下", "effects": "workspace_only",
             "acceptance": {"required_files": []}},
            idem="17b_profile",
        )
        row = await _wait_terminal(plane, view["execution_id"])
        assert row is not None and row["state"] == "SUCCEEDED", row["summary"]
        order = service._worker_manager.order(row["work_order_id"])
        assert order is not None
        assert order.inputs.get("worker_profile") == "developer", (
            f"workspace_only 任务被编译成 {order.inputs.get('worker_profile')}"
        )
        await service.close()


class TestCapabilityEscalation:
    async def test_blocked_escalates_same_execution(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """运行中缺能力（Worker 结构化 BLOCKED + missing capability）→
        控制面升级 profile 恢复同一 session：同一 execution、不新建任务、
        最终 SUCCEEDED（attempt 折叠在 execution 下）。"""
        service, mission = await _setup(tmp_path)
        counter = tmp_path / "esc.count"
        session_file = tmp_path / "esc-session.jsonl"
        session_file.write_text("{}\n")
        _enable_fake(
            service, tmp_path, monkeypatch,
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            f"N=$(cat {counter} 2>/dev/null || echo 0)\n"
            f"echo $((N+1)) > {counter}\n"
            'if [ "$N" -ge 1 ]; then\n'
            "  echo x > deliverable.txt\n"
            '  echo \'{"kind":"attempt_finished","report":"升级后完成"}\'\n'
            "else\n"
            f'  echo \'{{"kind":"session_persisted","session_file":"{session_file}"}}\'\n'
            '  echo \'{"kind":"attempt_finished","report":"只读无法写入'
            "\\nMISSING CAPABILITY: workspace.write"
            '\\nTERMINAL STATUS: BLOCKED"}\'\n'
            "  TERM_DIR=$(dirname \"$0\")/work\n"
            "  for d in \"$TERM_DIR\"/wo_*; do\n"
            '    echo \'{"cause":"BLOCKED","detail":"missing capability: '
            'workspace.write"}\' > "$d/termination.json.tmp"\n'
            '    mv "$d/termination.json.tmp" "$d/termination.json"\n'
            "  done\n"
            "fi\n",
            "fake-escalate",
        )
        plane = service._task_control_plane
        view = await plane.submit(
            mission.mission_id,
            # 只读编译（无 effects 声明 + repo.* → scout）——但任务实际
            # 需要写：worker 诚实 BLOCKED，控制面升级到 developer 恢复
            # 同一 session。
            {"goal": "分析并记录结论到 deliverable.txt",
             "required_capabilities": ["repo.inspect"],
             "deliverables": [{"type": "text/plain", "path": "deliverable.txt"}],
             "acceptance": {"required_files": ["deliverable.txt"]}},
            idem="17b_escalate",
        )
        row = await _wait_terminal(plane, view["execution_id"], timeout=90)
        assert row is not None
        assert row["state"] == "SUCCEEDED", (
            f"升级同 session 后应成功: {row['state']} {row['summary']}"
        )
        executions = plane.executions_for(mission.mission_id)
        assert len(executions) == 1, "能力升级不得裂变 execution"
        await service.close()
