"""0902 合并批对抗性复核的修复红测试（每发现一条闭环）。

- H1：project_deliverables 只投影当前 revision（旧 revision 产物
  不得进新 revision 的 outputs/——0902 事故的交付面复现）；
- M6：finish_task 的受信证据门（capability_touched/lineage）按
  当前 revision 过滤（rev1 产物不得满足 rev2 的 authority 门）；
- M7：allow_once 批准后首个读取者消费（并发等待者不得都放行）；
- M3：rosclaw_task 路径（模型面）过同样的覆盖率门禁——未覆盖
  条款拒绝执行（不是烧完资源才验收失败）；
- M4：形状词表单源——task_router 与 requirements 共享同一注册
  表；裸"圆"不命中"圆桌/圆珠笔"；画类指令无可识别形状 → 不自动
  路由（fail-closed——"画个心形"画成五角星 = 假成功）。
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _kernel(tmp_path: Path):
    import sqlite3

    from rosclaw.storage.migrations import MigrationRunner
    from rosclaw.task_kernel.service import TaskKernel

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = __import__("sqlite3").Row
    MigrationRunner().apply(conn, "sqlite")
    return TaskKernel(conn, tmp_path)


class TestH1ProjectionRevisionScope:
    def test_old_revision_artifacts_not_projected(self, tmp_path: Path) -> None:
        from rosclaw.task_kernel.projection import project_deliverables

        kernel = _kernel(tmp_path)
        bound = kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="s1",
            message_id="msg_1", text="画五角星", cwd=str(tmp_path),
        )
        task_id = str(bound["task_id"])
        # rev1 产物
        old = tmp_path / "old-scene.mp4"
        old.write_bytes(b"old")
        kernel.register_artifact(
            task_id=task_id, path=str(old), media_type="video/mp4",
            producer="kernel:test",
        )
        # 开 rev2
        kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="s1",
            message_id="msg_2", text="改成画圆", cwd=str(tmp_path),
        )
        new = tmp_path / "new-scene.mp4"
        new.write_bytes(b"new-content")
        kernel.register_artifact(
            task_id=task_id, path=str(new), media_type="video/mp4",
            producer="kernel:test",
        )
        assert project_deliverables(kernel, task_id) == "OK"
        outputs = tmp_path / "runs" / task_id / "r2" / "outputs"
        names = {p.name for p in outputs.iterdir()}
        assert "new-scene.mp4" in names
        assert "old-scene.mp4" not in names, (
            "rev1 旧产物投影进了 rev2 交付区——0902 事故交付面复现"
        )


class TestM6AuthorityGateRevisionScope:
    """受信证据门按 revision 过滤——rev1 的 kernel 产物不得满足
    rev2 的 PLAN_AUTHORITY_MISSING 门。"""

    def test_authority_gate_ignores_old_revision(self, tmp_path: Path) -> None:
        kernel = _kernel(tmp_path)
        bound = kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="s1",
            message_id="msg_1", text="画五角星", cwd=str(tmp_path),
        )
        task_id = str(bound["task_id"])
        # 具身化（M6 测的是具身任务的 authority 门）。
        kernel._conn.execute(
            "UPDATE tasks SET body_id = 'sim/ur5e' WHERE task_id = ?",
            (task_id,),
        )
        # rev1 有带血缘的 kernel 受信产物（rend●r receipt digest——
        # 无 revision 过滤时它会让 authority 门在 rev2 误判受信）。
        ev = tmp_path / "evidence-rev1.json"
        ev.write_text("{}")
        kernel.register_artifact(
            task_id=task_id, path=str(ev), media_type="application/json",
            producer="kernel:capability:ur5e.move",
            metadata={"lineage": {"kind": "preview_2d",
                                  "trace_id": "trace_rev1"}},
        )
        # 开 rev2——rev2 有具身痕迹（tool_used）但零受信证据。
        kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="s1",
            message_id="msg_2", text="改一下", cwd=str(tmp_path),
        )
        kernel.note_tool_use(task_id, "rosclaw_execute")
        # rev2 直接 finish：受信证据门必须认不出 rev1 的产物——
        # PLAN_AUTHORITY_MISSING（不是被旧证据满足的假 PASS）。
        verdict = kernel.finish_task(task_id=task_id, summary="t", artifact_ids=[])
        assert verdict.get("status") != "SUCCEEDED", verdict
        assert any(
            "PLAN_AUTHORITY_MISSING" in str(f) or "AUTHORITY" in str(f).upper()
            for f in verdict.get("failures", [])
        ), verdict.get("failures")


class TestM7AllowOnceConsumed:
    def test_allow_once_single_consumer(self, tmp_path: Path) -> None:
        from rosclaw.agentd.shell_gate import ShellGateBroker

        kernel = _kernel(tmp_path)
        bound = kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="s1",
            message_id="msg_1", text="写文件", cwd=str(tmp_path),
        )
        task_id = str(bound["task_id"])
        broker = ShellGateBroker(kernel)
        req = broker.request(
            task_id=task_id, revision=1, mission_id="m1", session_ref="s1",
            scope="shell.unsandboxed",
        )
        broker.decide(req["request_id"], "allow_once")
        # 第一个读取者拿到 APPROVED_ONCE——消费掉。
        assert broker.status(req["request_id"])["status"] == "APPROVED_ONCE"
        # 第二个并发等待者：不得也看到批准（"允许一次"≠"所有在飞者"）。
        second = broker.status(req["request_id"])["status"]
        assert second != "APPROVED_ONCE", f"allow_once 被消费两次: {second}"

class TestM3ModelPathCoverageGate:
    """rosclaw_task（模型面/直接执行）调 recipe 必须过同一覆盖率门禁
    ——未覆盖条款拒绝执行（不是烧完资源才验收失败）。"""

    def test_uncovered_requirement_blocks_recipe_execution(
        self, tmp_path: Path
    ) -> None:
        import sqlite3

        from rosclaw.agentd.task_execution import TaskExecutionService
        from rosclaw.storage.migrations import MigrationRunner
        from rosclaw.task_kernel.service import TaskKernel

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        MigrationRunner().apply(conn, "sqlite")
        kernel = TaskKernel(conn, tmp_path)
        kernel.persist_input(
            mission_id="m1", session_ref="s1",
            message_id="msg_1", text="画一个三角形",
        )
        bound = kernel.ensure_task_for_effect(
            mission_id="m1", session_ref="s1", backend_native_id="s1",
            cwd=str(tmp_path),
        )
        task_id = str(bound["task_id"])
        outcome = TaskExecutionService(
            kernel=kernel, conn=conn, home=tmp_path,
        ).execute(task_id, goal_hint="draw_shape")
        assert not outcome.ok, "三角形不在覆盖表竟执行了 recipe"
        assert "COVERAGE" in str(outcome.error_code), outcome.error_code
        # 零幽灵执行：无 plan/trace 产物落盘。
        assert not list((tmp_path / "sim").glob("**/*.json")), "未覆盖竟执行了 sim 链"


class TestM4ShapeRegistrySingleSource:
    def test_bare_yuan_matches_nothing(self) -> None:
        """裸"圆"不得命中圆桌/圆珠笔（双源漂移根治）。"""
        from rosclaw.task_kernel.task_router import compile_recipe_inputs

        assert "shape" not in compile_recipe_inputs("绕着圆桌画轨迹")
        assert "shape" not in compile_recipe_inputs("把圆珠笔递给我")
        assert compile_recipe_inputs("画一个圆形").get("shape") == "circle"

    def test_draw_directive_without_known_shape_not_auto_routed(self) -> None:
        """画类指令无可识别形状 → fail-closed（画个心形画成五角星
        = 假成功）。编译层先钉住：心形不产形状条款。"""
        from rosclaw.task_kernel.requirements import compile_requirements
        from rosclaw.task_kernel.task_router import RECIPE_COVERAGE

        reqs = compile_requirements("让机械臂画个心形")
        assert not any(r.verifier.startswith("shape.") for r in reqs)
        coverage = RECIPE_COVERAGE.get("recipe:sim.draw_path", frozenset())
        assert "shape.heart" not in coverage


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
