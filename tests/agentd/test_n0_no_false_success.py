"""PR-N0 红测试（总纲 N 方案 §15/§17 第一条指令）：事故固化与假成功熔断。

以 2026-08-20 UR5e 五角星事故为 P0：错误简化资产被 verifier 宣布成功、
task_finish 可临时修改 acceptance、artifact cwd 分裂、用户否定后旧成功
仍有效——四个问题必须先红后绿，绝不降低验收标准。

红测试清单（修复前必须全红）：
1. 机器人行为任务只有模型自产 artifact（无受信管道证据）→ 不得
   SUCCEEDED；
2. task_finish 不得携带 acceptance（验收在任务创建时冻结）；
3. artifact 相对路径只按任务 workspace 根解析（不存在时打印实际
   解析根）；
4. SUCCEEDED 未获用户接受（/done）时，用户修正消息 → 同一任务新
   revision + 旧 verification 立即作废；接受后新消息 → 新任务；
5. 同一 verifier 输入重复运行结果完全一致（确定性）。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from rosclaw.task_kernel.service import TaskKernel


def _real_ur5e_resource() -> dict:
    """真实权威 manifest 的资源证明（对照组用真值——不是编数字）。"""
    from rosclaw.cognition.resolver import resolve_resource

    repo = Path(__file__).resolve().parents[2]
    manifest = resolve_resource("robot", "ur5e", product_root=repo)
    assert manifest is not None
    return {
        "resource_id": "robot:ur5e",
        "manifest_digest": manifest["digests"].get("profile", ""),
        "model_path": manifest["paths"]["mjcf"],
        "model_digest": manifest["digests"]["mjcf"],
        "quality": "PRODUCTION",
        "canonical": True,
    }


def _kernel(tmp_path: Path) -> TaskKernel:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    from rosclaw.storage.migrations import MigrationRunner

    MigrationRunner().apply(conn, "sqlite")
    return TaskKernel(conn, tmp_path)


def _bind(kernel: TaskKernel, tmp_path: Path, *, text: str = "画五角星",
          body_id: str = "sim/ur5e", message_id: str = "m1") -> str:
    bound = kernel.bind_message(
        mission_id="mis_1", session_ref="s1", backend_native_id="s1",
        message_id=message_id, text=text, cwd=str(tmp_path),
        mode="SIMULATION", body_id=body_id,
    )
    return str(bound["task_id"])


def _write_artifact(kernel: TaskKernel, task_id: str, name: str,
                    content: bytes) -> dict:
    """经模型工具路径登记（producer 不可信——模拟事故中的手写脚本产物）。"""
    task = kernel.get_task(task_id)
    assert task is not None
    path = Path(task["workspace_path"]) / name
    path.write_bytes(content)
    return kernel.register_artifact(
        task_id=task_id, path=str(path), media_type="image/gif",
        producer="model:rosclaw_artifact_register",
    )


class TestNoFalseSuccess:
    def test_model_produced_only_artifacts_cannot_succeed_robot_task(
        self, tmp_path: Path
    ) -> None:
        """事故核心：全部产物由模型手写脚本产出（producer 不可信）——
        机器人行为任务不得 SUCCEEDED（需要受信管道的独立验证证据）。"""
        kernel = _kernel(tmp_path)
        task_id = _bind(kernel, tmp_path)
        # 行为任务=实际调用具身执行工具（body 在场只是绑定事实）。
        kernel.note_tool_use(task_id, "rosclaw_task")
        art = _write_artifact(kernel, task_id, "star.gif", b"GIF89a" + b"x" * 2048)
        result = kernel.finish_task(
            task_id=task_id, summary="五角星画好了", artifact_ids=[art["artifact_id"]],
        )
        assert result["status"] != "SUCCEEDED", (
            "模型自产证据竟让机器人行为任务 SUCCEEDED——幽灵成功"
        )
        task = kernel.get_task(task_id)
        assert task is not None and task["state"] != "SUCCEEDED"

    def test_trusted_pipeline_evidence_can_succeed(self, tmp_path: Path) -> None:
        """对照组：受信管道产物（kernel 内部登记）→ 可以 SUCCEEDED。"""
        kernel = _kernel(tmp_path)
        task_id = _bind(kernel, tmp_path)
        kernel.note_tool_use(task_id, "rosclaw_task")
        task = kernel.get_task(task_id)
        assert task is not None
        path = Path(task["workspace_path"]) / "star.gif"
        path.write_bytes(b"GIF89a" + b"x" * 2048)
        # WP-4：产品路径的受信媒体带 preview 血缘（trace 引用）——
        # 无血缘的受信媒体声明已不被接受（LINEAGE_MISSING）。
        trace_dir = tmp_path / "sim" / "traces" / "trace_test1"
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / "trace.json").write_text("{}", encoding="utf-8")
        art = kernel.register_artifact(
            task_id=task_id, path=str(path), media_type="image/gif",
            producer="kernel:sim_pipeline",
            metadata={
                "resource": _real_ur5e_resource(),
                "lineage": {"trace_id": "trace_test1", "kind": "preview_2d"},
            },
        )
        result = kernel.finish_task(
            task_id=task_id, summary="done", artifact_ids=[art["artifact_id"]],
        )
        assert result["status"] == "SUCCEEDED", result

    def test_finish_rejects_acceptance_param(self, tmp_path: Path) -> None:
        """task_finish 删除 acceptance 参数——验收在任务创建时冻结，
        做题人不得临时修改评分标准。"""
        import inspect

        from rosclaw.task_kernel.service import TaskKernel as K

        sig = inspect.signature(K.finish_task)
        assert "acceptance" not in sig.parameters, (
            "finish_task 仍接受 acceptance——模型可在收尾时自定义验收"
        )

    def test_finish_reads_frozen_creation_acceptance(self, tmp_path: Path) -> None:
        """验收条件在 bind 时冻结——finish 不接受新规则，也不丢创建时
        的规则（required_files 缺失必须 FAIL）。"""
        kernel = _kernel(tmp_path)
        task_id = _bind(kernel, tmp_path, body_id="")
        # 创建时冻结的验收：必须交付 report.txt。
        kernel.set_acceptance(task_id, {"required_files": ["report.txt"]})
        task = kernel.get_task(task_id)
        assert task is not None
        path = Path(task["workspace_path"]) / "other.txt"
        path.write_text("irrelevant", encoding="utf-8")
        art = kernel.register_artifact(
            task_id=task_id, path=str(path), media_type="text/plain",
        )
        result = kernel.finish_task(
            task_id=task_id, summary="done", artifact_ids=[art["artifact_id"]],
        )
        assert result["status"] != "SUCCEEDED", "创建时冻结的验收被忽略"
        # 补齐后同 revision 可通过。
        (Path(task["workspace_path"]) / "report.txt").write_text(
            "ok", encoding="utf-8"
        )
        result2 = kernel.finish_task(
            task_id=task_id, summary="done", artifact_ids=[art["artifact_id"]],
        )
        assert result2["status"] == "SUCCEEDED", result2

    def test_artifact_relative_path_single_root(self, tmp_path: Path) -> None:
        """相对路径只按任务 workspace 根解析（禁止按 session cwd 猜）；
        不存在时报错必须含实际解析根。"""
        kernel = _kernel(tmp_path)
        task_id = _bind(kernel, tmp_path)
        task = kernel.get_task(task_id)
        assert task is not None
        ws = Path(task["workspace_path"])
        (ws / "out.txt").write_text("ok", encoding="utf-8")
        art = kernel.register_artifact(
            task_id=task_id, path="out.txt", media_type="text/plain",
        )
        assert Path(art["path"]).resolve() == (ws / "out.txt").resolve()
        try:
            kernel.register_artifact(
                task_id=task_id, path="missing.txt", media_type="text/plain",
            )
            raise AssertionError("不存在的 artifact 竟登记成功")
        except ValueError as exc:
            assert str(ws) in str(exc), f"错误未含解析根: {exc}"

    def test_user_rejection_reopens_revision_and_invalidates_receipt(
        self, tmp_path: Path
    ) -> None:
        """SUCCEEDED 未获用户接受（/done）时，用户修正消息 → 同一任务
        revision+1 + 状态回 RUNNING + 旧 verification 立即作废。"""
        kernel = _kernel(tmp_path)
        task_id = _bind(kernel, tmp_path, body_id="")
        task = kernel.get_task(task_id)
        assert task is not None
        path = Path(task["workspace_path"]) / "report.txt"
        path.write_text("v1", encoding="utf-8")
        art = kernel.register_artifact(
            task_id=task_id, path=str(path), media_type="text/plain",
        )
        result = kernel.finish_task(
            task_id=task_id, summary="done", artifact_ids=[art["artifact_id"]],
        )
        assert result["status"] == "SUCCEEDED"
        verification_id = result["verification_id"]
        # 用户否定（未经 /done 接受）→ revision 2，旧 receipt 作废。
        bound = kernel.bind_message(
            mission_id="mis_1", session_ref="s1", backend_native_id="s1",
            message_id="m2", text="不对，画错了，重来", cwd=str(tmp_path),
        )
        assert bound["task_id"] == task_id, "否定竟裂变成新任务"
        assert bound["revision"] == 2
        task2 = kernel.get_task(task_id)
        assert task2 is not None and task2["state"] not in ("SUCCEEDED",), (
            f"用户否定后任务仍宣称成功: {task2['state']}"
        )
        row = kernel._conn.execute(
            "SELECT status FROM verifications WHERE verification_id = ?",
            (verification_id,),
        ).fetchone()
        assert row is not None and row["status"] != "PASS", (
            "旧 verification 在用户否定后仍有效"
        )

    def test_accepted_task_stays_closed(self, tmp_path: Path) -> None:
        """/done 接受后：新消息开新任务（不污染已接受结果）。"""
        kernel = _kernel(tmp_path)
        task_id = _bind(kernel, tmp_path, body_id="")
        task = kernel.get_task(task_id)
        assert task is not None
        path = Path(task["workspace_path"]) / "report.txt"
        path.write_text("v1", encoding="utf-8")
        art = kernel.register_artifact(
            task_id=task_id, path=str(path), media_type="text/plain",
        )
        result = kernel.finish_task(
            task_id=task_id, summary="done", artifact_ids=[art["artifact_id"]],
        )
        assert result["status"] == "SUCCEEDED"
        kernel.accept_task(task_id)  # /done
        bound = kernel.bind_message(
            mission_id="mis_1", session_ref="s1", backend_native_id="s1",
            message_id="m2", text="新目标", cwd=str(tmp_path),
        )
        assert bound["task_id"] != task_id, "已接受任务被新消息污染"
        task1 = kernel.get_task(task_id)
        assert task1 is not None and task1["state"] == "SUCCEEDED"

    def test_identical_input_deterministic_verdict(self, tmp_path: Path) -> None:
        """同一输入重复验证，结果完全一致（验收不得有随机性）。"""
        kernel = _kernel(tmp_path)
        task_id = _bind(kernel, tmp_path, body_id="")
        task = kernel.get_task(task_id)
        assert task is not None
        path = Path(task["workspace_path"]) / "report.txt"
        path.write_text("v1", encoding="utf-8")
        art = kernel.register_artifact(
            task_id=task_id, path=str(path), media_type="text/plain",
        )
        r1 = kernel.finish_task(
            task_id=task_id, summary="done", artifact_ids=[art["artifact_id"]],
        )
        r2 = kernel.finish_task(
            task_id=task_id, summary="done", artifact_ids=[art["artifact_id"]],
        )
        assert r1["status"] == r2["status"] == "SUCCEEDED"
        assert r2.get("already_terminal"), "终态重放必须幂等"
        assert r2["verification_id"] == r1["verification_id"], (
            "重复验证竟产生新 receipt"
        )


class TestFuseArmed:
    def test_bind_without_body_falls_back_to_mission_body(
        self, tmp_path: Path
    ) -> None:
        """bridge 回落：bind 不带 body_id 时用 mission 绑定 body——
        熔断在 chat 路径首条消息即武装（不自审前的惰性失效）。"""
        import asyncio

        from tests.agentd.test_pi_tool_bridge import _setup  # noqa: F401

        async def _run() -> None:
            service, mission = await _setup(tmp_path)
            from rosclaw.agentd.pi_bridge.server import PiBridgeServer

            bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
            result = await bridge._dispatch(
                "user:local:1000", 1, "pi.task.bind",
                {
                    "token": service.control_token,
                    "mission_id": mission.mission_id,
                    "session_ref": "pi_1", "backend_native_id": "pi_1",
                    "message_id": "msg_fuse", "text": "画五角星",
                    "cwd": str(tmp_path),
                },
            )
            assert result.get("ok"), result
            task = service._task_kernel.get_task(result["task_id"])
            assert task is not None
            assert task["body_id"], (
                "bind 未带 body_id 且未回落 mission body——熔断惰性失效"
            )
            await service.close()

        asyncio.run(_run())
