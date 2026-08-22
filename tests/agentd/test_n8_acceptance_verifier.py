"""PR-N8 红测试（调整方案 §七）：Acceptance/Verifier——把"完成任务"
从模型手中真正拿走。

红测试先行——AcceptanceSpecV2/编译器/插件注册表/轨迹验证不存在
时必须红。

1. AcceptanceCompilerV2 按优先级合并：安全底线 + 能力模板 + 用户
   显式 + 任务默认 + 模型建议——模型只能加严，不能放宽；
2. 每个 revision 冻结一份 AcceptanceSpecV2；
3. Verifier 插件注册表：verdict 由插件链产生；
4. 五角星不能只验"有 GIF/帧数/误差阈值"——还必须验：轨迹拓扑是
   五角星、轨迹闭合、正式模型、动力学确实执行、实际轨迹与规划
   一致、GIF/trace/metrics 同一 trace ID。
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


class TestAcceptanceSpecV2Contract:
    def test_round_trip_and_golden(self) -> None:
        from rosclaw.contracts.agent.acceptance import AcceptanceSpecV2

        spec = AcceptanceSpecV2.model_validate_contract({
            "schema_version": "rosclaw.acceptance_spec.v2",
            "spec_id": "acc_1",
            "task_id": "task_1",
            "revision": 1,
            "required_artifacts": ["star.gif"],
            "evidence_classes": ["SIM_DYN_ROLLOUT"],
            "resource_provenance_required": True,
            "numeric_thresholds": {"max_tracking_error_m": 0.01},
            "visual_requirements": ["closed_loop"],
            "postconditions": [],
            "allowed_execution_modes": ["SIMULATION"],
            "required_receipt": "sim_run_receipt",
            "verifier_refs": ["trajectory_tracking_v2"],
            "sources": {"safety_floor": [], "capability_template": [],
                        "user_explicit": [], "task_default": [],
                        "model_suggested": []},
        })
        assert spec.revision == 1
        assert spec.numeric_thresholds["max_tracking_error_m"] == 0.01
        golden = (
            REPO / "tests" / "contracts" / "golden"
            / "rosclaw.acceptance_spec.v2.json"
        )
        current = AcceptanceSpecV2.model_json_schema()
        current["$id"] = "rosclaw://schemas/rosclaw.acceptance_spec.v2"
        current["title"] = "rosclaw.acceptance_spec.v2"
        assert json.loads(golden.read_text(encoding="utf-8")) == current, (
            "acceptance_spec.v2 schema 漂移"
        )


class TestAcceptanceCompiler:
    def test_merge_priority_model_cannot_loosen(self) -> None:
        """模型只能加严：模型建议的宽松阈值/删除项不生效。"""
        from rosclaw.task_kernel.acceptance import compile_acceptance

        spec = compile_acceptance(
            task_id="t1", revision=1,
            safety_floor={
                "numeric_thresholds": {"max_tracking_error_m": 0.05},
                "required_artifacts": ["trace.json"],
            },
            capability_template={
                "numeric_thresholds": {"max_tracking_error_m": 0.02},
            },
            user_explicit={"required_artifacts": ["star.gif"]},
            task_default={"evidence_classes": ["SIM_DYN_ROLLOUT"]},
            model_suggested={
                # 模型想放宽到 0.5（不接受）+ 删掉 trace.json（不接受）
                # + 加严帧数（接受）。
                "numeric_thresholds": {"max_tracking_error_m": 0.5},
                "drop_required_artifacts": ["trace.json"],
                "visual_requirements": ["min_frames_60"],
            },
        )
        # 阈值取最严（min）——模型的 0.5 放宽不生效。
        assert spec.numeric_thresholds["max_tracking_error_m"] == 0.02
        # required_artifacts 并集——模型删除不生效。
        assert set(spec.required_artifacts) == {"trace.json", "star.gif"}
        # 模型加严被接受。
        assert "min_frames_60" in spec.visual_requirements
        assert spec.evidence_classes == ["SIM_DYN_ROLLOUT"]
        # 来源记录在案（可归因）。
        assert "model_suggested" in spec.sources

    def test_frozen_per_revision(self, tmp_path: Path) -> None:
        """kernel.set_acceptance 冻结 spec；revision+1 重编译——
        旧 revision 的 spec 不变。"""
        import sqlite3

        from rosclaw.storage.migrations import MigrationRunner
        from rosclaw.task_kernel.service import TaskKernel

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        MigrationRunner().apply(conn, "sqlite")
        kernel = TaskKernel(conn, tmp_path)
        bound = kernel.bind_message(
            mission_id="mis_1", session_ref="s1", backend_native_id="s1",
            message_id="m1", text="画五角星", cwd=str(tmp_path),
        )
        task_id = str(bound["task_id"])
        kernel.set_acceptance(task_id, {
            "required_artifacts": ["star.gif"],
            "numeric_thresholds": {"max_tracking_error_m": 0.01},
        })
        spec = kernel.get_acceptance_spec(task_id)
        assert spec is not None
        assert spec["revision"] == 1
        assert spec["numeric_thresholds"]["max_tracking_error_m"] == 0.01
        assert spec["spec_id"]
        # 冻结证据：再次读取完全一致（内容 hash）。
        again = kernel.get_acceptance_spec(task_id)
        assert again["spec_id"] == spec["spec_id"]


class TestTrajectoryVerifier:
    """五角星深度验证——用真实 SimTrajectoryService 产物。"""

    def _run_pipeline(self, tmp_path: Path, shape: str = "star5") -> dict:
        from rosclaw.agentd.sim_trajectory import SimTrajectoryService

        sim = SimTrajectoryService(tmp_path)
        plan = sim.generate_planar_path(
            shape=shape, center_m=[0.35, 0.25, 0.30], scale_m=0.05,
        )
        result = sim.simulate_cartesian_trajectory(plan["plan_id"])
        render = sim.render_trace(result["trace_id"])
        verify = sim.verify_tracking(
            result["trace_id"], max_tracking_error_m=0.05,
        )
        return {"plan": plan, "run": result, "render": render,
                "verify": verify, "home": tmp_path}

    def test_genuine_star_passes(self, tmp_path: Path) -> None:
        from rosclaw.task_kernel.verifier_plugins import TrajectoryVerifier

        ctx = self._run_pipeline(tmp_path)
        failures = TrajectoryVerifier().check(
            trace_json=ctx["run"]["artifacts"]["trace_json"],
            metrics_json=ctx["run"]["artifacts"]["metrics_json"],
            gif_path=ctx["render"]["artifact"]["path"],
            home=tmp_path,
            declared_shape="star5",
            max_tracking_error_m=0.05,
        )
        assert failures == [], failures

    def test_circle_for_star_fails(self, tmp_path: Path) -> None:
        """变异：声明五角星但跑的是圆——拓扑验证必须 FAIL。"""
        from rosclaw.task_kernel.verifier_plugins import TrajectoryVerifier

        ctx = self._run_pipeline(tmp_path, shape="circle")
        failures = TrajectoryVerifier().check(
            trace_json=ctx["run"]["artifacts"]["trace_json"],
            metrics_json=ctx["run"]["artifacts"]["metrics_json"],
            gif_path=ctx["render"]["artifact"]["path"],
            home=tmp_path,
            declared_shape="star5",
            max_tracking_error_m=0.05,
        )
        assert any("拓扑" in f or "TOPOLOGY" in f for f in failures), failures

    def test_tampered_trace_linkage_fails(self, tmp_path: Path) -> None:
        """metrics.json 换成别的 trace 的（同一 trace ID 被破坏）→
        FAIL。"""
        from rosclaw.task_kernel.verifier_plugins import TrajectoryVerifier

        ctx = self._run_pipeline(tmp_path)
        # 用不同参数的 run 的 metrics 替换（内容寻址下同参同 id——
        # 拼凑检测必须用不同 trace）。
        other_home = tmp_path / "other"
        self._run_pipeline(other_home, shape="star5")
        # 另一 home 里跑一个不同尺度的 run 得不同 trace_id。
        from rosclaw.agentd.sim_trajectory import SimTrajectoryService

        sim2 = SimTrajectoryService(other_home)
        plan2 = sim2.generate_planar_path(
            shape="star5", center_m=[0.35, 0.25, 0.30], scale_m=0.08,
        )
        run2 = sim2.simulate_cartesian_trajectory(plan2["plan_id"])
        other_metrics = run2["artifacts"]["metrics_json"]
        failures = TrajectoryVerifier().check(
            trace_json=ctx["run"]["artifacts"]["trace_json"],
            metrics_json=other_metrics,
            gif_path=ctx["render"]["artifact"]["path"],
            home=tmp_path,
            declared_shape="star5",
            max_tracking_error_m=0.05,
        )
        assert failures, "跨 trace 拼凑未被拒绝"
        assert any("trace" in f.lower() for f in failures)

    def test_dynamics_not_executed_fails(self, tmp_path: Path) -> None:
        """physics_executed=False（动力学没跑）→ FAIL。"""
        from rosclaw.task_kernel.verifier_plugins import TrajectoryVerifier

        ctx = self._run_pipeline(tmp_path)
        metrics_path = Path(ctx["run"]["artifacts"]["metrics_json"])
        json.loads(metrics_path.read_text(encoding="utf-8"))
        # metrics 不含 physics_executed——它在 run result 里；trace.json
        # 有 evidence_level。破坏证据等级模拟"非动力学"。
        trace_path = Path(ctx["run"]["artifacts"]["trace_json"])
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        trace["evidence_level"] = "COMMAND_REPLAY"
        trace_path.write_text(json.dumps(trace), encoding="utf-8")
        failures = TrajectoryVerifier().check(
            trace_json=str(trace_path),
            metrics_json=str(metrics_path),
            gif_path=ctx["render"]["artifact"]["path"],
            home=tmp_path,
            declared_shape="star5",
            max_tracking_error_m=0.05,
        )
        assert any("动力学" in f or "DYN" in f for f in failures), failures


class TestVerifierRegistry:
    def test_verdict_via_plugin_chain(self, tmp_path: Path) -> None:
        """verdict_for 由插件链产生——插件可注册/替换，输出形状不变
        （既有调用方无感）。"""
        from rosclaw.task_kernel.verifier import verdict_for
        from rosclaw.task_kernel.verifier_plugins import (
            default_registry,
        )

        registry = default_registry()
        names = [p.name for p in registry.plugins]
        for required in (
            "file_artifact", "acceptance_run", "trusted_evidence",
            "fixture_prohibition", "resource_provenance",
        ):
            assert required in names, f"注册表缺 {required}"
        # 同一输入经注册表与直接调用 verdict_for 结果一致。
        artifact = tmp_path / "a.txt"
        artifact.write_text("hello", encoding="utf-8")
        import hashlib

        sha = hashlib.sha256(b"hello").hexdigest()
        direct = verdict_for(
            artifacts=[{"path": str(artifact), "sha256": sha}],
            acceptance={}, workspace=tmp_path, summary="done",
        )
        via_registry = registry.verdict(
            artifacts=[{"path": str(artifact), "sha256": sha}],
            acceptance={}, workspace=tmp_path, summary="done",
        )
        assert via_registry["status"] == direct["status"] == "PASS"
