"""SimulationReceipt 测试（PR-MH5，ADR-0014，规格 §30/§31，红→绿）。

仿真证据永远 SIMULATED、usable_for_real_execution=false。
"""

from __future__ import annotations


def test_run_experiment_produces_receipt(loaded_backend) -> None:
    backend, ref = loaded_backend
    receipt = backend.run_experiment(
        ref.model_ref, controller={"position_targets": [0.3, 0.1]}, duration_s=0.2
    )

    assert receipt.schema_version == "rosclaw.sim.receipt.v1"
    assert receipt.backend == "mujoco"
    assert receipt.backend_version
    assert receipt.model_ref == ref.model_ref
    assert receipt.model_digest == ref.model_digest
    assert receipt.initial_state_ref.startswith("simsta_")
    assert receipt.action_digest.startswith("simact_")
    assert receipt.trace_ref.startswith("simtrc_")
    assert receipt.steps == 100
    assert receipt.simulation_time_s > 0
    assert receipt.trust_level == "SIMULATED"
    assert receipt.usable_for_real_execution is False
    assert receipt.metrics["tracking_rmse"] >= 0
    assert receipt.audit_ref  # 实验附带审计


def test_receipt_stored_idempotent(loaded_backend) -> None:
    backend, ref = loaded_backend
    r1 = backend.run_experiment(ref.model_ref, controller={"hold": True}, duration_s=0.05)
    r2 = backend.run_experiment(ref.model_ref, controller={"hold": True}, duration_s=0.05)
    assert r1.receipt_ref == r2.receipt_ref
    record = backend.store.get(r1.receipt_ref)
    assert record["kind"] == "simulation_receipt"
    assert record["trust_level"] == "SIMULATED"
    assert record["usable_for_real_execution"] is False
    assert record["semantic_digest"].startswith("simrcp_")


def test_experiment_metrics_computed(loaded_backend) -> None:
    backend, ref = loaded_backend
    receipt = backend.run_experiment(
        ref.model_ref, controller={"position_targets": [0.4, 0.0]}, duration_s=0.5
    )
    metrics = receipt.metrics
    assert metrics["tracking_rmse"] > 0
    assert metrics["peak_qvel"] > 0
    assert "overshoot" in metrics and "settling_time_s" in metrics
    assert metrics["energy_end"] != 0
    assert metrics["collision"] is False  # tiny_arm 无接触
    assert metrics["contact_max_penetration"] == 0.0


def test_experiment_success_semantics_split(loaded_backend) -> None:
    """0915 §三：audit PASS ≠ 任务成功——字段语义必须分开。"""
    backend, ref = loaded_backend
    receipt = backend.run_experiment(ref.model_ref, controller={"hold": True}, duration_s=0.05)
    assert receipt.physical_audit_pass is True  # tiny_arm strict audit PASS
    assert receipt.simulation_valid is True
    assert receipt.task_success is None  # 未给谓词 = 未评估
    assert receipt.verification_status == "NOT_EVALUATED"
    assert receipt.success is None  # 兼容字段 ≡ task_success


def test_experiment_task_predicates_machine_verdict(loaded_backend) -> None:
    """任务谓词机器判定：不诚实的 True/False 都由谓词决定。"""
    backend, ref = loaded_backend
    predicates = [
        {
            "channel": "site_pose:tool0",
            "field": "pos",
            "inside": {"min": [-1, -1, 0.6], "max": [1, 1, 0.8]},
        }
    ]
    passing = backend.run_experiment(
        ref.model_ref, controller={"hold": True}, duration_s=0.05, task_predicates=predicates
    )
    assert passing.task_success is True
    assert passing.verification_status == "PASS"
    assert passing.success is True  # 兼容字段

    failing_predicates = [
        {
            "channel": "site_pose:tool0",
            "field": "pos",
            "inside": {"min": [5, 5, 5], "max": [6, 6, 6]},
        }
    ]
    failing = backend.run_experiment(
        ref.model_ref,
        controller={"hold": True},
        duration_s=0.05,
        task_predicates=failing_predicates,
    )
    assert failing.task_success is False
    assert failing.verification_status == "FAIL"
    assert failing.success is False
    # 物理依然健康——物理与任务两分。
    assert failing.physical_audit_pass is True
