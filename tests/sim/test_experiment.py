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


def test_experiment_success_from_audit(loaded_backend) -> None:
    backend, ref = loaded_backend
    receipt = backend.run_experiment(ref.model_ref, controller={"hold": True}, duration_s=0.05)
    assert receipt.success is True  # tiny_arm strict audit PASS
