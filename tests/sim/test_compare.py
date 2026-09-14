"""sim_compare 测试（PR-MH5，规格 §20/§21，红→绿）。

不让 LLM 自己从 10 个 JSON 文件里"肉眼比较"。
"""

from __future__ import annotations


def _experiment_at_kp(backend, ref, kp: float):
    patched = backend.patch_model(
        ref.model_ref,
        [
            {
                "op": "set",
                "target": {"type": "actuator", "name": "shoulder_servo"},
                "field": "kp",
                "value": kp,
            },
            {
                "op": "set",
                "target": {"type": "actuator", "name": "elbow_servo"},
                "field": "kp",
                "value": kp,
            },
        ],
    )
    return backend.run_experiment(
        patched.new_model_ref, controller={"position_targets": [0.4, 0.2]}, duration_s=1.0
    )


def test_compare_metric_table_and_best(loaded_backend) -> None:
    backend, ref = loaded_backend
    weak = _experiment_at_kp(backend, ref, 10.0)
    strong = _experiment_at_kp(backend, ref, 400.0)

    result = backend.compare_experiments([weak.receipt_ref, strong.receipt_ref])
    assert result.schema_version == "rosclaw.sim.comparison_result.v1"
    assert len(result.metric_table) == 2
    rows = {row["receipt_ref"]: row for row in result.metric_table}
    assert weak.receipt_ref in rows and strong.receipt_ref in rows
    # 强 kp 跟踪误差显著小于弱 kp（1s 窗口物理实测比值 ~0.69，
    # 阈值取 0.8 留物理余量——不编造 0.5 之类的漂亮数字）。
    assert rows[strong.receipt_ref]["tracking_rmse"] < rows[weak.receipt_ref]["tracking_rmse"] * 0.8
    assert result.best_ref == strong.receipt_ref
    assert result.pareto_refs  # Pareto 候选非空


def test_compare_pareto_candidates(loaded_backend) -> None:
    backend, ref = loaded_backend
    a = _experiment_at_kp(backend, ref, 10.0)
    b = _experiment_at_kp(backend, ref, 100.0)
    c = _experiment_at_kp(backend, ref, 400.0)
    result = backend.compare_experiments([a.receipt_ref, b.receipt_ref, c.receipt_ref])
    # Pareto 候选 ⊆ 输入；best ∈ Pareto。
    assert set(result.pareto_refs) <= {a.receipt_ref, b.receipt_ref, c.receipt_ref}
    assert result.best_ref in result.pareto_refs


def test_compare_requires_two_refs(loaded_backend) -> None:
    backend, ref = loaded_backend
    receipt = backend.run_experiment(ref.model_ref, controller={"hold": True}, steps=10)
    import pytest

    with pytest.raises(ValueError, match="COMPARE_REFS_REQUIRED"):
        backend.compare_experiments([receipt.receipt_ref])
