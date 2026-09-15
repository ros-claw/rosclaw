"""Rollout 引擎测试（PR-MH3，ADR-0014，规格 §15，红→绿）。"""

from __future__ import annotations

import pytest


def test_rollout_hold_produces_trace(loaded_backend) -> None:
    backend, ref = loaded_backend
    trace = backend.rollout(ref.model_ref, controller={"hold": True}, duration_s=0.1)

    assert trace.model_ref == ref.model_ref
    assert trace.model_digest == ref.model_digest
    assert trace.steps == 50  # 0.1s / 0.002s
    assert trace.timestep_s == pytest.approx(0.002)
    assert trace.states_digest.startswith("sha256:")
    assert trace.trace_ref.startswith("simtrc_")
    assert trace.final_state_ref.startswith("simsta_")

    record = backend.store.get(trace.trace_ref)
    assert record["kind"] == "simulation_trace"
    assert record["controller"] == {"hold": True}
    assert record["seed"] == 0
    assert len(record["states"]) >= 2
    first = record["states"][0]
    assert set(first) >= {"t", "qpos", "qvel", "ctrl"}


def test_rollout_deterministic(loaded_backend) -> None:
    backend, ref = loaded_backend
    t1 = backend.rollout(ref.model_ref, controller={"hold": True}, duration_s=0.05, seed=7)
    t2 = backend.rollout(ref.model_ref, controller={"hold": True}, duration_s=0.05, seed=7)
    assert t1.trace_ref == t2.trace_ref
    assert t1.states_digest == t2.states_digest


def test_rollout_ctrl_series_moves_joints(loaded_backend) -> None:
    backend, ref = loaded_backend
    series = [[0.5, 0.0]] * 100
    trace = backend.rollout(ref.model_ref, controller={"ctrl_series": series})
    assert trace.steps == 100
    final = backend.store.get(trace.final_state_ref)
    assert final["qpos"][0] > 0.05  # shoulder 向 0.5 目标运动
    # hold 基线不动。
    hold = backend.rollout(ref.model_ref, controller={"hold": True}, steps=100)
    hold_final = backend.store.get(hold.final_state_ref)
    assert hold_final["qpos"][0] == pytest.approx(0.0, abs=1e-6)


def test_rollout_position_targets(loaded_backend) -> None:
    backend, ref = loaded_backend
    trace = backend.rollout(
        ref.model_ref, controller={"position_targets": [0.4, 0.1]}, duration_s=0.2
    )
    final = backend.store.get(trace.final_state_ref)
    assert final["ctrl"] == [0.4, 0.1]
    assert final["qpos"][0] > 0.05


def test_rollout_controller_invalid(loaded_backend) -> None:
    backend, ref = loaded_backend
    with pytest.raises(ValueError, match="CONTROLLER_INVALID"):
        backend.rollout(ref.model_ref, controller={"magic": True})
    with pytest.raises(ValueError, match="CONTROLLER_INVALID"):
        backend.rollout(ref.model_ref, controller={"ctrl_series": [[0.1]]})  # nu=2


def test_rollout_budget_enforced(loaded_backend) -> None:
    backend, ref = loaded_backend
    with pytest.raises(ValueError, match="SIM_BUDGET_EXCEEDED"):
        backend.rollout(ref.model_ref, controller={"hold": True}, steps=10_000_000)
    with pytest.raises(ValueError, match="SIM_BUDGET_EXCEEDED"):
        backend.rollout(ref.model_ref, controller={"hold": True}, duration_s=10_000.0)


def test_rollout_record_points_bounded(loaded_backend) -> None:
    backend, ref = loaded_backend
    trace = backend.rollout(ref.model_ref, controller={"hold": True}, steps=2000)
    record = backend.store.get(trace.trace_ref)
    assert len(record["states"]) <= 480  # max_record_points 有界


def test_rollout_from_state_ref(loaded_backend) -> None:
    backend, ref = loaded_backend
    t1 = backend.rollout(ref.model_ref, controller={"position_targets": [0.3, 0.0]}, duration_s=0.1)
    # 从 t1 终点继续 rollout（snapshot → 继续实验链）。
    t2 = backend.rollout(
        ref.model_ref,
        state_ref=t1.final_state_ref,
        controller={"hold": True},
        duration_s=0.05,
    )
    assert t2.model_ref == ref.model_ref
    start = backend.store.get(t2.trace_ref)["states"][0]
    assert start["qpos"][0] == pytest.approx(
        backend.store.get(t1.final_state_ref)["qpos"][0], abs=1e-9
    )


def test_rollout_cross_model_state_rejected(loaded_backend) -> None:
    backend, ref = loaded_backend
    other = backend.patch_model(
        ref.model_ref,
        [
            {
                "op": "set",
                "target": {"type": "joint", "name": "shoulder"},
                "field": "damping",
                "value": 2.0,
            }
        ],
    )
    state_ref = backend.initial_state(other.new_model_ref)
    with pytest.raises(ValueError, match="CROSS_MODEL_REF"):
        backend.rollout(ref.model_ref, state_ref=state_ref, controller={"hold": True}, steps=10)
