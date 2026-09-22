"""GPU Semantic Agreement 测试（MH24，讨论总纲 §34-§38，红→绿）。

MH18 的 final-qpos 比对太弱：起点终点一致但中间碰撞/力/谓词
可能全不同。升级全语义：final state + 轨迹检查点 + task
success + collision events + peak contact force + tracking
RMSE + audit verdict。双态 SEMANTIC_AGREEMENT/DIVERGENCE；
分歧按类型入 counterexample 语料（§38——不扔）。

GPU live 诚实 NOT_RUN（本机无 jax-cuda/aarch64 与 mjx/warp——
门逻辑以 CPU 产出的"GPU-like"候选数据验证，真实 GPU 数据
接入即同一判据）。
"""

from __future__ import annotations

import pytest

TINY_ARM = """<mujoco model="tiny_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="0.5"/>
      <geom name="base_geom" type="capsule" size="0.05 0.2" mass="1.0"/>
      <body name="forearm" pos="0 0 0.4">
        <joint name="elbow" type="hinge" axis="0 1 0" damping="0.1"/>
        <geom name="forearm_geom" type="capsule" size="0.03 0.2" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_servo" joint="shoulder" kp="10" ctrlrange="-1.57 1.57"/>
    <position name="elbow_servo" joint="elbow" kp="5"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def backend(tmp_path):
    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    (tmp_path / "arm.xml").write_text(TINY_ARM, encoding="utf-8")
    return MujocoBackend(tmp_path)


def _gpu_like_candidate(backend, model_ref: str, *, tampered: bool = False) -> dict:
    """CPU 产出的"GPU 候选结果"（门逻辑验证数据）。"""
    receipt = backend.rollout(model_ref, controller={"position_targets": [0.4, 0.2]}, steps=100)
    trace = backend.store.get(receipt.trace_ref)
    final_qpos = trace["states"][-1]["qpos"]
    final_qvel = trace["states"][-1]["qvel"]
    if tampered:
        final_qvel = [v + 0.01 for v in final_qvel]  # 速度场分歧
    checkpoints = [
        {"step": 50, "qpos": trace["states"][50]["qpos"] if len(trace["states"]) > 50 else final_qpos},
    ]
    return {
        "name": "cand",
        "patches": [],
        "gpu_final_qpos": final_qpos,
        "gpu_final_qvel": final_qvel,
        "gpu_checkpoints": checkpoints,
        "gpu_controller": {"position_targets": [0.4, 0.2]},
        "gpu_steps": 100,
        "gpu_task_success": True,
        "gpu_peak_contact_force": 0.0,
    }


def test_semantic_agreement_full_pass(backend) -> None:
    """全字段一致 → SEMANTIC_AGREEMENT → promoted。"""
    from rosclaw.sim.acceleration import evaluate_gpu_candidates_semantic

    ref = backend.load_model("arm.xml")
    candidate = _gpu_like_candidate(backend, ref.model_ref)
    verdict = evaluate_gpu_candidates_semantic(backend, ref.model_ref, [candidate])
    assert verdict["promoted"] == ["cand"]
    assert verdict["agreements"]["cand"]["status"] == "SEMANTIC_AGREEMENT"


def test_semantic_divergence_on_qvel(backend) -> None:
    """qvel 分歧（终点 qpos 一致但速度场不同）→ DIVERGENCE +
    counterexample 落库（不扔）。"""
    from rosclaw.sim.acceleration import evaluate_gpu_candidates_semantic

    ref = backend.load_model("arm.xml")
    candidate = _gpu_like_candidate(backend, ref.model_ref, tampered=True)
    verdict = evaluate_gpu_candidates_semantic(backend, ref.model_ref, [candidate])
    assert verdict["promoted"] == []
    agreement = verdict["agreements"]["cand"]
    assert agreement["status"] == "SEMANTIC_DIVERGENCE"
    assert "qvel" in str(agreement["diverged_fields"])
    stored = backend.store.get(verdict["counterexamples"][0]["counterexample_ref"])
    assert stored["kind"] == "gpu_cpu_counterexample"


def test_semantic_divergence_on_task_success(backend) -> None:
    """task_success 不一致即分歧（物理语义不同，不只状态量）。"""
    from rosclaw.sim.acceleration import evaluate_gpu_candidates_semantic

    ref = backend.load_model("arm.xml")
    candidate = _gpu_like_candidate(backend, ref.model_ref)
    # CPU 侧谓词判定为 True（shoulder 在容差内），GPU 声称 False → 分歧。
    candidate["task_predicates"] = [
        {"channel": "joint_positions", "joint_in_range": {"index": 0, "min": -1.0, "max": 1.0}}
    ]
    candidate["gpu_task_success"] = False
    verdict = evaluate_gpu_candidates_semantic(backend, ref.model_ref, [candidate])
    assert verdict["promoted"] == []
    assert "task_success" in str(verdict["agreements"]["cand"]["diverged_fields"])


def test_counterexample_corpus_classification(backend) -> None:
    """§38：counterexample 按类型分类（solver/contact/friction/
    high-stiffness/constraint）落 benchmarks 语料。"""
    from rosclaw.sim.acceleration import (
        classify_counterexample,
        evaluate_gpu_candidates_semantic,
    )

    ref = backend.load_model("arm.xml")
    candidate = _gpu_like_candidate(backend, ref.model_ref, tampered=True)
    verdict = evaluate_gpu_candidates_semantic(backend, ref.model_ref, [candidate])
    categories = [classify_counterexample(c) for c in verdict["counterexamples"]]
    assert all(c in ("contact", "solver", "friction", "high_stiffness", "constraint", "state") for c in categories)


def test_gpu_execution_status_honest_not_run() -> None:
    """本机 GPU 执行面诚实状态：jax-cuda 缺失/mjx/warp 未装
    → NOT_RUN（不假装 GPU qualified）。"""
    from rosclaw.sim.acceleration import gpu_execution_status

    status = gpu_execution_status()
    assert status["status"] in ("NOT_RUN", "AVAILABLE_CPU_BACKEND")
    if status["status"] == "NOT_RUN":
        assert status["reason"]
    # 绝不输出 GPU_QUALIFIED（本机无 cuda jaxlib）。
    assert status.get("gpu_qualified") is not True
