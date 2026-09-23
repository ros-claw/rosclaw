"""MJX GPU live 候选产出测试（MH24 G41/G42 live，红→绿）。

诚实纪律：GPU 执行面不可用（无 jax-cuda / 无 GPU 设备 / mjx 未装）
时测试 skip 并打印机器可读原因——CI（无 GPU）走 skip 路径，
GB10 aarch64（jax 0.10.2 cuda12 + mujoco-mjx 3.13.0）走真实路径。
绝不拿 CPU 数据冒充 GPU live。
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


def test_mjx_candidate_not_run_when_gpu_absent(backend, monkeypatch) -> None:
    """无 GPU 执行面时诚实 NOT_RUN（结构化原因，绝不抛裸异常）。"""
    from rosclaw.sim import acceleration

    monkeypatch.setattr(acceleration, "gpu_execution_status", lambda: {"status": "NOT_RUN", "reason": "test", "gpu_qualified": False})
    ref = backend.load_model("arm.xml")
    result = acceleration.run_mjx_candidate(
        backend, ref.model_ref, controller={"position_targets": [0.4, 0.2]}, steps=100
    )
    assert result["status"] == "NOT_RUN"
    assert result["reason"]


def test_mjx_candidate_live_real_gpu(backend) -> None:
    """真 GPU：MJX 轨迹产出 → 与 CPU authoritative 全语义比对。

    GB10 aarch64 实测入口；CI 无 GPU → skip（诚实留档不伪造）。
    """
    from rosclaw.sim.acceleration import gpu_execution_status

    status = gpu_execution_status()
    if status["status"] != "AVAILABLE":
        pytest.skip(f"GPU execution unavailable: {status.get('reason')}")

    from rosclaw.sim.acceleration import (
        evaluate_gpu_candidates_semantic,
        run_mjx_candidate,
    )

    ref = backend.load_model("arm.xml")
    result = run_mjx_candidate(
        backend, ref.model_ref, controller={"position_targets": [0.4, 0.2]}, steps=100
    )
    assert result["status"] == "RAN"
    candidate = result["candidate"]
    assert candidate["gpu_runtime"]["backend"] == "mjx"
    assert len(candidate["gpu_final_qpos"]) == 2

    verdict = evaluate_gpu_candidates_semantic(backend, ref.model_ref, [candidate])
    agreement = verdict["agreements"][candidate["name"]]
    # 不预设一致/分歧——记录真实结果；分歧则 counterexample 必须落库。
    assert agreement["status"] in ("SEMANTIC_AGREEMENT", "SEMANTIC_DIVERGENCE")
    if agreement["status"] == "SEMANTIC_DIVERGENCE":
        assert verdict["counterexamples"], "分歧必须落 counterexample 语料（§38）"
        stored = backend.store.get(verdict["counterexamples"][0]["counterexample_ref"])
        assert stored["kind"] == "gpu_cpu_counterexample"
        assert stored["gpu_payload"]["gpu_runtime"]["backend"] == "mjx"
    else:
        assert verdict["promoted"] == [candidate["name"]]


def test_gpu_execution_status_probe_consistent() -> None:
    """状态探测与实际环境一致（不硬编码机器假设）：

    - AVAILABLE ⇒ gpu_qualified=True + devices 非空；
    - NOT_RUN ⇒ reason 非空且 gpu_qualified 不为 True；
    - 任何情况下 status 必须是三态之一。
    """
    from rosclaw.sim.acceleration import gpu_execution_status

    status = gpu_execution_status()
    assert status["status"] in ("NOT_RUN", "AVAILABLE_CPU_BACKEND", "AVAILABLE")
    if status["status"] == "AVAILABLE":
        assert status["gpu_qualified"] is True
        assert status["devices"]
    else:
        assert status.get("reason")
        assert status.get("gpu_qualified") is not True
