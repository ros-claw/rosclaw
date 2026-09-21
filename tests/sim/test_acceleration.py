"""Backend Fidelity Gate 测试（MH18，0916 优化 §二十八-§三十，红→绿）。

GPU = exploration，CPU = authoritative verification（ADR-0014 冻结）：
- `sim_acceleration_compatibility(model_ref)`：静态兼容性分级
  （CPU_ONLY / MJX_JAX_COMPATIBLE / MJX_WARP_COMPATIBLE + 具体原因），
  检查项全部来自 MJX/Warp 官方文档限制（PGS/noslip/plugins/
  flexcomp/muscle/custom sensor/Euler-only integrator）；
- GPU candidate 必须 CPU agreement：top-K 候选经 CPU strict
  replay 复算，分歧保留为 counterexample（不扔——§三十）。
"""

from __future__ import annotations

import pytest

TINY_ARM = """<mujoco model="tiny_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <joint name="shoulder" type="hinge" axis="0 1 0" damping="0.5"/>
      <geom name="base_geom" type="capsule" size="0.05 0.2" mass="1.0"/>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_servo" joint="shoulder" kp="10" ctrlrange="-1.57 1.57"/>
  </actuator>
</mujoco>
"""


def _variant(option_attrs: str = "", actuator_xml: str | None = None) -> str:
    actuator = actuator_xml or '<position name="shoulder_servo" joint="shoulder" kp="10"/>'
    return TINY_ARM.replace("<option timestep=\"0.002\"/>", f"<option timestep=\"0.002\" {option_attrs}/>").replace(
        '<position name="shoulder_servo" joint="shoulder" kp="10" ctrlrange="-1.57 1.57"/>', actuator
    )


@pytest.fixture
def backend(tmp_path):
    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    (tmp_path / "arm.xml").write_text(TINY_ARM, encoding="utf-8")
    (tmp_path / "pgs.xml").write_text(_variant('solver="PGS"'), encoding="utf-8")
    (tmp_path / "noslip.xml").write_text(_variant('noslip_iterations="2"'), encoding="utf-8")
    (tmp_path / "implicit.xml").write_text(_variant('integrator="implicit"'), encoding="utf-8")
    # muscle 必须挂 tendon（实证：joint 直连 muscle 的 lengthrange 不收敛）。
    (tmp_path / "muscle.xml").write_text(
        """<mujoco model="muscle_bot">
  <worldbody>
    <body name="b" pos="0 0 0.3">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05 0.2" mass="1.0"/>
      <site name="s0" pos="0 0 0"/>
      <site name="s1" pos="0 0 0.3"/>
    </body>
  </worldbody>
  <tendon><fixed name="t"><joint joint="j" coef="0.1"/></fixed></tendon>
  <actuator><muscle name="m1" tendon="t" range="0.05 0.4" lengthrange="0.05 0.5" scale="50"/></actuator>
</mujoco>
""",
        encoding="utf-8",
    )
    return MujocoBackend(tmp_path)


def test_clean_model_is_warp_compatible(backend) -> None:
    ref = backend.load_model("arm.xml")
    result = backend.acceleration_compatibility(ref.model_ref)
    assert result["classification"] == "MJX_WARP_COMPATIBLE"
    assert result["reasons"] == []


def test_pgs_is_cpu_only(backend) -> None:
    """PGS solver：MJX/Warp 均不支持（官方文档）。"""
    ref = backend.load_model("pgs.xml")
    result = backend.acceleration_compatibility(ref.model_ref)
    assert result["classification"] == "CPU_ONLY"
    assert any("pgs" in r.lower() for r in result["reasons"])


def test_noslip_is_cpu_only(backend) -> None:
    ref = backend.load_model("noslip.xml")
    result = backend.acceleration_compatibility(ref.model_ref)
    assert result["classification"] == "CPU_ONLY"
    assert any("noslip" in r.lower() for r in result["reasons"])


def test_muscle_actuator_is_cpu_only(backend) -> None:
    ref = backend.load_model("muscle.xml")
    result = backend.acceleration_compatibility(ref.model_ref)
    assert result["classification"] == "CPU_ONLY"
    assert any("muscle" in r.lower() for r in result["reasons"])


def test_implicit_integrator_warp_blocked_jax_ok(backend) -> None:
    """implicit integrator：Warp 仅 Euler（文档限制），MJX(JAX) 可
    ——分级 MJX_JAX_COMPATIBLE 而非 CPU_ONLY。"""
    ref = backend.load_model("implicit.xml")
    result = backend.acceleration_compatibility(ref.model_ref)
    assert result["classification"] == "MJX_JAX_COMPATIBLE"
    assert any("euler" in r.lower() or "integrator" in r.lower() for r in result["reasons"])


def test_gpu_candidate_cpu_agreement_gate(backend) -> None:
    """§三十：GPU 候选必须 CPU agreement——用合成"GPU 结果集"
    （真 rollout 数据假装来自 GPU sweep）验证提升门：
    一致的 PROMOTE，分歧的存 counterexample 不提升。"""
    from rosclaw.sim.acceleration import evaluate_gpu_candidates

    ref = backend.load_model("arm.xml")
    # 合成 sweep：branch A 的"GPU 结果"取自 kp=20 patch 模型的真
    # rollout（与 CPU 重放一致），branch B 被篡改（GPU/CPU 分歧）。
    patched = backend.patch_model(
        ref.model_ref,
        [{"op": "set", "target": {"type": "actuator", "name": "shoulder_servo"}, "field": "kp", "value": 20.0}],
    )
    good = backend.rollout(patched.new_model_ref, controller={"position_targets": [0.3]}, steps=60)
    trace = backend.store.get(good.trace_ref)
    candidates = [
        {
            "name": "kp20_honest",
            "patches": [
                {"op": "set", "target": {"type": "actuator", "name": "shoulder_servo"}, "field": "kp", "value": 20.0}
            ],
            "gpu_final_qpos": trace["states"][-1]["qpos"],
            "gpu_controller": {"position_targets": [0.3]},
            "gpu_steps": 60,
        },
        {
            "name": "kp20_diverged",
            "patches": [
                {"op": "set", "target": {"type": "actuator", "name": "shoulder_servo"}, "field": "kp", "value": 20.0}
            ],
            "gpu_final_qpos": [9.99],  # GPU/CPU 分歧
            "gpu_controller": {"position_targets": [0.3]},
            "gpu_steps": 60,
        },
    ]
    verdict = evaluate_gpu_candidates(backend, ref.model_ref, candidates)
    assert verdict["promoted"] == ["kp20_honest"]
    assert len(verdict["counterexamples"]) == 1
    counter = verdict["counterexamples"][0]
    assert counter["name"] == "kp20_diverged"
    assert counter["counterexample_ref"].startswith("simexp_")
    # counterexample 落库保留（§三十：分歧本身是有价值数据，不扔）。
    stored = backend.store.get(counter["counterexample_ref"])
    assert stored["kind"] == "gpu_cpu_counterexample"


def test_gpu_candidate_all_diverged_promotes_nothing(backend) -> None:
    from rosclaw.sim.acceleration import evaluate_gpu_candidates

    ref = backend.load_model("arm.xml")
    candidates = [
        {
            "name": "fake",
            "patches": [],
            "gpu_final_qpos": [123.0],
            "gpu_controller": {"position_targets": [0.3]},
            "gpu_steps": 60,
        }
    ]
    verdict = evaluate_gpu_candidates(backend, ref.model_ref, candidates)
    assert verdict["promoted"] == []
    assert len(verdict["counterexamples"]) == 1
