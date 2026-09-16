"""限值与数值健壮性 audit 测试（MH15，0916 优化 §二十一-§二十三，红→绿）。

限值来源纪律：模型自带 ctrlrange/forcerange（物理事实）与
e-URDF Safety Profile；无声明 → NOT_EVALUATED（中性，不用万能阈值）。
"""

from __future__ import annotations

import pytest

SATURATION_MODEL = """<mujoco model="sat_bot">
  <compiler autolimits="true"/>
  <worldbody>
    <body name="arm" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom name="g" type="capsule" size="0.05 0.2" mass="1.0"/>
    </body>
  </worldbody>
  <actuator>
    <position name="s1" joint="j1" kp="50" ctrlrange="-0.01 0.01"/>
  </actuator>
</mujoco>
"""

BUFFER_MODEL = """<mujoco model="bad_buffer_bot">
  <worldbody>
    <body name="arm" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom name="g" type="capsule" size="0.05 0.2" mass="1.0"/>
    </body>
  </worldbody>
  <actuator><position name="s1" joint="j1" kp="10"/></actuator>
  <sensor>
    <jointpos name="jp" joint="j1" nsample="10" delay="0.1"/>
  </sensor>
</mujoco>
"""

TILTED_MODEL = """<mujoco model="tilted">
  <option gravity="9.81 0 0"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
  </worldbody>
</mujoco>
"""

DROP_MODEL = """<mujoco model="drop_bot">
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="rock" pos="0 0 0.5">
      <freejoint name="rf"/>
      <geom name="rock_g" type="box" size="0.1 0.1 0.1" mass="30.0"/>
    </body>
  </worldbody>
</mujoco>
"""

# A23 实证（0916）：简单模型（含 08_unstable_controller 单自由度）
# Newton/CG/PGS 逐位一致——无约束或精确线性求解时 solver 不敏感是物理事实，
# 不是检查缺陷。真正 solver 敏感的形态：多接触堆叠 + 严重受限的
# iterations 预算（CG 迭代法收敛不足，Newton 一两步即收敛）→ 末态实质不同。
_SOLVER_STACK_BODIES = """
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="b1" pos="0 0 0.05"><freejoint name="f1"/><geom type="box" size="0.05 0.05 0.05" mass="1.0"/></body>
    <body name="b2" pos="0.005 0 0.15"><freejoint name="f2"/><geom type="box" size="0.05 0.05 0.05" mass="1.0"/></body>
    <body name="b3" pos="-0.005 0 0.25"><freejoint name="f3"/><geom type="box" size="0.05 0.05 0.05" mass="1.0"/></body>
  </worldbody>
"""

SOLVER_SENSITIVE_MODEL = (
    '<mujoco model="solver_fragile">\n'
    '  <option timestep="0.002" iterations="2"><flag warmstart="disable"/></option>\n'
    + _SOLVER_STACK_BODIES
    + "</mujoco>\n"
)

SOLVER_STABLE_MODEL = (
    '<mujoco model="solver_robust">\n'
    '  <option timestep="0.002"><flag warmstart="disable"/></option>\n'
    + _SOLVER_STACK_BODIES
    + "</mujoco>\n"
)


@pytest.fixture
def backend(tmp_path):
    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    for name, xml in (
        ("sat", SATURATION_MODEL),
        ("buffer", BUFFER_MODEL),
        ("tilted", TILTED_MODEL),
        ("drop", DROP_MODEL),
        ("solver_fragile", SOLVER_SENSITIVE_MODEL),
        ("solver_robust", SOLVER_STABLE_MODEL),
    ):
        (tmp_path / f"{name}.xml").write_text(xml, encoding="utf-8")
    return MujocoBackend(tmp_path)


def test_a09_saturation_red(backend) -> None:
    ref = backend.load_model("sat.xml")
    trace = backend.rollout(ref.model_ref, controller={"position_targets": [1.0]}, steps=200)
    result = backend.audit(
        ref.model_ref, checks=["A09_actuator_saturation"], trace_ref=trace.trace_ref
    )
    check = result.checks["A09_actuator_saturation"]
    assert check["status"] == "FAIL"
    assert check["saturation_ratio"] > 0.2


def test_a09_saturation_green(loaded_backend) -> None:
    backend, ref = loaded_backend
    trace = backend.rollout(ref.model_ref, controller={"position_targets": [0.3, 0.1]}, steps=100)
    result = backend.audit(
        ref.model_ref, checks=["A09_actuator_saturation"], trace_ref=trace.trace_ref
    )
    assert result.checks["A09_actuator_saturation"]["status"] == "PASS"


def test_a10_force_limit_red(backend, loaded_backend) -> None:
    # tiny_arm forcerange=-50 50；kp patch 到 5000 后位置伺服力超限。
    backend, ref = loaded_backend
    stiff = backend.patch_model(
        ref.model_ref,
        [
            {
                "op": "set",
                "target": {"type": "actuator", "name": "shoulder_servo"},
                "field": "kp",
                "value": 5000.0,
            },
        ],
    )
    trace = backend.rollout(
        stiff.new_model_ref, controller={"position_targets": [0.5, 0.1]}, steps=100
    )
    result = backend.audit(
        stiff.new_model_ref, checks=["A10_force_limit"], trace_ref=trace.trace_ref
    )
    check = result.checks["A10_force_limit"]
    assert check["status"] == "FAIL"
    # forcerange 是物理裁剪——力恰好钉在 50.0，FAIL 由持续饱和比判定。
    assert check["max_actuator_force"] >= 50.0


def test_a10_force_limit_green(loaded_backend) -> None:
    backend, ref = loaded_backend
    trace = backend.rollout(ref.model_ref, controller={"hold": True}, steps=50)
    result = backend.audit(ref.model_ref, checks=["A10_force_limit"], trace_ref=trace.trace_ref)
    assert result.checks["A10_force_limit"]["status"] == "PASS"


def test_a11_velocity_limit_red(backend) -> None:
    ref = backend.load_model("sat.xml")
    trace = backend.rollout(ref.model_ref, controller={"position_targets": [1.0]}, steps=200)
    result = backend.audit(ref.model_ref, checks=["A11_velocity_limit"], trace_ref=trace.trace_ref)
    # 无 e-URDF 声明 → NOT_EVALUATED（中性，不用万能阈值）
    assert result.checks["A11_velocity_limit"]["status"] == "NOT_EVALUATED"


def test_a11_velocity_limit_with_declaration(backend) -> None:
    """经 ctx.extra 声明 velocity_limits 后超速即 FAIL（e-URDF 同款机制）。"""
    from rosclaw.sim.audit.context import AuditContext
    from rosclaw.sim.audit.limits import a11_velocity_limit

    ref = backend.load_model("sat.xml")
    manifest = backend._manifest(ref.model_ref)
    spec = backend._spec_from_manifest(manifest)
    model, _ = backend._compile_smoke(spec)
    # 必须带 trace_record——否则 hold 扫描无运动，qvel=0 测不到超速。
    trace = backend.rollout(ref.model_ref, controller={"position_targets": [1.0]}, steps=200)
    ctx = AuditContext(
        model=model,
        spec=spec,
        xml_text=manifest["mjcf_xml"],
        trace_record=backend.store.get(trace.trace_ref),
        extra={"safety_limits": {"velocity_limits": {"j1": 0.01}}},
    )
    outcome = a11_velocity_limit(ctx)
    assert outcome["status"] == "FAIL"
    assert outcome["violations"][0]["joint"] == "j1"


def test_a12_acceleration_spike_red(backend) -> None:
    from rosclaw.sim.audit.context import AuditContext
    from rosclaw.sim.audit.limits import a12_acceleration_spike

    ref = backend.load_model("sat.xml")
    manifest = backend._manifest(ref.model_ref)
    spec = backend._spec_from_manifest(manifest)
    model, _ = backend._compile_smoke(spec)
    trace = backend.rollout(ref.model_ref, controller={"position_targets": [1.0]}, steps=50)
    ctx_trace = AuditContext(
        model=model,
        spec=spec,
        xml_text=manifest["mjcf_xml"],
        trace_record=backend.store.get(trace.trace_ref),
        extra={"acceleration_limit": 10.0},
    )
    outcome = a12_acceleration_spike(ctx_trace)
    assert outcome["status"] == "FAIL"
    assert outcome["violations"][0]["reason"] == "acceleration_spike"


def test_a14_peak_contact_force_red(backend) -> None:
    from rosclaw.sim.audit.context import AuditContext
    from rosclaw.sim.audit.limits import a14_peak_contact_force

    ref = backend.load_model("drop.xml")
    manifest = backend._manifest(ref.model_ref)
    spec = backend._spec_from_manifest(manifest)
    model, _ = backend._compile_smoke(spec)
    ctx = AuditContext(
        model=model,
        spec=spec,
        xml_text=manifest["mjcf_xml"],
        extra={"safety_limits": {"force_limits": {"max_tcp_force": 10.0}}},
    )
    outcome = a14_peak_contact_force(ctx)
    assert outcome["status"] == "FAIL"
    assert outcome["violations"][0]["max_contact_force"] > 10.0


def test_a14_peak_contact_force_green(backend) -> None:
    from rosclaw.sim.audit.context import AuditContext
    from rosclaw.sim.audit.limits import a14_peak_contact_force

    ref = backend.load_model("drop.xml")
    manifest = backend._manifest(ref.model_ref)
    spec = backend._spec_from_manifest(manifest)
    model, _ = backend._compile_smoke(spec)
    ctx = AuditContext(
        model=model,
        spec=spec,
        xml_text=manifest["mjcf_xml"],
        extra={"safety_limits": {"force_limits": {"max_tcp_force": 1e6}}},
    )
    outcome = a14_peak_contact_force(ctx)
    assert outcome["status"] == "PASS"


def test_a21_sensor_buffer_undersized_red(backend) -> None:
    ref = backend.load_model("buffer.xml")
    result = backend.audit(ref.model_ref, checks=["A21_sensor_validity"])
    check = result.checks["A21_sensor_validity"]
    assert check["status"] == "FAIL"
    assert check["violations"][0]["reason"] == "sensor_buffer_undersized"


def test_a21_sensor_validity_green(loaded_backend) -> None:
    backend, ref = loaded_backend
    result = backend.audit(ref.model_ref, checks=["A21_sensor_validity"])
    assert result.checks["A21_sensor_validity"]["status"] == "PASS"


def test_a22_frame_convention_warn(backend) -> None:
    ref = backend.load_model("tilted.xml")
    result = backend.audit(ref.model_ref, checks=["A22_frame_convention"])
    check = result.checks["A22_frame_convention"]
    assert check["status"] == "WARN"
    assert result.status == "WARN"


def test_a22_frame_convention_green(loaded_backend) -> None:
    backend, ref = loaded_backend
    result = backend.audit(ref.model_ref, checks=["A22_frame_convention"])
    assert result.checks["A22_frame_convention"]["status"] == "PASS"


def test_a23_solver_sensitivity_warn(backend) -> None:
    """多接触堆叠 + iterations=2：CG 收敛不足，末态与 Newton 实质不同
    → ROBUSTNESS_WARNING（实测偏差 ≈2e-3 > sensitivity_rel=1e-3）。"""
    ref = backend.load_model("solver_fragile.xml")
    result = backend.audit(ref.model_ref, checks=["A23_solver_sensitivity"])
    check = result.checks["A23_solver_sensitivity"]
    assert check["status"] == "WARN"
    assert check["solver_deviations"]["cg"] > 1e-3


def test_a23_solver_green(backend) -> None:
    """同堆叠、默认 iterations 预算：solver 均收敛，末态一致 → PASS。"""
    ref = backend.load_model("solver_robust.xml")
    result = backend.audit(ref.model_ref, checks=["A23_solver_sensitivity"])
    assert result.checks["A23_solver_sensitivity"]["status"] == "PASS"


def test_a24_timestep_fragility_warn(fixture_backend) -> None:
    backend, ref = fixture_backend("broken_models", "08_unstable_controller")
    result = backend.audit(ref.model_ref, checks=["A24_timestep_sensitivity"])
    check = result.checks["A24_timestep_sensitivity"]
    assert check["status"] == "WARN"
    assert check["timestep_deviation"] > 1e-3


def test_a24_timestep_green(fixture_backend) -> None:
    backend, ref = fixture_backend("fixed_models", "08_unstable_controller")
    result = backend.audit(ref.model_ref, checks=["A24_timestep_sensitivity"])
    assert result.checks["A24_timestep_sensitivity"]["status"] == "PASS"


def test_not_evaluated_is_neutral(loaded_backend) -> None:
    """无声明限值的 body：NOT_EVALUATED 不拉低总状态。"""
    backend, ref = loaded_backend
    result = backend.audit(
        ref.model_ref,
        checks=["A11_velocity_limit", "A12_acceleration_spike", "A14_peak_contact_force"],
    )
    assert result.status == "PASS"
    for name in ("A11_velocity_limit", "A12_acceleration_spike", "A14_peak_contact_force"):
        assert result.checks[name]["status"] == "NOT_EVALUATED"
