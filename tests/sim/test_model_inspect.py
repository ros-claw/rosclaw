"""全量模型检查测试（PR-MH2，ADR-0014，规格 §12.2，红→绿）。

summary 只用于 Agent 理解；真实判断必须使用结构化数据。
"""

from __future__ import annotations

import pytest

from rosclaw.sim.backends.mujoco.inspect import inspect_model_full
from rosclaw.sim.model_inspect import inspect_mjcf


def test_full_inspection_fields(loaded_backend) -> None:
    backend, ref = loaded_backend
    inspection = backend.inspect_model(ref.model_ref)
    detail = inspection.detail

    # §12.2 清单逐项。
    assert detail["nbody"] == 3  # world + base + forearm
    assert detail["njnt"] == 2
    assert inspection.nq == 2 and inspection.nv == 2 and inspection.nu == 2

    tree = detail["body_tree"]
    assert tree["name"] == "world"
    child_names = [c["name"] for c in tree["children"]]
    assert "base" in child_names

    joints = {j["name"]: j for j in detail["joints_detail"]}
    shoulder = joints["shoulder"]
    assert shoulder["type"] == "hinge"
    assert shoulder["qpos_addr"] == 0 and shoulder["dof_addr"] == 0
    assert shoulder["limited"] is True
    assert shoulder["range"] == pytest.approx([-1.57, 1.57])
    assert joints["elbow"]["limited"] is False

    actuators = {a["name"]: a for a in detail["actuators_detail"]}
    servo = actuators["shoulder_servo"]
    assert servo["trntype"] == "joint"
    assert servo["target"] == "shoulder"
    assert servo["ctrlrange"] == pytest.approx([-1.57, 1.57])
    assert servo["forcerange"] == pytest.approx([-50.0, 50.0])
    assert len(servo["gear"]) == 6
    assert servo["kp"] == pytest.approx(10.0)  # 从 gainprm/biasprm 反推

    geoms = {g["name"]: g for g in detail["geoms"]}
    base_geom = geoms["base_geom"]
    assert base_geom["body"] == "base"
    assert base_geom["type"] == "capsule"
    assert base_geom["mass"] == pytest.approx(1.0)
    assert "contype" in base_geom and "conaffinity" in base_geom

    assert detail["sensors_detail"][0]["name"] == "shoulder_pos"
    assert "top" in [c["name"] for c in detail["cameras_detail"]]
    assert "tool0" in [s["name"] for s in detail["sites_detail"]]

    for key in ("equality", "tendons", "contact_excludes", "keyframes"):
        assert key in detail

    options = detail["options"]
    assert options["timestep"] == pytest.approx(0.002)
    assert options["solver"] and options["integrator"]
    assert options["iterations"] > 0

    assert inspection.model_ref == ref.model_ref


def test_summary_is_chinese_and_nonauthoritative(loaded_backend) -> None:
    backend, ref = loaded_backend
    inspection = backend.inspect_model(ref.model_ref)
    summary = inspection.summary
    assert isinstance(summary, str)
    assert "关节" in summary and "执行器" in summary
    assert "2" in summary  # 2 个关节
    assert "tool0" in summary


def test_legacy_inspect_mjcf_unchanged(tiny_task_root) -> None:
    """回归护栏：inspect_mjcf 输出形状冻结（w02 精确相等测试同款断言）。"""
    info = inspect_mjcf(tiny_task_root / "arm.xml")
    assert set(info.to_dict()) == {
        "model_digest",
        "nq",
        "nv",
        "nu",
        "joints",
        "actuators",
        "sensors",
        "cameras",
        "sites",
        "gripper",
    }
    assert info.joints[0] == {
        "name": "shoulder",
        "type": "hinge",
        "qpos_addr": 0,
        "dof_addr": 0,
    }
    assert info.actuators[0] == {
        "name": "shoulder_servo",
        "joint": "shoulder",
        "ctrlrange": [-1.57, 1.57],
    }


def test_inspect_model_full_direct(tiny_task_root) -> None:
    """inspect_model_full 可直接对编译后 MjModel 使用。"""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(tiny_task_root / "arm.xml"))
    detail = inspect_model_full(model, model_digest="sha256:test")
    assert detail["model_digest"] == "sha256:test"
    assert detail["nq"] == 2


def test_inspection_reports_compiled_motor_defaults_and_body_inertia() -> None:
    """Identical names do not imply identical compiled motor/body dynamics."""
    import mujoco

    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <default><joint armature="0.012" damping="0.05" frictionloss="0.2"/></default>
      <option gravity="0 0 -8" tolerance="1e-7" ls_tolerance="0.02"/>
      <worldbody><body name="link">
        <inertial pos="0.01 0.02 0.03" mass="2" diaginertia="0.01 0.02 0.025"/>
        <joint name="hinge" type="hinge"/>
        <joint name="slide" type="slide" axis="1 0 0" armature="0.025"/>
        <geom type="sphere" size="0.1"/>
      </body></worldbody>
    </mujoco>""")
    detail = inspect_model_full(model)
    hinge, slide = detail["joints_detail"]
    assert hinge["armature"] == pytest.approx(0.012)
    assert slide["armature"] == pytest.approx(0.025)
    assert hinge["damping"] == pytest.approx(0.05)
    assert slide["frictionloss"] == pytest.approx(0.2)
    body = next(b for b in detail["bodies"] if b["name"] == "link")
    assert body["mass"] == 2
    assert body["inertia"] == pytest.approx([0.01, 0.02, 0.025])
    assert body["inertial_pos"] == pytest.approx([0.01, 0.02, 0.03])
    assert body["inertial_quat"] == pytest.approx([1, 0, 0, 0])
    assert detail["options"]["gravity"] == [0, 0, -8]
    assert detail["options"]["tolerance"] == pytest.approx(1e-7)
    assert detail["options"]["ls_tolerance"] == pytest.approx(0.02)
    # Inspection reports the live compiled model, not a cached XML attribute.
    model.dof_armature[1] = 0.04
    assert inspect_model_full(model)["joints_detail"][1]["armature"] == pytest.approx(0.04)
    assert model.dof_armature[0] == pytest.approx(0.012)


def test_multidof_joint_does_not_report_misleading_scalar_motor_properties() -> None:
    import mujoco

    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body>
      <freejoint/><geom type="sphere" size="0.1"/>
      <body pos="0 0 .3"><joint type="ball"/><geom type="sphere" size=".1"/></body>
    </body></worldbody></mujoco>""")
    for joint in inspect_model_full(model)["joints_detail"]:
        assert joint["type"] in {"free", "ball"}
        assert not {"armature", "damping", "frictionloss"} & joint.keys()
