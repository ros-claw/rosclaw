"""WorldSpec / Interaction Contract 测试（PR-MH7，规格 §22-§24/§42，红→绿）。"""

from __future__ import annotations

import pytest

from rosclaw.sim.world.compiler import compile_world
from rosclaw.sim.world.interactions import evaluate_predicates
from rosclaw.sim.world.validation import validate_worldspec


def _t1_world(**overrides):
    """T1（button + cube + box）最小世界（规格 §42）。"""
    spec = {
        "schema_version": "rosclaw.sim.worldspec.v1",
        "world": {"gravity": [0, 0, -9.81], "ground": True, "seed": 0},
        "objects": [
            {
                "id": "button",
                "shape": "cylinder",
                "size": [0.03, 0.02, 0.02],
                "pos": [0.2, 0, 0.02],
                "mass": 0.2,
                "dynamic": False,
                "rgba": [0.2, 0.2, 0.8, 1],
            },
            {
                "id": "cube",
                "shape": "box",
                "size": [0.03, 0.03, 0.03],
                "pos": [0.4, 0, 0.03],
                "mass": 0.1,
                "rgba": [1, 0, 0, 1],
            },
            {
                "id": "bin",
                "shape": "box",
                "size": [0.05, 0.05, 0.05],
                "pos": [0.6, 0, 0.05],
                "mass": 1.0,
                "dynamic": False,
                "rgba": [0.9, 0.9, 0.9, 1],
            },
        ],
        "sensors": [{"type": "camera", "name": "top", "pos": [0.4, 0, 1.5], "quat": [1, 0, 0, 0]}],
        "interaction_points": [
            {
                "id": "press_button",
                "affordance": "press",
                "target": {"type": "body", "name": "button"},
                "action_schema": {
                    "type": "object",
                    "properties": {"force": {"type": "number", "minimum": 0, "maximum": 50}},
                    "required": ["force"],
                },
                "success": [],
                "depends_on": [],
            },
            {
                "id": "inspect_cube",
                "affordance": "inspect",
                "target": {"type": "body", "name": "cube"},
                "depends_on": ["press_button"],
            },
        ],
        "task": {
            "goal": "把红色方块放进箱子区域",
            "success": [
                {
                    "channel": "body_pose:cube",
                    "field": "pos",
                    "inside": {"min": [0.5, -0.1, 0.0], "max": [0.7, 0.1, 0.2]},
                }
            ],
        },
    }
    spec.update(overrides)
    return spec


# --- validation ---------------------------------------------------------------


def test_validate_t1_ok() -> None:
    normalized = validate_worldspec(_t1_world())
    assert normalized["world"]["timestep_s"] == 0.002
    assert len(normalized["objects"]) == 3
    assert normalized["interaction_points"][1]["depends_on"] == ["press_button"]


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda s: s.update({"schema_version": "v0"}), "schema_version"),
        (lambda s: s["objects"][0].update({"shape": "torus"}), "shape"),
        (lambda s: s["objects"][0].update({"size": [0.03, 0.03]}), "size"),
        (lambda s: s["objects"][0].update({"quat": [1, 1, 0, 0]}), "quat norm"),
        (lambda s: s["objects"][0].update({"mass": -1}), "mass"),
        (lambda s: s["interaction_points"][0].update({"affordance": "teleport"}), "affordance"),
        (
            lambda s: s["interaction_points"][0].update({"target": {"type": "mesh", "name": "x"}}),
            "target.type",
        ),
        (lambda s: s["interaction_points"][1].update({"depends_on": ["ghost"]}), "depends_on"),
        (
            lambda s: s["interaction_points"][0].update({"depends_on": ["press_button"]}),
            "self dependency",
        ),
        (
            lambda s: s["interaction_points"][0]["action_schema"].update({"required": ["ghost"]}),
            "required not in properties",
        ),
        (
            lambda s: s["task"]["success"].append(
                {"channel": "body_pose:cube", "field": "pos", "teleport_to": [0, 0, 0]}
            ),
            "exactly one of",
        ),
        (lambda s: s["objects"][0].update({"id": "/etc/passwd"}), "absolute path"),
    ],
)
def test_validate_rejections(mutate, match) -> None:
    spec = _t1_world()
    mutate(spec)
    with pytest.raises(ValueError, match=match):
        validate_worldspec(spec)


def test_validate_body_ref_kind_and_pose() -> None:
    spec = _t1_world(
        body_refs=[{"id": "robot", "kind": "menagerie", "ref": "unitree_go2"}],
    )
    with pytest.raises(ValueError, match="kind unsupported"):
        validate_worldspec(spec)


# --- compile ------------------------------------------------------------------


def test_compile_t1_world_and_audit(fixture_backend, tmp_path) -> None:
    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    backend = MujocoBackend(tmp_path)
    world = compile_world(backend, _t1_world(), name="t1")
    assert world["model_ref"].startswith("simmdl_")
    assert world["interaction_order"] == ["press_button", "inspect_cube"]

    inspection = backend.inspect_model(world["model_ref"])
    body_names = [b["name"] for b in inspection.detail["bodies"]]
    assert {"button", "cube", "bin"} <= set(body_names)
    assert "top" in [c["name"] for c in inspection.detail["cameras_detail"]]
    markers = [
        s["name"] for s in inspection.detail["sites_detail"] if s["name"].startswith("marker_")
    ]
    assert set(markers) == {"marker_press_button", "marker_inspect_cube"}

    # 编译出的世界必须物理诚实（A08 marker 贴面不埋入）。
    result = backend.audit(world["model_ref"])
    assert result.status == "PASS", result.violations


def test_compile_with_robot_attach(fixture_backend, tmp_path) -> None:
    pytest.importorskip("mujoco")
    from rosclaw.runtime.eurdf_loader import _default_zoo_path

    if not (_default_zoo_path() / "ur5e" / "robot.mjcf.xml").exists():
        pytest.skip("e-urdf-zoo ur5e not available")

    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    backend = MujocoBackend(tmp_path)
    world = compile_world(
        backend,
        _t1_world(body_refs=[{"id": "ur5e", "kind": "eurdf", "ref": "ur5e"}]),
        name="t1_ur5e",
    )
    inspection = backend.inspect_model(world["model_ref"])  # 含 mesh 资产重建
    assert inspection.nq >= 6
    assert inspection.detail["njnt"] >= 6
    # ur5e：e-URDF 声明了 grasp（required_hardware: gripper），但 MJCF
    # 是手臂本体、无夹爪执行器 → UNPROVEN（诚实，不冒充 AVAILABLE）。
    assert world["capabilities"] == {"ur5e": "UNPROVEN"}


def test_grasp_without_gripper_rejected(fixture_backend, tmp_path) -> None:
    """§24/H05：无夹爪 body 的世界声明 grasp → CAPABILITY_UNAVAILABLE。"""
    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    spec = _t1_world(body_refs=[{"id": "ur5e", "kind": "eurdf", "ref": "ur5e"}])
    spec["interaction_points"].append(
        {
            "id": "grasp_cube",
            "affordance": "grasp",
            "target": {"type": "body", "name": "cube"},
            "depends_on": ["inspect_cube"],
        }
    )
    backend = MujocoBackend(tmp_path)
    from rosclaw.runtime.eurdf_loader import _default_zoo_path

    if not (_default_zoo_path() / "ur5e" / "robot.mjcf.xml").exists():
        pytest.skip("e-urdf-zoo ur5e not available")
    with pytest.raises(ValueError, match="CAPABILITY_UNAVAILABLE"):
        compile_world(backend, spec, name="t1_grasp")


def test_grasp_without_sidecar_undeclared(fixture_backend, tmp_path) -> None:
    """0915 §五：raw MJCF 无声明 = UNDECLARED——名字像 gripper 也不行。"""
    gripper_bot = """<mujoco model="gripper_bot">
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <joint name="slide" type="slide" axis="0 0 1"/>
      <geom name="palm" type="box" size="0.04 0.04 0.02" mass="0.5"/>
      <body name="finger_l" pos="0.05 0 0">
        <joint name="gripper" type="slide" axis="1 0 0" range="-0.03 0.03"/>
        <geom name="fl" type="box" size="0.01 0.02 0.02" mass="0.05"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="gripper_servo" joint="gripper" kp="20"/>
  </actuator>
</mujoco>
"""
    (tmp_path / "bot.xml").write_text(gripper_bot, encoding="utf-8")
    # 无 sidecar：即使 actuator 名字叫 gripper，也不允许 grasp。
    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    spec = _t1_world(body_refs=[{"id": "bot", "kind": "task", "ref": "bot.xml"}])
    spec["interaction_points"].append(
        {
            "id": "grasp_cube",
            "affordance": "grasp",
            "target": {"type": "body", "name": "cube"},
            "depends_on": ["inspect_cube"],
        }
    )
    backend = MujocoBackend(tmp_path)
    with pytest.raises(ValueError, match="CAPABILITY_UNAVAILABLE"):
        compile_world(backend, spec, name="t1_undeclared")


def test_grasp_with_real_gripper_allowed(fixture_backend, tmp_path) -> None:
    gripper_bot = """<mujoco model="gripper_bot">
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <joint name="slide" type="slide" axis="0 0 1"/>
      <geom name="palm" type="box" size="0.04 0.04 0.02" mass="0.5"/>
      <body name="finger_l" pos="0.05 0 0">
        <joint name="gripper" type="slide" axis="1 0 0" range="-0.03 0.03"/>
        <geom name="fl" type="box" size="0.01 0.02 0.02" mass="0.05"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="gripper_servo" joint="gripper" kp="20"/>
  </actuator>
</mujoco>
"""
    (tmp_path / "bot.xml").write_text(gripper_bot, encoding="utf-8")
    # 声明→证明绑定：sidecar 声明 grasp 由 joint "gripper" 承担，
    # 模型负责证明该 joint 存在且有 actuator——不是从名字猜。
    (tmp_path / "bot.capabilities.yaml").write_text(
        "grasp:\n  actuator_joints: [gripper]\n", encoding="utf-8"
    )
    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    spec = _t1_world(body_refs=[{"id": "bot", "kind": "task", "ref": "bot.xml"}])
    spec["interaction_points"].append(
        {
            "id": "grasp_cube",
            "affordance": "grasp",
            "target": {"type": "body", "name": "cube"},
            "depends_on": ["inspect_cube"],
        }
    )
    backend = MujocoBackend(tmp_path)
    world = compile_world(backend, spec, name="t1_gripper")
    assert world["capabilities"] == {"bot": "AVAILABLE"}


# --- predicate evaluation -------------------------------------------------------


def test_evaluate_predicates_inside_and_near() -> None:
    observations = {"body_pose:cube": {"pos": [0.6, 0.0, 0.05], "quat": [0, 0, 0, 1]}}
    predicates = [
        {
            "channel": "body_pose:cube",
            "field": "pos",
            "inside": {"min": [0.5, -0.1, 0.0], "max": [0.7, 0.1, 0.2]},
        },
        {
            "channel": "body_pose:cube",
            "field": "pos",
            "near": {"target": [0.6, 0.0, 0.05], "tolerance": 0.01},
        },
        {
            "channel": "body_pose:cube",
            "field": "pos",
            "near": {"target": [0.0, 0.0, 0.0], "tolerance": 0.01},
        },
        {
            "channel": "body_pose:ghost",
            "field": "pos",
            "inside": {"min": [0, 0, 0], "max": [1, 1, 1]},
        },
    ]
    results = evaluate_predicates(observations, predicates)
    assert results[0]["ok"] is True
    assert results[1]["ok"] is True
    assert results[2]["ok"] is False
    assert results[3]["ok"] is False and results[3]["reason"] == "channel_missing"
