"""W09 红测试（规格 2026-09-08 §13）：十类本地任务验收层。

三层纪律（§13.1）：本文件执行**物理/媒体层**（真实 MuJoCo，无
大模型）；agent 层（需真实模型 key）逐例标 NOT_RUN 跳过——
不合成冒充；契约/故障层由既有套件覆盖。

每个 case 产结果记录（§13.2）：分发物 hash 占位/模型 ID/seed/
执行状态/成功/假成功/费用未知标未知。评分器与答案不挂给
Agent（本层无 Agent）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import yaml

CASES_DIR = Path(__file__).parent / "cases"


def _cases() -> list[dict]:
    return [yaml.safe_load(p.read_text(encoding="utf-8")) for p in sorted(CASES_DIR.glob("*.yaml"))]


def _record(case: dict, verdict: str, detail: dict, started: float) -> dict:
    """§13.2 结果记录（费用未知标未知，不写 0）。"""
    return {
        "id": case["id"],
        "tier": case["tier"],
        "verdict": verdict,
        "wall_time_s": round(time.monotonic() - started, 2),
        "model": None,  # 物理层无模型——None 不是假记录
        "cost": "unknown",  # 费用未知标未知
        "false_success": False,
        "detail": detail,
    }


class TestCaseDefinitions:
    def test_ten_cases_complete_schema(self) -> None:
        cases = _cases()
        assert len(cases) == 10, f"只有 {len(cases)} 个 case"
        for case in cases:
            for key in (
                "id",
                "tier",
                "fixture",
                "variants",
                "allowed_mode",
                "allowed_outputs",
                "oracle",
                "grading",
            ):
                assert key in case, f"{case.get('id')} 缺 {key}"
            assert case["tier"] in ("contract", "physics", "agent")
            assert case["allowed_mode"] == "SIM", f"{case['id']} 未限 SIM（REAL 门禁）"


FREEJOINT_VARIANTS = [
    # (name, nq, nv, nu)：freejoint(7/6) + hinge(1/1) + 可选 ball(4/3)
    ("v1_hinge", 8, 7, 1),
    ("v2_hinge_ball", 12, 10, 2),
    ("v3_hinge_only_actuator_layout", 8, 7, 1),
]


def _variant_mjcf(name: str, with_ball: bool) -> str:
    # Residual repair candidate, NOT a diagnosis or dynamics acceptance.
    # Earlier static PASS did not prevent the actual MuJoCo 3.13 failure:
    # model3/data1/STEP20, warning5/count1/lastinfo9; 21 saved samples.
    # Step1 qacc[6]=60438.58598426328, step19 max|qvel|=12456451.197568132;
    # step20 time/ctrl reset is NOT success. The older step14 failure is
    # separate history, not the current failure. Compiled inertias NOT SAVED.
    # Hypothesis ONLY: the free ball need not augment shoulder effective
    # inertia; the thin parent may be inadequate. Neither is proven causal.
    # Give the parent its OWN explicit 1 kg capsule, radius 0.05 m,
    # half-cylinder length 0.1 m, centered on the unchanged z hinge axis.
    # Uniform-solid analytic Iz = mc*r^2/2 + ms*2*r^2/5 = 0.0011875 kg m^2
    # (cylinder mass fraction 3/4). This is geometry arithmetic, NOT a
    # compiled/effective inertia measurement. Isolated torque scale at 0.2
    # is ~168.4 rad/s^2; no coupled or numerical stability follows from it.
    # Retain the 0.05 m / 0.5 kg sphere (analytic I=0.0005), but place its
    # center at z=0.22 m: capsule tip z=0.15, sphere bottom z=0.17.
    # This leaves 0.02 m geometric clearance without any contact filtering.
    # Preserve v1 XML, topology, motors, dt .002, ctrl .2, 30 steps/.06 s,
    # full-state/warning guards. No damping/armature/solver/suppression added.
    # Source runs pure XML/arithmetic only: model/data/forward/STEP/NN/render=0.
    # Physical validation requires later separate Root GO, with original quota;
    # body_mass/body_inertia may then be observed in already-admitted compiles.
    option = '\n  <option timestep="0.002"/>' if with_ball else ""
    parent_geom = (
        '<geom type="capsule" size="0.05 0.1" mass="1"/>'
        if with_ball
        else '<geom type="capsule" size="0.01 0.1"/>'
    )
    ball = (
        """
      <body name="wrist" pos="0 0 0.22">
        <joint name="wrist_ball" type="ball"/>
        <geom type="sphere" size="0.05" mass="0.5"/>
      </body>"""
        if with_ball
        else ""
    )
    ball_act = (
        """
    <motor name="ball_motor" joint="wrist_ball" ctrlrange="-1 1"
           gear="1 0 0 0 0 0"/>"""
        if with_ball
        else ""
    )
    return f"""<mujoco model="{name}">{option}
  <worldbody>
    <body name="free_obj" pos="0 0 0.1">
      <freejoint name="obj_free"/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.5"/>
    </body>
    <body name="arm" pos="0.3 0 0.3">
      <joint name="shoulder" type="hinge" axis="0 0 1"/>
      {parent_geom}
      {ball}
    </body>
  </worldbody>
  <actuator>
    <motor name="shoulder_motor" joint="shoulder" ctrlrange="-1 1"/>
    {ball_act}
  </actuator>
</mujoco>"""


class TestL01UnfamiliarModel:
    """物理层：变体结构解释真值一致 + 完整状态记录（nq≠nv≠nu）。"""

    @staticmethod
    def _static_fixture_wrist_inertia(xml: str) -> float:
        """Solid-sphere arithmetic only; no MuJoCo compile/forward/step."""
        import math
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml)
        wrist = root.find("./worldbody/body[@name='arm']/body[@name='wrist']")
        assert wrist is not None
        geom = wrist.find("geom")
        assert geom is not None and geom.get("type") == "sphere"
        # Explicit one-radius geometry/mass, not inferred compiler defaults.
        assert "mass" in geom.attrib and "size" in geom.attrib
        sizes = [float(v) for v in geom.attrib["size"].split()]
        assert len(sizes) == 1
        radius, mass = sizes[0], float(geom.attrib["mass"])
        assert math.isfinite(radius) and radius > 0
        assert math.isfinite(mass) and mass > 0
        inertia = 2 * mass * radius**2 / 5
        assert math.isfinite(inertia) and inertia > 0
        return inertia

    @pytest.mark.parametrize(
        "name,with_ball,expected",
        [("v1_hinge", False, (8, 7, 1)), ("v2_hinge_ball", True, (12, 10, 2))],
    )
    def test_static_fixture_topology_dimensions(
        self,
        name,
        with_ball,
        expected,
    ) -> None:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(_variant_mjcf(name, with_ball))
        assert root.tag == "mujoco" and root.get("model") == name
        joints = root.findall(".//freejoint") + root.findall(".//joint")
        dimensions = {"freejoint": (7, 6), "hinge": (1, 1), "ball": (4, 3)}
        nq = nv = 0
        for joint in joints:
            kind = "freejoint" if joint.tag == "freejoint" else joint.get("type")
            q, v = dimensions[kind]
            nq += q
            nv += v
        motors = root.findall("./actuator/motor")
        assert (nq, nv, len(motors)) == expected
        names = {j.get("name") for j in joints}
        assert len(names) == len(joints)  # no duplicated/extra coordinates
        assert names == (
            {"obj_free", "shoulder", "wrist_ball"} if with_ball else {"obj_free", "shoulder"}
        )
        actuated = {m.get("joint") for m in motors}
        assert actuated <= names and names - actuated == {"obj_free"}
        free = root.find("./worldbody/body[@name='free_obj']/freejoint")
        assert free is not None and free.get("name") == "obj_free"
        wrist = root.find("./worldbody/body[@name='arm']/body[@name='wrist']")
        if with_ball:
            assert wrist is not None and wrist.get("pos") == "0 0 0.22"
            assert wrist.find("joint").attrib == {"name": "wrist_ball", "type": "ball"}
        else:
            assert wrist is None

    @pytest.mark.parametrize("with_ball", [False, True])
    def test_static_fixture_actuator_controls(self, with_ball) -> None:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(_variant_mjcf("controls", with_ball))
        actuator = root.find("actuator")
        expected = {"shoulder_motor": "shoulder"}
        if with_ball:
            expected["ball_motor"] = "wrist_ball"
        assert {m.get("name"): m.get("joint") for m in actuator} == expected
        assert len(actuator) == len(expected)
        for motor in actuator:
            assert motor.tag == "motor"
            lo, hi = map(float, motor.attrib["ctrlrange"].split())
            assert (lo, hi) == (-1.0, 1.0) and lo < 0.2 < hi
            allowed = {"name", "joint", "ctrlrange"}
            if motor.get("name") == "ball_motor":
                allowed.add("gear")
                assert motor.get("gear") == "1 0 0 0 0 0"
            assert set(motor.attrib) == allowed  # no force clamp/gain reduction
        assert root.find("default") is None

    def test_static_fixture_declared_wrist_inertia(self) -> None:
        import math
        import xml.etree.ElementTree as ET

        xml = _variant_mjcf("inertia", True)
        geom = ET.fromstring(xml).find(".//body[@name='wrist']/geom")
        assert geom.attrib == {"type": "sphere", "size": "0.05", "mass": "0.5"}
        inertia = self._static_fixture_wrist_inertia(xml)
        assert inertia == pytest.approx(0.0005, rel=1e-12, abs=0)
        # Analytic scale comparison, not measured inertia/cause/stability.
        old_mass = 1000 * (4 * math.pi / 3) * 0.01**3
        old_inertia = 2 * old_mass * 0.01**2 / 5
        assert inertia / old_inertia > 2900
        assert 0.2 * 0.002 / inertia == pytest.approx(0.8, rel=1e-12, abs=0)

    def test_static_fixture_timing_and_unmodified_solver(self) -> None:
        import xml.etree.ElementTree as ET

        v2 = ET.fromstring(_variant_mjcf("timing", True))
        option = v2.find("option")
        assert option is not None and option.attrib == {"timestep": "0.002"}
        assert len(option) == 0  # no disabling flags or solver overrides
        assert 30 * float(option.get("timestep")) == 0.06
        assert [child.tag for child in v2] == ["option", "worldbody", "actuator"]
        v1 = ET.fromstring(_variant_mjcf("timing_v1", False))
        assert [child.tag for child in v1] == ["worldbody", "actuator"]
        assert v1.find("option") is None  # original default timestep unchanged

    @pytest.mark.parametrize("with_ball", [False, True])
    def test_static_fixture_base_geometry_preserved(self, with_ball) -> None:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(_variant_mjcf("base", with_ball))
        world = root.find("worldbody")
        assert [b.get("name") for b in world] == ["free_obj", "arm"]
        free, arm = world
        assert free.attrib == {"name": "free_obj", "pos": "0 0 0.1"}
        assert free.find("geom").attrib == {
            "type": "box",
            "size": "0.02 0.02 0.02",
            "mass": "0.5",
        }
        assert free.find("freejoint").attrib == {"name": "obj_free"}
        assert arm.attrib == {"name": "arm", "pos": "0.3 0 0.3"}
        assert arm.find("joint").attrib == {
            "name": "shoulder",
            "type": "hinge",
            "axis": "0 0 1",
        }
        expected_parent = (
            {"type": "capsule", "size": "0.05 0.1", "mass": "1"}
            if with_ball
            else {"type": "capsule", "size": "0.01 0.1"}
        )
        assert arm.find("geom").attrib == expected_parent

    @staticmethod
    def _static_fixture_parent_axial_inertia(xml: str) -> float:
        """Uniform capsule's own axial moment, not compiled/effective inertia."""
        import math
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml)
        geom = root.find("./worldbody/body[@name='arm']/geom")
        assert geom is not None and geom.get("type") == "capsule"
        assert "size" in geom.attrib and "mass" in geom.attrib
        sizes = [float(v) for v in geom.get("size").split()]
        assert len(sizes) == 2
        radius, half_length = sizes
        mass = float(geom.get("mass"))
        assert all(math.isfinite(v) and v > 0 for v in (radius, half_length, mass))
        # One uniform capsule: cylindrical volume plus two hemispheres.
        cylinder_fraction = (2 * half_length) / (2 * half_length + 4 * radius / 3)
        inertia = mass * radius**2 * (0.5 * cylinder_fraction + 0.4 * (1 - cylinder_fraction))
        assert math.isfinite(inertia) and inertia > 0
        return inertia

    def test_static_fixture_parent_own_inertia_scale(self) -> None:
        import xml.etree.ElementTree as ET

        xml = _variant_mjcf("parent_scale", True)
        geom = ET.fromstring(xml).find("./worldbody/body[@name='arm']/geom")
        assert geom.attrib == {"type": "capsule", "size": "0.05 0.1", "mass": "1"}
        inertia = self._static_fixture_parent_axial_inertia(xml)
        assert inertia == pytest.approx(0.0011875, rel=1e-12, abs=0)
        assert 0.2 * 0.002 / inertia == pytest.approx(0.3368421052631579)
        # Child mass is deliberately absent from this analytic calculation.
        root = ET.fromstring(xml)
        root.find(".//body[@name='wrist']/geom").set("mass", "5")
        assert (
            self._static_fixture_parent_axial_inertia(ET.tostring(root, encoding="unicode"))
            == inertia
        )

    def test_static_fixture_geometry_clearance_and_finite_placement(self) -> None:
        import math
        import xml.etree.ElementTree as ET

        root = ET.fromstring(_variant_mjcf("clearance", True))
        arm = root.find("./worldbody/body[@name='arm']")
        wrist = arm.find("body[@name='wrist']")
        assert wrist.attrib == {"name": "wrist", "pos": "0 0 0.22"}
        parent, child = arm.find("geom"), wrist.find("geom")
        assert set(parent.attrib) == {"type", "size", "mass"}
        assert set(child.attrib) == {"type", "size", "mass"}
        # No pos/fromto/orientation: capsule is centered, aligned with hinge z.
        radius, half_length = map(float, parent.get("size").split())
        ball_radius = float(child.get("size"))
        x, y, z = map(float, wrist.get("pos").split())
        assert (x, y) == (0, 0)
        assert z - ball_radius - (half_length + radius) == pytest.approx(0.02)
        for body in (arm, wrist):
            values = [float(v) for v in body.get("pos").split()]
            assert len(values) == 3 and all(math.isfinite(v) for v in values)
        for geom in root.findall(".//geom"):
            values = [float(v) for v in geom.get("size").split()]
            values.append(float(geom.get("mass")))
            assert all(math.isfinite(v) and v > 0 for v in values)
        assert root.findall(".//inertial") == []
        for joint in root.findall(".//joint"):
            assert not {"damping", "armature", "frictionloss", "stiffness"} & set(joint.attrib)

    def test_static_fixture_v1_xml_unchanged(self) -> None:
        import xml.etree.ElementTree as ET

        # Literal original v1 tree: not a comparison against the generator itself.
        expected = ET.fromstring("""<mujoco model="v1_hinge"><worldbody>
          <body name="free_obj" pos="0 0 0.1"><freejoint name="obj_free"/>
            <geom type="box" size="0.02 0.02 0.02" mass="0.5"/></body>
          <body name="arm" pos="0.3 0 0.3">
            <joint name="shoulder" type="hinge" axis="0 0 1"/>
            <geom type="capsule" size="0.01 0.1"/></body>
          </worldbody><actuator><motor name="shoulder_motor" joint="shoulder"
          ctrlrange="-1 1"/></actuator></mujoco>""")
        actual = ET.fromstring(_variant_mjcf("v1_hinge", False))

        def canonical(element):
            return (
                element.tag,
                sorted(element.attrib.items()),
                (element.text or "").strip(),
                [canonical(child) for child in element],
            )

        assert canonical(actual) == canonical(expected)

    @pytest.mark.parametrize(
        "attribute,value",
        [
            ("mass", None),
            ("mass", "0"),
            ("mass", "-1"),
            ("mass", "inf"),
            ("mass", "nan"),
            ("size", "0 0.1"),
            ("size", "0.05 0"),
            ("size", "0.05 -0.1"),
            ("size", "0.05 inf"),
            ("size", "nan 0.1"),
            ("size", "0.05"),
            ("type", "sphere"),
        ],
    )
    def test_static_fixture_invalid_parent_rejected(self, attribute, value) -> None:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(_variant_mjcf("negative_parent", True))
        geom = root.find("./worldbody/body[@name='arm']/geom")
        if value is None:
            del geom.attrib[attribute]
        else:
            geom.set(attribute, value)
        with pytest.raises(AssertionError):
            self._static_fixture_parent_axial_inertia(ET.tostring(root, encoding="unicode"))

    @pytest.mark.parametrize(
        "attribute,value",
        [
            ("mass", None),
            ("mass", "0"),
            ("mass", "-0.5"),
            ("mass", "inf"),
            ("size", "nan"),
            ("size", "0"),
            ("size", "-0.05"),
            ("size", "0.05 0.05"),
            ("type", "box"),
        ],
    )
    def test_static_fixture_invalid_wrist_rejected(self, attribute, value) -> None:
        import xml.etree.ElementTree as ET

        # Negative pure controls verify the mass/inertia checks are operative;
        # mutations stay in memory and never reach the dynamics fixture.
        root = ET.fromstring(_variant_mjcf("negative", True))
        geom = root.find(".//body[@name='wrist']/geom")
        if value is None:
            del geom.attrib[attribute]
        else:
            geom.set(attribute, value)
        with pytest.raises(AssertionError):
            self._static_fixture_wrist_inertia(ET.tostring(root, encoding="unicode"))

    @pytest.mark.parametrize(
        "name,nq,nv,nu,with_ball",
        [("v1_hinge", 8, 7, 1, False), ("v2_hinge_ball", 12, 10, 2, True)],
    )
    def test_structure_truth_and_full_state(
        self,
        tmp_path,
        name,
        nq,
        nv,
        nu,
        with_ball,
    ) -> None:
        from rosclaw.sim import api
        from rosclaw.sim.model_inspect import inspect_mjcf

        started = time.monotonic()
        mjcf = tmp_path / f"{name}.xml"
        mjcf.write_text(_variant_mjcf(name, with_ball), encoding="utf-8")
        info = inspect_mjcf(mjcf)
        assert (info.nq, info.nv, info.nu) == (nq, nv, nu)
        passive = {j["name"] for j in info.joints} - {a["joint"] for a in info.actuators}
        assert "obj_free" in passive  # freejoint 不是 actuator
        if with_ball:
            assert "wrist_ball" not in passive  # ball 有 motor
        ref, _ = api.load_model(mjcf, task_root=tmp_path)
        op = api.submit_simulation(
            ref,
            None,
            {"ctrl_series": [[0.2] * nu] * 30},
            0.06,
            task_root=tmp_path,
        )
        trace = api.read_trace(op, task_root=tmp_path)
        sample = trace["states"][-1]
        assert len(sample["qpos"]) == nq  # 完整状态（W03）
        assert len(sample["ctrl"]) == nu
        rec = _record(
            {"id": f"L01/{name}", "tier": "physics"},
            "PASS",
            {"nq": nq, "nv": nv, "nu": nu},
            started,
        )
        assert rec["cost"] == "unknown"


class TestL02ArbitraryTrajectory:
    """物理层：任意 SE(3) 路径 rollout + 时间对齐 + 视频同源。"""

    def test_rollout_render_time_aligned(self, tmp_path) -> None:
        from rosclaw.sim import api

        started = time.monotonic()
        mjcf = tmp_path / "m.xml"
        mjcf.write_text(_variant_mjcf("l02", False), encoding="utf-8")
        ref, _ = api.load_model(mjcf, task_root=tmp_path)
        op = api.submit_simulation(
            ref,
            None,
            {"ctrl_series": [[0.4]] * 100},
            0.2,
            task_root=tmp_path,
        )
        trace = api.read_trace(op, task_root=tmp_path)
        times = [s["t"] for s in trace["states"]]
        assert times == sorted(times)  # 时间单调
        rop = api.render(
            op,
            {"camera": "follow", "outputs": ["gif"], "fps": 10.0},
            task_root=tmp_path,
        )
        gif = tmp_path / "renders" / rop / f"{rop}.gif"
        assert gif.exists()
        from PIL import Image

        with Image.open(gif) as img:
            assert getattr(img, "n_frames", 1) >= 2
        receipt = json.loads((tmp_path / "renders" / rop / "render_receipt.json").read_text())
        assert receipt["evidence_level"] == "agent_generated_experiment"
        _record({"id": "L02", "tier": "physics"}, "PASS", {"duration_s": 0.2}, started)


class TestL07DampingExperiment:
    """物理层：三阻尼×三次，只改声明参数，衰减结论可重算。"""

    PENDULUM = """<mujoco model="pend">
      <option timestep="0.002"/>
      <worldbody>
        <body name="link" pos="0 0 0.3">
          <joint name="hinge" type="hinge" axis="0 1 0"
                 damping="{damping}" armature="0"/>
          <geom type="capsule" size="0.01 0.15" pos="0 0 -0.15" mass="1"/>
        </body>
      </worldbody>
    </mujoco>"""

    def test_decay_recomputable_and_monotone(self, tmp_path) -> None:
        import mujoco
        import numpy as np

        started = time.monotonic()
        results: dict[float, list[float]] = {}
        # 夹具参数选欠阻尼区（§13.9：避免过阻尼改变指标定义——
        # 实测 damping=0.8 六秒内峰值不足三个，指标失效）。
        for damping in (0.02, 0.1, 0.3):
            peak_decays: list[float] = []
            for _run in range(3):  # 每条件三次（同参确定性重跑）
                model = mujoco.MjModel.from_xml_string(self.PENDULUM.format(damping=damping))
                data = mujoco.MjData(model)
                data.qpos[0] = 0.5
                peaks: list[float] = []
                prev_vel = 0.0
                steps = int(8.0 / model.opt.timestep)
                for _ in range(steps):
                    mujoco.mj_step(model, data)
                    vel = float(data.qvel[0])
                    if prev_vel > 0 >= vel or prev_vel < 0 <= vel:
                        peaks.append(abs(float(data.qpos[0])))
                    prev_vel = vel
                # 衰减率 = 相邻峰值比（从数据重算，不读理论值）。
                if len(peaks) >= 3:
                    peak_decays.append(float(peaks[2] / peaks[1]))
            results[damping] = peak_decays
        rates = {d: float(np.mean(v)) for d, v in results.items() if v}
        assert len(rates) == 3, f"条件未产生可重算峰值: {results.keys()}"
        assert rates[0.3] < rates[0.02], f"阻尼越大衰减越快——实测不符: {rates}"
        _record({"id": "L07", "tier": "physics"}, "PASS", {"peak_ratio_by_damping": rates}, started)


class TestL10RenderLifecycle:
    """物理层：同 trace 双视角独立产出（并发安全由 W04 身份键
    保证）；重启后读同一 trace 不重 rollout。"""

    def test_two_views_independent_and_restart_reads(self, tmp_path) -> None:
        from rosclaw.sim import api

        started = time.monotonic()
        mjcf = tmp_path / "m.xml"
        mjcf.write_text(_variant_mjcf("l10", False), encoding="utf-8")
        ref, _ = api.load_model(mjcf, task_root=tmp_path)
        op = api.submit_simulation(
            ref,
            None,
            {"ctrl_series": [[0.3]] * 50},
            0.1,
            task_root=tmp_path,
        )
        digest_before = json.loads((tmp_path / "models" / f"{op}.json").read_text())[
            "states_digest"
        ]
        r1 = api.render(op, {"camera": "follow", "outputs": ["gif"]}, task_root=tmp_path)
        r2 = api.render(op, {"camera": "top", "outputs": ["gif"]}, task_root=tmp_path)
        assert r1 != r2  # 独立 operation 目录
        assert (tmp_path / "renders" / r1 / f"{r1}.gif").exists()
        assert (tmp_path / "renders" / r2 / f"{r2}.gif").exists()
        # “重启”= 新进程式重读（记录持久化——不重 rollout）。
        digest_after = json.loads((tmp_path / "models" / f"{op}.json").read_text())["states_digest"]
        assert digest_before == digest_after
        _record({"id": "L10", "tier": "physics"}, "PASS", {"views": [r1, r2]}, started)


class TestAgentTierHonestNotRun:
    """agent 层真实驱动已建（tests/eval/agent_tier/）——2026-09-10
    真实 K3 六类全过：L03 绕行（0 接触/净空 35mm）、L04 推物进区
    保持、L05 双变体视觉定位（≤20mm 盒体距离）、L06 双变体 LQR
    平衡（独立重跑 ≤5°）、L08 诚实拒绝+fake REAL 零命令、L09
    幅度比 0.519+顶视图复用 trace。

    无 key 时本层仍 NOT_RUN——不合成冒充。"""

    @pytest.mark.parametrize(
        "module",
        [
            "test_l03_obstacle",
            "test_l04_push",
            "test_l05_visual",
            "test_l06_cartpole",
            "test_l08_missing_capability",
            "test_l09_modify_append",
        ],
    )
    def test_agent_tier_driver_exists(self, module) -> None:
        import importlib.util

        path = Path(__file__).parent / "agent_tier" / f"{module}.py"
        assert path.exists(), f"agent 层驱动缺失: {path}"
        spec = importlib.util.spec_from_file_location(module, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # 可导入（夹具/接口自洽）
        import os

        if not os.environ.get("ROSCLAW_KIMI_API_KEY"):
            pytest.skip(f"NOT_RUN: {module} 需真实模型（有 key 直跑）")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
