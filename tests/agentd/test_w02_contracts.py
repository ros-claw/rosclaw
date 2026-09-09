"""W02 红测试（规格 2026-09-08 §6）：

6.1 可推导的身体档案——从加载的 MJCF 推出（不是从注册表/名字猜）：
- freejoint（nq=7, nv=6）、ball joint（nq=4, nv=3）、被动铰链、
  执行器少于关节数的 fixture 均能 inspect；
- joint 名称/类型/qpos 地址/dof 地址、执行器→关节映射、
  控制类型与限制、传感器、camera/site 全齐；
- 没有夹爪不从机器人名称猜有夹爪。

6.3/§2.2：渲染回放不得默认 UR5e——trace/plan 记录什么模型就
用什么模型；记录缺失=legacy/unverified 标记，不静默默认。

6.4：最小 Python API（load_model/observe/submit_simulation/
read_trace/render/export）实际可导入可组合——task_local_mjcf
可加载（任务根内解析，不执行外部插件）。
"""

from __future__ import annotations

import pytest

FIXTURE_MJCF = """<mujoco model="w02_fixture">
  <worldbody>
    <body name="free_obj" pos="0 0 0.1">
      <freejoint name="obj_free"/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.5"/>
    </body>
    <body name="arm" pos="0.3 0 0.3">
      <joint name="shoulder" type="hinge" axis="0 0 1"/>
      <geom type="capsule" size="0.01 0.1"/>
      <body name="forearm" pos="0 0 0.1">
        <joint name="elbow" type="ball"/>
        <geom type="capsule" size="0.008 0.08"/>
        <site name="tip" pos="0 0 0.08"/>
      </body>
    </body>
    <camera name="overhead" pos="0.3 0 1.2" euler="0 0 0"/>
  </worldbody>
  <actuator>
    <motor name="shoulder_motor" joint="shoulder" ctrlrange="-1 1"/>
  </actuator>
  <sensor>
    <jointpos joint="shoulder"/>
  </sensor>
</mujoco>
"""


class TestModelInspection:
    def test_fixture_dimensions_and_mapping(self, tmp_path) -> None:
        """nq≠nv≠nu 的 fixture：freejoint(7/6) + hinge(1/1) + ball(4/3)
        → nq=12, nv=10；执行器 1 个（少于关节数）。"""
        from rosclaw.sim.model_inspect import inspect_mjcf

        path = tmp_path / "fixture.xml"
        path.write_text(FIXTURE_MJCF, encoding="utf-8")
        info = inspect_mjcf(path)
        assert info.nq == 12, info
        assert info.nv == 10, info
        assert info.nu == 1, info
        assert info.model_digest.startswith("sha256:")
        joint_names = [j["name"] for j in info.joints]
        assert "obj_free" in joint_names and "shoulder" in joint_names
        assert "elbow" in joint_names
        types = {j["name"]: j["type"] for j in info.joints}
        assert types["obj_free"] == "free"
        assert types["elbow"] == "ball"
        # 执行器→关节映射：shoulder_motor → shoulder（elbow 是被动
        # 关节——不得把 ball joint 说成 actuator）。
        assert info.actuators == [
            {"name": "shoulder_motor", "joint": "shoulder",
             "ctrlrange": [-1.0, 1.0]}
        ]
        passive = {j["name"] for j in info.joints} - {
            a["joint"] for a in info.actuators
        }
        assert "elbow" in passive
        # camera/site/传感器
        assert "overhead" in info.cameras
        assert "tip" in info.sites
        assert any(s.get("type") == "jointpos" for s in info.sensors)

    def test_no_gripper_invention(self, tmp_path) -> None:
        """无夹爪模型 → 不得出现夹爪声明（不从名字猜）。"""
        from rosclaw.sim.model_inspect import inspect_mjcf

        path = tmp_path / "fixture.xml"
        path.write_text(FIXTURE_MJCF, encoding="utf-8")
        info = inspect_mjcf(path)
        assert info.gripper is False


class TestNoDefaultRobot:
    def test_missing_model_identity_is_legacy_not_ur5e(self) -> None:
        """无 spec 且无模型身份记录 → legacy/unverified 标记或诚实
        拒绝——绝不静默默认 UR5e。"""
        import inspect

        from rosclaw.agentd import sim_render

        src = inspect.getsource(sim_render)
        # 默认机器人必须是"从记录推导"——裸字面量默认不再存在。
        assert 'robot_id = "ur5e"' not in src, (
            "回放仍存在裸 UR5e 默认（记录什么模型才用什么模型）"
        )


class TestPythonApi:
    def test_api_importable_and_composable(self, tmp_path) -> None:
        """六类入口存在且可组合：load_model → submit_simulation →
        read_trace → render → export。"""
        from rosclaw.sim import api

        for name in (
            "load_model", "observe", "submit_simulation",
            "read_trace", "render", "export",
        ):
            assert callable(getattr(api, name, None)), f"缺 api.{name}"

    def test_load_task_local_mjcf(self, tmp_path) -> None:
        """任务根内 MJCF 可加载并返回 model_ref + body_description
        （nq/nv/nu/站点）。"""
        from rosclaw.sim import api

        mjcf = tmp_path / "task_model.xml"
        mjcf.write_text(FIXTURE_MJCF, encoding="utf-8")
        ref, desc = api.load_model(mjcf, task_root=tmp_path)
        assert ref
        assert desc["nq"] == 12 and desc["nu"] == 1
        assert "tip" in desc["sites"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


class TestQuaternionContract:
    def test_pose_spec_uses_named_xyzw(self, tmp_path) -> None:
        """§6.2：位姿契约的对外四元数必须命名明确
        （orientation_xyzw / quaternion_xyzw）——禁止匿名四元素
        数组在边界自由流通。"""
        import json

        from rosclaw.agentd.sim_trajectory import SimTrajectoryService

        svc = SimTrajectoryService(tmp_path)
        plan = svc.generate_planar_path(
            shape="star5", center_m=[0.35, 0.25, 0.30], scale_m=0.05,
        )
        payload = svc.get_plan_payload(str(plan["plan_id"]))
        spec = payload.get("spec") or {}
        text = json.dumps(spec)
        # 显式命名的 xyzw 字段在场；不允许裸 "orientation": [w,x,y,z]
        # 无后缀字段（匿名四元素数组）。
        assert "orientation_xyzw" in text or "quaternion_xyzw" in text, text[:300]
        import re as _re

        assert not _re.search(r'"orientation"\s*:\s*\[', text), (
            "出现匿名 orientation 数组（必须 *_xyzw 命名）"
        )


class TestApiEndToEnd:
    def test_load_simulate_render_export(self, tmp_path) -> None:
        """六入口组合闭环：load_model → submit_simulation（完整
        状态）→ read_trace → render（子进程正确后端+时间驱动）
        → export（真实可解码 GIF）。"""
        from rosclaw.sim import api

        mjcf = tmp_path / "task_model.xml"
        mjcf.write_text(FIXTURE_MJCF, encoding="utf-8")
        ref, desc = api.load_model(mjcf, task_root=tmp_path)
        op = api.submit_simulation(
            ref, None, {"ctrl_series": [[0.3]] * 50}, 0.1,
            task_root=tmp_path,
        )
        trace = api.read_trace(op, task_root=tmp_path)
        assert len(trace["states"]) == 50
        assert len(trace["states"][0]["qpos"]) == 12  # 完整 nq
        assert len(trace["states"][0]["ctrl"]) == 1  # 完整 nu
        rop = api.render(
            op, {"camera": "top", "outputs": ["gif"], "fps": 8.0},
            task_root=tmp_path,
        )
        out = api.export(
            rop + "/" + rop + ".gif", tmp_path / "out.gif",
            task_root=tmp_path,
        )
        assert out.exists() and out.stat().st_size > 500
        # GIF 可解码（PIL 打开 + 帧数>1）。
        from PIL import Image

        with Image.open(out) as img:
            assert getattr(img, "n_frames", 1) >= 2
        # 渲染 receipt 标 agent-generated（不冒充受信验证）。
        import json

        receipt = json.loads(
            (tmp_path / "renders" / rop / "render_receipt.json").read_text()
        )
        assert receipt["evidence_level"] == "agent_generated_experiment"

    def test_cross_model_reference_rejected(self, tmp_path) -> None:
        """§6.3：跨模型引用（qpos 维度不符）直接拒绝——不静默
        截断或补零。"""
        from rosclaw.sim import api

        mjcf = tmp_path / "task_model.xml"
        mjcf.write_text(FIXTURE_MJCF, encoding="utf-8")
        ref, _ = api.load_model(mjcf, task_root=tmp_path)
        op = api.submit_simulation(
            ref, None, {"ctrl_series": [[0.3]] * 5}, 0.01,
            task_root=tmp_path,
        )
        # 篡改 states 维度 → 渲染必须拒绝（STATE_DIMENSION）。
        import json as _json

        record = tmp_path / "models" / f"{op}.json"
        payload = _json.loads(record.read_text())
        payload["states"][0]["qpos"] = [0.0, 0.0, 0.0]  # 3 != nq=12
        record.write_text(_json.dumps(payload))
        import pytest as _pt

        with _pt.raises(ValueError, match="STATE_DIMENSION"):
            api.render(op, {"outputs": ["gif"]}, task_root=tmp_path)
