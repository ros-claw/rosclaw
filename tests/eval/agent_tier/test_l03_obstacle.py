"""L03 避障与失败后修正（规格 §13.5）——agent 层真实驱动。

fixture：双连杆平面臂 + 障碍球恰好挡在直线 q 插值路径上
（可行性已验证：q-straight 净空 −0.119；绕行净空 +0.067）。
模型交 ctrl_series.json（关节位置目标序列——位置伺服式电机），
oracle 独立重放判定：
- 末态 eef 距目标 ≤10 mm（该夹具工程目标，不泛化）；
- 全程与障碍球的接触为零 + 最小净空 ≥20 mm；
- 模型应关注失败事实并改计划（prompt 明确告知障碍存在——
  我们不检查它的"修正过程"，只检查最终环境结局）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.eval.agent_tier import driver

ARM = """<mujoco model="arm2_l03">
  <option timestep="0.002"/>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" size="0.01 0.15" pos="0 0.15 0" mass="0.5"/>
      <body name="link2" pos="0 0.3 0">
        <joint name="j2" type="hinge" axis="0 0 1"/>
        <geom type="capsule" size="0.008 0.15" pos="0 0.15 0" mass="0.3"/>
        <site name="eef" pos="0 0.3 0"/>
      </body>
    </body>
    <geom name="obstacle" type="sphere" size="0.12" pos="-0.531 0.215 0"/>
  </worldbody>
  <actuator>
    <position name="m1" joint="j1" kp="30" ctrlrange="-3 3"/>
    <position name="m2" joint="j2" kp="30" ctrlrange="-3 3"/>
  </actuator>
</mujoco>"""

PROMPT = """工作目录里有 arm.xml（双连杆平面臂，关节 j1/j2 是位置伺服（kp=30，目标限幅 ±3 rad）——ctrl 即关节目标角）。

场景事实：工作空间内有一个障碍球（在 arm.xml 里，请自己读取理解其位置与大小）。

任务：把末端（site "eef"）从初始位形（q=0）移动到目标点 (-0.35, 0.35)，全程不得接触障碍球，且与障碍球表面保持至少 20mm 净空。

注意：关节直线插值会撞障碍——需要绕行。你可以多次仿真试验：每次失败的真实碰撞事实就在仿真里，分析它再改计划。

要求：
1. 在 ctrl_series.json 里给出关节目标序列，格式严格为：
   {"dt": 每步秒数, "nu": 2, "series": [[j1, j2], ...]}
2. 自行仿真验证后再交——我们会用同一模型独立重放：
   末端误差 ≤10mm、障碍接触为零、最小净空 ≥20mm 才算完成。
3. 完成后简述你的绕行思路与实测指标。"""


def _oracle(run: driver.AgentRun) -> None:
    import mujoco
    import numpy as np

    series_path = run.ws / "ctrl_series.json"
    assert series_path.exists(), "模型未交 ctrl_series.json（见 PTY 日志）"
    dt, series = driver.load_ctrl_series(series_path)
    model = mujoco.MjModel.from_xml_string(ARM)
    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef")
    obs = np.array([-0.531, 0.215])
    target = np.array([-0.35, 0.35])
    min_clearance = float("inf")
    obstacle_contacts = 0
    for row in series:
        assert len(row) == 2, f"控制行维度 {len(row)} != 2"
        assert all(abs(v) <= 3.0 + 1e-9 for v in row), "关节目标越限幅 ±3"
        data.ctrl[:] = row
        steps = max(1, int(round(dt / model.opt.timestep)))
        for _ in range(steps):
            mujoco.mj_step(model, data)
            for i in range(data.ncon):
                c = data.contact[i]
                g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
                g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
                if "obstacle" in (g1, g2):
                    obstacle_contacts += 1
            # 净空：eef 与 link2 中点到障碍表面（近似——夹具小步长
            # 下段间穿越由接触系统兜底）。
            for p in (data.site_xpos[sid][:2], data.xpos[2][:2]):
                d = float(np.linalg.norm(p - obs)) - 0.12
                min_clearance = min(min_clearance, d)
    # 伺服滞后补偿：在末行目标上保持 1s 再量末端误差（位置伺服
    # 的收敛时间——参考解 kp=30 下约 50mm 瞬态滞后）。
    for _ in range(int(1.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        for i in range(data.ncon):
            c = data.contact[i]
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
            if "obstacle" in (g1, g2):
                obstacle_contacts += 1
        for p in (data.site_xpos[sid][:2], data.xpos[2][:2]):
            min_clearance = min(
                min_clearance, float(np.linalg.norm(p - obs)) - 0.12,
            )
    final_err = float(np.linalg.norm(data.site_xpos[sid][:2] - target))
    assert obstacle_contacts == 0, f"与障碍球接触 {obstacle_contacts} 次"
    assert min_clearance >= 0.02, (
        f"最小净空 {min_clearance * 1000:.0f}mm < 20mm"
    )
    assert final_err <= 0.01, f"末端误差 {final_err * 1000:.1f}mm > 10mm"


@pytest.mark.slow
@pytest.mark.skipif(
    not (driver.has_key() and driver.has_runtime()),
    reason="NOT_RUN: 无真实 key/Node——不合成冒充",
)
class TestL03ObstacleRetryAgent:
    def test_avoid_obstacle_reach_target(self, tmp_path: Path) -> None:
        run = driver.AgentRun(tmp_path, settle_timeout=1500)
        try:
            run.run(PROMPT, files={"arm.xml": ARM})
        finally:
            run.stop()
        _oracle(run)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
