"""L04 预期接触推物（规格 §13.6）——agent 层真实驱动。

fixture：平面 pusher + 可滑动方块 + 目标区。模型交 ctrl_series.json
（pusher 电机序列），oracle 用同一夹具独立重放：
- 方块完全进入目标区（x ∈ [0.25, 0.45], |y| ≤ 0.08）并保持 1 s；
- 允许 pusher–block、block/pusher–table 接触；**其他障碍接触为零**
  （本夹具无其他障碍——约束即"只有这两类接触对"）；
- 方块运动来自仿真状态（重放所得，不是模型自述）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.eval.agent_tier import driver

PUSHER = """<mujoco model="pusher_l04">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="table" type="plane" size="2 2 0.1" friction="0.3 0.005 0.0001"/>
    <body name="pusher" pos="-0.3 0 0.02">
      <joint name="px" type="slide" axis="1 0 0"/>
      <geom name="pusher_geom" type="box" size="0.02 0.05 0.02" mass="2"
            friction="0.5 0.005 0.0001"/>
    </body>
    <body name="block" pos="0 0 0.02">
      <joint name="bx" type="slide" axis="1 0 0" damping="3.0"/>
      <joint name="by" type="slide" axis="0 1 0" damping="3.0"/>
      <geom name="block_geom" type="box" size="0.02 0.02 0.02" mass="0.3"
            friction="0.4 0.005 0.0001"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="push" joint="px" gear="1" ctrlrange="-5 5"/>
  </actuator>
</mujoco>"""

PROMPT = """工作目录里有 pusher.xml（平面推物场景：pusher 可沿 x 被电机驱动，block 可在桌面滑动）。

任务：把 block 推到目标区域（x ∈ [0.25, 0.45]，|y| ≤ 0.08），并在目标区内停住至少 1 秒。

要求：
1. 在 ctrl_series.json 里给出 pusher 电机的控制序列，格式严格为：
   {"dt": 每步秒数, "nu": 1, "series": [[力], [力], ...]}
2. 自行仿真验证后再交——我们会用同一模型独立重放你的序列：
   方块完全进区并保持 1 秒才算完成；只允许 pusher–block、
   方块/pusher–桌面接触，不得引入别的碰撞。
3. 完成后简述策略与实测末态位置。"""


def _oracle(run: driver.AgentRun) -> None:
    import mujoco

    series_path = run.ws / "ctrl_series.json"
    assert series_path.exists(), "模型未交 ctrl_series.json（见 PTY 日志）"
    dt, series = driver.load_ctrl_series(series_path)
    model = mujoco.MjModel.from_xml_string(PUSHER)
    data = mujoco.MjData(model)
    allowed = {
        frozenset(("pusher_geom", "block_geom")),
        frozenset(("table", "block_geom")),
        frozenset(("table", "pusher_geom")),
    }
    states: list[tuple[float, float, float]] = []
    forbidden: list[tuple] = []
    for row in series:
        assert len(row) == 1, f"控制行维度 {len(row)} != 1"
        assert abs(row[0]) <= 5.0 + 1e-9, "电机力越限幅 ±5"
        data.ctrl[0] = row[0]
        steps = max(1, int(round(dt / model.opt.timestep)))
        for _ in range(steps):
            mujoco.mj_step(model, data)
            for i in range(data.ncon):
                c = data.contact[i]
                g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom{c.geom1}"
                g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom{c.geom2}"
                if frozenset((g1, g2)) not in allowed:
                    forbidden.append((g1, g2, float(c.dist)))
        states.append((float(data.time), float(data.qpos[1]), float(data.qpos[2])))
    assert not forbidden, f"出现非允许接触: {forbidden[:3]}"
    # 进区 + 保持 1s：末段连续 1s 在区内。
    bx, by = states[-1][1], states[-1][2]
    assert 0.25 <= bx <= 0.45 and abs(by) <= 0.08, (
        f"方块末态 ({bx:.3f}, {by:.3f}) 不在目标区"
    )
    hold = [s for s in states if 0.25 <= s[1] <= 0.45 and abs(s[2]) <= 0.08]
    assert hold and (states[-1][0] - hold[0][0]) >= 1.0 - dt, (
        f"方块在区内保持不足 1s（末态在区 {states[-1][0] - hold[0][0]:.2f}s）"
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not (driver.has_key() and driver.has_runtime()),
    reason="NOT_RUN: 无真实 key/Node——不合成冒充",
)
class TestL04ContactPushAgent:
    def test_push_block_to_target(self, tmp_path: Path) -> None:
        run = driver.AgentRun(tmp_path, settle_timeout=1200)
        try:
            run.run(PROMPT, files={"pusher.xml": PUSHER})
        finally:
            run.stop()
        _oracle(run)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
