"""L06 cart-pole 平衡（规格 §13.8）——agent 层真实驱动。

fixture：近直立 cart-pole（质量/杆长在夹具文件里，模型自己读；
两个参数变体）。模型写本地控制器（接口 control(state)->力），
oracle 用**我们的 mujoco + 模型的控制器**重跑 10 s 判定：
- t>1 s 后 |角度| ≤ 5°；车体不越轨（|x| ≤ 0.9 m，轨道 ±1.0 m）；
- 角度曲线数据与视频存在且视频可解码非空白；
- 控制失败必须如实记（不返回预制平衡演示——本测试就是判定）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.eval.agent_tier import driver

CARTPOLE = """<mujoco model="cartpole_l06">
  <option timestep="0.002"/>
  <worldbody>
    <body name="cart" pos="0 0 0.1">
      <joint name="slide" type="slide" axis="1 0 0" range="-1.0 1.0"/>
      <geom type="box" size="0.05 0.03 0.02" mass="{cart_mass}"/>
      <body name="pole" pos="0 0 0.02">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.006 {half_len}" pos="0 0 {half_len}"
              mass="{pole_mass}"/>
      </body>
    </body>
    <camera name="side" pos="0 -1.2 0.2" euler="90 0 0"/>
  </worldbody>
  <actuator>
    <motor name="force" joint="slide" gear="1" ctrlrange="-10 10"/>
  </actuator>
</mujoco>
"""

# 变体（预先验证可解——小车 1.0kg/杆 0.2kg·0.4m 与 1.2kg/杆 0.15kg·0.3m）
VARIANTS = [
    {"id": "m1_l04", "cart_mass": "1.0", "pole_mass": "0.2", "half_len": "0.2",
     "force_limit": 10.0},
    {"id": "m12_l03", "cart_mass": "1.2", "pole_mass": "0.15",
     "half_len": "0.15", "force_limit": 10.0},
]

PROMPT = """工作目录里有 cartpole.xml（cart-pole 模型：参数就在文件里，请自己读取理解）。

任务：让摆杆在近直立附近保持平衡 10 秒。

要求：
1. 在 cartpole_controller.py 里实现一个本地控制器，接口严格为：
   def control(state: dict) -> float
   state = {"theta": 摆角rad（0=直立）, "theta_dot": 角速度, "x": 小车位置, "x_dot": 小车速度}
   返回值是电机力（限幅 ±10 N）。
   （控制循环必须本地确定性运行——不允许在控制循环里逐步调用大模型。）
2. 用你的控制器跑 10 秒仿真（初始 theta=0.05 rad, 其余 0），产出：
   - angle_curve.csv：每行 t,theta,x（整个 10 秒）；
   - balance.mp4 或 balance.gif：仿真过程视频（可自选渲染方式）。
3. 完成后直接简述你的控制器思路和实测指标。"""


def _oracle(run: driver.AgentRun, variant: dict) -> None:
    import mujoco
    import numpy as np

    ctrl_path = run.ws / "cartpole_controller.py"
    assert ctrl_path.exists(), (
        "模型未交 cartpole_controller.py（见 PTY 日志）"
    )
    control = driver.load_controller(ctrl_path)
    model = mujoco.MjModel.from_xml_string(CARTPOLE.format(**variant))
    data = mujoco.MjData(model)
    data.qpos[1] = 0.05  # theta
    steps = int(10.0 / model.opt.timestep)
    series: list[tuple[float, float, float]] = []
    for _ in range(steps):
        state = {
            "theta": float(data.qpos[1]),
            "theta_dot": float(data.qvel[1]),
            "x": float(data.qpos[0]),
            "x_dot": float(data.qvel[0]),
        }
        force = float(np.clip(control(state), -10.0, 10.0))
        assert abs(force) <= 10.0 + 1e-9, "控制器越限幅"
        data.ctrl[0] = force
        mujoco.mj_step(model, data)
        series.append((float(data.time), float(data.qpos[1]), float(data.qpos[0])))
    tail = [s for s in series if s[0] > 1.0]
    max_angle = max(abs(s[1]) for s in tail)
    max_x = max(abs(s[2]) for s in tail)
    assert max_angle <= 5.0 * 3.14159 / 180.0, (
        f"t>1s 最大摆角 {max_angle * 180 / 3.14159:.1f}° > 5°"
    )
    assert max_x <= 0.9, f"小车越轨 |x|={max_x:.2f} > 0.9"
    # 模型自产数据/视频存在且视频可解码非空白。
    assert (run.ws / "angle_curve.csv").exists(), "缺 angle_curve.csv"
    videos = list(run.ws.glob("balance.*"))
    assert videos, "缺平衡视频"
    import imageio.v3 as iio

    frames = list(iio.imiter(str(videos[0])))
    assert len(frames) >= 10
    assert float(np.asarray(frames[len(frames) // 2]).std()) > 1.0, (
        "视频全空白"
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not (driver.has_key() and driver.has_runtime()),
    reason="NOT_RUN: 无真实 key/Node——不合成冒充",
)
class TestL06CartpoleAgent:
    @pytest.mark.parametrize("variant", VARIANTS, ids=[v["id"] for v in VARIANTS])
    def test_balance_10s(self, tmp_path: Path, variant: dict) -> None:
        run = driver.AgentRun(tmp_path, settle_timeout=1200)
        try:
            run.run(PROMPT, files={"cartpole.xml": CARTPOLE.format(**variant)})
        finally:
            run.stop()
        _oracle(run, variant)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
