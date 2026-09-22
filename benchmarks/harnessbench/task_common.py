"""HarnessBench 任务公共件（BenchTask dataclass + v1 模型引用）。

tasks.py / tasks_v2.py 共同从这里取——不许互相 import（循环实证）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchTask:
    task_id: str
    category: str  # understanding|repair|experiment|honesty|vision|interaction|shadow|dynamic
    prompt: str  # 发给 Agent 的任务（零答案泄漏 + 交付契约）
    staged_files: dict[str, str] = field(default_factory=dict)  # 相对路径 → 内容
    oracle: dict = field(default_factory=dict)  # {"kind": ..., ...}——judge 按 kind 分发


# R02（真正的 Broken Model Doctor）：三处注入缺陷——隐式质量 +
# 初始穿透 + 伺服下垂（数值与 tests/sim 的 H02 不同，防语料记忆）。


# R02（真正的 Broken Model Doctor）：三处注入缺陷——隐式质量 +
# 初始穿透 + 伺服下垂（数值与 tests/sim 的 H02 不同，防语料记忆）。
DOCTOR_MODEL = """<mujoco model="sick_bot">
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="base" pos="0 0 0.035">
      <geom name="base_geom" type="box" size="0.05 0.05 0.05" mass="2.0"/>
      <body name="arm" pos="0.05 0 0">
        <joint name="elbow" type="hinge" axis="0 1 0" damping="0.02"/>
        <geom name="heavy" type="box" size="0.06 0.06 0.06" pos="0.06 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="elbow_servo" joint="elbow" kp="0.02" ctrlrange="-3 3"/>
  </actuator>
</mujoco>
"""


# E01（Parameter Scientist）：欠阻尼单关节——hold 明显抖动/超调。
SCIENTIST_MODEL = """<mujoco model="jitter_bot">
  <compiler autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <body name="link" pos="0 0 0.4">
      <joint name="hip" type="hinge" axis="0 1 0" damping="0.01"/>
      <geom name="link_g" type="capsule" size="0.04 0.25" mass="1.5"/>
    </body>
  </worldbody>
  <actuator>
    <position name="hip_servo" joint="hip" kp="8" ctrlrange="-3 3"/>
  </actuator>
</mujoco>
"""
