"""证据等级合约（总纲 §8.3，WP-P0-6）。

顺序即强度——任何 receipt/trace/task 结果必须声明其一；文案不得
超出等级能力：

- PLANNED：只有规划，未执行。
- COMMAND_REPLAY：把命令/规划点写入 sandbox state——能证明路径数据
  自洽，不能证明动力学或真实机械臂完成运动。
- KINEMATIC_SIM：运动学求解与关节/末端状态演化。
- PHYSICS_SIM：物理引擎（MuJoCo/Gazebo）返回的模拟状态。
- INDEPENDENT_SIM_OBSERVATION：独立观测器从 simulator state/传感器
  读取（不是 executor 自己复制）。
- REAL_COMMAND_ACCEPTED：真机控制栈接受目标（不证明完成）。
- REAL_OBSERVED：独立传感/状态反馈证明执行结果。
"""

from __future__ import annotations

EVIDENCE_LEVELS: tuple[str, ...] = (
    "PLANNED",
    "COMMAND_REPLAY",
    "KINEMATIC_SIM",
    "PHYSICS_SIM",
    "INDEPENDENT_SIM_OBSERVATION",
    "REAL_COMMAND_ACCEPTED",
    "REAL_OBSERVED",
)

#: 各等级的诚实局限文案（用户可见）。
LIMITATION_TEXT: dict[str, str] = {
    "PLANNED": "仅规划，未执行。",
    "COMMAND_REPLAY": (
        "trace 与计划来自同一 sandbox——能证明路径数据自洽，"
        "不能证明 MuJoCo 动力学或真实机械臂完成运动"
    ),
    "KINEMATIC_SIM": "运动学仿真结果，不含动力学/接触/真实传感。",
    "PHYSICS_SIM": "物理仿真结果，非真机证据。",
    "INDEPENDENT_SIM_OBSERVATION": "独立仿真观测，非真机证据。",
    "REAL_COMMAND_ACCEPTED": "真机已接受目标——完成需独立观测确认。",
    "REAL_OBSERVED": "真机独立观测确认。",
}
