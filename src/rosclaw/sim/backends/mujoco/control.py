"""Canonical ControlMapper（MH20-B，讨论总纲 §7/§十九）。

**所有执行路径写 ctrl 的唯一业务通道**：

```text
ControlSchema（control_channels 结构化真相）
      ↓
ControlMapper（按 (actuator, role) / joint 解析槽位）
      ↓
MjData.ctrl
```

MH10 已实证：actuator 个数 ≠ ctrl 维度（PID 一个 actuator 占
pos/vel/ff 多槽）——把 actuator 序号当 ctrl 槽位在多输入模型上
语义全错（第二个执行器的 joint_target 会写进第一个 PID 的 vel
槽）。architecture test（test_no_raw_ctrl_business_write）锁定：
业务模块不得出现 `data.ctrl[...]`。
"""

from __future__ import annotations

import math
from typing import Any


def channel_map_for(model, spec) -> dict[tuple[str, str], int]:  # noqa: ANN001
    """(actuator, role) → ctrl 槽位索引（ControlSchema 结构化真相）。"""
    from rosclaw.sim.backends.mujoco.inspect import _control_channels

    return {(c["actuator"], c["role"]): int(c["index"]) for c in _control_channels(model, spec)}


def _finite(value: Any, what: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"INTERACTION_PAYLOAD_INVALID: {what} must be a finite number")
    return float(value)


def write_setpoint(
    model,  # noqa: ANN001
    data,  # noqa: ANN001
    channel_map: dict[tuple[str, str], int],
    *,
    actuator: str,
    role: str,
    value: float,
) -> float:
    """按 (actuator, role) 写 ctrl 槽位（schema 缺失即
    CONTROLLER_SCHEMA_MISMATCH，绝不猜槽位）。"""
    slot = channel_map.get((actuator, role))
    if slot is None:
        raise ValueError(f"CONTROLLER_SCHEMA_MISMATCH: no control channel {actuator}:{role}")
    value = _finite(value, f"setpoint {actuator}:{role}")
    data.ctrl[slot] = value
    return value


def actuator_for_joint(model, joint_name: str) -> str:  # noqa: ANN001
    """关节 → 驱动它的执行器名（无执行器诚实 CAPABILITY_UNAVAILABLE）。"""
    import mujoco

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"INTERACTION_TARGET_NOT_FOUND: joint {joint_name!r}")
    for i in range(model.actuator_trnid.shape[0]):
        if (
            int(model.actuator_trntype[i]) == int(mujoco.mjtTrn.mjTRN_JOINT)
            and int(model.actuator_trnid[i][0]) == joint_id
        ):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                return name
    raise ValueError(
        f"CAPABILITY_UNAVAILABLE: joint {joint_name!r} has no actuator for joint_target"
    )


def resolve_position_role(channel_map: dict[tuple[str, str], int], actuator: str) -> str:
    """位置目标的角色解析：PID 多输入是 "pos"；普通单输入执行器
    的唯一通道是 "ctrl"（其位置目标语义）。两者都没有即
    CONTROLLER_SCHEMA_MISMATCH（绝不猜槽位）。"""
    if (actuator, "pos") in channel_map:
        return "pos"
    if (actuator, "ctrl") in channel_map:
        return "ctrl"
    raise ValueError(
        f"CONTROLLER_SCHEMA_MISMATCH: no position channel for {actuator!r}"
        "（既无 pos（PID）也无 ctrl（单输入）通道）"
    )


def write_joint_target(
    model,  # noqa: ANN001
    data,  # noqa: ANN001
    channel_map: dict[tuple[str, str], int],
    *,
    joint_name: str,
    value: float,
) -> tuple[str, float]:
    """joint_target 的唯一正确写法：关节 → 执行器 → 位置槽位
    （PID="pos" / 单输入="ctrl"）。返回 (actuator_name, applied_value)。"""
    actuator = actuator_for_joint(model, joint_name)
    role = resolve_position_role(channel_map, actuator)
    applied = write_setpoint(model, data, channel_map, actuator=actuator, role=role, value=value)
    return actuator, applied
