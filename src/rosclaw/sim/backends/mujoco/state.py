"""状态快照 / 恢复 / 分叉（PR-MH3，ADR-0014，规格 §9/§14）。

Maturity: experimental（ADR-0000 §4）。

快照保存 MuJoCo 可继续仿真的完整状态（time/qpos/qvel/act/ctrl/
mocap_pos/mocap_quat），并绑定 model_digest。恢复强约束：
跨模型 ``CROSS_MODEL_REF``、维度不符 ``STATE_DIMENSION`` /
``CTRL_DIMENSION``、非有限值 ``STATE_INVALID``——绝不 silent
truncate 或 pad。
"""

from __future__ import annotations

import math
from typing import Any

#: 快照字段（规格 §9）。
STATE_ARRAY_KEYS = ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat")


def capture_state(model, data) -> dict[str, Any]:  # noqa: ANN001
    """从 MjData 捕获完整状态（纯值，无 model 绑定——调用方补）。"""
    return {
        "time": float(data.time),
        "qpos": [float(v) for v in data.qpos],
        "qvel": [float(v) for v in data.qvel],
        "act": [float(v) for v in data.act],
        "ctrl": [float(v) for v in data.ctrl],
        "mocap_pos": [float(v) for v in data.mocap_pos.reshape(-1)],
        "mocap_quat": [float(v) for v in data.mocap_quat.reshape(-1)],
    }


def validate_state_values(model, values: dict[str, Any]) -> None:  # noqa: ANN001
    """维度与有限性校验（fail closed）。"""
    dims = {
        "qpos": model.nq,
        "qvel": model.nv,
        "act": model.na,
        "ctrl": model.nu,
        "mocap_pos": model.nmocap * 3,
        "mocap_quat": model.nmocap * 4,
    }
    for key, expected in dims.items():
        actual = values.get(key)
        if not isinstance(actual, list) or len(actual) != expected:
            code = "CTRL_DIMENSION" if key == "ctrl" else "STATE_DIMENSION"
            raise ValueError(
                f"{code}: {key} length {len(actual) if isinstance(actual, list) else '?'} != {expected}"
            )
    for key in STATE_ARRAY_KEYS:
        for v in values[key]:
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                raise ValueError(f"STATE_INVALID: {key} contains non-finite value {v!r}")
    time_value = values.get("time", 0.0)
    if not isinstance(time_value, (int, float)) or not math.isfinite(time_value):
        raise ValueError(f"STATE_INVALID: time must be finite, got {time_value!r}")


def apply_state(model, data, values: dict[str, Any]) -> None:  # noqa: ANN001
    """把校验过的状态写回 MjData 并 mj_forward。"""
    import mujoco

    validate_state_values(model, values)
    data.time = float(values.get("time", 0.0))
    for i, v in enumerate(values["qpos"]):
        data.qpos[i] = v
    for i, v in enumerate(values["qvel"]):
        data.qvel[i] = v
    for i, v in enumerate(values["act"]):
        data.act[i] = v
    for i, v in enumerate(values["ctrl"]):
        data.ctrl[i] = v
    if model.nmocap:
        for i, v in enumerate(values["mocap_pos"]):
            data.mocap_pos.reshape(-1)[i] = v
        for i, v in enumerate(values["mocap_quat"]):
            data.mocap_quat.reshape(-1)[i] = v
    mujoco.mj_forward(model, data)
