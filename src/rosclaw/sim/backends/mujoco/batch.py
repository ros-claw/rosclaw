"""多模态批次 rollout（MH14，0916 优化 §十七-§十九，ADR-0014）。

Maturity: experimental（ADR-0000 §4）。

`mujoco.rollout` 原生批量：多个 homogeneous MjModel（shape 相同、
浮点参数不同——kp/damping/friction/mass sweep 的天然接口）+ 多个
MjData + native thread pool。**CPU batch 优先于 MJX**（与权威 CPU
truth 同域、无需 CPU/GPU agreement、更易 strict replay）。
实测：batch 轨迹与串行轨迹逐步一致。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def check_homogeneous(models: list[Any]) -> None:
    """homogeneous 校验：nq/nv/nu/nstate + 结构签名全等，
    否则 BATCH_NOT_HOMOGENEOUS（mujoco 的 compatibility 更严——
    同维度异结构也不兼容，必须提前拦截）。"""
    import mujoco

    if len(models) < 1:
        raise ValueError("BATCH_EMPTY: no models")
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    signature = (
        models[0].nq,
        models[0].nv,
        models[0].nu,
        mujoco.mj_stateSize(models[0], spec),
        tuple(int(t) for t in models[0].jnt_type),
        tuple(int(t) for t in models[0].actuator_trntype),
    )
    for index, model in enumerate(models[1:], start=1):
        other = (
            model.nq,
            model.nv,
            model.nu,
            mujoco.mj_stateSize(model, spec),
            tuple(int(t) for t in model.jnt_type),
            tuple(int(t) for t in model.actuator_trntype),
        )
        if other != signature:
            raise ValueError(
                f"BATCH_NOT_HOMOGENEOUS: models[0] != models[{index}] "
                f"(dimensions or joint/actuator signature differ)"
            )


def initial_vectors(models: list[Any]) -> np.ndarray:
    """每个模型 qpos0 + mj_forward 的 FULLPHYSICS 初始向量。"""
    import mujoco

    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    nstate = mujoco.mj_stateSize(models[0], spec)
    initial = np.zeros((len(models), nstate))
    for index, model in enumerate(models):
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        mujoco.mj_getState(model, data, initial[index], spec)
    return initial


def run_batch(
    models: list[Any],
    *,
    ctrl_rows: np.ndarray,
    record_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """执行批量 rollout，返回 (state_traj, sensordata_traj)。

    ctrl_rows: (nstep, nu) 或 (nbatch, nstep, nu)——open-loop 控制序列。
    """
    import mujoco
    import mujoco.rollout as rollout_lib

    check_homogeneous(models)
    datas = [mujoco.MjData(model) for model in models]
    initial = initial_vectors(models)
    if ctrl_rows.ndim == 2:
        ctrl = ctrl_rows[np.newaxis, :, :]
    else:
        ctrl = ctrl_rows
    try:
        state_traj, sensordata_traj = rollout_lib.rollout(models, datas, initial, ctrl)
    except ValueError as exc:
        raise ValueError(f"BATCH_NOT_HOMOGENEOUS: {exc}") from exc
    return state_traj[:, ::record_stride, :], sensordata_traj


def trajectory_to_states(
    model,
    state_traj: np.ndarray,  # noqa: ANN001
    ctrl_rows: np.ndarray,
    *,
    record_stride: int,
) -> list[dict[str, Any]]:
    """把 batch 轨迹转回 trace states 形态（与串行 rollout 记录一致）。

    FULLPHYSICS 布局：time(1) + qpos(nq) + qvel(nv) + act(na)。
    """
    nq, nv = model.nq, model.nv
    states = []
    for step, row in enumerate(state_traj):
        ctrl_index = min(step * record_stride, len(ctrl_rows) - 1)
        states.append(
            {
                "t": float(row[0]),
                "qpos": [float(v) for v in row[1 : 1 + nq]],
                "qvel": [float(v) for v in row[1 + nq : 1 + nq + nv]],
                "ctrl": [float(v) for v in ctrl_rows[ctrl_index]],
            }
        )
    return states
