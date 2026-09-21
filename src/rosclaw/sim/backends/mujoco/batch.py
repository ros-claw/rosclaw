"""多模态批次 rollout（MH14，0916 优化 §十七-§十九，ADR-0014）。

Maturity: experimental（ADR-0000 §4）。

`mujoco.rollout` 原生批量：多个 homogeneous MjModel（shape 相同、
浮点参数不同——kp/damping/friction/mass sweep 的天然接口）+ 多个
MjData + native thread pool。**CPU batch 优先于 MJX**（与权威 CPU
truth 同域、无需 CPU/GPU agreement、更易 strict replay）。
实测：batch 轨迹与串行轨迹逐步一致。
"""

from __future__ import annotations

from typing import Any

import numpy as np


def check_homogeneous(models: list[Any]) -> None:
    """homogeneous 校验（MH20-A §6 升级版）。

    结构签名：nq/nv/nu/nstate + jnt_type/actuator_trntype 全等
    （mujoco 的 compatibility 更严——同维度异结构也不兼容，必须
    提前拦截）→ 违反即 BATCH_NOT_HOMOGENEOUS。

    **语义签名（新增）**：timestep/integrator/solver/gain/bias 类型
    全等——500 steps × dt=0.002 与 × dt=0.001 不是同一个实验，
    违反即 BATCH_SEMANTICS_INCOMPATIBLE（调用方诚实串行回退，
    不得混入同一批比较）。"""
    import mujoco

    if len(models) < 1:
        raise ValueError("BATCH_EMPTY: no models")
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS

    def signature(model: Any) -> tuple:
        return (
            model.nq,
            model.nv,
            model.nu,
            mujoco.mj_stateSize(model, spec),
            tuple(int(t) for t in model.jnt_type),
            tuple(int(t) for t in model.actuator_trntype),
            tuple(int(t) for t in model.actuator_gaintype),
            tuple(int(t) for t in model.actuator_biastype),
        )

    def semantics(model: Any) -> tuple:
        return (
            float(model.opt.timestep),
            int(model.opt.integrator),
            int(model.opt.solver),
        )

    base_sig = signature(models[0])
    base_sem = semantics(models[0])
    for index, model in enumerate(models[1:], start=1):
        if signature(model) != base_sig:
            raise ValueError(
                f"BATCH_NOT_HOMOGENEOUS: models[0] != models[{index}] "
                f"(dimensions or joint/actuator signature differ)"
            )
        if semantics(model) != base_sem:
            raise ValueError(
                f"BATCH_SEMANTICS_INCOMPATIBLE: models[0] != models[{index}] "
                f"(timestep/integrator/solver differ——同 steps 不同物理时间)"
            )


def initial_vectors(models: list[Any], vectors: Any | None = None) -> np.ndarray:
    """每个模型的 FULLPHYSICS 初始向量。

    vectors 为 None：各模型 qpos0 + mj_forward（默认初态）。
    vectors 给定（MH20-A）：直接使用调用方提供的 FULLPHYSICS 向量
    （transplant 后的 branch state——并行与串行同一实验起点）。"""
    import mujoco

    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    nstate = mujoco.mj_stateSize(models[0], spec)
    initial = np.zeros((len(models), nstate))
    if vectors is not None:
        if len(vectors) != len(models):
            raise ValueError(
                f"BATCH_STATE_COUNT_MISMATCH: {len(vectors)} vectors != {len(models)} models"
            )
        for index, vector in enumerate(vectors):
            if len(vector) != nstate:
                raise ValueError(
                    f"BATCH_STATE_DIMENSION: vectors[{index}] size {len(vector)} != {nstate}"
                )
            initial[index] = np.asarray(vector, dtype=float)
        return initial
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
    initial: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """执行批量 rollout，返回 (state_traj, sensordata_traj)。

    ctrl_rows: (nstep, nu) 或 (nbatch, nstep, nu)——open-loop 控制序列。
    initial: 可选调用方提供的 FULLPHYSICS 初始向量（MH20-A）。
    """
    import mujoco
    import mujoco.rollout as rollout_lib

    check_homogeneous(models)
    datas = [mujoco.MjData(model) for model in models]
    initial_vectors_ = initial if initial is not None else initial_vectors(models)
    ctrl = ctrl_rows[np.newaxis, :, :] if ctrl_rows.ndim == 2 else ctrl_rows
    try:
        state_traj, sensordata_traj = rollout_lib.rollout(models, datas, initial_vectors_, ctrl)
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
