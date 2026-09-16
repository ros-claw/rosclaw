"""状态真值 v2（MH10，0916 优化 §二，ADR-0014）。

Maturity: experimental（ADR-0000 §4）。

核心原则：**MuJoCo 自己定义 MuJoCo 的 state truth**——不再人工
维护"完整状态字段列表"。默认 authoritative snapshot 为
``mjSTATE_INTEGRATION``（含 history / plugin_state / eq_active /
userdata / warmstart）；v1 手工字段快照自动标 ``LEGACY_PARTIAL``。
"""

from __future__ import annotations

FIDELITY_FULL_INTEGRATION = "FULL_INTEGRATION"
FIDELITY_FULL_PHYSICS = "FULL_PHYSICS"
FIDELITY_LEGACY_PARTIAL = "LEGACY_PARTIAL"

FIDELITIES = (FIDELITY_FULL_INTEGRATION, FIDELITY_FULL_PHYSICS, FIDELITY_LEGACY_PARTIAL)


def _spec_enum(fidelity: str):  # noqa: ANN202
    import mujoco

    if fidelity == FIDELITY_FULL_INTEGRATION:
        return mujoco.mjtState.mjSTATE_INTEGRATION
    if fidelity == FIDELITY_FULL_PHYSICS:
        return mujoco.mjtState.mjSTATE_FULLPHYSICS
    raise ValueError(f"STATE_FIDELITY_UNKNOWN: {fidelity!r}")


def spec_name(spec_value: int) -> str:
    import mujoco

    if spec_value == int(mujoco.mjtState.mjSTATE_INTEGRATION):
        return "MJSTATE_INTEGRATION"
    if spec_value == int(mujoco.mjtState.mjSTATE_FULLPHYSICS):
        return "MJSTATE_FULLPHYSICS"
    return f"MJSTATE({spec_value})"


def capture_state_v2(model, data, *, fidelity: str = FIDELITY_FULL_INTEGRATION):  # noqa: ANN001, ANN202
    """mj_getState 捕获完整状态向量（float64）。"""
    import mujoco
    import numpy as np

    spec = _spec_enum(fidelity)
    size = mujoco.mj_stateSize(model, spec)
    vector = np.zeros(size, dtype=np.float64)
    mujoco.mj_getState(model, data, vector, spec)
    return vector, int(spec)


def apply_state_v2(model, data, vector, spec_value: int) -> None:  # noqa: ANN001
    """mj_setState 恢复；维度/有限性 fail closed。"""
    import mujoco
    import numpy as np

    spec = mujoco.mjtState(spec_value)
    expected = mujoco.mj_stateSize(model, spec)
    if len(vector) != expected:
        raise ValueError(f"STATE_DIMENSION: state vector length {len(vector)} != {expected}")
    if not np.isfinite(vector).all():
        raise ValueError("STATE_INVALID: state vector contains non-finite values")
    mujoco.mj_setState(model, data, np.asarray(vector, dtype=np.float64), spec)
    # mj_setState 不重算派生量（sensordata/contact 等）——恢复后立即
    # forward，让延迟传感器从 history buffer 重建读数。
    mujoco.mj_forward(model, data)
