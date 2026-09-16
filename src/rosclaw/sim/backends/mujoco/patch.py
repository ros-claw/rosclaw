"""MjSpec 结构化模型补丁（PR-MH2，ADR-0014，规格 §12.3/§8/§39）。

Maturity: experimental（ADR-0000 §4）。

P0 只开放 ``set`` 白名单；``add`` / ``remove`` / ``attach`` 显式
``MODEL_FIELD_UNSUPPORTED``（拓扑与命名空间语义在后续里程碑开放）。
值校验 fail closed：NaN/Inf、倒挂 range、越界 rgba、非法四元数
全部拒绝。母模型不可变——补丁在 ``MjSpec`` 上进行，编译失败不
触碰母对象。
"""

from __future__ import annotations

import math
from typing import Any

_TARGET_COLLECTIONS = {
    "joint": "joints",
    "geom": "geoms",
    "actuator": "actuators",
    "body": "bodies",
    "camera": "cameras",
}

#: P0 白名单（规格 §12.3.1）：(target_type, field)。
SET_WHITELIST: frozenset[tuple[str, str]] = frozenset(
    {
        ("joint", "damping"),
        ("joint", "range"),
        ("geom", "friction"),
        ("geom", "mass"),
        ("geom", "density"),
        ("geom", "rgba"),
        ("actuator", "kp"),
        ("actuator", "kv"),
        ("actuator", "ctrlrange"),
        ("body", "pos"),
        ("body", "quat"),
        ("camera", "pos"),
        ("camera", "quat"),
        ("option", "timestep"),
        ("option", "integrator"),
    }
)


def _finite_scalar(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"MODEL_PATCH_INVALID: {field} expects a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"MODEL_PATCH_INVALID: {field} must be finite, got {value!r}")
    return result


def _finite_vec(value: Any, field: str, lengths: tuple[int, ...]) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) not in lengths:
        raise ValueError(
            f"MODEL_PATCH_INVALID: {field} expects a vector of length {lengths}, got {value!r}"
        )
    return [_finite_scalar(v, field) for v in value]


def _ordered_pair(value: Any, field: str) -> list[float]:
    pair = _finite_vec(value, field, (2,))
    if pair[0] > pair[1]:
        raise ValueError(f"MODEL_PATCH_INVALID: {field} range inverted: {pair}")
    return pair


def _set_damping(element, value: Any) -> None:  # noqa: ANN001, ANN202
    scalar = _finite_scalar(value, "joint.damping")
    try:
        # MuJoCo 3.11 Python 绑定中 damping 是 shape-(3,) ndarray，
        # 标量直赋抛 TypeError——原位写 [0]。
        element.damping[0] = scalar
    except TypeError:
        element.damping = scalar


def _set_range(element, value: Any) -> None:  # noqa: ANN001, ANN202
    element.range[:] = _ordered_pair(value, "joint.range")


def _set_friction(element, value: Any) -> None:  # noqa: ANN001, ANN202
    vec = _finite_vec(value, "geom.friction", (1, 3))
    element.friction[:] = vec * 3 if len(vec) == 1 else vec


def _set_mass(element, value: Any) -> None:  # noqa: ANN001, ANN202
    mass = _finite_scalar(value, "geom.mass")
    if mass <= 0:
        raise ValueError(f"MODEL_PATCH_INVALID: geom.mass must be > 0, got {mass}")
    element.mass = mass


def _set_density(element, value: Any) -> None:  # noqa: ANN001, ANN202
    density = _finite_scalar(value, "geom.density")
    if density < 0:
        raise ValueError(f"MODEL_PATCH_INVALID: geom.density must be >= 0, got {density}")
    element.density = density


def _set_rgba(element, value: Any) -> None:  # noqa: ANN001, ANN202
    vec = _finite_vec(value, "geom.rgba", (4,))
    if any(v < 0.0 or v > 1.0 for v in vec):
        raise ValueError(f"MODEL_PATCH_INVALID: geom.rgba components must be in [0,1], got {vec}")
    element.rgba[:] = vec


def _servo_gain_guard(element, field: str) -> None:  # noqa: ANN001, ANN202
    """kp/kv 只映射到 FIXED 增益 + NONE/AFFINE 偏置的 position 类执行器。"""
    import mujoco

    if element.gaintype != mujoco.mjtGain.mjGAIN_FIXED:
        raise ValueError(
            f"MODEL_FIELD_UNSUPPORTED: {field} requires gaintype=fixed, got {element.gaintype}"
        )
    if element.biastype not in (mujoco.mjtBias.mjBIAS_NONE, mujoco.mjtBias.mjBIAS_AFFINE):
        raise ValueError(
            f"MODEL_FIELD_UNSUPPORTED: {field} requires biastype none/affine, got {element.biastype}"
        )


def _set_kp(element, value: Any) -> None:  # noqa: ANN001, ANN202
    import mujoco

    _servo_gain_guard(element, "actuator.kp")
    kp = _finite_scalar(value, "actuator.kp")
    element.gainprm[0] = kp
    if element.biastype == mujoco.mjtBias.mjBIAS_AFFINE:
        element.biasprm[1] = -kp


def _set_kv(element, value: Any) -> None:  # noqa: ANN001, ANN202
    import mujoco

    _servo_gain_guard(element, "actuator.kv")
    if element.biastype != mujoco.mjtBias.mjBIAS_AFFINE:
        raise ValueError("MODEL_FIELD_UNSUPPORTED: actuator.kv requires biastype=affine")
    element.biasprm[2] = -_finite_scalar(value, "actuator.kv")


def _set_ctrlrange(element, value: Any) -> None:  # noqa: ANN001, ANN202
    element.ctrlrange[:] = _ordered_pair(value, "actuator.ctrlrange")


def _set_pos(element, value: Any) -> None:  # noqa: ANN001, ANN202
    element.pos[:] = _finite_vec(value, "body/camera.pos", (3,))


def _set_quat(element, value: Any) -> None:  # noqa: ANN001, ANN202
    vec = _finite_vec(value, "body/camera.quat", (4,))
    norm = math.sqrt(sum(v * v for v in vec))
    if abs(norm - 1.0) > 1e-6:
        raise ValueError(f"MODEL_PATCH_INVALID: quat norm must be 1 (wxyz), got |q|={norm}")
    element.quat[:] = vec


def _set_timestep(spec, value: Any) -> None:  # noqa: ANN001, ANN202
    timestep = _finite_scalar(value, "option.timestep")
    if timestep <= 0:
        raise ValueError(f"MODEL_PATCH_INVALID: option.timestep must be > 0, got {timestep}")
    spec.option.timestep = timestep


def _set_integrator(spec, value: Any) -> None:  # noqa: ANN001, ANN202
    import mujoco

    members = {
        name.removeprefix("mjINT_").lower(): getattr(mujoco.mjtIntegrator, name)
        for name in dir(mujoco.mjtIntegrator)
        if name.startswith("mjINT_")
    }
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in members:
            raise ValueError(
                f"MODEL_PATCH_INVALID: unknown integrator {value!r} (known: {sorted(members)})"
            )
        spec.option.integrator = members[key]
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"MODEL_PATCH_INVALID: option.integrator expects int or name, got {value!r}"
        )
    if value not in {int(v) for v in members.values()}:
        raise ValueError(f"MODEL_PATCH_INVALID: unknown integrator value {value}")
    spec.option.integrator = mujoco.mjtIntegrator(value)


_SET_APPLIERS = {
    ("joint", "damping"): _set_damping,
    ("joint", "range"): _set_range,
    ("geom", "friction"): _set_friction,
    ("geom", "mass"): _set_mass,
    ("geom", "density"): _set_density,
    ("geom", "rgba"): _set_rgba,
    ("actuator", "kp"): _set_kp,
    ("actuator", "kv"): _set_kv,
    ("actuator", "ctrlrange"): _set_ctrlrange,
    ("body", "pos"): _set_pos,
    ("body", "quat"): _set_quat,
    ("camera", "pos"): _set_pos,
    ("camera", "quat"): _set_quat,
}


def _find_target(spec, target: Any):  # noqa: ANN001, ANN202
    if not isinstance(target, dict):
        raise ValueError(f"MODEL_PATCH_INVALID: target must be a mapping, got {target!r}")
    ttype = target.get("type")
    collection_name = _TARGET_COLLECTIONS.get(ttype) if isinstance(ttype, str) else None
    if collection_name is None:
        raise ValueError(f"MODEL_PATCH_INVALID: unknown target type {ttype!r}")
    name = target.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("MODEL_PATCH_INVALID: target.name must be a non-empty string")
    for element in getattr(spec, collection_name):
        if element.name == name:
            return element
    raise ValueError(f"MODEL_TARGET_NOT_FOUND: {target.get('type')}:{name}")


def apply_patches(spec, patches: list[dict[str, Any]]) -> None:  # noqa: ANN001
    """把补丁序列应用到 MjSpec（P0：仅 set 白名单）。

    全部错误 fail closed：MODEL_PATCH_INVALID / MODEL_TARGET_NOT_FOUND /
    MODEL_FIELD_UNSUPPORTED。编译由调用方负责（失败不触碰母模型）。
    """
    if not isinstance(patches, list) or not patches:
        raise ValueError("MODEL_PATCH_INVALID: patches must be a non-empty list")
    for patch in patches:
        if not isinstance(patch, dict):
            raise ValueError(f"MODEL_PATCH_INVALID: patch must be a mapping, got {patch!r}")
        op = patch.get("op")
        if op in ("add", "remove", "attach"):
            raise ValueError(
                f"MODEL_FIELD_UNSUPPORTED: op={op!r} 将在后续里程碑开放（P0 仅 set 白名单）"
            )
        if op != "set":
            raise ValueError(f"MODEL_PATCH_INVALID: unknown op {op!r}")
        field = patch.get("field")
        if not isinstance(field, str):
            raise ValueError(f"MODEL_PATCH_INVALID: field must be a string, got {field!r}")
        target = patch.get("target")
        ttype = target.get("type") if isinstance(target, dict) else None
        key = (ttype, field)
        if key not in SET_WHITELIST:
            raise ValueError(f"MODEL_FIELD_UNSUPPORTED: {ttype}.{field} not in P0 whitelist")
        if "value" not in patch:
            raise ValueError("MODEL_PATCH_INVALID: set requires value")
        if ttype == "option":
            applier = {"timestep": _set_timestep, "integrator": _set_integrator}[field]
            applier(spec, patch["value"])
            continue
        _SET_APPLIERS[key](_find_target(spec, target), patch["value"])
