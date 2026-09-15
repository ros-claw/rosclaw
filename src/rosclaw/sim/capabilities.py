"""MuJoCo 运行时能力探测（PR-MH0，ADR-0014）。

Maturity: experimental（ADR-0000 §4）。

原则：
- 真探测（属性 / 子模块 / import 检查），不做版本字符串比较；
- 必需能力缺失 → ``SIM_CAPABILITY_UNAVAILABLE`` fail-fast，绝不静默降级；
- 可选能力缺失只记录 ``details``，不改变语义；
- GL 探测只 import 子模块、不创建上下文（import 成功 ≠ 上下文可建，
  渲染可用性由后续渲染 PR 二次确认）；
- 模块级不 import mujoco（惰性加载，便于测试与无 mujoco 环境导入）。
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from collections.abc import Iterable
from datetime import UTC, datetime

from rosclaw.sim.contracts import SimulationBackendCapabilities

SIM_CAPABILITY_UNAVAILABLE = "SIM_CAPABILITY_UNAVAILABLE"

#: 模型不可变 + MjSpec 编辑再编译工作流的最小前提。
DEFAULT_REQUIRED: tuple[str, ...] = ("mjspec", "model_editing", "recompile")

ALL_CAPABILITIES: tuple[str, ...] = (
    "mjspec",
    "model_editing",
    "recompile",
    "rollout_multithreading",
    "egl",
    "osmesa",
    "glfw",
    "threadpool",
    "surfacevel",
    "pid_actuator",
    "mjx_available",
    "menagerie_available",
)


class SimCapabilityError(RuntimeError):
    """必需仿真能力缺失（fail closed）。"""

    code = SIM_CAPABILITY_UNAVAILABLE

    def __init__(self, missing: list[str], details: dict[str, str]) -> None:
        self.missing = list(missing)
        self.details = dict(details)
        super().__init__(
            f"{SIM_CAPABILITY_UNAVAILABLE}: missing required capabilities: "
            + ", ".join(self.missing)
        )


def _try_import(name: str) -> tuple[bool, str]:
    """尝试 import 子模块；任何异常（含 dlopen 失败）都记为不可用。"""
    try:
        importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 —— 探测面必须全捕获
        return False, f"import failed: {type(exc).__name__}: {exc}"
    return True, ""


_GL_LIBS = {
    "egl": ("EGL", "mujoco.egl"),
    "osmesa": ("OSMesa", "mujoco.osmesa"),
    "glfw": ("glfw", "mujoco.glfw"),
}


def _gl_backend_available(name: str) -> tuple[bool, str]:
    """GL 探测：模块 find_spec + 系统库存在性——**不 import**。

    实证教训（#547 CI）：import mujoco.glfw 会让后续渲染的 auto
    后端选择翻转为 glfw（test_wp3_scene_render 在 CI 由 egl 变
    glfw 而失败）。探测必须零副作用。
    """
    import ctypes.util

    lib_name, module_name = _GL_LIBS[name]
    if importlib.util.find_spec(module_name) is None:
        return False, f"{module_name} module not found"
    if ctypes.util.find_library(lib_name) is None:
        return False, f"lib{lib_name} not found"
    return True, ""


def _detect(mujoco) -> tuple[dict[str, bool], dict[str, str]]:  # noqa: ANN001
    caps: dict[str, bool] = {}
    details: dict[str, str] = {}

    mjspec_cls = getattr(mujoco, "MjSpec", None)
    caps["mjspec"] = mjspec_cls is not None
    if not caps["mjspec"]:
        details["mjspec"] = "mujoco.MjSpec not found"

    # 真实例化空 spec，验证可编辑面（worldbody / compile）。
    try:
        spec = mjspec_cls() if mjspec_cls is not None else None
        caps["model_editing"] = spec is not None and hasattr(spec, "worldbody")
        caps["recompile"] = spec is not None and hasattr(spec, "compile")
    except Exception as exc:  # noqa: BLE001
        caps["model_editing"] = False
        caps["recompile"] = False
        details["model_editing"] = f"MjSpec instantiation failed: {exc}"
    if not caps["model_editing"]:
        details.setdefault("model_editing", "MjSpec editing surface unavailable")
    if not caps["recompile"]:
        details.setdefault("recompile", "MjSpec.compile unavailable")

    rollout_ok, rollout_detail = _try_import("mujoco.rollout")
    if rollout_ok:
        import mujoco.rollout as rollout_mod

        rollout_ok = hasattr(rollout_mod, "rollout")
        if not rollout_ok:
            rollout_detail = "mujoco.rollout.rollout not found"
    caps["rollout_multithreading"] = rollout_ok
    if rollout_detail:
        details["rollout_multithreading"] = rollout_detail

    for name in ("egl", "osmesa", "glfw"):
        if name in ("egl", "osmesa") and sys.platform != "linux":
            caps[name] = False
            details[name] = f"not applicable on {sys.platform}"
            continue
        ok, detail = _gl_backend_available(name)
        caps[name] = ok
        if detail:
            details[name] = detail

    caps["threadpool"] = importlib.util.find_spec("mujoco.threadpool") is not None or hasattr(
        mujoco, "mj_bindThreadPool"
    )
    if not caps["threadpool"]:
        details["threadpool"] = "mujoco.threadpool module unavailable"

    # surfacevel 是 geom 属性（3.11 引入），不是传感器枚举——
    # 探测通道为 spec geom 的 surfacevel 字段（0915 §九实证修正）。
    try:
        probe_geom = mjspec_cls().worldbody.add_geom() if mjspec_cls is not None else None
        caps["surfacevel"] = probe_geom is not None and hasattr(probe_geom, "surfacevel")
    except Exception:  # noqa: BLE001
        caps["surfacevel"] = False
    if not caps["surfacevel"]:
        details["surfacevel"] = "geom surfacevel property not in this MuJoCo"

    # 待 MuJoCo 3.12+ 复核确切枚举名；3.13 实测 mjGAIN_PID 存在。
    caps["pid_actuator"] = hasattr(mujoco.mjtGain, "mjGAIN_PID")
    if not caps["pid_actuator"]:
        details["pid_actuator"] = "mjGAIN_PID not in this MuJoCo"

    mjx_ok, mjx_detail = _try_import("mujoco.mjx")
    caps["mjx_available"] = mjx_ok
    if mjx_detail:
        details["mjx_available"] = mjx_detail

    caps["menagerie_available"] = importlib.util.find_spec("mujoco_menagerie") is not None
    if not caps["menagerie_available"]:
        details["menagerie_available"] = "mujoco_menagerie package not installed"

    return caps, details


def probe_mujoco_capabilities(
    *,
    required: Iterable[str] = DEFAULT_REQUIRED,
) -> SimulationBackendCapabilities:
    """探测本机 MuJoCo 能力面；必需能力缺失时 fail-fast。

    返回已盖章（digest）的 ``SimulationBackendCapabilities``。
    """
    required_tuple = tuple(required)
    unknown = [name for name in required_tuple if name not in ALL_CAPABILITIES]
    if unknown:
        raise ValueError(f"unknown capability names: {unknown}")

    import mujoco

    caps, details = _detect(mujoco)
    missing = [name for name in required_tuple if not caps.get(name, False)]
    if missing:
        raise SimCapabilityError(missing, details)

    return SimulationBackendCapabilities(
        backend="mujoco",
        backend_version=str(getattr(mujoco, "__version__", "")),
        created_at=datetime.now(UTC).isoformat(),
        capabilities=caps,
        details=details,
        platform=sys.platform,
        gl_backend=os.environ.get("MUJOCO_GL", ""),
        missing_required=missing,
    ).with_digest()
