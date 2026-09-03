"""OS 级隔离与图形后端探测（0902 审计 R1-c，§5.3）。

与 firstboot.doctor 的 runtime.sandbox（MuJoCo 物理沙箱）不同——
这里探测的是 shell 执行的 OS 隔离（bwrap/容器）与渲染图形后端
（OSMesa/EGL）。§5.3：启动前一次探测 + 真实 smoke test，无可用
隔离时任务开始前提示一次，而不是运行到一半甩给用户。

探测结果落盘 <home>/agent/os-isolation.json，chat 会话启动时消费
（一次性提示），doctor 输出给结论+修复命令。
"""

from __future__ import annotations

import ctypes.util
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

_BWRAP_SMOKE_TIMEOUT_S = 5.0


def _bwrap_smoke(bwrap_path: str) -> tuple[bool, str]:
    """真实 smoke：present 不等于 usable——云 VM 常见 user namespace
    受限（bwrap 装了但 --unshare-all 直接失败）。"""
    try:
        proc = subprocess.run(
            [bwrap_path, "--unshare-all", "--ro-bind", "/usr", "/usr",
             "--", "true"],
            capture_output=True, text=True, timeout=_BWRAP_SMOKE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return False, "smoke 超时"
    except OSError as exc:
        return False, str(exc)
    if proc.returncode == 0:
        return True, "smoke ok"
    detail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return False, detail[-1][:200] if detail else f"exit {proc.returncode}"


def _graphics_backend() -> str | None:
    """headless 渲染后端：MUJOCO_GL 显式指定 > OSMesa > EGL。

    （CI 实证：装 mujoco ≠ 能渲染——libosmesa6 缺失时渲染崩。）
    """
    env = (os.environ.get("MUJOCO_GL") or "").strip().lower()
    if env in ("osmesa", "egl", "glfw"):
        return env
    if ctypes.util.find_library("OSMesa"):
        return "osmesa"
    if ctypes.util.find_library("EGL"):
        return "egl"
    return None


def _container_runtime() -> str | None:
    for name in ("docker", "podman"):
        path = shutil.which(name)
        if path:
            return path
    return None


def probe_os_isolation() -> dict:
    """一次探测（§5.3）——bwrap 存在性+smoke、容器后端、图形后端。"""
    bwrap_path = shutil.which("bwrap")
    smoke_ok, smoke_detail = (
        _bwrap_smoke(bwrap_path) if bwrap_path else (False, "bwrap 未安装")
    )
    graphics = _graphics_backend()
    container = _container_runtime()
    return {
        "checked_at": datetime.now(UTC).isoformat(),
        "bwrap": {
            "path": bwrap_path,
            "smoke_ok": smoke_ok,
            "detail": smoke_detail,
        },
        "container": container,
        "graphics": graphics,
        # shell 强隔离可用 = bwrap 装且 smoke 过（容器 backend 是后续
        # 可选项，当前正式路径只有 bwrap）。
        "isolation_ready": bool(bwrap_path and smoke_ok),
    }


def write_readiness(home: Path, probe: dict) -> Path:
    out = home / "agent" / "os-isolation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(probe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return out


def probe_and_persist(home: Path) -> dict:
    probe = probe_os_isolation()
    write_readiness(home, probe)
    return probe
