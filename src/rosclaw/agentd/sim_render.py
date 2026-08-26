"""原生离线场景渲染（WP-3，0823 审计 §四.WP-3）。

`simulation.render` 是 ROSClaw 正式能力，不是模型自写脚本：

- canonical MJCF（Sandbox/Resource Resolver 链）；
- trajectory state replay（rollout 落盘的 trajectory_states.json —
  动力学真实 qpos，不是命令回放）；
- 相机预设（follow/free/top）；
- 渲染后端 EGL→OSMesa→Xvfb 自动探测——**子进程隔离**（本机实证：
  进程内逐个试会被 glfw 递归初始化崩掉 libc++abi）；
- RenderReceipt：renderer build digest + input trace digest +
  backend + camera + frames；
- 离线：不访问 PyPI/网络。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

#: 探测顺序（经验证的最小到最简）。
_BACKEND_ORDER = ("egl", "osmesa", "xvfb")

_PROBE_SNIPPET = (
    "import os,mujoco;"
    "m=mujoco.MjModel.from_xml_string("
    "'<mujoco><worldbody><geom type=\"plane\" size=\"2 2 0.1\"/></worldbody></mujoco>');"
    "d=mujoco.MjData(m);r=mujoco.Renderer(m,64,64);"
    "r.update_scene(d);r.render();r.close();print('OK')"
)


def _probe_xvfb(*, timeout_sec: float = 30.0) -> tuple[bool, str]:
    """Xvfb 后端探测（P0-F）：必须经 xvfb-run 包一层虚拟显示 +
    MUJOCO_GL=glfw——MUJOCO_GL=xvfb 是无效值（0824 事故：手工
    Xvfb 成功、官方却判断不可用的根因）。无 xvfb-run 即不可用
    （诚实明细，不猜）。"""
    import shutil

    wrapper = shutil.which("xvfb-run")
    if wrapper is None:
        return False, "xvfb-run 不在 PATH（未安装 xvfb）"
    env = dict(os_environ(), MUJOCO_GL="glfw")
    try:
        proc = subprocess.run(
            [wrapper, "-a", sys.executable, "-c", _PROBE_SNIPPET],
            env=env, capture_output=True, timeout=timeout_sec,
        )
        if proc.returncode == 0 and b"OK" in proc.stdout:
            return True, "ok"
        return False, (
            proc.stderr.decode(errors="replace").strip().splitlines() or ["?"]
        )[-1][:200]
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}"[:200]


def probe_render_backend(
    *, timeout_sec: float = 30.0,
) -> tuple[str | None, dict[str, str]]:
    """EGL→OSMesa→Xvfb 子进程隔离探测（进程内探测会崩宿主——
    本机 glfw 实证；每个后端真实渲染一帧 smoke test）。返回
    (backend or None, 每后端明细)。"""
    detail: dict[str, str] = {}
    for backend in _BACKEND_ORDER:
        ok, note = _probe_backend(backend, timeout_sec=timeout_sec)
        detail[backend] = note
        if ok:
            return backend, detail
    return None, detail


def os_environ() -> dict[str, str]:
    import os

    return dict(os.environ)


def _renderer_build_digest() -> str:
    import mujoco

    payload = f"mujoco:{mujoco.__version__}|sim_render:v1"
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def render_scene_trace(
    home: Path,
    trace_id: str,
    *,
    camera: str = "follow",
    max_frames: int = 60,
    width: int = 640,
    height: int = 360,
    world_id: str = "empty",
    tool_ref: str = "",
) -> dict[str, Any]:
    """trace → 场景 GIF+MP4（真实 MuJoCo 离屏渲染）+ RenderReceipt。

    R0-3（0826 体验审计 §5.R0-3）：
    - 结构化 IPC：子进程写原子 result JSON 文件，stdout/stderr 只
      作诊断——空输出/噪声/rc=0 无结果都是稳定错误码，绝不向
      调用方泄漏裸 JSONDecodeError；
    - 后端降级只在 supervisor 内部执行一次（EGL→OSMesa→Xvfb
      顺序）——模型只收到最终结果；
    - world_id/tool_ref 来自 TaskSpec——声明了工具但资产不存在
      时 TOOL_ASSET_MISSING 诚实失败（不假装持笔）。
    """
    home = Path(home)
    trace_dir = home / "sim" / "traces" / trace_id
    trace_path = trace_dir / "trace.json"
    states_path = trace_dir / "trajectory_states.json"
    if not trace_path.exists() or not states_path.exists():
        raise ValueError(
            f"RENDER_INPUT_MISSING: trace {trace_id!r} 缺 trace.json/"
            "trajectory_states.json"
        )
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    states_doc = json.loads(states_path.read_text(encoding="utf-8"))
    states = states_doc.get("states") or []
    if not states:
        raise ValueError(f"RENDER_INPUT_MISSING: {trace_id} 无 trajectory states")
    # 血缘：states digest 必须与 trace.json 记录一致（篡改即拒）。
    states_digest = "sha256:" + hashlib.sha256(
        states_path.read_bytes()
    ).hexdigest()
    declared = str(trace.get("states_digest", ""))
    if declared and declared != states_digest:
        raise ValueError(
            f"RENDER_INPUT_DIGEST_MISMATCH: trajectory_states 与 trace 记录"
            f"不符（{declared[:19]}… != {states_digest[:19]}…）"
        )
    if tool_ref:
        _require_tool_asset(tool_ref)
    backend, probe_detail = probe_render_backend()
    if backend is None:
        raise ValueError(
            "RENDER_BACKEND_UNAVAILABLE: EGL/OSMesa/Xvfb 全部不可用——"
            + json.dumps(probe_detail, ensure_ascii=False)[:300]
        )
    # supervisor 内部一次降级：首选后端渲染失败 → 下一个后端再试
    # 一次；之后把最终错误抛出（调用方只见一次有语义的结果）。
    candidates = [backend, *[b for b in _BACKEND_ORDER if b != backend]]
    last_error: ValueError | None = None
    for attempt, candidate in enumerate(candidates[:2]):
        if attempt > 0:
            ok, note = _probe_backend(candidate)
            if not ok:
                last_error = ValueError(
                    f"RENDER_BACKEND_UNAVAILABLE: 降级后端 {candidate} "
                    f"不可用（{note}）"
                )
                continue
        try:
            return _render_attempt(
                home, trace_id, candidate,
                camera=camera, max_frames=max_frames,
                width=width, height=height,
                world_id=world_id, tool_ref=tool_ref,
            )
        except ValueError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def _probe_backend(backend: str, *, timeout_sec: float = 30.0) -> tuple[bool, str]:
    """单后端探测（egl/osmesa 直接；xvfb 经 xvfb-run+glfw）。"""
    if backend == "xvfb":
        return _probe_xvfb(timeout_sec=timeout_sec)
    env = dict(os_environ(), MUJOCO_GL=backend)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_SNIPPET],
            env=env, capture_output=True, timeout=timeout_sec,
        )
        if proc.returncode == 0 and b"OK" in proc.stdout:
            return True, "ok"
        return False, (
            proc.stderr.decode(errors="replace").strip().splitlines() or ["?"]
        )[-1][:200]
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}"[:200]


def _render_attempt(
    home: Path,
    trace_id: str,
    backend: str,
    *,
    camera: str,
    max_frames: int,
    width: int,
    height: int,
    world_id: str,
    tool_ref: str,
) -> dict[str, Any]:
    """单次渲染尝试（结构化 IPC：原子 result 文件为唯一结果
    通道——stdout/stderr 只作诊断日志）。"""
    import os
    import shutil

    result_path = (
        home / "sim" / "traces" / trace_id / f"{trace_id}-scene-result.json"
    )
    result_path.unlink(missing_ok=True)
    argv = [
        sys.executable, "-m", "rosclaw.agentd.sim_render",
        str(home), trace_id, camera, str(max_frames),
        str(width), str(height),
        "--world", world_id, "--tool", tool_ref,
        "--result", str(result_path),
    ]
    if backend == "xvfb":
        env = dict(os.environ, MUJOCO_GL="glfw")
        argv = [shutil.which("xvfb-run") or "xvfb-run", "-a", *argv]
    else:
        env = dict(os.environ, MUJOCO_GL=backend)
    proc = subprocess.run(argv, env=env, capture_output=True, timeout=600)
    if not result_path.exists():
        if proc.returncode != 0:
            tail = proc.stderr.decode(errors="replace")[-300:]
            raise ValueError(f"RENDER_FAILED: 子进程渲染失败: {tail}")
        raise ValueError(
            "RENDER_RESULT_MISSING: 子进程 rc=0 但未写 result 文件"
            f"（stdout 尾部: {proc.stdout.decode(errors='replace')[-200:]!r}）"
        )
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(
            f"RENDER_RESULT_CORRUPT: result 文件不是合法 JSON（{exc}）"
        ) from exc
    if not isinstance(result, dict) or not result.get("ok"):
        code = str(result.get("error_code", "RENDER_FAILED")) if isinstance(
            result, dict
        ) else "RENDER_FAILED"
        message = (
            str(result.get("message", "")) if isinstance(result, dict) else ""
        )
        raise ValueError(f"{code}: {message}"[:400])
    missing = [k for k in ("artifact", "artifacts", "receipt")
               if k not in result]
    if missing:
        raise ValueError(
            f"RENDER_RESULT_INCOMPLETE: result 缺字段 {missing}"
        )
    return result


def _require_tool_asset(tool_ref: str) -> None:
    """工具资产断言（TOOL_ASSET_MISSING 诚实失败——不假装持笔）。

    资产查找：packaged/repo zoo 的 tools/<name>/ 目录（当前无任何
    工具资产——声明即失败，等资产包落地后自然解锁）。
    """
    from rosclaw.runtime.eurdf_loader import _default_zoo_path

    name = tool_ref.removeprefix("tool:")
    candidate = _default_zoo_path().parent / "tools" / name
    if not candidate.is_dir():
        raise ValueError(
            f"TOOL_ASSET_MISSING: {tool_ref} 无权威工具资产（查找 "
            f"{candidate}）——不得假装持笔渲染"
        )


def _render_impl(
    home: Path,
    trace_id: str,
    *,
    camera: str,
    max_frames: int,
    width: int,
    height: int,
    world_id: str = "empty",
    tool_ref: str = "",
) -> dict[str, Any]:
    """子进程内真实渲染（MUJOCO_GL 已由父进程设定）。"""
    backend = os_environ().get("MUJOCO_GL", "")
    home = Path(home)
    trace_dir = home / "sim" / "traces" / trace_id
    trace_path = trace_dir / "trace.json"
    states_path = trace_dir / "trajectory_states.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    states = json.loads(states_path.read_text(encoding="utf-8"))["states"]
    states_digest = "sha256:" + hashlib.sha256(
        states_path.read_bytes()
    ).hexdigest()
    # canonical MJCF（与 rollout 同一资源链）；world 来自 TaskSpec
    # （tabletop/empty——empty 不得冒充桌面场景）。
    from rosclaw.sandbox.sandbox_api import Sandbox

    sandbox = Sandbox.create("ur5e", world_id, "mujoco")
    if not sandbox.has_physics:
        raise ValueError(f"RENDER_INPUT: canonical 模型不可用: {sandbox.load_error}")
    import mujoco
    import numpy as np

    model = sandbox.physics_model
    data = sandbox.physics_data
    renderer = mujoco.Renderer(model, height=height, width=width)
    # 相机预设：follow=跟踪工作区中心；top=俯视；free=固定斜视角。
    cam = mujoco.MjvCamera()
    positions = [s["qpos"] for s in states]
    n = len(positions)
    step = max(1, n // max_frames)
    frames_idx = list(range(0, n, step))[: max(2, n // step)]
    # 工作区取景（eef 路径包围盒中心）。
    trace_pts = trace.get("actual") or []
    if trace_pts:
        cx = sum(p["x"] for p in trace_pts) / len(trace_pts)
        cy = sum(p["y"] for p in trace_pts) / len(trace_pts)
        cz = sum(p["z"] for p in trace_pts) / len(trace_pts)
    else:
        cx, cy, cz = 0.35, 0.25, 0.30
    if camera == "top":
        cam.lookat[:] = [cx, cy, cz]
        cam.distance = 1.2
        cam.azimuth = 90.0
        cam.elevation = -89.0
    elif camera == "follow":
        cam.lookat[:] = [cx, cy, cz]
        cam.distance = 0.9
        cam.azimuth = 135.0
        cam.elevation = -25.0
    else:  # free
        cam.lookat[:] = [cx, cy, cz]
        cam.distance = 1.4
        cam.azimuth = 45.0
        cam.elevation = -30.0
    try:
        from PIL import Image

        images = []
        for idx in frames_idx:
            q = positions[idx]
            data.qpos[: int(model.nu)] = np.array(q[: int(model.nu)])
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            images.append(Image.fromarray(renderer.render()))
        out = trace_dir / f"{trace_id}-scene.gif"
        images[0].save(
            out, save_all=True, append_images=images[1:],
            duration=int(1000 / 12), loop=0,
        )
        # P0-F：官方渲染同时产出 MP4（imageio + imageio-ffmpeg
        # 自带静态 ffmpeg——离线，不需要系统 ffmpeg）。
        import imageio.v3 as iio
        import numpy as _np

        mp4 = trace_dir / f"{trace_id}-scene.mp4"
        frames_arr = [_np.asarray(img) for img in images]
        iio.imwrite(mp4, frames_arr, fps=12)
    finally:
        renderer.close()
        sandbox.close()
    receipt = {
        "schema_version": "rosclaw.render_receipt.v1",
        "backend": backend,
        "camera": camera,
        "world_id": world_id,
        "tool_ref": tool_ref,
        "renderer_build_digest": _renderer_build_digest(),
        "input_trace_digest": "sha256:" + hashlib.sha256(
            trace_path.read_bytes()
        ).hexdigest(),
        "states_digest": states_digest,
        "outputs": ["gif", "mp4"],
        "resource": trace.get("resource") or {},
    }
    (trace_dir / "render_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return {
        "ok": True,
        "artifact": {
            "path": str(out),
            "frames": len(images),
            "format": "gif",
            "bytes": out.stat().st_size,
            "evidence_level": "SIM_DYN_ROLLOUT",
        },
        "artifacts": {
            "gif": {
                "path": str(out),
                "frames": len(images),
                "format": "gif",
                "bytes": out.stat().st_size,
                "evidence_level": "SIM_DYN_ROLLOUT",
            },
            "mp4": {
                "path": str(mp4),
                "frames": len(images),
                "format": "mp4",
                "bytes": mp4.stat().st_size,
                "evidence_level": "SIM_DYN_ROLLOUT",
            },
        },
        "receipt": receipt,
    }


__all__ = ["probe_render_backend", "render_scene_trace"]


if __name__ == "__main__":
    # R0-3 结构化 IPC：结果写原子 result 文件（tmp+replace），
    # stdout/stderr 只作诊断——父进程不解析 stdout。
    import argparse

    _parser = argparse.ArgumentParser()
    _parser.add_argument("home")
    _parser.add_argument("trace_id")
    _parser.add_argument("camera")
    _parser.add_argument("max_frames", type=int)
    _parser.add_argument("width", type=int)
    _parser.add_argument("height", type=int)
    _parser.add_argument("--world", default="empty")
    _parser.add_argument("--tool", default="")
    _parser.add_argument("--result", required=True)
    _args = _parser.parse_args()
    _result_path = Path(_args.result)
    try:
        _payload: dict = _render_impl(
            Path(_args.home), _args.trace_id, camera=_args.camera,
            max_frames=_args.max_frames, width=_args.width,
            height=_args.height, world_id=_args.world, tool_ref=_args.tool,
        )
    except Exception as _exc:  # noqa: BLE001 - 失败也是结构化结果
        _message = str(_exc)[:300]
        _code = "RENDER_FAILED"
        if ":" in _message:
            _head = _message.split(":", 1)[0]
            if _head.replace("_", "").isalnum() and _head.isupper():
                _code, _message = _head, _message.split(":", 1)[1].strip()
        _payload = {"ok": False, "error_code": _code, "message": _message}
    _result_path.parent.mkdir(parents=True, exist_ok=True)
    _tmp = _result_path.with_suffix(".tmp")
    _tmp.write_text(json.dumps(_payload, ensure_ascii=False), encoding="utf-8")
    _tmp.replace(_result_path)  # 原子替换——父进程只认完整文件
    sys.exit(0 if _payload.get("ok") else 1)
