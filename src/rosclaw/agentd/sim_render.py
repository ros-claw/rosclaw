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


def probe_render_backend(
    *, timeout_sec: float = 30.0,
) -> tuple[str | None, dict[str, str]]:
    """EGL→OSMesa→Xvfb 子进程隔离探测（进程内探测会崩宿主——
    本机 glfw 实证）。返回 (backend or None, 每后端明细)。"""
    detail: dict[str, str] = {}
    for backend in _BACKEND_ORDER:
        env = dict(os_environ(), MUJOCO_GL=backend)
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _PROBE_SNIPPET],
                env=env, capture_output=True, timeout=timeout_sec,
            )
            if proc.returncode == 0 and b"OK" in proc.stdout:
                detail[backend] = "ok"
                return backend, detail
            detail[backend] = (
                proc.stderr.decode(errors="replace").strip().splitlines() or ["?"]
            )[-1][:200]
        except (subprocess.TimeoutExpired, OSError) as exc:
            detail[backend] = f"{type(exc).__name__}: {exc}"[:200]
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
) -> dict[str, Any]:
    """trace → 场景 GIF（真实 MuJoCo 离屏渲染）+ RenderReceipt。"""
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
    backend, probe_detail = probe_render_backend()
    if backend is None:
        raise ValueError(
            "RENDER_BACKEND_UNAVAILABLE: EGL/OSMesa/Xvfb 全部不可用——"
            + json.dumps(probe_detail, ensure_ascii=False)[:300]
        )
    # GL 平台在 mujoco import 时初始化——MUJOCO_GL 必须在子进程首
    # 次 import 前设定；渲染在子进程执行（GL 崩不跨进程伤宿主）。
    import os

    env = dict(os.environ, MUJOCO_GL=backend)
    proc = subprocess.run(
        [
            sys.executable, "-m", "rosclaw.agentd.sim_render",
            str(home), trace_id, camera, str(max_frames),
            str(width), str(height),
        ],
        env=env, capture_output=True, timeout=600,
    )
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace")[-300:]
        raise ValueError(f"RENDER_FAILED: 子进程渲染失败: {tail}")
    result = json.loads(proc.stdout.decode())
    return result


def _render_impl(
    home: Path,
    trace_id: str,
    *,
    camera: str,
    max_frames: int,
    width: int,
    height: int,
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
    # canonical MJCF（与 rollout 同一资源链）。
    from rosclaw.sandbox.sandbox_api import Sandbox

    sandbox = Sandbox.create("ur5e", "empty", "mujoco")
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
    finally:
        renderer.close()
        sandbox.close()
    receipt = {
        "schema_version": "rosclaw.render_receipt.v1",
        "backend": backend,
        "camera": camera,
        "renderer_build_digest": _renderer_build_digest(),
        "input_trace_digest": "sha256:" + hashlib.sha256(
            trace_path.read_bytes()
        ).hexdigest(),
        "states_digest": states_digest,
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
        "receipt": receipt,
    }


__all__ = ["probe_render_backend", "render_scene_trace"]


if __name__ == "__main__":

    _home, _trace, _camera, _maxf, _w, _h = sys.argv[1:7]
    print(json.dumps(_render_impl(
        Path(_home), _trace, camera=_camera,
        max_frames=int(_maxf), width=int(_w), height=int(_h),
    )))
