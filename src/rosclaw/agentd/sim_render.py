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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rosclaw.contracts.agent.render_spec import RenderSpecV1

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


def render_from_spec(
    home: Path,
    spec: RenderSpecV1,
    trace_id: str,
    *,
    max_frames: int = 60,
    width: int = 640,
    height: int = 360,
) -> dict[str, Any]:
    """RenderSpecV1 驱动的通用渲染（0902 R2-2，§4.3）。

    与 render_scene_trace 的差异：本体来自 RenderProfile 注册表
    （不再 hardcode ur5e）；overlay 真实绘制且证据绑定；outputs
    遵守 spec 声明。

    父进程职责（子进程只做像素）：
    - 本体档案解析（RENDER_PROFILE_MISSING 诚实失败）；
    - overlay 证据绑定：trace 源 == 本次 trace_id；plan 源 ==
      trace 的 plan_hash（RENDER_EVIDENCE_MISMATCH——旧证据不得
      冒充本次，0902 假成功教训）；
    - 未实现 overlay 类型 RENDER_OVERLAY_UNSUPPORTED（不静默跳过
      宣称已画）。
    """
    from rosclaw.agentd.render_profiles import resolve_render_profile

    body_id = spec.body_ref.removeprefix("robot:")
    profile = resolve_render_profile(body_id)  # 未登记即抛

    home = Path(home)
    trace_path = home / "sim" / "traces" / trace_id / "trace.json"
    if not trace_path.exists():
        raise ValueError(f"RENDER_INPUT_MISSING: trace {trace_id!r} 不存在")
    plan_hash = str(json.loads(trace_path.read_text(encoding="utf-8")).get("plan_hash", ""))

    supported = {"actual_eef_trace", "planned_trace", "waypoints", "contact_points"}
    for overlay in spec.overlays:
        if overlay.kind not in supported:
            raise ValueError(
                f"RENDER_OVERLAY_UNSUPPORTED: {overlay.kind} 渲染未实现"
                f"（已支持：{sorted(supported)}）——不静默跳过"
            )
        if overlay.kind == "actual_eef_trace":
            if overlay.source_ref != f"trace:{trace_id}":
                raise ValueError(
                    f"RENDER_EVIDENCE_MISMATCH: overlay source_ref "
                    f"{overlay.source_ref!r} != 本次 trace:{trace_id}"
                )
        else:
            if overlay.source_ref != f"plan:{plan_hash[:16]}":
                raise ValueError(
                    f"RENDER_EVIDENCE_MISMATCH: overlay source_ref "
                    f"{overlay.source_ref!r} != 本次 plan:{plan_hash[:16]}"
                )
    if len(spec.attachments) > 1:
        raise ValueError(
            "RENDER_ATTACHMENTS_UNSUPPORTED: 多附件渲染未实现（当前单附件）"
        )
    tool_ref = spec.attachments[0].tool_ref if spec.attachments else ""
    camera = spec.cameras[0].preset if spec.cameras else (
        profile.default_cameras[0].preset if profile.default_cameras else "follow"
    )
    # 世界引用 → world_id（world:tabletop → tabletop）。
    world_id = spec.world_ref.removeprefix("world:")

    # spec 落盘进 trace 目录——子进程从文件读（CLI 参数不塞 JSON）。
    spec_payload = spec.model_dump(mode="json")
    spec_path = trace_path.parent / f"{trace_id}-render-spec.json"
    spec_path.write_text(
        json.dumps(spec_payload, ensure_ascii=False), encoding="utf-8"
    )
    result = render_scene_trace(
        home, trace_id,
        camera=camera, max_frames=max_frames, width=width, height=height,
        world_id=world_id, tool_ref=tool_ref,
        render_spec_path=spec_path,
    )
    return result


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
    render_spec_path: Path | None = None,
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
    if world_id:
        _require_world_asset(world_id)
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
                render_spec_path=render_spec_path,
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
    render_spec_path: Path | None = None,
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
    if render_spec_path is not None:
        argv += ["--spec", str(render_spec_path)]
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


def _require_world_asset(world_id: str) -> None:
    """world 资产断言（WORLD_ASSET_MISSING 诚实失败——"桌面
    移除"注入时不得假装有桌面）。"""
    from rosclaw.sandbox.sandbox_api import SUPPORTED_MUJOCO_WORLDS

    if world_id not in SUPPORTED_MUJOCO_WORLDS:
        raise ValueError(
            f"WORLD_ASSET_MISSING: world {world_id!r} 不可用"
            f"（支持：{sorted(SUPPORTED_MUJOCO_WORLDS)}）"
        )


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


def _overlay_scene_geoms(
    spec_overlays: list[dict],
    trace: dict,
    plan_doc: dict | None,
) -> list[tuple[str, Any]]:
    """overlay → 装饰几何序列（mujoco mjvGeom 配置元组）。

    返回 (kind, payload) 列表由调用方逐帧应用到 scene——真实绘制，
    绘制成功的 kind 才进 receipt 的 overlays_applied。
    """

    # 渲染成本有界（CI 软件光栅实证：全量轨迹点 >600s 超时）——
    # 折线/点列抽稀到 240 段以内，形状语义不变。
    def _decimate(pts: list, cap: int = 240) -> list:
        if len(pts) <= cap:
            return pts
        step = (len(pts) - 1) / (cap - 1)
        return [pts[round(i * step)] for i in range(cap)]

    applied: list[tuple[str, Any]] = []
    for overlay in spec_overlays:
        kind = overlay.get("kind", "")
        if kind == "actual_eef_trace":
            pts = _decimate(
                [(p["x"], p["y"], p["z"]) for p in trace.get("actual") or []]
            )
            applied.append((kind, ("polyline", pts, (1.0, 0.2, 0.2, 0.9))))
        elif kind == "planned_trace":
            pts = _decimate([
                (p["x"], p["y"], p["z"]) for p in trace.get("planned") or []
            ])
            applied.append((kind, ("polyline", pts, (0.2, 0.6, 1.0, 0.9))))
        elif kind == "waypoints":
            wps = (plan_doc or {}).get("spec", {}).get("waypoints") or []
            pts = [tuple(w["position_m"]) for w in wps]
            applied.append((kind, ("points", pts, (1.0, 0.8, 0.1, 0.95))))
        elif kind == "contact_points":
            wps = (plan_doc or {}).get("spec", {}).get("waypoints") or []
            pts = [tuple(w["position_m"]) for w in wps if w.get("kind") == "contact"]
            applied.append((kind, ("points", pts, (0.1, 1.0, 0.3, 1.0))))
    return applied


def _apply_overlay_geoms(renderer: Any, overlays: list[tuple[str, Any]]) -> list[str]:
    """把 overlay 几何 append 进 renderer.scene（装饰几何，不碰物理）。
    返回实际画上的 kind 列表。"""
    import mujoco
    import numpy as np

    scn = renderer.scene
    drawn: list[str] = []
    for kind, payload in overlays:
        shape, pts, rgba = payload
        if not pts:
            continue
        if shape == "polyline":
            for a, b in zip(pts, pts[1:], strict=False):
                if scn.ngeom >= scn.maxgeom:
                    break
                # mujoco 3.x：mjv_connector 只设 size/pos/mat——
                # rgba 等其余属性必须先 mjv_initGeom（实证：
                # mjv_makeConnector 在 3.11 已改名为 mjv_connector）。
                geom = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    np.zeros(3), np.zeros(3),
                    np.eye(3).flatten(),
                    np.array(rgba, dtype=np.float32),
                )
                mujoco.mjv_connector(
                    geom,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    1.6,  # 线宽（像素级近似）
                    np.array(a, dtype=float), np.array(b, dtype=float),
                )
                scn.ngeom += 1
            drawn.append(kind)
        else:  # points
            for p in pts:
                if scn.ngeom >= scn.maxgeom:
                    break
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.004, 0.0, 0.0]),
                    np.array(p, dtype=float),
                    np.eye(3).flatten(),
                    np.array(rgba, dtype=np.float32),
                )
                scn.ngeom += 1
            drawn.append(kind)
    return drawn


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
    render_spec_path: str = "",
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
    # RenderSpec（R2-2）：本体档案 + overlay + outputs 全部由 spec
    # 驱动；无 spec = 旧路径（ur5e 默认，向后兼容）。
    spec_doc: dict = {}
    if render_spec_path:
        spec_doc = json.loads(Path(render_spec_path).read_text(encoding="utf-8"))
    spec_overlays = spec_doc.get("overlays") or []
    spec_outputs = spec_doc.get("outputs") or ["gif", "mp4"]
    spec_body_ref = str(spec_doc.get("body_ref") or "")
    # canonical MJCF（与 rollout 同一资源链）；world 来自 TaskSpec
    # （tabletop/empty——empty 不得冒充桌面场景）。本体：spec 驱动时
    # 走 RenderProfile 注册表（不 hardcode）；无 spec 保持 ur5e 兼容。
    from rosclaw.sandbox.sandbox_api import Sandbox

    robot_id = "ur5e"
    if spec_body_ref:
        from rosclaw.agentd.render_profiles import sandbox_robot_id

        robot_id = sandbox_robot_id(spec_body_ref.removeprefix("robot:"))
    sandbox = Sandbox.create(robot_id, world_id, "mujoco")
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

        # overlay 几何（静态点列，逐帧重挂——update_scene 每帧重置
        # scene.ngeom）。plan 源 overlay 需要 plan 文档。
        plan_doc: dict | None = None
        if any(o.get("kind") != "actual_eef_trace" for o in spec_overlays):
            plan_hash = str(trace.get("plan_hash", ""))
            plan_path = home / "sim" / "plans" / f"plan_{plan_hash[:16]}.json"
            if plan_path.exists():
                plan_doc = json.loads(plan_path.read_text(encoding="utf-8"))
        overlay_geoms = _overlay_scene_geoms(spec_overlays, trace, plan_doc)
        overlays_applied: list[str] = []

        images = []
        for idx in frames_idx:
            q = positions[idx]
            data.qpos[: int(model.nu)] = np.array(q[: int(model.nu)])
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            if overlay_geoms:
                overlays_applied = _apply_overlay_geoms(renderer, overlay_geoms)
            images.append(Image.fromarray(renderer.render()))
        artifacts: dict[str, Any] = {}
        if "gif" in spec_outputs:
            out = trace_dir / f"{trace_id}-scene.gif"
            images[0].save(
                out, save_all=True, append_images=images[1:],
                duration=int(1000 / 12), loop=0,
            )
            artifacts["gif"] = {
                "path": str(out),
                "frames": len(images),
                "format": "gif",
                "bytes": out.stat().st_size,
                "evidence_level": "SIM_DYN_ROLLOUT",
            }
        if "mp4" in spec_outputs:
            # P0-F：官方渲染同时产出 MP4（imageio + imageio-ffmpeg
            # 自带静态 ffmpeg——离线，不需要系统 ffmpeg）。
            import imageio.v3 as iio
            import numpy as _np

            mp4 = trace_dir / f"{trace_id}-scene.mp4"
            frames_arr = [_np.asarray(img) for img in images]
            iio.imwrite(mp4, frames_arr, fps=12)
            artifacts["mp4"] = {
                "path": str(mp4),
                "frames": len(images),
                "format": "mp4",
                "bytes": mp4.stat().st_size,
                "evidence_level": "SIM_DYN_ROLLOUT",
            }
    finally:
        renderer.close()
        sandbox.close()
    if not artifacts:
        raise ValueError("RENDER_OUTPUT_EMPTY: spec.outputs 未产生任何产物")
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
        "outputs": sorted(artifacts),
        "resource": trace.get("resource") or {},
    }
    if spec_doc:
        # R2-2：spec 锚点进 receipt（body/spec digest/真实绘制的
        # overlay——宣称与画面一致可审计）。
        receipt["body_ref"] = spec_body_ref
        receipt["overlays_applied"] = overlays_applied
        receipt["spec_digest"] = "sha256:" + hashlib.sha256(
            json.dumps(spec_doc, sort_keys=True).encode()
        ).hexdigest()
    (trace_dir / "render_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    primary = artifacts.get("gif") or artifacts["mp4"]
    return {
        "ok": True,
        "artifact": primary,
        "artifacts": artifacts,
        "receipt": receipt,
    }


__all__ = ["probe_render_backend", "render_from_spec", "render_scene_trace"]


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
    _parser.add_argument("--spec", default="")
    _parser.add_argument("--result", required=True)
    _args = _parser.parse_args()
    _result_path = Path(_args.result)
    try:
        _payload: dict = _render_impl(
            Path(_args.home), _args.trace_id, camera=_args.camera,
            max_frames=_args.max_frames, width=_args.width,
            height=_args.height, world_id=_args.world, tool_ref=_args.tool,
            render_spec_path=_args.spec,
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
