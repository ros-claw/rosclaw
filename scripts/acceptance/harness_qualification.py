#!/usr/bin/env python3
"""B-3（0916 三审 P0-5）：新 Harness（SimulationRuntime/MujocoBackend）
渲染/动力学资格认证——每台机器真实跑，机器生成资格矩阵。

三审稿：新 Harness 需要在 Jetson/DGX/x86 做 EGL/Xvfb 资格认证。
本脚本在当前机器真实执行三后端单帧渲染（egl/osmesa/xvfb——
逐个真实尝试，绝不走 auto/glfw：本机 glfw 会崩宿主，MH 实证）
+ 一次短 rollout（动力学闭环），输出机器可读 JSON。不可达的
平台（本机之外的 DGX/x86）由矩阵行 NOT_RUN 诚实占位——不合成。

用法：
  python scripts/acceptance/harness_qualification.py --out /tmp/harness-qual.json
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def _env_facts() -> dict:
    import mujoco

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "mujoco_gl_env": __import__("os").environ.get("MUJOCO_GL", "<unset>"),
    }


def _smoke_ur5e_xml() -> str:
    """最小单关节摆模型（不依赖 zoo 资产——资格只验 GL/动力学）。"""
    return (
        '<mujoco><option gravity="0 0 -9.81"/>'
        '<worldbody><body name="p" pos="0 0 1">'
        '<joint name="j" type="hinge" axis="0 1 0" damping="0.1"/>'
        '<geom type="capsule" size="0.02 0.3" mass="0.5"/>'
        '<camera name="cam" pos="0 -2 1" euler="90 0 0"/>'
        '</body></worldbody></mujoco>'
    )


def _render_one_frame(backend: str, timeout: int = 60) -> tuple[bool, str]:
    """指定后端真实渲一帧（子进程隔离——glfw/egl 崩溃不炸宿主）。

    xvfb 特殊：mujoco 3.13 的 MUJOCO_GL 无 xvfb 值——经 xvfb-run
    包装 + glfw 渲染（无 xvfb-run 即不可用，诚实记录）。"""
    worker = (
        "import os,sys;"
        "os.environ['MUJOCO_GL']=sys.argv[1];"
        "import mujoco,numpy as np;"
        "m=mujoco.MjModel.from_xml_string(sys.argv[2]);"
        "d=mujoco.MjData(m);"
        "mujoco.mj_forward(m,d);"
        "r=mujoco.Renderer(m,64,64);"
        "r.update_scene(d,'cam');"
        "img=r.render();"
        "assert img.sum()>0,'black frame';"
        "print('OK',int(img.sum()))"
    )
    argv = [sys.executable, "-c", worker, backend, _smoke_ur5e_xml()]
    if backend == "xvfb":
        import shutil

        xvfb_run = shutil.which("xvfb-run")
        if not xvfb_run:
            return False, "xvfb-run 不在 PATH（未安装 xorg-x11-server-Xvfb）"
        argv = [xvfb_run, "-a", sys.executable, "-c", worker,
                "glfw", _smoke_ur5e_xml()]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if proc.returncode == 0 and "OK" in proc.stdout:
        return True, proc.stdout.strip().splitlines()[-1][:80]
    tail = (proc.stderr or "").strip().splitlines()
    return False, (tail[-1] if tail else "native crash")[:120]


def _rollout_smoke() -> tuple[bool, str]:
    """经 SimulationRuntime 的真实 rollout（动力学闭环资格）。"""
    try:
        from rosclaw.sim.runtime import SimulationRuntime
    except Exception as exc:  # noqa: BLE001
        return False, f"import: {exc}"
    try:
        runtime = SimulationRuntime(task_root=None)
        loaded = runtime.backend.load_model_xml(
            _smoke_ur5e_xml(),
            source={"kind": "qualification", "id": "harness_qual_pendulum"},
        )
        model_ref = loaded.model_ref
        result = runtime.rollout(
            model_ref,
            controller={"hold": True},
            duration_s=0.5,
            seed=0,
        )
        states = result.get("states") or result.get("trace") or {}
        digest = str(result.get("states_digest") or result.get("trace_digest") or "")
        ok = bool(digest) or bool(states)
        return ok, f"digest={digest[:24]} keys={sorted(result)[:6]}"
    except Exception as exc:  # 失败如实记录
        return False, f"{type(exc).__name__}: {str(exc)[:160]}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/harness-qual.json"))
    args = parser.parse_args()

    report: dict = {
        "schema_version": "rosclaw.harness_qualification.v1",
        "generated_by": "scripts/acceptance/harness_qualification.py",
        "host": _env_facts(),
        "render_backends": {},
        "rollout": {},
        "matrix_not_run": [
            {"platform": "DGX x86_64", "state": "NOT_RUN", "reason": "本机之外——不合成"},
            {"platform": "x86_64 workstation", "state": "NOT_RUN", "reason": "本机之外——不合成"},
        ],
    }
    started = time.monotonic()
    any_ok = False
    for backend in ("egl", "osmesa", "xvfb"):
        ok, note = _render_one_frame(backend)
        report["render_backends"][backend] = {"ok": ok, "detail": note}
        any_ok = any_ok or ok
        print(f"[render:{backend}] {'OK' if ok else 'FAIL'} {note}", flush=True)
    ok, note = _rollout_smoke()
    report["rollout"] = {"ok": ok, "detail": note}
    print(f"[rollout] {'OK' if ok else 'FAIL'} {note}", flush=True)
    report["verdict"] = "QUALIFIED" if (any_ok and ok) else "NOT_QUALIFIED"
    report["wall_time_s"] = round(time.monotonic() - started, 1)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8",
    )
    print(json.dumps({"verdict": report["verdict"], "out": str(args.out)}))
    return 0 if report["verdict"] == "QUALIFIED" else 1


if __name__ == "__main__":
    sys.exit(main())
