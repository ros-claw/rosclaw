"""P0-F 红测试（0824 总纲 §19.P0-F）：官方离线 RendererSupervisor。

红测试先行——Xvfb 正确探测/MP4/依赖闭包不存在时必须红。

验收（文档原文）：
- 无网络环境 10/10 渲染成功；
- Xvfb 已安装时必须被正确识别（0824 事故：手工 Xvfb 成功、
  官方却判断不可用——MUJOCO_GL=xvfb 是无效值，必须走
  xvfb-run + MUJOCO_GL=glfw）；
- input trace digest 与 artifact lineage 一致；
- renderer 失败只影响 delivery/media outcome（P0-D 已钉住）。
- 任务期间绝不安装依赖（安装阶段完成依赖闭包）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_trace(home: Path) -> dict:
    from rosclaw.agentd.sim_trajectory import SimTrajectoryService

    sim = SimTrajectoryService(home)
    plan = sim.generate_planar_path(
        shape="star5", center_m=[0.35, 0.25, 0.30], scale_m=0.05,
    )
    return sim.simulate_cartesian_trajectory(plan["plan_id"])


class TestXvfbProbe:
    def test_xvfb_probe_uses_xvfb_run_with_glfw(self) -> None:
        """Xvfb 探测必须经 xvfb-run + MUJOCO_GL=glfw——
        MUJOCO_GL=xvfb 是无效值（0824 官方误判不可用的根因）。"""
        import shutil as _sh

        from rosclaw.agentd import sim_render

        calls: list[dict] = []

        def fake_run(argv, **kwargs):
            calls.append({"argv": argv, "env": kwargs.get("env", {})})
            class Proc:
                returncode = 0
                stdout = b"OK"
                stderr = b""
            return Proc()

        original_run = sim_render.subprocess.run
        original_which = _sh.which
        sim_render.subprocess.run = fake_run  # type: ignore[assignment]
        _sh.which = lambda name: "/usr/bin/xvfb-run" if name == "xvfb-run" else None  # type: ignore[assignment]
        try:
            sim_render._probe_xvfb()
        finally:
            sim_render.subprocess.run = original_run  # type: ignore[assignment]
            _sh.which = original_which  # type: ignore[assignment]
        assert calls, "_probe_xvfb 未发起探测"
        argv = calls[0]["argv"]
        assert "xvfb-run" in argv[0] or "xvfb-run" in argv, (
            f"Xvfb 探测未走 xvfb-run: {argv}"
        )
        assert calls[0]["env"].get("MUJOCO_GL") == "glfw", (
            f"Xvfb 内应 MUJOCO_GL=glfw，实际 {calls[0]['env'].get('MUJOCO_GL')!r}"
        )

    def test_xvfb_recognized_when_installed(self) -> None:
        """Xvfb 已安装时必须被正确识别（有 xvfb-run 的主机）。"""
        import shutil

        import pytest as _pt

        if not shutil.which("xvfb-run"):
            _pt.skip("本机无 xvfb-run（CI 装 xvfb 后实证）")
        from rosclaw.agentd.sim_render import probe_render_backend

        backend, detail = probe_render_backend()
        assert backend in ("egl", "osmesa", "xvfb"), detail


class TestOfflineRenderOutputs:
    def test_render_produces_gif_and_mp4_with_receipt(
        self, tmp_path: Path
    ) -> None:
        """官方渲染自动产出 GIF + MP4 + receipt（同一 trace digest
        血缘）。"""
        from rosclaw.agentd.sim_render import render_scene_trace

        run = _make_trace(tmp_path)
        result = render_scene_trace(tmp_path, run["trace_id"])
        artifacts = result.get("artifacts") or {}
        assert "gif" in artifacts and "mp4" in artifacts, (
            f"缺 GIF/MP4 产物: {list(artifacts)}"
        )
        mp4 = Path(artifacts["mp4"]["path"])
        assert mp4.exists() and mp4.stat().st_size > 1000
        # MP4 可解码（不是假文件）。
        import imageio.v3 as iio

        frames = iio.imread(mp4, index=0)
        assert frames.size > 0, "MP4 不可解码"
        receipt = result["receipt"]
        assert receipt["input_trace_digest"].startswith("sha256:")
        assert "mp4" in json.dumps(receipt)

    def test_no_pip_install_during_render(self, tmp_path: Path) -> None:
        """任务期间绝不安装依赖：渲染路径零 pip/uv install 调用。"""
        from rosclaw.agentd import sim_render

        install_calls: list[list[str]] = []
        real_run = sim_render.subprocess.run

        def spy(argv, **kwargs):
            cmd = [str(a) for a in argv]
            if any("pip" in a or a == "uv" for a in cmd) and "install" in cmd:
                install_calls.append(cmd)
            return real_run(argv, **kwargs)

        sim_render.subprocess.run = spy  # type: ignore[assignment]
        try:
            run = _make_trace(tmp_path)
            sim_render.render_scene_trace(tmp_path, run["trace_id"])
        finally:
            sim_render.subprocess.run = real_run  # type: ignore[assignment]
        assert install_calls == [], f"任务期安装依赖: {install_calls}"

    def test_dependency_closure_importable(self) -> None:
        """安装阶段依赖闭包：imageio/imageio-ffmpeg 随主包安装
        （不经 runtime manager 任务期装）。"""
        import imageio  # noqa: F401
        import imageio_ffmpeg  # noqa: F401

    def test_pil_missing_honest_error_no_install(self) -> None:
        """PIL 缺失 → RENDER_DEPS_MISSING 诚实失败，不发起安装
        （P0-F：Pillow 是主依赖——任务期间绝不安装）。"""
        import builtins

        from rosclaw.agentd.sim_trajectory import SimTrajectoryService

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "PIL" or name.startswith("PIL."):
                raise ImportError("No module named 'PIL'")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = fake_import
        try:
            sim = SimTrajectoryService(Path("/tmp/p0f"))
            with pytest.raises(ValueError, match="RENDER_DEPS_MISSING"):
                sim._import_pil()
        finally:
            builtins.__import__ = real_import
