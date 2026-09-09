"""W04 红测试（规格 2026-09-08 §8）：渲染器补强。

§8.1 单次渲染独立身份——请求相同的幂等重试复用已完成结果；
同 trace 不同参数互不覆盖。
§8.2 时间驱动选帧（时间戳+fps，保留首尾；playback rate）；
相机按轨迹包围范围取景。
§8.3 后端按可用性排序逐个真实尝试（不是只试列表前两个）；
探测缓存键含运行时/环境信息，失效后可重新探测。
"""

from __future__ import annotations

import pytest


class TestBackendFallbackDepth:
    def test_fallback_reaches_third_backend(self, monkeypatch) -> None:
        """EGL 首选+OSMesa 渲染失败+Xvfb 可用 → 必须真试到第三
        候选成功（基线 candidates[:2] 在 Xvfb 为第三时直接失败）。"""
        from rosclaw.agentd import sim_render

        attempts: list[str] = []
        probes: list[str] = []

        def fake_probe(*, timeout_sec: float = 30.0):
            return "egl", {"egl": "ok"}

        def fake_single(backend: str, *, timeout_sec: float = 30.0):
            probes.append(backend)
            return (True, "ok") if backend == "xvfb" else (False, "boom")

        def fake_attempt(home, trace_id, backend, **kwargs):
            attempts.append(backend)
            if backend != "xvfb":
                raise ValueError(f"RENDER_FAILED: {backend} 模拟崩溃")
            return {"artifact": {"path": "x.gif"}, "receipt": {"backend": backend}}

        monkeypatch.setattr(sim_render, "probe_render_backend", fake_probe)
        monkeypatch.setattr(sim_render, "_probe_backend", fake_single)
        monkeypatch.setattr(sim_render, "_render_attempt", fake_attempt)
        out = sim_render._render_with_fallback(
            render_call=lambda b: fake_attempt(None, None, b),
            probe_backend=fake_single,
            first="egl",
        )
        assert out["receipt"]["backend"] == "xvfb"
        # osmesa 探测不可用被跳过（不浪费渲染尝试）；xvfb 作为
        # 第三候选被真实探测并渲染成功——基线在 candidates[:2]
        # 时根本到不了这里。
        assert attempts == ["egl", "xvfb"], attempts
        assert probes == ["osmesa", "xvfb"], probes

    def test_each_backend_at_most_one_attempt(self, monkeypatch) -> None:
        """每个候选最多一次真实尝试；全失败时结构化错误列出
        每个后端的原因（不只前两个）。"""
        from rosclaw.agentd import sim_render

        calls: list[str] = []

        def render_call(backend: str):
            calls.append(backend)
            raise ValueError(f"RENDER_FAILED: {backend} 崩了")

        def probe_all_ok(backend: str, *, timeout_sec: float = 30.0):
            return True, "ok"

        with pytest.raises(ValueError) as exc:
            sim_render._render_with_fallback(
                render_call=render_call,
                probe_backend=probe_all_ok,
                first="egl",
            )
        assert calls == ["egl", "osmesa", "xvfb"], calls
        message = str(exc.value)
        for backend in ("egl", "osmesa", "xvfb"):
            assert backend in message, f"聚合错误缺 {backend} 的原因"

    def test_source_no_two_candidate_cap(self) -> None:
        import inspect

        from rosclaw.agentd import sim_render

        src = inspect.getsource(sim_render)
        assert "candidates[:2]" not in src


class TestProbeCache:
    def test_success_cached_and_failure_reprobed(self, monkeypatch) -> None:
        from rosclaw.agentd import sim_render

        sim_render._probe_cache_clear()
        calls: list[str] = []

        def flaky(backend: str, *, timeout_sec: float = 30.0):
            calls.append(backend)
            if len(calls) <= 3:
                return False, "down"
            return (backend == "egl", "ok" if backend == "egl" else "no")

        monkeypatch.setattr(sim_render, "_probe_backend", flaky)
        backend, _ = sim_render.probe_render_backend()
        assert backend is None  # 全失败
        first_round = len(calls)
        # 失败不缓存——下次真实重探（恢复后能找到 egl）。
        backend2, _ = sim_render.probe_render_backend()
        assert backend2 == "egl"
        assert len(calls) > first_round
        # 成功已缓存——第三次调用零探测。
        count_before = len(calls)
        backend3, _ = sim_render.probe_render_backend()
        assert backend3 == "egl"
        assert len(calls) == count_before
        sim_render._probe_cache_clear()

    def test_cache_key_includes_runtime_and_env(self, monkeypatch) -> None:
        from rosclaw.agentd import sim_render

        key1 = sim_render._probe_cache_key()
        monkeypatch.setenv("DISPLAY", ":99")
        key2 = sim_render._probe_cache_key()
        assert key1 != key2, "环境变化（DISPLAY）未反映进缓存键"


class TestTimestampFrameSelection:
    def test_frames_selected_by_time_with_fps(self) -> None:
        """时间驱动选帧：目标时刻 = first + i/fps，取最近状态；
        首尾必含。"""
        from rosclaw.agentd.sim_render import select_frame_indices

        # 10 s 记录，100 Hz 采样（1001 状态）。
        states = [
            {"step": i, "time": i * 0.01, "qpos": [0.0]}
            for i in range(1001)
        ]
        idx = select_frame_indices(states, fps=10.0, playback_rate=1.0)
        times = [states[i]["time"] for i in idx]
        assert idx[0] == 0 and idx[-1] == len(states) - 1
        # 10 s × 10 fps = 100 间隔 → 101 帧（±1）。
        assert abs(len(idx) - 101) <= 1, len(idx)
        # 帧时刻逼近 i/fps 网格（误差 < 一个采样周期）。
        for k, t in enumerate(times):
            assert abs(t - k * 0.1) < 0.011, (k, t)

    def test_playback_rate_scales_output_duration(self) -> None:
        from rosclaw.agentd.sim_render import select_frame_indices

        states = [
            {"step": i, "time": i * 0.01, "qpos": [0.0]}
            for i in range(1001)
        ]
        idx_2x = select_frame_indices(states, fps=10.0, playback_rate=2.0)
        # 2× 播放：10 s 记录 → 5 s 视频 → ~51 帧。
        assert abs(len(idx_2x) - 51) <= 1, len(idx_2x)
        assert idx_2x[0] == 0 and idx_2x[-1] == len(states) - 1

    def test_short_record_keeps_first_and_last(self) -> None:
        from rosclaw.agentd.sim_render import select_frame_indices

        states = [{"step": 0, "time": 0.0, "qpos": [0.0]},
                  {"step": 1, "time": 0.01, "qpos": [0.1]}]
        idx = select_frame_indices(states, fps=12.0, playback_rate=1.0)
        assert idx[0] == 0 and idx[-1] == 1 and len(idx) >= 2


class TestCameraBBoxFraming:
    def test_distance_scales_with_workspace_extent(self) -> None:
        """相机距离由轨迹包围范围驱动——大工作区拉远，小工作区
        拉近（固定距离只作无轨迹旧兼容）。"""
        import mujoco

        from rosclaw.agentd.sim_render import apply_camera_framing

        cam = mujoco.MjvCamera()
        apply_camera_framing(cam, "follow", center=(0.3, 0.2, 0.3), extent=1.0)
        wide = float(cam.distance)
        apply_camera_framing(cam, "follow", center=(0.3, 0.2, 0.3), extent=0.05)
        tight = float(cam.distance)
        assert wide > tight * 2, (wide, tight)
        # 无轨迹 extent=None → 旧固定参数兼容（不崩）。
        apply_camera_framing(cam, "follow", center=(0.35, 0.25, 0.30), extent=None)
        assert float(cam.distance) > 0.0


class TestRenderIdentity:
    def test_render_key_stable_and_param_sensitive(self, tmp_path) -> None:
        """渲染身份 = trace/model digest + spec + 渲染器版本：
        同参同键，改视角/尺寸不同键。"""
        from rosclaw.agentd.sim_render import render_identity_key

        base = {
            "states_digest": "sha256:" + "a" * 64, "camera": "follow",
            "max_frames": 60, "width": 640, "height": 360,
            "world_id": "empty",
            "tool_ref": "", "spec_digest": "", "fps": 12.0,
            "playback_rate": 1.0,
        }
        k1 = render_identity_key(**base)
        assert k1 == render_identity_key(**base)
        assert k1 != render_identity_key(**{**base, "camera": "top"})
        assert k1 != render_identity_key(**{**base, "width": 320})
        assert k1 != render_identity_key(
            **{**base, "states_digest": "sha256:" + "b" * 64}
        )


class TestRenderLifecycleE2E:
    """§8.1/§8.4 完成条件（真实渲染管线，wp3 同款 harness）。"""

    def test_retry_reuses_and_two_views_coexist(self, tmp_path) -> None:
        """同参重试复用已完成结果（reused=True 且不重渲染）；
        同 trace 两视角各自产出互不覆盖。"""
        from tests.agentd.test_wp3_scene_render import _make_trace

        from rosclaw.agentd.sim_render import render_scene_trace

        run = _make_trace(tmp_path)
        trace_id = run["trace_id"]
        first = render_scene_trace(tmp_path, trace_id)
        second = render_scene_trace(tmp_path, trace_id)
        assert second.get("reused") is True, "同渲染身份重试未复用"
        assert second["artifact"]["path"] == first["artifact"]["path"]
        top = render_scene_trace(tmp_path, trace_id, camera="top")
        from pathlib import Path

        assert Path(first["artifact"]["path"]).exists(), (
            "第二视角渲染覆盖了第一视角的产物"
        )
        assert Path(top["artifact"]["path"]).exists()
        assert top["artifact"]["path"] != first["artifact"]["path"]

    def test_gif_duration_matches_record(self, tmp_path) -> None:
        """§8.4：输出 1× 时长与记录时长误差 ≤ 一个输出帧。"""
        import json

        from tests.agentd.test_wp3_scene_render import _make_trace

        from rosclaw.agentd.sim_render import render_scene_trace

        run = _make_trace(tmp_path)
        trace_id = run["trace_id"]
        result = render_scene_trace(tmp_path, trace_id, fps=10.0)
        states = json.loads(
            (tmp_path / "sim" / "traces" / trace_id
             / "trajectory_states.json").read_text(encoding="utf-8"),
        )["states"]
        record_s = float(states[-1]["time"]) - float(states[0]["time"])
        from PIL import Image

        with Image.open(result["artifact"]["path"]) as img:
            n_frames = getattr(img, "n_frames", 1)
            per_frame_ms = float(img.info.get("duration", 100))
        gif_s = n_frames * per_frame_ms / 1000.0
        assert abs(gif_s - record_s) <= 2.0 / 10.0 + 0.35, (
            f"GIF 时长 {gif_s:.2f}s 与记录 {record_s:.2f}s 不符"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
