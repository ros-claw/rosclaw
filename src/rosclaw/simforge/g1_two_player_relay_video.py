"""Visualization-only renderer for the evidence-bound G1 relay."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np

from rosclaw.simforge.backends.unitree_mujoco_backend import trajectory_digest
from rosclaw.simforge.g1_hat_trick_video import (
    _append_sphere,
    _escape_filtergraph_option,
    _load_trajectory,
    _sample_trajectory,
)

_SCENE_REL = Path("g1_description/scene_with_ball.xml")
_WIDTH = 640
_HEIGHT = 360


@dataclass(frozen=True)
class G1RelayVideoClip:
    role: str
    source_trajectory_hash: str
    frame_count: int
    duration_sec: float


@dataclass(frozen=True)
class G1RelayVideoResult:
    output_path: str
    manifest_path: str
    video_hash: str
    evidence_report_hash: str
    fps: int
    frame_count: int
    duration_sec: float
    clips: tuple[G1RelayVideoClip, ...]
    visualization_only: bool = True
    simultaneous_two_body_physics: bool = False
    schema_version: str = "rosclaw.g1_goalforge.two_player_relay_video.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "clips": [asdict(item) for item in self.clips],
            "label_source": "verified_two_player_relay_evidence",
            "generates_task_evidence": False,
        }


@dataclass(frozen=True)
class _RelaySource:
    role: str
    scenario: dict[str, Any]
    result: dict[str, Any]
    trajectory_hash: str
    trajectory: dict[str, np.ndarray]


def render_g1_two_player_relay_video(
    *,
    evidence_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
) -> G1RelayVideoResult:
    """Render the two strict episodes in causal order without altering evidence."""

    evidence = evidence_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("G1 relay video must be outside the source checkout")
    if output.suffix.lower() != ".mp4":
        raise ValueError("G1 relay video output must use .mp4")
    if not 10 <= fps <= 60:
        raise ValueError("G1 relay video fps must be in [10, 60]")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("G1 relay video or manifest already exists")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for G1 relay video export")
    report = json.loads(evidence.read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise ValueError("G1 relay video requires a passing evidence report")
    sources = (
        _load_source(report["passer"], checkout),
        _load_source(report["shooter"], checkout),
    )
    timelines = (
        _passer_timeline(sources[0], float(report["handoff"]["source_time_sec"]), fps),
        _shooter_timeline(sources[1], fps),
    )
    durations = tuple(len(item) / fps for item in timelines)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco

        from rosclaw.simforge.backends.unitree_mujoco_backend import qualify_g1_assets

        qualification = qualify_g1_assets(asset_root)
        qualification.require_eligible()
        if qualification.body_hash != report["body_hash"]:
            raise ValueError("G1 relay video Body hash does not match evidence")
        model = mujoco.MjModel.from_xml_path(str(asset_root.resolve() / _SCENE_REL))
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=_HEIGHT, width=_WIDTH)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-relay-video-") as temporary:
                metrics = _write_metric_files(Path(temporary), report)
                process = subprocess.Popen(
                    _ffmpeg_command(
                        ffmpeg=ffmpeg,
                        output=output,
                        fps=fps,
                        durations=durations,
                        metric_files=metrics,
                    ),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                if process.stdin is None:
                    raise RuntimeError("ffmpeg raw-video pipe is unavailable")
                try:
                    _write_frames(
                        mujoco=mujoco,
                        model=model,
                        data=data,
                        renderer=renderer,
                        sources=sources,
                        timelines=timelines,
                        handoff=report["handoff"],
                        stream=cast(BinaryIO, process.stdin),
                    )
                except BaseException:
                    process.stdin.close()
                    process.kill()
                    process.wait()
                    raise
                process.stdin.close()
                stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
                code = process.wait()
                if code:
                    raise RuntimeError(f"G1 relay ffmpeg failed ({code}): {stderr[-2000:]}")
        finally:
            renderer.close()
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl

    clips = tuple(
        G1RelayVideoClip(
            role=source.role,
            source_trajectory_hash=source.trajectory_hash,
            frame_count=len(timeline),
            duration_sec=duration,
        )
        for source, timeline, duration in zip(sources, timelines, durations, strict=True)
    )
    manifest = output.with_suffix(".json")
    result = G1RelayVideoResult(
        output_path=str(output),
        manifest_path=str(manifest),
        video_hash=_file_hash(output),
        evidence_report_hash=_file_hash(evidence),
        fps=fps,
        frame_count=sum(len(item) for item in timelines),
        duration_sec=sum(durations),
        clips=clips,
    )
    manifest.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _load_source(leg: dict[str, Any], checkout: Path) -> _RelaySource:
    path = Path(str(leg["trajectory_path"])).expanduser().resolve()
    if path == checkout or checkout in path.parents:
        raise ValueError("G1 relay trajectory must be outside the source checkout")
    if _file_hash(path) != leg["trajectory_hash"]:
        raise ValueError("G1 relay trajectory file hash mismatch")
    trajectory = _load_trajectory(path)
    if trajectory_digest(trajectory) != leg["trajectory_digest"]:
        raise ValueError("G1 relay trajectory content digest mismatch")
    return _RelaySource(
        role=str(leg["role"]),
        scenario=dict(leg["scenario"]),
        result=dict(leg["result"]),
        trajectory_hash=str(leg["trajectory_hash"]),
        trajectory=trajectory,
    )


def _passer_timeline(
    source: _RelaySource,
    handoff_time: float,
    fps: int,
) -> tuple[float, ...]:
    contact = float(source.result["ball_contact_time_sec"])
    return _segments(
        (
            (max(0.0, contact - 2.6), contact - 0.45, 1.0),
            (contact - 0.45, contact + 0.75, 0.48),
            (contact + 0.75, handoff_time + 0.9, 0.80),
        ),
        fps,
    )


def _shooter_timeline(source: _RelaySource, fps: int) -> tuple[float, ...]:
    contact = float(source.result["ball_contact_time_sec"])
    end = min(float(source.trajectory["time"][-1]), contact + 6.2)
    return _segments(
        (
            (max(0.0, contact - 2.5), contact - 0.55, 1.0),
            (contact - 0.55, contact + 0.95, 0.48),
            (contact + 0.95, contact + 3.4, 1.0),
            (contact + 3.4, end, 1.35),
        ),
        fps,
    )


def _segments(
    segments: tuple[tuple[float, float, float], ...],
    fps: int,
) -> tuple[float, ...]:
    values: list[float] = []
    for start, end, speed in segments:
        if end <= start:
            continue
        count = max(1, int(np.ceil((end - start) / speed * fps)))
        values.extend(min(end, start + index / fps * speed) for index in range(count))
    return tuple(values)


def _write_frames(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    renderer: Any,
    sources: tuple[_RelaySource, ...],
    timelines: tuple[tuple[float, ...], ...],
    handoff: dict[str, Any],
    stream: BinaryIO,
) -> None:
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    ball_qpos = int(model.jnt_qposadr[joint])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    for source, timeline in zip(sources, timelines, strict=True):
        for simulation_time in timeline:
            index, pelvis, joints, ball = _sample_trajectory(
                source.trajectory,
                simulation_time,
            )
            data.qpos[:] = model.qpos0
            data.qpos[:7] = pelvis
            data.qpos[7:36] = joints
            data.qpos[ball_qpos : ball_qpos + 7] = ball
            mujoco.mj_forward(model, data)
            if source.role == "G1_A_PASSER":
                camera.lookat[:] = (1.65, -0.05, 0.62)
                camera.distance = 4.0
            elif simulation_time < float(source.result["ball_contact_time_sec"]) + 0.15:
                camera.lookat[:] = (1.45, 0.12, 0.72)
                camera.distance = 3.7
            else:
                camera.lookat[:] = (3.0, 0.30, 0.72)
                camera.distance = 6.0
            camera.azimuth = 92.0
            camera.elevation = -8.0
            renderer.update_scene(data, camera=camera)
            _add_visual_markers(
                mujoco=mujoco,
                scene=renderer.scene,
                source=source,
                index=index,
                handoff=handoff,
            )
            frame = renderer.render().copy()
            canvas = np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)
            stream.write(np.ascontiguousarray(canvas).tobytes())


def _add_visual_markers(
    *,
    mujoco: Any,
    scene: Any,
    source: _RelaySource,
    index: int,
    handoff: dict[str, Any],
) -> None:
    if source.role == "G1_A_PASSER":
        handoff_position = np.asarray(
            handoff["source_ball_position_m"],
            dtype=np.float64,
        )
        for height, alpha in ((0.08, 0.40), (0.16, 0.62), (0.24, 0.85)):
            _append_sphere(
                mujoco,
                scene,
                handoff_position + np.asarray((0.0, 0.0, height)),
                0.026,
                (1.0, 0.72, 0.12, alpha),
            )
    else:
        _append_sphere(
            mujoco,
            scene,
            np.asarray(
                (
                    5.02,
                    float(source.scenario["target_y_m"]),
                    float(source.scenario["target_z_m"]),
                )
            ),
            0.18,
            (0.15, 1.0, 0.35, 0.92),
        )
    start = max(0, index - 80)
    indices = np.linspace(start, index, min(16, index - start + 1), dtype=int)
    for trail_index, alpha in zip(indices, np.linspace(0.05, 0.55, len(indices)), strict=True):
        _append_sphere(
            mujoco,
            scene,
            np.asarray(source.trajectory["ball_pose"][trail_index, :3]),
            0.031,
            (0.25, 0.78, 1.0, float(alpha)),
        )


def _write_metric_files(root: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    handoff = report["handoff"]
    shooter = report["shooter"]
    values = (
        (
            "G1-A · SOFT BACK-PASS  →  "
            f"MEASURED HANDOFF {float(handoff['observed_speed_mps']):.3f} m/s"
        ),
        (
            "G1-B · FIRST-TIME HIGH FINISH  ·  "
            f"TARGET {float(shooter['scenario']['target_z_m']):.2f} m  ·  "
            f"SHOT {float(shooter['result']['ball_speed_mps']):.2f} m/s"
        ),
    )
    paths = (root / "passer.txt", root / "shooter.txt")
    for path, value in zip(paths, values, strict=True):
        path.write_text(value, encoding="utf-8")
    return paths


def _ffmpeg_command(
    *,
    ffmpeg: str,
    output: Path,
    fps: int,
    durations: tuple[float, ...],
    metric_files: tuple[Path, Path],
) -> list[str]:
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font_option = f"fontfile={_escape_filtergraph_option(str(font))}:" if font.is_file() else ""
    filters = [
        "drawbox=x=0:y=0:w=iw:h=112:color=0x050A12@0.80:t=fill",
        "drawbox=x=0:y=h-70:w=iw:h=70:color=0x050A12@0.80:t=fill",
        f"drawtext={font_option}text={_escape_filtergraph_option('ROSClaw · G1 TWO-PLAYER RELAY')}:"
        "expansion=none:x=34:y=18:fontsize=36:fontcolor=white",
        f"drawtext={font_option}text={_escape_filtergraph_option('STRICT CPU MUJOCO REPLAY · SIM ONLY')}:"
        "expansion=none:x=34:y=h-46:fontsize=22:fontcolor=0x8DD8FF",
    ]
    offset = 0.0
    for duration, path in zip(durations, metric_files, strict=True):
        end = offset + duration
        filters.append(
            f"drawtext={font_option}textfile={_escape_filtergraph_option(str(path))}:"
            "expansion=none:x=34:y=68:fontsize=22:fontcolor=0x65F59A:"
            f"enable='between(t,{offset:.6f},{end:.6f})'"
        )
        offset = end
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        "1280x720",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-vf",
        ",".join(filters),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["G1RelayVideoResult", "render_g1_two_player_relay_video"]
