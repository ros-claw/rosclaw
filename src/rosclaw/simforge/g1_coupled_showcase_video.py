"""Cinematic visualization-only export for the five coupled G1 challenges."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np

from rosclaw.simforge.backends.unitree_mujoco_backend import (
    qualify_g1_assets,
    trajectory_digest,
)
from rosclaw.simforge.g1_coupled_relay import _coupled_model
from rosclaw.simforge.g1_coupled_relay_video import (
    _id,
    _joint_qpos,
    _load_coupled_trajectory,
)
from rosclaw.simforge.g1_hat_trick_video import (
    _append_sphere,
    _escape_filtergraph_option,
)

_WIDTH = 640
_HEIGHT = 360


@dataclass(frozen=True)
class G1CoupledShowcaseVideoClip:
    case_id: str
    title: str
    trajectory_hash: str
    trajectory_digest: str
    frame_count: int
    duration_sec: float


@dataclass(frozen=True)
class G1CoupledShowcaseVideoResult:
    output_path: str
    manifest_path: str
    video_hash: str
    evidence_report_hash: str
    renderer_hash: str
    fps: int
    frame_count: int
    duration_sec: float
    clips: tuple[G1CoupledShowcaseVideoClip, ...]
    visualization_only: bool = True
    simultaneous_two_body_physics: bool = True
    pixels_used_for_promotion: bool = False
    schema_version: str = "rosclaw.g1_goalforge.coupled_showcase_video.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "clips": [asdict(clip) for clip in self.clips],
            "generates_task_evidence": False,
            "label_source": "five_strict_coupled_physics_challenges",
        }


@dataclass(frozen=True)
class _Source:
    case_id: str
    title: str
    subtitle: str
    camera_azimuth_deg: float
    result: dict[str, Any]
    trajectory_hash: str
    trajectory_digest: str
    trajectory: dict[str, np.ndarray]


def render_g1_coupled_showcase_video(
    *,
    evidence_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
) -> G1CoupledShowcaseVideoResult:
    """Render five verified challenges as a roughly one-minute showcase."""

    evidence = evidence_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("coupled showcase video must be outside the source checkout")
    if output.suffix.lower() != ".mp4":
        raise ValueError("coupled showcase video output must use .mp4")
    if not 10 <= fps <= 60:
        raise ValueError("coupled showcase video fps must be in [10, 60]")
    manifest = output.with_suffix(".json")
    if output.exists() or manifest.exists():
        raise FileExistsError("coupled showcase video or manifest already exists")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for coupled showcase export")

    report = json.loads(evidence.read_text(encoding="utf-8"))
    if report.get("passed") is not True or len(report.get("cases", ())) != 5:
        raise ValueError("coupled showcase video requires five passing challenges")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if qualification.body_hash != report["body_hash"]:
        raise ValueError("coupled showcase Body hash does not match evidence")
    sources = tuple(_load_source(case, checkout) for case in report["cases"])
    timelines = tuple(_cinematic_timeline(source, fps) for source in sources)
    durations = tuple(len(timeline) / fps for timeline in timelines)

    output.parent.mkdir(parents=True, exist_ok=True)
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco

        model = _coupled_model(asset_root.expanduser().resolve())
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=_HEIGHT, width=_WIDTH)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-five-challenge-video-") as temp:
                labels = _write_label_files(Path(temp), sources)
                process = subprocess.Popen(
                    _ffmpeg_command(
                        ffmpeg=ffmpeg,
                        output=output,
                        fps=fps,
                        durations=durations,
                        label_files=labels,
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
                    raise RuntimeError(f"coupled showcase ffmpeg failed ({code}): {stderr[-2000:]}")
        finally:
            renderer.close()
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl

    clips = tuple(
        G1CoupledShowcaseVideoClip(
            case_id=source.case_id,
            title=source.title,
            trajectory_hash=source.trajectory_hash,
            trajectory_digest=source.trajectory_digest,
            frame_count=len(timeline),
            duration_sec=duration,
        )
        for source, timeline, duration in zip(sources, timelines, durations, strict=True)
    )
    result = G1CoupledShowcaseVideoResult(
        output_path=str(output),
        manifest_path=str(manifest),
        video_hash=_file_hash(output),
        evidence_report_hash=_file_hash(evidence),
        renderer_hash=_file_hash(Path(__file__)),
        fps=fps,
        frame_count=sum(clip.frame_count for clip in clips),
        duration_sec=sum(clip.duration_sec for clip in clips),
        clips=clips,
    )
    manifest.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _load_source(case: dict[str, Any], checkout: Path) -> _Source:
    if case.get("passed") is not True or case.get("strict_replay") is not True:
        raise ValueError("showcase source case is not a strict passing rollout")
    path = Path(str(case["trajectory_path"])).expanduser().resolve()
    if path == checkout or checkout in path.parents:
        raise ValueError("showcase trajectory must be outside the source checkout")
    if _file_hash(path) != case["trajectory_hash"]:
        raise ValueError("showcase trajectory file hash mismatch")
    trajectory = _load_coupled_trajectory(path)
    digest = trajectory_digest(trajectory)
    if digest != case["trajectory_digest"]:
        raise ValueError("showcase trajectory content digest mismatch")
    spec = case["spec"]
    return _Source(
        case_id=str(spec["case_id"]),
        title=str(spec["title"]),
        subtitle=str(spec["subtitle"]),
        camera_azimuth_deg=float(spec["camera_azimuth_deg"]),
        result=dict(case["result"]),
        trajectory_hash=str(case["trajectory_hash"]),
        trajectory_digest=digest,
        trajectory=trajectory,
    )


def _cinematic_timeline(source: _Source, fps: int) -> tuple[float, ...]:
    pass_time = float(source.result["pass_contact_time_sec"])
    shot_time = float(source.result["shot_contact_time_sec"])
    end = min(float(source.trajectory["time"][-1]), shot_time + 5.0)
    return _segments(
        (
            (max(0.0, pass_time - 1.30), pass_time - 0.35, 1.0),
            (pass_time - 0.35, pass_time + 0.65, 0.42),
            (pass_time + 0.65, shot_time - 0.45, 1.0),
            (shot_time - 0.45, shot_time + 0.90, 0.38),
            (shot_time + 0.90, end, 0.85),
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
        count = max(1, int(math.ceil((end - start) / speed * fps)))
        values.extend(min(end, start + index / fps * speed) for index in range(count))
    return tuple(values)


def _write_frames(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    renderer: Any,
    sources: tuple[_Source, ...],
    timelines: tuple[tuple[float, ...], ...],
    stream: BinaryIO,
) -> None:
    ball_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_joint = int(model.body_jntadr[ball_body])
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    shooter_joint_qpos = _joint_qpos(model, mujoco, "")
    passer_joint_qpos = _joint_qpos(model, mujoco, "passer_")
    passer_free = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "passer_floating_base_joint")
    passer_qpos = int(model.jnt_qposadr[passer_free])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    for source, timeline in zip(sources, timelines, strict=True):
        times = np.asarray(source.trajectory["time"], dtype=np.float64)
        shot_time = float(source.result["shot_contact_time_sec"])
        for simulation_time in timeline:
            index = int(np.argmin(np.abs(times - simulation_time)))
            data.qpos[:] = model.qpos0
            data.qpos[:7] = source.trajectory["shooter_pelvis_pose"][index]
            data.qpos[shooter_joint_qpos] = source.trajectory["shooter_joint_position"][index]
            data.qpos[passer_qpos : passer_qpos + 7] = source.trajectory["passer_pelvis_pose"][
                index
            ]
            data.qpos[passer_joint_qpos] = source.trajectory["passer_joint_position"][index]
            data.qpos[ball_qpos : ball_qpos + 7] = source.trajectory["ball_pose"][index]
            mujoco.mj_forward(model, data)
            if simulation_time < shot_time + 0.20:
                camera.lookat[:] = (2.25, 0.10, 0.72)
                camera.distance = 6.15
            else:
                camera.lookat[:] = (2.85, 0.38, 0.82)
                camera.distance = 6.85
            camera.azimuth = source.camera_azimuth_deg
            camera.elevation = -8.5
            renderer.update_scene(data, camera=camera)
            _add_markers(mujoco, renderer.scene, source, index)
            frame = renderer.render().copy()
            canvas = np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)
            stream.write(np.ascontiguousarray(canvas).tobytes())


def _add_markers(mujoco: Any, scene: Any, source: _Source, index: int) -> None:
    target = np.asarray((5.02, 1.10, 1.09), dtype=np.float64)
    _append_sphere(mujoco, scene, target, 0.13, (0.12, 1.0, 0.34, 0.92))
    for angle in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False):
        ring = target + np.asarray((0.0, 0.30 * math.cos(angle), 0.30 * math.sin(angle)))
        _append_sphere(mujoco, scene, ring, 0.025, (0.10, 0.95, 0.55, 0.72))
    start = max(0, index - 90)
    indices = np.linspace(start, index, min(22, index - start + 1), dtype=int)
    for trail_index, alpha in zip(
        indices,
        np.linspace(0.03, 0.60, len(indices)),
        strict=True,
    ):
        _append_sphere(
            mujoco,
            scene,
            np.asarray(source.trajectory["ball_pose"][trail_index, :3]),
            0.029,
            (0.18, 0.76, 1.0, float(alpha)),
        )
    if int(source.trajectory["shooter_phase_correction"][index]) != 0:
        shooter = np.asarray(source.trajectory["shooter_pelvis_pose"][index, :3])
        _append_sphere(
            mujoco,
            scene,
            shooter + np.asarray((0.0, 0.0, 1.25)),
            0.055,
            (1.0, 0.48, 0.08, 0.95),
        )


def _write_label_files(
    root: Path,
    sources: tuple[_Source, ...],
) -> tuple[tuple[Path, Path], ...]:
    root.mkdir(parents=True, exist_ok=True)
    result: list[tuple[Path, Path]] = []
    for index, source in enumerate(sources, start=1):
        heading = root / f"heading-{index}.txt"
        metric = root / f"metric-{index}.txt"
        heading.write_text(
            f"CHALLENGE {index}/5 · {source.title}",
            encoding="utf-8",
        )
        metric.write_text(
            f"{source.subtitle} · SHOT {float(source.result['shot_peak_ball_speed_mps']):.2f} m/s "
            f"· CROSS {float(source.result['goal_crossing_z_m']):.2f} m · "
            f"ERROR {float(source.result['target_error_m']):.2f} m",
            encoding="utf-8",
        )
        result.append((heading, metric))
    return tuple(result)


def _ffmpeg_command(
    *,
    ffmpeg: str,
    output: Path,
    fps: int,
    durations: tuple[float, ...],
    label_files: tuple[tuple[Path, Path], ...],
) -> list[str]:
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font_option = f"fontfile={_escape_filtergraph_option(str(font))}:" if font.is_file() else ""
    title = _escape_filtergraph_option("ROSClaw · G1 FIVE-CHALLENGE SHOWCASE")
    footer = _escape_filtergraph_option("STRICT CPU MUJOCO · SIM ONLY · FIVE REAL PHYSICS ROLLOUTS")
    filters = [
        "drawbox=x=0:y=0:w=iw:h=138:color=0x040913@0.84:t=fill",
        "drawbox=x=0:y=h-66:w=iw:h=66:color=0x040913@0.84:t=fill",
        f"drawtext={font_option}text={title}:expansion=none:x=32:y=14:fontsize=34:fontcolor=white",
        f"drawtext={font_option}text={footer}:expansion=none:x=32:y=h-43:fontsize=20:fontcolor=0x8DD8FF",
    ]
    offset = 0.0
    for duration, (heading, metric) in zip(durations, label_files, strict=True):
        end = offset + duration
        enable = f"enable='between(t,{offset:.6f},{end:.6f})'"
        filters.extend(
            (
                f"drawtext={font_option}textfile={_escape_filtergraph_option(str(heading))}:"
                f"expansion=none:x=32:y=55:fontsize=23:fontcolor=0x65F59A:{enable}",
                f"drawtext={font_option}textfile={_escape_filtergraph_option(str(metric))}:"
                f"expansion=none:x=32:y=94:fontsize=18:fontcolor=0xFFD166:{enable}",
            )
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


__all__ = [
    "G1CoupledShowcaseVideoResult",
    "render_g1_coupled_showcase_video",
]
