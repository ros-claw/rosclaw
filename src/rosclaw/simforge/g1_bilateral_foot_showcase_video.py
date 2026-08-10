"""Unified-stadium 1080p render of physical left/right-foot evidence."""

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
from rosclaw.simforge.g1_coupled_relay_video import _id, _joint_qpos
from rosclaw.simforge.g1_hat_trick_video import (
    _append_sphere,
    _escape_filtergraph_option,
)
from rosclaw.simforge.g1_stadium_scene import G1TrainingGoalSpec, build_g1_stadium_model

_WIDTH = 1280
_HEIGHT = 720
_OUTPUT_WIDTH = 1920
_OUTPUT_HEIGHT = 1080


@dataclass(frozen=True)
class G1BilateralFootVideoResult:
    output_path: str
    manifest_path: str
    video_hash: str
    evidence_hash: str
    renderer_hash: str
    fps: int
    width: int
    height: int
    frame_count: int
    duration_sec: float
    physical_left_foot: bool = True
    physical_right_foot: bool = True
    visualization_only: bool = True
    pixels_used_for_scoring: bool = False
    schema_version: str = "rosclaw.g1_goalforge.bilateral_foot_video.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Case:
    foot: str
    corner: str
    target: np.ndarray
    result: dict[str, Any]
    trajectory: dict[str, np.ndarray]


@dataclass(frozen=True)
class _Frame:
    simulation_time_sec: float
    view: str


def render_g1_bilateral_foot_showcase_video(
    *,
    evidence_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
) -> G1BilateralFootVideoResult:
    """Render both strict physical cases; no image mirroring is performed."""

    evidence = evidence_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("bilateral-foot video must be outside the source checkout")
    if output.suffix.lower() != ".mp4":
        raise ValueError("bilateral-foot video output must use .mp4")
    if not 10 <= fps <= 60:
        raise ValueError("bilateral-foot video fps must be in [10, 60]")
    manifest = output.with_suffix(".json")
    if output.exists() or manifest.exists():
        raise FileExistsError("bilateral-foot video or manifest already exists")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for bilateral-foot video export")

    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        report = json.loads(evidence.read_text(encoding="utf-8"))
        if report.get("passed") is not True or len(report.get("cases", ())) != 2:
            raise ValueError("bilateral-foot video requires two passing cases")
        qualification = qualify_g1_assets(asset_root)
        qualification.require_eligible()
        if qualification.body_hash != report["body_hash"]:
            raise ValueError("bilateral-foot video Body hash does not match evidence")
        cases = tuple(_load_case(case, checkout) for case in report["cases"])
        if {case.foot for case in cases} != {"left", "right"}:
            raise ValueError("bilateral-foot video requires actual left and right cases")
        timelines = tuple(_timeline(case, fps) for case in cases)
        output.parent.mkdir(parents=True, exist_ok=True)

        import mujoco

        with tempfile.TemporaryDirectory(prefix="rosclaw-bilateral-video-") as temp:
            labels = _write_labels(Path(temp), cases)
            process = subprocess.Popen(
                _ffmpeg_command(
                    ffmpeg=ffmpeg,
                    output=output,
                    fps=fps,
                    labels=labels,
                    durations=tuple(len(value) / fps for value in timelines),
                ),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if process.stdin is None:
                raise RuntimeError("ffmpeg raw-video pipe is unavailable")
            try:
                for case, timeline in zip(cases, timelines, strict=True):
                    _write_case_frames(
                        mujoco=mujoco,
                        asset_root=asset_root,
                        case=case,
                        timeline=timeline,
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
                raise RuntimeError(f"bilateral-foot ffmpeg failed ({code}): {stderr[-2000:]}")
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl

    frame_count = sum(len(value) for value in timelines)
    result = G1BilateralFootVideoResult(
        output_path=str(output),
        manifest_path=str(manifest),
        video_hash=_file_hash(output),
        evidence_hash=_file_hash(evidence),
        renderer_hash=_file_hash(Path(__file__)),
        fps=fps,
        width=_OUTPUT_WIDTH,
        height=_OUTPUT_HEIGHT,
        frame_count=frame_count,
        duration_sec=frame_count / fps,
    )
    manifest.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _load_case(value: dict[str, Any], checkout: Path) -> _Case:
    if value.get("passed") is not True or value.get("strict_replay") is not True:
        raise ValueError("bilateral-foot source case is not strict passing evidence")
    path = Path(str(value["trajectory_path"])).expanduser().resolve()
    if path == checkout or checkout in path.parents:
        raise ValueError("bilateral-foot trajectory must be outside the source checkout")
    if _file_hash(path) != value["trajectory_hash"]:
        raise ValueError("bilateral-foot trajectory file hash mismatch")
    with np.load(path, allow_pickle=False) as archive:
        trajectory = {name: archive[name] for name in archive.files}
    if trajectory_digest(trajectory) != value["trajectory_digest"]:
        raise ValueError("bilateral-foot trajectory digest mismatch")
    for name, shape in {
        "time": (),
        "pelvis_pose": (7,),
        "joint_position": (29,),
        "ball_pose": (7,),
    }.items():
        array = np.asarray(trajectory.get(name))
        if array.ndim != len(shape) + 1 or array.shape[1:] != shape:
            raise ValueError(f"bilateral-foot trajectory {name} has invalid shape")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"bilateral-foot trajectory {name} is non-finite")
    return _Case(
        foot=str(value["kick_foot"]),
        corner=str(value["declared_corner"]),
        target=np.asarray(value["target_m"], dtype=np.float64),
        result=dict(value["result"]),
        trajectory=trajectory,
    )


def _timeline(case: _Case, fps: int) -> tuple[_Frame, ...]:
    contact = float(case.result["ball_contact_time_sec"])
    end = min(float(case.trajectory["time"][-1]), contact + 4.0)
    frames: list[_Frame] = []

    def add(start: float, stop: float, speed: float, view: str) -> None:
        count = max(1, int(math.ceil((stop - start) / speed * fps)))
        frames.extend(
            _Frame(min(stop, start + index / fps * speed), view) for index in range(count)
        )

    add(0.0, contact - 0.85, 1.45, "wide")
    add(contact - 0.85, contact + 0.90, 0.48, "strike")
    add(contact + 0.90, end, 0.82, "goal")
    add(contact - 0.70, contact + 1.25, 0.38, "replay")
    return tuple(frames)


def _write_case_frames(
    *,
    mujoco: Any,
    asset_root: Path,
    case: _Case,
    timeline: tuple[_Frame, ...],
    stream: BinaryIO,
) -> None:
    goal = G1TrainingGoalSpec(
        plane_x_m=5.0,
        width_m=2.8,
        height_m=1.7,
        depth_m=1.0,
        target_y_m=float(case.target[1]),
        target_z_m=float(case.target[2]),
        precision_radius_m=0.10,
    )
    model = build_g1_stadium_model(asset_root.expanduser().resolve(), goal)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), _WIDTH)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), _HEIGHT)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=_HEIGHT, width=_WIDTH)
    try:
        ball_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        ball_joint = int(model.body_jntadr[ball_body])
        ball_qpos = int(model.jnt_qposadr[ball_joint])
        joints = _joint_qpos(model, mujoco, "")
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        time = np.asarray(case.trajectory["time"], dtype=np.float64)
        for frame in timeline:
            index = int(np.argmin(np.abs(time - frame.simulation_time_sec)))
            data.qpos[:] = model.qpos0
            data.qpos[:7] = case.trajectory["pelvis_pose"][index]
            data.qpos[joints] = case.trajectory["joint_position"][index]
            data.qpos[ball_qpos : ball_qpos + 7] = case.trajectory["ball_pose"][index]
            mujoco.mj_forward(model, data)
            _set_camera(camera, frame.view, case, index)
            renderer.update_scene(data, camera=camera)
            _add_markers(mujoco, renderer.scene, case, index)
            stream.write(np.ascontiguousarray(renderer.render()).tobytes())
    finally:
        renderer.close()


def _set_camera(camera: Any, view: str, case: _Case, index: int) -> None:
    ball = np.asarray(case.trajectory["ball_pose"][index, :3], dtype=np.float64)
    side = 1.0 if case.foot == "right" else -1.0
    if view == "wide":
        camera.lookat[:] = (2.45, 0.0, 0.72)
        camera.distance, camera.azimuth, camera.elevation = 7.0, 91.0, -9.0
    elif view == "strike":
        camera.lookat[:] = (1.05, side * 0.03, 0.58)
        camera.distance, camera.azimuth, camera.elevation = 3.7, 105.0, -6.0
    elif view == "replay":
        camera.lookat[:] = ball + np.asarray((0.0, 0.0, 0.35))
        camera.distance, camera.azimuth, camera.elevation = 4.4, 132.0, -4.0
    else:
        camera.lookat[:] = (3.8, side * 0.35, 0.72)
        camera.distance, camera.azimuth, camera.elevation = 5.7, 110.0, -7.0


def _add_markers(mujoco: Any, scene: Any, case: _Case, index: int) -> None:
    target = case.target + np.asarray((0.02, 0.0, 0.0))
    for angle in np.linspace(0.0, 2.0 * math.pi, 26, endpoint=False):
        point = target + np.asarray((0.0, 0.10 * math.cos(angle), 0.10 * math.sin(angle)))
        _append_sphere(mujoco, scene, point, 0.011, (0.16, 1.0, 0.38, 0.88))
    start = max(0, index - 80)
    indices = np.linspace(start, index, min(24, index - start + 1), dtype=int)
    for trail_index, alpha in zip(indices, np.linspace(0.03, 0.62, len(indices)), strict=True):
        _append_sphere(
            mujoco,
            scene,
            np.asarray(case.trajectory["ball_pose"][trail_index, :3]),
            0.025,
            (0.20, 0.78, 1.0, float(alpha)),
        )


def _write_labels(root: Path, cases: tuple[_Case, ...]) -> tuple[tuple[Path, Path], ...]:
    root.mkdir(parents=True, exist_ok=True)
    values: list[tuple[Path, Path]] = []
    for index, case in enumerate(cases):
        heading = root / f"heading-{index}.txt"
        metric = root / f"metric-{index}.txt"
        heading.write_text(
            f"{case.foot.upper()} FOOT · ACTUAL MUJOCO CONTACT · {case.corner.upper()}",
            encoding="utf-8",
        )
        metric.write_text(
            f"ERROR {float(case.result['target_error_m']) * 100:.1f} cm  ·  "
            f"BALL {float(case.result['ball_speed_mps']):.2f} m/s  ·  "
            "NO FALL / NO JOINT OR TORQUE VIOLATION",
            encoding="utf-8",
        )
        values.append((heading, metric))
    return tuple(values)


def _ffmpeg_command(
    *,
    ffmpeg: str,
    output: Path,
    fps: int,
    labels: tuple[tuple[Path, Path], ...],
    durations: tuple[float, ...],
) -> list[str]:
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font_option = f"fontfile={_escape_filtergraph_option(str(font))}:" if font.is_file() else ""
    title = _escape_filtergraph_option("ROSClaw · G1 BILATERAL PHYSICAL FOOTWORK")
    footer = _escape_filtergraph_option(
        "STRICT REPLAY · SIM ONLY · LEFT FOOT IS PHYSICS, NOT HFLIP · UNIFIED THIN-NET STADIUM"
    )
    filters = [
        "scale=1920:1080:flags=lanczos",
        "drawbox=x=0:y=0:w=iw:h=190:color=0x030711@0.84:t=fill",
        "drawbox=x=0:y=ih-74:w=iw:h=74:color=0x030711@0.84:t=fill",
        f"drawtext={font_option}text={title}:expansion=none:x=46:y=18:fontsize=43:fontcolor=white",
        f"drawtext={font_option}text={footer}:expansion=none:x=46:y=h-48:fontsize=23:fontcolor=0x8DD8FF",
    ]
    offset = 0.0
    for duration, (heading, metric) in zip(durations, labels, strict=True):
        end = offset + duration
        enable = f"enable='between(t,{offset:.6f},{end:.6f})'"
        filters.extend(
            (
                f"drawtext={font_option}textfile={_escape_filtergraph_option(str(heading))}:"
                f"expansion=none:x=46:y=78:fontsize=30:fontcolor=0x65F59A:{enable}",
                f"drawtext={font_option}textfile={_escape_filtergraph_option(str(metric))}:"
                f"expansion=none:x=46:y=133:fontsize=24:fontcolor=0xFFD166:{enable}",
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
        f"{_WIDTH}x{_HEIGHT}",
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
        "17",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["G1BilateralFootVideoResult", "render_g1_bilateral_foot_showcase_video"]
