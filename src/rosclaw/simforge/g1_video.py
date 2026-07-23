"""MuJoCo offscreen video export for recorded G1 GoalForge evidence.

The renderer consumes immutable trajectory artifacts after verification. It
cannot change task labels, receipts, Champion state, or Promotion decisions.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np

from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_bytes

_SCENE_REL = Path("g1_description/scene_with_ball.xml")
_VIDEO_KEYS = ("baseline", "same_seed_retry", "new_location_first_shot")
_VIDEO_TITLES = (
    "SHOT 1  FIXED PRIOR  TARGET MISS",
    "SHOT 2  SAME SEED RETRY  SUCCESS",
    "SHOT 3  NEW LOCATION  FIRST SHOT SUCCESS",
)
_VIDEO_COLORS = ("0xF87171", "0x4ADE80", "0x60A5FA")
_RENDER_WIDTH = 640
_RENDER_HEIGHT = 360


@dataclass(frozen=True)
class PlaybackSample:
    simulation_time_sec: float
    playback_time_sec: float


@dataclass(frozen=True)
class GoalForgeVideoClip:
    name: str
    title: str
    status: str
    target_error_m: float
    support_slip_m: float
    source_episode_id: str
    trajectory_hash: str
    playback_start_sec: float
    playback_duration_sec: float
    source_duration_sec: float
    frame_count: int


@dataclass(frozen=True)
class GoalForgeVideoResult:
    output_path: Path
    manifest_path: Path
    video_hash: str
    demo_hash: str
    width: int
    height: int
    fps: int
    duration_sec: float
    frame_count: int
    clips: tuple[GoalForgeVideoClip, ...]
    visualization_only: bool = True
    schema_version: str = "rosclaw.g1_goalforge.video_export.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "output_path": str(self.output_path),
            "manifest_path": str(self.manifest_path),
            "video_hash": self.video_hash,
            "demo_hash": self.demo_hash,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "duration_sec": self.duration_sec,
            "frame_count": self.frame_count,
            "clips": [asdict(clip) for clip in self.clips],
            "visualization_only": self.visualization_only,
            "label_source": "independently_verified_goalforge_results",
        }


@dataclass(frozen=True)
class _ClipSource:
    key: str
    title: str
    color: str
    artifact_root: Path
    trajectory_path: Path
    request_path: Path
    status: str
    target_error_m: float
    support_slip_m: float
    episode_id: str
    trajectory_hash: str
    target_y_m: float
    target_z_m: float
    contact_time_sec: float
    trajectory: dict[str, np.ndarray]


def build_playback_timeline(
    source_duration_sec: float,
    *,
    fps: int,
) -> tuple[PlaybackSample, ...]:
    """Build a deterministic timeline with a slow-motion contact window."""

    if not math.isfinite(source_duration_sec) or source_duration_sec <= 0.0:
        raise ValueError("source duration must be positive and finite")
    if not 10 <= fps <= 60:
        raise ValueError("video fps must be in [10, 60]")
    boundaries = (0.0, min(4.3, source_duration_sec), min(6.4, source_duration_sec))
    segments = (
        (boundaries[0], boundaries[1], 1.5),
        (boundaries[1], boundaries[2], 0.4),
        (boundaries[2], source_duration_sec, 1.7),
    )
    values: list[PlaybackSample] = []
    playback_offset = 0.0
    for source_start, source_end, speed in segments:
        if source_end <= source_start:
            continue
        playback_duration = (source_end - source_start) / speed
        frame_count = max(1, int(math.ceil(playback_duration * fps)))
        for frame in range(frame_count):
            playback_delta = frame / fps
            source_time = min(source_end, source_start + playback_delta * speed)
            values.append(
                PlaybackSample(
                    simulation_time_sec=source_time,
                    playback_time_sec=playback_offset + playback_delta,
                )
            )
        playback_offset += playback_duration
    if not values:
        raise ValueError("video timeline is empty")
    return tuple(values)


def render_goalforge_video(
    *,
    demo_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
) -> GoalForgeVideoResult:
    """Render a recorded three-shot GoalForge demo to H.264 MP4."""

    checkout = source_checkout.expanduser().resolve()
    demo_file = demo_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    if checkout == output or checkout in output.parents:
        raise ValueError("GoalForge video output must be outside the source checkout")
    if output.suffix.lower() != ".mp4":
        raise ValueError("GoalForge video output must use the .mp4 suffix")
    if width < 640 or height < 360 or width % 2 or height % 2:
        raise ValueError("video dimensions must be even and at least 640x360")
    if not 10 <= fps <= 60:
        raise ValueError("video fps must be in [10, 60]")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for GoalForge video export")
    scene_path = asset_root.expanduser().resolve() / _SCENE_REL
    if not scene_path.is_file():
        raise FileNotFoundError(f"GoalForge MuJoCo scene is missing: {scene_path}")
    if not demo_file.is_file():
        raise FileNotFoundError(f"GoalForge demo report is missing: {demo_file}")

    demo = _read_object(demo_file)
    if demo.get("passed") is not True:
        raise ValueError("GoalForge video requires a passing demo report")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"GoalForge video output already exists: {output}")
    manifest = output.with_suffix(".json")
    if manifest.exists():
        raise FileExistsError(f"GoalForge video manifest already exists: {manifest}")

    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco

        from rosclaw.simforge.backends.unitree_mujoco_backend import qualify_g1_assets

        qualification = qualify_g1_assets(asset_root)
        qualification.require_eligible()
        clips = _load_clips(
            demo,
            checkout,
            expected_body_hash=qualification.body_hash,
            expected_kick_prior_hash=qualification.kick_prior_hash,
        )
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(
            model,
            height=_RENDER_HEIGHT,
            width=_RENDER_WIDTH,
        )
        try:
            timelines = tuple(
                build_playback_timeline(
                    float(clip.trajectory["time"][-1]),
                    fps=fps,
                )
                for clip in clips
            )
            clip_durations = tuple(len(timeline) / fps for timeline in timelines)
            command = _ffmpeg_command(
                ffmpeg=ffmpeg,
                output=output,
                width=width,
                height=height,
                fps=fps,
                clips=clips,
                clip_durations=clip_durations,
            )
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if process.stdin is None:
                raise RuntimeError("ffmpeg raw-video input pipe is unavailable")
            try:
                _write_video_frames(
                    mujoco=mujoco,
                    model=model,
                    data=data,
                    renderer=renderer,
                    clips=clips,
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
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"ffmpeg video export failed ({return_code}): {stderr[-2000:]}")
        finally:
            renderer.close()
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl

    clip_results: list[GoalForgeVideoClip] = []
    offset = 0.0
    for clip, timeline, duration in zip(
        clips,
        timelines,
        clip_durations,
        strict=True,
    ):
        clip_results.append(
            GoalForgeVideoClip(
                name=clip.key,
                title=clip.title,
                status=clip.status,
                target_error_m=clip.target_error_m,
                support_slip_m=clip.support_slip_m,
                source_episode_id=clip.episode_id,
                trajectory_hash=clip.trajectory_hash,
                playback_start_sec=offset,
                playback_duration_sec=duration,
                source_duration_sec=float(clip.trajectory["time"][-1]),
                frame_count=len(timeline),
            )
        )
        offset += duration
    result = GoalForgeVideoResult(
        output_path=output,
        manifest_path=manifest,
        video_hash=hash_bytes(output.read_bytes()),
        demo_hash=hash_bytes(demo_file.read_bytes()),
        width=width,
        height=height,
        fps=fps,
        duration_sec=offset,
        frame_count=sum(len(timeline) for timeline in timelines),
        clips=tuple(clip_results),
    )
    manifest.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _load_clips(
    demo: dict[str, Any],
    checkout: Path,
    *,
    expected_body_hash: str,
    expected_kick_prior_hash: str,
) -> tuple[_ClipSource, ...]:
    clips = []
    for key, title, color in zip(
        _VIDEO_KEYS,
        _VIDEO_TITLES,
        _VIDEO_COLORS,
        strict=True,
    ):
        value = demo.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"GoalForge demo is missing {key}")
        artifact_root = Path(str(value.get("artifact_root", ""))).expanduser().resolve()
        if checkout == artifact_root or checkout in artifact_root.parents:
            raise ValueError("GoalForge video source artifacts must be outside the checkout")
        trajectory_path = artifact_root / "trajectory.npz"
        request_path = artifact_root / "trajectory-request.json"
        result_path = artifact_root / "result.json"
        if not all(path.is_file() for path in (trajectory_path, request_path, result_path)):
            raise FileNotFoundError(f"GoalForge episode artifacts are incomplete: {artifact_root}")
        request = _read_object(request_path)
        recorded_result = _read_object(result_path)
        scenario = request.get("scenario")
        result = value.get("result")
        receipt = value.get("receipt")
        if not isinstance(scenario, dict) or not isinstance(result, dict):
            raise ValueError(f"GoalForge episode metadata is incomplete: {key}")
        if not isinstance(receipt, dict):
            raise ValueError(f"GoalForge video requires a simulation receipt: {key}")
        if (
            receipt.get("strict_replay") is not True
            or receipt.get("independently_verified") is not True
        ):
            raise ValueError(
                f"GoalForge video requires independently verified strict replay: {key}"
            )
        if (
            receipt.get("body_hash") != expected_body_hash
            or request.get("body_hash") != expected_body_hash
        ):
            raise ValueError(f"GoalForge video body hash mismatch: {key}")
        if (
            receipt.get("kick_prior_hash") != expected_kick_prior_hash
            or request.get("kick_prior_hash") != expected_kick_prior_hash
        ):
            raise ValueError(f"GoalForge video kick-prior hash mismatch: {key}")
        expected_hashes = {
            "request_hash": hash_bytes(request_path.read_bytes()),
            "trajectory_hash": hash_bytes(trajectory_path.read_bytes()),
            "result_hash": hash_bytes(result_path.read_bytes()),
        }
        for receipt_key, observed_hash in expected_hashes.items():
            if receipt.get(receipt_key) != observed_hash:
                raise ValueError(f"GoalForge video {receipt_key} mismatch: {key}")
        if recorded_result != result:
            raise ValueError(f"GoalForge video result metadata mismatch: {key}")
        with np.load(trajectory_path, allow_pickle=False) as trajectory_archive:
            trajectory = {name: trajectory_archive[name] for name in trajectory_archive.files}
        _validate_trajectory(trajectory)
        clips.append(
            _ClipSource(
                key=key,
                title=title,
                color=color,
                artifact_root=artifact_root,
                trajectory_path=trajectory_path,
                request_path=request_path,
                status=str(result["status"]),
                target_error_m=float(result["target_error_m"]),
                support_slip_m=float(result["support_foot_slip_m"]),
                episode_id=str(receipt["episode_id"]),
                trajectory_hash=str(receipt["trajectory_hash"]),
                target_y_m=float(scenario["target_y_m"]),
                target_z_m=float(scenario["target_z_m"]),
                contact_time_sec=float(result.get("ball_contact_time_sec") or 5.25),
                trajectory=trajectory,
            )
        )
    return tuple(clips)


def _validate_trajectory(trajectory: dict[str, np.ndarray]) -> None:
    required_shapes = {
        "time": (1,),
        "pelvis_pose": (7,),
        "joint_position": (29,),
        "ball_pose": (7,),
    }
    lengths = set()
    for name, trailing_shape in required_shapes.items():
        value = np.asarray(trajectory.get(name))
        if value.ndim != len(trailing_shape) + 1 and name != "time":
            raise ValueError(f"GoalForge trajectory {name} has invalid rank")
        if name == "time":
            if value.ndim != 1:
                raise ValueError("GoalForge trajectory time has invalid rank")
        elif value.shape[1:] != trailing_shape:
            raise ValueError(f"GoalForge trajectory {name} has invalid shape")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"GoalForge trajectory {name} contains non-finite values")
        lengths.add(int(value.shape[0]))
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise ValueError("GoalForge trajectory channels are not aligned")
    times = np.asarray(trajectory["time"])
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("GoalForge trajectory time must be strictly increasing")


def _write_video_frames(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    renderer: Any,
    clips: tuple[_ClipSource, ...],
    timelines: tuple[tuple[PlaybackSample, ...], ...],
    stream: BinaryIO,
) -> None:
    ball_joint = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "ball_free",
    )
    if ball_joint < 0:
        raise ValueError("GoalForge video scene does not contain ball_free")
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    for clip, timeline in zip(clips, timelines, strict=True):
        times = clip.trajectory["time"]
        ball_poses = clip.trajectory["ball_pose"]
        for sample in timeline:
            index = int(np.searchsorted(times, sample.simulation_time_sec, side="left"))
            index = min(index, len(times) - 1)
            data.qpos[:] = model.qpos0
            data.qpos[:7] = clip.trajectory["pelvis_pose"][index]
            data.qpos[7:36] = clip.trajectory["joint_position"][index]
            data.qpos[ball_qpos : ball_qpos + 7] = ball_poses[index]
            mujoco.mj_forward(model, data)
            _position_camera(camera, sample.simulation_time_sec, clip.contact_time_sec)
            renderer.update_scene(data, camera=camera)
            _add_target_and_trail(
                mujoco=mujoco,
                scene=renderer.scene,
                target_y=clip.target_y_m,
                target_z=clip.target_z_m,
                ball_poses=ball_poses,
                index=index,
            )
            stream.write(renderer.render().tobytes())


def _position_camera(camera: Any, simulation_time: float, contact_time: float) -> None:
    if contact_time - 0.12 <= simulation_time <= contact_time + 1.15:
        camera.lookat[:] = (3.0, 0.0, 0.62)
        camera.distance = 6.2
        camera.azimuth = 90.0
        camera.elevation = -9.0
    else:
        camera.lookat[:] = (1.05, 0.0, 0.73)
        camera.distance = 3.15
        camera.azimuth = 90.0
        camera.elevation = -7.0


def _add_target_and_trail(
    *,
    mujoco: Any,
    scene: Any,
    target_y: float,
    target_z: float,
    ball_poses: np.ndarray,
    index: int,
) -> None:
    _append_visual_sphere(
        mujoco=mujoco,
        scene=scene,
        position=np.asarray((5.02, target_y, target_z), dtype=np.float64),
        radius=0.16,
        rgba=(0.20, 1.0, 0.35, 0.90),
    )
    start = max(0, index - 35)
    trail_indices = np.linspace(start, index, num=min(10, index - start + 1), dtype=int)
    for trail_index, alpha in zip(
        trail_indices,
        np.linspace(0.08, 0.45, num=len(trail_indices)),
        strict=True,
    ):
        _append_visual_sphere(
            mujoco=mujoco,
            scene=scene,
            position=np.asarray(ball_poses[trail_index, :3], dtype=np.float64),
            radius=0.035,
            rgba=(0.35, 0.80, 1.0, float(alpha)),
        )


def _append_visual_sphere(
    *,
    mujoco: Any,
    scene: Any,
    position: np.ndarray,
    radius: float,
    rgba: tuple[float, float, float, float],
) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray((radius, radius, radius), dtype=np.float64),
        position,
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _ffmpeg_command(
    *,
    ffmpeg: str,
    output: Path,
    width: int,
    height: int,
    fps: int,
    clips: tuple[_ClipSource, ...],
    clip_durations: tuple[float, ...],
) -> list[str]:
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font_option = f"fontfile={font}:" if font.is_file() else ""
    filters = [
        f"scale={width}:{height}:flags=lanczos",
        "drawbox=x=0:y=0:w=iw:h=86:color=black@0.62:t=fill",
        "drawbox=x=0:y=h-48:w=iw:h=48:color=black@0.55:t=fill",
        (
            f"drawtext={font_option}text='ROSClaw G1 GoalForge':"
            "x=28:y=18:fontsize=30:fontcolor=white"
        ),
        (
            f"drawtext={font_option}text='MuJoCo evidence replay  visualization only':"
            "x=28:y=h-35:fontsize=20:fontcolor=white@0.88"
        ),
    ]
    offset = 0.0
    for clip, duration in zip(clips, clip_durations, strict=True):
        end = offset + duration
        metrics = (
            f"{clip.title}   error {clip.target_error_m:.3f} m   slip {clip.support_slip_m:.3f} m"
        )
        filters.append(
            f"drawtext={font_option}text='{metrics}':x=28:y=55:"
            f"fontsize=19:fontcolor={clip.color}:"
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
        f"{_RENDER_WIDTH}x{_RENDER_HEIGHT}",
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
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"GoalForge JSON root must be an object: {path}")
    return value


__all__ = [
    "GoalForgeVideoClip",
    "GoalForgeVideoResult",
    "PlaybackSample",
    "build_playback_timeline",
    "render_goalforge_video",
]
