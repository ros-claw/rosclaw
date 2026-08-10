"""Cinematic 1080p export of strict three-player G1 showcase evidence."""

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
from rosclaw.simforge.g1_stadium_scene import build_g1_three_player_stadium_model
from rosclaw.simforge.g1_three_player_showcase import (
    three_player_goal_spec,
    three_player_goalkeeper_config,
)

_WIDTH = 1280
_HEIGHT = 720
_OUTPUT_WIDTH = 1920
_OUTPUT_HEIGHT = 1080


@dataclass(frozen=True)
class G1ThreePlayerShowcaseVideoResult:
    output_path: str
    manifest_path: str
    video_hash: str
    evidence_report_hash: str
    source_trajectory_hash: str
    source_trajectory_digest: str
    renderer_hash: str
    fps: int
    width: int
    height: int
    frame_count: int
    duration_sec: float
    visualization_only: bool = True
    simultaneous_three_body_physics: bool = True
    pixels_used_for_promotion: bool = False
    schema_version: str = "rosclaw.g1_goalforge.three_player_showcase_video.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "label_source": "strict_three_player_shared_world_evidence",
            "generates_task_evidence": False,
        }


@dataclass(frozen=True)
class _Frame:
    simulation_time_sec: float
    view: str


def render_g1_three_player_showcase_video(
    *,
    evidence_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
) -> G1ThreePlayerShowcaseVideoResult:
    """Render one long three-player sequence plus two alternate-angle replays."""

    evidence = evidence_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("three-player video must be outside the source checkout")
    if output.suffix.lower() != ".mp4":
        raise ValueError("three-player video output must use .mp4")
    if not 10 <= fps <= 60:
        raise ValueError("three-player video fps must be in [10, 60]")
    manifest = output.with_suffix(".json")
    if output.exists() or manifest.exists():
        raise FileExistsError("three-player video or manifest already exists")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for three-player video export")

    # This must precede asset qualification because qualification imports
    # MuJoCo. Once imported, changing MUJOCO_GL cannot repair a GLFW context.
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        report = json.loads(evidence.read_text(encoding="utf-8"))
        if report.get("passed") is not True or report.get("strict_replay") is not True:
            raise ValueError("three-player video requires passing strict-replay evidence")
        if report.get("simultaneous_three_body_physics") is not True:
            raise ValueError("evidence is not a simultaneous three-body rollout")
        request_path = evidence.parent / "request.json"
        if _file_hash(request_path) != report["request_hash"]:
            raise ValueError("three-player request file hash mismatch")
        request = json.loads(request_path.read_text(encoding="utf-8"))
        trajectory_path = evidence.parent / "trajectory.npz"
        if _file_hash(trajectory_path) != report["trajectory_hash"]:
            raise ValueError("three-player trajectory file hash mismatch")
        trajectory = _load_trajectory(trajectory_path)
        digest = trajectory_digest(trajectory)
        if digest != report["trajectory_digest"]:
            raise ValueError("three-player trajectory content digest mismatch")
        qualification = qualify_g1_assets(asset_root)
        qualification.require_eligible()
        if qualification.body_hash != report["body_hash"]:
            raise ValueError("three-player video Body hash does not match evidence")
        timeline = _timeline(report, trajectory, fps)
        output.parent.mkdir(parents=True, exist_ok=True)
        import mujoco

        goal = three_player_goal_spec()
        keeper = three_player_goalkeeper_config()
        request_passer_origin = request["passer_origin_m"]
        passer_origin = (
            float(request_passer_origin[0]),
            float(request_passer_origin[1]),
            float(request_passer_origin[2]),
        )
        model = build_g1_three_player_stadium_model(
            asset_root.expanduser().resolve(),
            passer_origin_m=passer_origin,
            goalkeeper_origin_m=(
                goal.plane_x_m - keeper.depth_from_goal_line_m,
                0.0,
                0.0,
            ),
            spec=goal,
        )
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), _WIDTH)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), _HEIGHT)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=_HEIGHT, width=_WIDTH)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-three-player-video-") as temp:
                labels = _write_labels(Path(temp), report)
                process = subprocess.Popen(
                    _ffmpeg_command(
                        ffmpeg=ffmpeg,
                        output=output,
                        fps=fps,
                        labels=labels,
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
                        trajectory=trajectory,
                        timeline=timeline,
                        target=np.asarray(request["physical_scoring_target_m"], dtype=np.float64),
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
                    raise RuntimeError(f"three-player ffmpeg failed ({code}): {stderr[-2000:]}")
        finally:
            renderer.close()
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl

    result = G1ThreePlayerShowcaseVideoResult(
        output_path=str(output),
        manifest_path=str(manifest),
        video_hash=_file_hash(output),
        evidence_report_hash=_file_hash(evidence),
        source_trajectory_hash=_file_hash(trajectory_path),
        source_trajectory_digest=digest,
        renderer_hash=_file_hash(Path(__file__)),
        fps=fps,
        width=_OUTPUT_WIDTH,
        height=_OUTPUT_HEIGHT,
        frame_count=len(timeline),
        duration_sec=len(timeline) / fps,
    )
    manifest.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _timeline(
    report: dict[str, Any],
    trajectory: dict[str, np.ndarray],
    fps: int,
) -> tuple[_Frame, ...]:
    result = report["result"]
    pass_time = float(result["pass_contact_time_sec"])
    shot_time = float(result["shot_contact_time_sec"])
    end = min(float(trajectory["time"][-1]), shot_time + 4.8)
    frames: list[_Frame] = []

    def add(start: float, stop: float, speed: float, view: str) -> None:
        count = max(1, int(math.ceil((stop - start) / speed * fps)))
        frames.extend(
            _Frame(min(stop, start + index / fps * speed), view) for index in range(count)
        )

    # Full event: wide establishes all three physical agents and long spacing.
    add(0.0, pass_time - 0.75, 1.45, "wide")
    add(pass_time - 0.75, pass_time + 0.55, 0.62, "pass")
    add(pass_time + 0.55, shot_time - 0.45, 1.0, "follow")
    add(shot_time - 0.45, shot_time + 1.15, 0.48, "goal")
    add(shot_time + 1.15, end, 0.85, "wide")
    # Alternate-angle replays keep the video long enough for close inspection.
    add(pass_time - 0.80, pass_time + 1.25, 0.45, "pass_close")
    add(shot_time - 0.70, shot_time + 1.60, 0.38, "keeper")
    add(end - 0.70, end, 0.55, "wide")
    return tuple(frames)


def _load_trajectory(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        value = {name: archive[name] for name in archive.files}
    expected = {
        "time": (),
        "ball_pose": (7,),
        "passer_pelvis_pose": (7,),
        "shooter_pelvis_pose": (7,),
        "goalkeeper_pelvis_pose": (7,),
        "passer_joint_position": (29,),
        "shooter_joint_position": (29,),
        "goalkeeper_joint_position": (29,),
    }
    for name, shape in expected.items():
        array = np.asarray(value.get(name))
        if array.ndim != len(shape) + 1 or array.shape[1:] != shape:
            raise ValueError(f"three-player trajectory {name} has invalid shape")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"three-player trajectory {name} is non-finite")
    time = np.asarray(value["time"], dtype=np.float64)
    if len(time) < 2 or not np.all(np.diff(time) > 0.0):
        raise ValueError("three-player trajectory time must be strictly increasing")
    if len({len(np.asarray(value[name])) for name in expected}) != 1:
        raise ValueError("three-player trajectory arrays do not share one timeline")
    return value


def _write_frames(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    renderer: Any,
    trajectory: dict[str, np.ndarray],
    timeline: tuple[_Frame, ...],
    target: np.ndarray,
    stream: BinaryIO,
) -> None:
    ball_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_joint = int(model.body_jntadr[ball_body])
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    joint_qpos = {
        role: _joint_qpos(model, mujoco, prefix)
        for role, prefix in (
            ("shooter", ""),
            ("passer", "passer_"),
            ("goalkeeper", "goalkeeper_"),
        )
    }
    free_qpos = {"shooter": 0}
    for role, prefix in (("passer", "passer_"), ("goalkeeper", "goalkeeper_")):
        free = _id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint")
        free_qpos[role] = int(model.jnt_qposadr[free])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    time = np.asarray(trajectory["time"], dtype=np.float64)
    for frame in timeline:
        index = int(np.argmin(np.abs(time - frame.simulation_time_sec)))
        data.qpos[:] = model.qpos0
        for role in ("shooter", "passer", "goalkeeper"):
            data.qpos[free_qpos[role] : free_qpos[role] + 7] = trajectory[f"{role}_pelvis_pose"][
                index
            ]
            data.qpos[joint_qpos[role]] = trajectory[f"{role}_joint_position"][index]
        data.qpos[ball_qpos : ball_qpos + 7] = trajectory["ball_pose"][index]
        mujoco.mj_forward(model, data)
        _set_camera(camera, frame.view, trajectory, index)
        renderer.update_scene(data, camera=camera)
        _add_markers(mujoco, renderer.scene, trajectory, target, index)
        rendered = renderer.render().copy()
        stream.write(np.ascontiguousarray(rendered).tobytes())


def _set_camera(camera: Any, view: str, trajectory: dict[str, np.ndarray], index: int) -> None:
    ball = np.asarray(trajectory["ball_pose"][index, :3], dtype=np.float64)
    if view == "wide":
        camera.lookat[:] = (3.55, 0.0, 0.72)
        camera.distance, camera.azimuth, camera.elevation = 10.0, 91.0, -11.0
    elif view == "pass":
        camera.lookat[:] = (2.55, -0.05, 0.66)
        camera.distance, camera.azimuth, camera.elevation = 6.3, 92.0, -8.0
    elif view == "pass_close":
        camera.lookat[:] = ball + np.asarray((0.0, 0.0, 0.42))
        camera.distance, camera.azimuth, camera.elevation = 4.7, 116.0, -6.0
    elif view == "follow":
        camera.lookat[:] = (3.6, 0.18, 0.72)
        camera.distance, camera.azimuth, camera.elevation = 7.8, 93.0, -8.0
    elif view == "keeper":
        camera.lookat[:] = (6.45, 0.35, 0.78)
        camera.distance, camera.azimuth, camera.elevation = 4.9, 196.0, -5.0
    else:
        camera.lookat[:] = (5.75, 0.35, 0.82)
        camera.distance, camera.azimuth, camera.elevation = 6.4, 110.0, -7.0


def _add_markers(
    mujoco: Any,
    scene: Any,
    trajectory: dict[str, np.ndarray],
    target: np.ndarray,
    index: int,
) -> None:
    marker_target = target + np.asarray((0.02, 0.0, 0.0))
    for angle in np.linspace(0.0, 2.0 * math.pi, 28, endpoint=False):
        ring = marker_target + np.asarray((0.0, 0.10 * math.cos(angle), 0.10 * math.sin(angle)))
        _append_sphere(mujoco, scene, ring, 0.012, (0.16, 1.0, 0.38, 0.88))
    start = max(0, index - 100)
    indices = np.linspace(start, index, min(28, index - start + 1), dtype=int)
    for trail_index, alpha in zip(indices, np.linspace(0.03, 0.64, len(indices)), strict=True):
        _append_sphere(
            mujoco,
            scene,
            np.asarray(trajectory["ball_pose"][trail_index, :3]),
            0.026,
            (0.20, 0.78, 1.0, float(alpha)),
        )


def _write_labels(root: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    heading = root / "heading.txt"
    metric = root / "metric.txt"
    result = report["result"]
    heading.write_text(
        "THREE G1 · LONG RELAY · REACTIVE GOALKEEPER",
        encoding="utf-8",
    )
    metric.write_text(
        f"PASS {float(report['pass_distance_m']):.2f} m / {float(result['pass_delivery_error_m']) * 100:.1f} cm  ·  "
        f"SHOT {float(report['shot_distance_m']):.2f} m / {float(result['target_error_m']) * 1000:.1f} mm  ·  "
        f"KEEPER MOVE {float(result['goalkeeper_lateral_displacement_m']):.2f} m",
        encoding="utf-8",
    )
    return heading, metric


def _ffmpeg_command(*, ffmpeg: str, output: Path, fps: int, labels: tuple[Path, Path]) -> list[str]:
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font_option = f"fontfile={_escape_filtergraph_option(str(font))}:" if font.is_file() else ""
    footer = _escape_filtergraph_option(
        "STRICT CPU MUJOCO REPLAY · SIM ONLY · ONE SHARED PHYSICS WORLD · NO PIXEL SCORING"
    )
    filters = (
        "scale=1920:1080:flags=lanczos",
        "drawbox=x=0:y=0:w=iw:h=168:color=0x030711@0.82:t=fill",
        "drawbox=x=0:y=ih-74:w=iw:h=74:color=0x030711@0.82:t=fill",
        f"drawtext={font_option}textfile={_escape_filtergraph_option(str(labels[0]))}:"
        "expansion=none:x=46:y=22:fontsize=45:fontcolor=white",
        f"drawtext={font_option}textfile={_escape_filtergraph_option(str(labels[1]))}:"
        "expansion=none:x=46:y=92:fontsize=27:fontcolor=0x65F59A",
        f"drawtext={font_option}text={footer}:expansion=none:"
        "x=46:y=h-48:fontsize=24:fontcolor=0x8DD8FF",
    )
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


__all__ = [
    "G1ThreePlayerShowcaseVideoResult",
    "render_g1_three_player_showcase_video",
]
