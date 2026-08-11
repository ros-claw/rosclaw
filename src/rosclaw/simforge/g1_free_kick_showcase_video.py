"""Cinematic evidence-downstream export for the learned G1 free kick."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np

from rosclaw.simforge.backends.unitree_mujoco_backend import (
    qualify_g1_assets,
    trajectory_digest,
)
from rosclaw.simforge.g1_hat_trick_video import (
    _append_sphere,
    _escape_filtergraph_option,
    _load_trajectory,
    _sample_trajectory,
)
from rosclaw.simforge.g1_stadium_scene import (
    G1TrainingGoalSpec,
    build_g1_stadium_model,
    g1_stadium_scene_hash,
)

_RESOLUTIONS = {"720p": (1280, 720), "1080p": (1920, 1080)}


@dataclass(frozen=True)
class G1FreeKickVideoClip:
    clip_id: str
    title: str
    frame_count: int
    duration_sec: float
    playback_kind: str


@dataclass(frozen=True)
class G1FreeKickVideoResult:
    output_path: str
    manifest_path: str
    video_hash: str
    evidence_hash: str
    trajectory_hash: str
    renderer_hash: str
    fps: int
    width: int
    height: int
    frame_count: int
    duration_sec: float
    clips: tuple[G1FreeKickVideoClip, ...]
    source_evidence_passed: bool
    candidate_only: bool
    visualization_only: bool = True
    pixels_used_for_scoring: bool = False
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw.simforge.g1_free_kick_video.v5"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "clips": [asdict(clip) for clip in self.clips]}


def render_g1_free_kick_showcase_video(
    *,
    evidence_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
    resolution: str = "720p",
    allow_rejected_candidate: bool = False,
) -> G1FreeKickVideoResult:
    """Render the verified continuous rollout plus a slow-motion goal replay."""

    evidence_file = evidence_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("free-kick video must be outside the source checkout")
    if output.suffix.lower() != ".mp4":
        raise ValueError("free-kick video output must use .mp4")
    if not 10 <= fps <= 60:
        raise ValueError("free-kick video fps must be in [10, 60]")
    try:
        width, height = _RESOLUTIONS[resolution]
    except KeyError as error:
        raise ValueError("free-kick video resolution must be 720p or 1080p") from error
    manifest = output.with_suffix(".json")
    if output.exists() or manifest.exists():
        raise FileExistsError("free-kick video or manifest already exists")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for free-kick video export")
    evidence = json.loads(evidence_file.read_text(encoding="utf-8"))
    evidence_passed = evidence.get("passed") is True
    if evidence.get("strict_replay") is not True:
        raise ValueError("free-kick video requires strict-replay evidence")
    if not evidence_passed and not allow_rejected_candidate:
        raise ValueError(
            "free-kick video requires passing evidence unless rejected-candidate review is explicit"
        )
    if evidence.get("evidence_domain") != "DEVELOPMENT_SHOWCASE":
        raise ValueError("free-kick video only accepts declared development showcase evidence")
    # Asset qualification imports MuJoCo.  Select EGL before that first
    # import; setting MUJOCO_GL only at Renderer construction is too late in a
    # headless process because MuJoCo has already selected its GL backend.
    qualification = _qualify_g1_assets_headless(asset_root)
    qualification.require_eligible()
    if qualification.body_hash != evidence.get("body_hash"):
        raise ValueError("free-kick video Body hash does not match evidence")
    goal = G1TrainingGoalSpec(**dict(evidence["goal_spec"]))
    if g1_stadium_scene_hash(asset_root, goal) != evidence.get("stadium_scene_hash"):
        raise ValueError("free-kick stadium scene hash mismatch")
    trajectory_path = Path(str(evidence["trajectory_path"])).expanduser().resolve()
    if trajectory_path == checkout or checkout in trajectory_path.parents:
        raise ValueError("free-kick trajectory must be outside the source checkout")
    if _file_hash(trajectory_path) != evidence.get("trajectory_hash"):
        raise ValueError("free-kick trajectory artifact hash mismatch")
    trajectory = _load_trajectory(trajectory_path)
    if trajectory_digest(trajectory) != evidence.get("trajectory_digest"):
        raise ValueError("free-kick trajectory digest mismatch")
    result = dict(evidence["result"])
    contact_value = result.get("contact_time_sec")
    if contact_value is None:
        swing = np.flatnonzero(np.asarray(trajectory["event_phase"]) == 4)
        contact_time = (
            float(trajectory["time"][-1])
            if not swing.size
            else float(trajectory["time"][int(swing[0])]) + 0.60
        )
    else:
        contact_time = float(contact_value)
    crossing_value = result.get("goal_crossing_xyz_m")
    crossing = (
        None
        if crossing_value is None
        else (
            float(crossing_value[0]),
            float(crossing_value[1]),
            float(crossing_value[2]),
        )
    )
    intro = tuple(float(trajectory["time"][0]) for _ in range(int(1.5 * fps)))
    continuous = _uniform_timeline(
        float(trajectory["time"][0]), float(trajectory["time"][-1]), fps, 1.0
    )
    slow_motion = _uniform_timeline(contact_time - 0.55, contact_time + 0.95, fps, 0.35)
    finale = tuple(float(trajectory["time"][-1]) for _ in range(int(2.5 * fps)))
    timelines = (intro, continuous, slow_motion, finale)
    durations = tuple(len(timeline) / fps for timeline in timelines)

    output.parent.mkdir(parents=True, exist_ok=True)
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco

        model = build_g1_stadium_model(asset_root, goal)
        _configure_offscreen_framebuffer(model, width=width, height=height)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        process = subprocess.Popen(
            _ffmpeg_command(
                ffmpeg=ffmpeg,
                output=output,
                fps=fps,
                durations=durations,
                evidence=evidence,
                width=width,
                height=height,
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if process.stdin is None:
            renderer.close()
            raise RuntimeError("free-kick ffmpeg pipe is unavailable")
        try:
            _write_frames(
                mujoco=mujoco,
                model=model,
                data=data,
                renderer=renderer,
                camera=camera,
                trajectory=trajectory,
                goal=goal,
                crossing=crossing,
                contact_time=contact_time,
                timelines=timelines,
                stream=cast(BinaryIO, process.stdin),
            )
        except BaseException:
            process.stdin.close()
            process.kill()
            process.wait()
            raise
        finally:
            renderer.close()
        process.stdin.close()
        stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
        code = process.wait()
        if code:
            raise RuntimeError(f"free-kick ffmpeg failed ({code}): {stderr[-2000:]}")
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl

    target_label = (
        str(result.get("declared_target_corner", "precision_target")).replace("_", "-").upper()
    )
    challenge_label = "REGULATION PRECISION" if goal.regulation_field_enabled else target_label
    clip_specs = (
        ("01-intro", f"{challenge_label} CHALLENGE", "VERIFIED_POSE_HOLD"),
        ("02-continuous", "RUN-UP → STRIKE → RECOVERY", "STRICT_PHYSICS_REPLAY"),
        ("03-goal-cam", "TARGET AND ACTUAL CROSSING", "INTERPOLATED_SLOW_MOTION_REPLAY"),
        ("04-scorecard", f"{challenge_label} SCORECARD", "VERIFIED_FINAL_POSE_HOLD"),
    )
    clips = tuple(
        G1FreeKickVideoClip(
            clip_id=clip_id,
            title=title,
            frame_count=len(timeline),
            duration_sec=duration,
            playback_kind=kind,
        )
        for (clip_id, title, kind), timeline, duration in zip(
            clip_specs, timelines, durations, strict=True
        )
    )
    value = G1FreeKickVideoResult(
        output_path=str(output),
        manifest_path=str(manifest),
        video_hash=_file_hash(output),
        evidence_hash=_file_hash(evidence_file),
        trajectory_hash=str(evidence["trajectory_hash"]),
        renderer_hash=_file_hash(Path(__file__)),
        fps=fps,
        width=width,
        height=height,
        frame_count=sum(clip.frame_count for clip in clips),
        duration_sec=sum(clip.duration_sec for clip in clips),
        clips=clips,
        source_evidence_passed=evidence_passed,
        candidate_only=not evidence_passed,
    )
    manifest.write_text(
        json.dumps(value.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return value


def _configure_offscreen_framebuffer(model: Any, *, width: int, height: int) -> None:
    """Ensure MuJoCo's native offscreen target can hold the requested frame."""

    visual = model.vis.global_
    visual.offwidth = max(int(visual.offwidth), width)
    visual.offheight = max(int(visual.offheight), height)


def _uniform_timeline(start: float, end: float, fps: int, speed: float) -> tuple[float, ...]:
    if end <= start or speed <= 0.0:
        raise ValueError("free-kick timeline segment is invalid")
    count = max(1, int(math.ceil((end - start) / speed * fps)))
    return tuple(min(end, start + index / fps * speed) for index in range(count))


def _write_frames(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    renderer: Any,
    camera: Any,
    trajectory: dict[str, np.ndarray],
    goal: G1TrainingGoalSpec,
    crossing: tuple[float, float, float] | None,
    contact_time: float,
    timelines: tuple[tuple[float, ...], ...],
    stream: BinaryIO,
) -> None:
    ball_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    for clip_index, timeline in enumerate(timelines):
        for simulation_time in timeline:
            frame = _render_pose(
                mujoco=mujoco,
                model=model,
                data=data,
                renderer=renderer,
                camera=camera,
                trajectory=trajectory,
                goal=goal,
                crossing=crossing,
                simulation_time=simulation_time,
                contact_time=contact_time,
                ball_qpos=ball_qpos,
                intro_camera=clip_index == 0,
                goal_camera=clip_index == 2,
                final_camera=clip_index == 3,
            )
            stream.write(np.ascontiguousarray(frame).tobytes())


def _render_pose(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    renderer: Any,
    camera: Any,
    trajectory: dict[str, np.ndarray],
    goal: G1TrainingGoalSpec,
    crossing: tuple[float, float, float] | None,
    simulation_time: float,
    contact_time: float,
    ball_qpos: int,
    intro_camera: bool,
    goal_camera: bool,
    final_camera: bool,
) -> np.ndarray:
    index, pelvis, joints, ball = _sample_trajectory(trajectory, simulation_time)
    data.qpos[:] = model.qpos0
    data.qpos[:7] = pelvis
    data.qpos[7:36] = joints
    data.qpos[ball_qpos : ball_qpos + 7] = ball
    mujoco.mj_forward(model, data)
    if intro_camera:
        camera.lookat[:] = ((goal.plane_x_m - 3.4) * 0.5, 0.0, 0.64)
        camera.distance = max(9.25, goal.plane_x_m + 3.0)
        camera.azimuth = 92.0
        camera.elevation = -8.0
    elif goal_camera:
        camera.lookat[:] = (goal.plane_x_m - 1.75, 0.38, 0.72)
        camera.distance = 4.15
        camera.azimuth = 132.0
        camera.elevation = -10.0
    elif final_camera:
        camera.lookat[:] = (goal.plane_x_m - 2.6, 0.30, 0.76)
        camera.distance = max(6.3, goal.plane_x_m - 0.8)
        camera.azimuth = 112.0
        camera.elevation = -9.0
    elif simulation_time < contact_time - 1.0:
        # A fixed sideline camera makes the measured 3.4 m traversal visible;
        # a pelvis-tracking camera made the same physics look like walking in
        # place against the textureless pitch.
        camera.lookat[:] = (-1.22, 0.0, 0.68)
        camera.distance = 4.70
        camera.azimuth = 92.0
        camera.elevation = -9.0
    elif simulation_time < contact_time + 0.15:
        camera.lookat[:] = (1.55, 0.16, 0.70)
        camera.distance = 4.10
        camera.azimuth = 100.0
        camera.elevation = -9.0
    else:
        camera.lookat[:] = (goal.plane_x_m - 2.75, 0.34, 0.72)
        camera.distance = max(6.25, goal.plane_x_m - 1.2)
        camera.azimuth = 108.0
        camera.elevation = -9.0
    renderer.update_scene(data, camera=camera)
    _add_precision_ring(mujoco, renderer.scene, goal, crossing)
    _add_ball_trail(mujoco, renderer.scene, trajectory, index, contact_time)
    return renderer.render().copy()


def _add_precision_ring(
    mujoco: Any,
    scene: Any,
    goal: G1TrainingGoalSpec,
    crossing: tuple[float, float, float] | None,
) -> None:
    for angle in np.linspace(0.0, 2.0 * math.pi, 28, endpoint=False):
        position = np.asarray(
            (
                goal.plane_x_m - 0.035,
                goal.target_y_m + goal.precision_radius_m * math.cos(angle),
                goal.target_z_m + goal.precision_radius_m * math.sin(angle),
            )
        )
        _append_sphere(mujoco, scene, position, 0.014, (0.18, 1.0, 0.38, 0.92))
    _append_sphere(
        mujoco,
        scene,
        np.asarray((goal.plane_x_m - 0.04, goal.target_y_m, goal.target_z_m)),
        0.025,
        (1.0, 0.82, 0.18, 0.96),
    )
    # Cyan corners mark the physics-derived ball centre at the goal plane;
    # the green ring and yellow centre remain the declared target contract.
    if crossing is None:
        return
    for delta_y, delta_z in ((-0.04, -0.04), (-0.04, 0.04), (0.04, -0.04), (0.04, 0.04)):
        _append_sphere(
            mujoco,
            scene,
            np.asarray(
                (
                    goal.plane_x_m - 0.055,
                    crossing[1] + delta_y,
                    crossing[2] + delta_z,
                )
            ),
            0.012,
            (0.20, 0.78, 1.0, 0.98),
        )


def _add_ball_trail(
    mujoco: Any,
    scene: Any,
    trajectory: dict[str, np.ndarray],
    index: int,
    contact_time: float,
) -> None:
    if float(trajectory["time"][index]) < contact_time:
        return
    contact_index = int(np.searchsorted(trajectory["time"], contact_time, side="left"))
    start = max(contact_index, index - 65)
    indices = np.linspace(start, index, min(18, index - start + 1), dtype=int)
    for trail_index, alpha in zip(indices, np.linspace(0.08, 0.62, len(indices)), strict=True):
        _append_sphere(
            mujoco,
            scene,
            np.asarray(trajectory["ball_pose"][trail_index, :3]),
            0.027,
            (0.18, 0.74, 1.0, float(alpha)),
        )


def _ffmpeg_command(
    *,
    ffmpeg: str,
    output: Path,
    fps: int,
    durations: tuple[float, ...],
    evidence: dict[str, Any],
    width: int,
    height: int,
) -> list[str]:
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font_option = f"fontfile={_escape_filtergraph_option(str(font))}:" if font.is_file() else ""
    passed = evidence.get("passed") is True
    result = evidence["result"]
    flow = evidence.get("flow_config", {})
    regulation_field = bool(
        dict(evidence.get("goal_spec", {})).get("regulation_field_enabled", False)
    )
    teacher = result.get("loft_teacher_executed") is True
    if passed:
        title_text = "ROSClaw GoalForge · G1 LEARNED FREE KICK"
        footer_text = "CPU MUJOCO · STRICT REPLAY · CONTINUOUS PHYSICS · SIM ONLY"
    elif teacher:
        title_text = "ROSClaw GoalForge · SIM-ONLY LOFT TEACHER"
        footer_text = "DISTILLATION DATA · NOT PROMOTED · STRICT REPLAY · SIM ONLY"
    elif result.get("football_outcome_retry_recovery_executed") is True:
        title_text = "ROSClaw GoalForge · CONTINUOUS RECOVER-REASSESS-KICK"
        footer_text = "DEVELOPMENT EVIDENCE · NOT PROMOTED · STRICT REPLAY · SIM ONLY"
    elif result.get("football_outcome_model_executed") is True:
        title_text = "ROSClaw GoalForge · SUCCESS-FAILURE OUTCOME MEMORY"
        footer_text = "DEVELOPMENT EVIDENCE · NOT PROMOTED · STRICT REPLAY · SIM ONLY"
    elif result.get("ballistic_skill_memory_executed") is True:
        title_text = "ROSClaw GoalForge · FULL-STATE SKILL MEMORY"
        footer_text = "SUPPORTED SKILL ISLAND · NOT PROMOTED · STRICT REPLAY · SIM ONLY"
    elif result.get("ballistic_contact_impulse_actor_executed") is True:
        title_text = "ROSClaw GoalForge · G1 LEARNED CONTACT ACTOR"
        footer_text = "DATA-DRIVEN MUSCLE MEMORY · DEVELOPMENT · NOT PROMOTED · SIM ONLY"
    elif flow.get("contextual_phase_calibration_hash") is not None:
        title_text = "ROSClaw GoalForge · PROPRIOCEPTIVE STRIKE ROUTER"
        footer_text = "DEVELOPMENT CANDIDATE · NOT PROMOTED · STRICT REPLAY · SIM ONLY"
    else:
        title_text = "ROSClaw GoalForge · REJECTED SONIC CANDIDATE"
        footer_text = "DIAGNOSTIC ONLY · NOT PROMOTED · STRICT REPLAY · SIM ONLY"
    title = _escape_filtergraph_option(title_text)
    footer = _escape_filtergraph_option(footer_text)
    shot_distance = float(result["shot_distance_m"])
    runup = float(result["runup_distance_m"])
    runup_peak = float(result["runup_peak_speed_mps"])
    transition_delay = _optional_finite_float(result.get("handoff_to_contact_sec"))
    handoff_yaw = _optional_finite_float(result.get("handoff_yaw_rad"))
    selected_phase = int(result.get("selected_kick_phase_start_frame", -1))
    if result.get("football_outcome_retry_recovery_executed") is True:
        expert_label = "RECOVERED OUTCOME EXPERT"
    elif result.get("football_outcome_model_executed") is True:
        expert_label = "LEARNED OUTCOME EXPERT"
    elif result.get("contextual_phase_expert_executed") is True:
        expert_label = "HIGH-YAW EXPERT"
    elif result.get("ballistic_skill_memory_executed") is True:
        expert_label = str(result.get("ballistic_skill_id", "MEMORY SKILL")).upper()
    else:
        expert_label = "BASE EXPERT"
    speed = float(result["ball_speed_peak_mps"])
    plane_error = _optional_finite_float(result.get("goal_plane_target_error_m"))
    net_error = _optional_finite_float(result.get("net_capture_target_error_m"))
    final_error = float(result["final_ball_yz_target_error_m"])
    threshold = float(result["precision_radius_m"])
    skill_distance = _optional_finite_float(result.get("ballistic_skill_nearest_distance"))
    memory_suffix = (
        ""
        if result.get("ballistic_skill_memory_executed") is not True
        else f" · STATE D {'N/A' if skill_distance is None else f'{skill_distance:.3f}'}"
    )
    target_label = (
        str(result.get("declared_target_corner", "precision_target")).replace("_", "-").upper()
    )
    regulation_label = "REGULATION GOAL · " if regulation_field else ""
    headings = (
        f"{shot_distance:.2f} m SET PIECE · {regulation_label}{target_label}",
        f"{runup:.2f} m APPROACH · {runup_peak:.2f} m/s PEAK · "
        f"HANDOFF-CONTACT {_duration_text(transition_delay)}",
        f"GOAL PLANE {_metric_text(plane_error)} · LIMIT {threshold:.2f} m · SHOT {speed:.2f} m/s",
        f"{expert_label} P{selected_phase}{memory_suffix} · YAW {_angle_text(handoff_yaw)} · "
        f"PLANE {_metric_text(plane_error)} · NET {_metric_text(net_error)} · "
        f"FINAL {final_error:.3f} m",
    )
    colors = ("0xFFD166", "0x65F59A", "0x8DD8FF", "0x65F59A")
    scale = height / 720.0
    header_height = round(118 * scale)
    footer_height = round(64 * scale)
    left = round(30 * scale)
    filters = [
        f"drawbox=x=0:y=0:w=iw:h={header_height}:color=0x040913@0.84:t=fill",
        f"drawbox=x=0:y=h-{footer_height}:w=iw:h={footer_height}:color=0x040913@0.84:t=fill",
        f"drawtext={font_option}text={title}:expansion=none:x={left}:y={round(13 * scale)}:"
        f"fontsize={round(33 * scale)}:fontcolor=white",
        f"drawtext={font_option}text={footer}:expansion=none:x={left}:y=h-{round(42 * scale)}:"
        f"fontsize={round(19 * scale)}:fontcolor=0x8DD8FF",
    ]
    offset = 0.0
    for heading, color, duration in zip(headings, colors, durations, strict=True):
        end = offset + duration
        filters.append(
            f"drawtext={font_option}text={_escape_filtergraph_option(heading)}:"
            f"expansion=none:x={left}:y={round(61 * scale)}:"
            f"fontsize={round(22 * scale)}:fontcolor={color}:"
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
        f"{width}x{height}",
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


def _optional_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _qualify_g1_assets_headless(asset_root: Path) -> Any:
    """Import/qualify MuJoCo assets under EGL without leaking environment state."""

    previous = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        return qualify_g1_assets(asset_root)
    finally:
        if previous is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous


def _metric_text(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f} m"


def _angle_text(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.3f} rad"


def _duration_text(value: float | None) -> str:
    """Format a measured transition duration without inventing zero delay."""

    return "N/A" if value is None else f"{value:.2f} s"


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["G1FreeKickVideoResult", "render_g1_free_kick_showcase_video"]
