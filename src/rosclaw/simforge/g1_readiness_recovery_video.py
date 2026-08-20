"""Evidence-downstream video for frozen G1 readiness recovery episodes."""

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

_WIDTH = 640
_HEIGHT = 360


@dataclass(frozen=True)
class G1ReadinessRecoveryVideoClip:
    planner_seed: int
    source_evidence_hash: str
    source_trajectory_hash: str
    initial_speed_mps: float
    final_speed_mps: float
    initial_joint_velocity_rms_rad_s: float
    final_joint_velocity_rms_rad_s: float
    peak_tilt_rad: float
    frame_count: int
    duration_sec: float


@dataclass(frozen=True)
class G1ReadinessRecoveryVideoResult:
    output_path: str
    manifest_path: str
    video_hash: str
    evaluation_hash: str
    renderer_hash: str
    fps: int
    frame_count: int
    duration_sec: float
    clips: tuple[G1ReadinessRecoveryVideoClip, ...]
    visualization_only: bool = True
    pixels_used_for_scoring: bool = False
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw.simforge.g1_readiness_recovery_video.v1"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "clips": [asdict(clip) for clip in self.clips]}


@dataclass(frozen=True)
class _Source:
    evidence_hash: str
    trajectory_hash: str
    result: dict[str, Any]
    trajectory: dict[str, np.ndarray]


def render_g1_readiness_recovery_video(
    *,
    evaluation_path: Path,
    evidence_paths: tuple[Path, ...],
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
) -> G1ReadinessRecoveryVideoResult:
    evaluation_file = evaluation_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("readiness recovery video must be outside the checkout")
    if output.suffix.lower() != ".mp4":
        raise ValueError("readiness recovery video output must use .mp4")
    if not 10 <= fps <= 60:
        raise ValueError("readiness recovery video fps must be in [10, 60]")
    manifest = output.with_suffix(".json")
    if output.exists() or manifest.exists():
        raise FileExistsError("readiness recovery video or manifest already exists")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for readiness recovery video")
    evaluation = json.loads(evaluation_file.read_text(encoding="utf-8"))
    if (
        evaluation.get("schema_version")
        != "rosclaw.growth.g1_readiness_recovery_evaluation.v1"
        or evaluation.get("accepted") is not True
        or evaluation.get("strict_replay_all") is not True
        or evaluation.get("evidence_domain")
        != "SIM_ONLY_FROZEN_CONDITIONAL_VALIDATION"
    ):
        raise ValueError("readiness recovery video requires an accepted frozen evaluation")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if qualification.body_hash != evaluation.get("body_hash"):
        raise ValueError("readiness recovery video Body hash mismatch")
    scene_hash = g1_stadium_scene_hash(asset_root, G1TrainingGoalSpec())
    sources = _load_sources(evidence_paths, evaluation, checkout, scene_hash)
    timelines = tuple(_timeline(source.trajectory, fps) for source in sources)
    durations = tuple(len(item) / fps for item in timelines)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco

        model = build_g1_stadium_model(asset_root, G1TrainingGoalSpec())
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=_HEIGHT, width=_WIDTH)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        process = subprocess.Popen(
            _ffmpeg_command(
                ffmpeg=ffmpeg,
                output=output,
                fps=fps,
                sources=sources,
                durations=durations,
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if process.stdin is None:
            renderer.close()
            raise RuntimeError("readiness recovery ffmpeg pipe is unavailable")
        try:
            _write_frames(
                mujoco=mujoco,
                model=model,
                data=data,
                renderer=renderer,
                camera=camera,
                sources=sources,
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
            raise RuntimeError(f"readiness recovery ffmpeg failed ({code}): {stderr[-2000:]}")
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl

    clips = tuple(
        G1ReadinessRecoveryVideoClip(
            planner_seed=int(source.result["planner_seed"]),
            source_evidence_hash=source.evidence_hash,
            source_trajectory_hash=source.trajectory_hash,
            initial_speed_mps=float(source.result["initial_speed_mps"]),
            final_speed_mps=float(source.result["final_speed_mps"]),
            initial_joint_velocity_rms_rad_s=float(
                source.result["initial_joint_velocity_rms_rad_s"]
            ),
            final_joint_velocity_rms_rad_s=float(
                source.result["final_joint_velocity_rms_rad_s"]
            ),
            peak_tilt_rad=float(source.result["recovery_peak_tilt_rad"]),
            frame_count=len(timeline),
            duration_sec=duration,
        )
        for source, timeline, duration in zip(sources, timelines, durations, strict=True)
    )
    value = G1ReadinessRecoveryVideoResult(
        output_path=str(output),
        manifest_path=str(manifest),
        video_hash=_file_hash(output),
        evaluation_hash=_file_hash(evaluation_file),
        renderer_hash=_file_hash(Path(__file__)),
        fps=fps,
        frame_count=sum(item.frame_count for item in clips),
        duration_sec=sum(item.duration_sec for item in clips),
        clips=clips,
    )
    manifest.write_text(
        json.dumps(value.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return value


def _load_sources(
    evidence_paths: tuple[Path, ...],
    evaluation: dict[str, Any],
    checkout: Path,
    expected_scene_hash: str,
) -> tuple[_Source, ...]:
    if len(evidence_paths) != int(evaluation["episode_count"]):
        raise ValueError("readiness recovery video evidence count differs")
    expected_hashes = {str(item) for item in evaluation["source_evidence_hashes"]}
    sources: list[_Source] = []
    for raw_path in evidence_paths:
        path = raw_path.expanduser().resolve()
        evidence_hash = _file_hash(path)
        if evidence_hash not in expected_hashes:
            raise ValueError("readiness recovery video source is not in the evaluation")
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if (
            evidence.get("passed") is not True
            or evidence.get("strict_replay") is not True
            or evidence.get("evidence_domain")
            != "FROZEN_READINESS_RECOVERY_VALIDATION"
        ):
            raise ValueError("readiness recovery video episode is not eligible")
        if evidence.get("stadium_scene_hash") != expected_scene_hash:
            raise ValueError("readiness recovery video stadium scene hash mismatch")
        trajectory_path = Path(str(evidence["trajectory_path"])).resolve()
        if trajectory_path == checkout or checkout in trajectory_path.parents:
            raise ValueError("readiness recovery video trajectory must be outside checkout")
        trajectory_hash = _file_hash(trajectory_path)
        if trajectory_hash != evidence.get("trajectory_hash"):
            raise ValueError("readiness recovery video trajectory hash mismatch")
        trajectory = _load_trajectory(trajectory_path)
        if trajectory_digest(trajectory) != evidence.get("trajectory_digest"):
            raise ValueError("readiness recovery video trajectory digest mismatch")
        sources.append(
            _Source(
                evidence_hash=evidence_hash,
                trajectory_hash=trajectory_hash,
                result=dict(evidence["result"]),
                trajectory=trajectory,
            )
        )
    if {source.evidence_hash for source in sources} != expected_hashes:
        raise ValueError("readiness recovery video sources are incomplete")
    return tuple(sorted(sources, key=lambda item: int(item.result["planner_seed"])))


def _timeline(trajectory: dict[str, np.ndarray], fps: int) -> tuple[float, ...]:
    start = float(trajectory["time"][0])
    end = float(trajectory["time"][-1])
    intro = tuple(start for _ in range(int(round(0.70 * fps))))
    count = int(math.ceil((end - start) * fps))
    replay = tuple(min(end, start + index / fps) for index in range(count))
    finale = tuple(end for _ in range(int(round(1.10 * fps))))
    return intro + replay + finale


def _write_frames(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    renderer: Any,
    camera: Any,
    sources: tuple[_Source, ...],
    timelines: tuple[tuple[float, ...], ...],
    stream: BinaryIO,
) -> None:
    ball_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    azimuths = (92.0, 103.0, 81.0)
    for source, timeline, azimuth in zip(sources, timelines, azimuths, strict=True):
        for simulation_time in timeline:
            index, pelvis, joints, ball = _sample_trajectory(
                source.trajectory, simulation_time
            )
            data.qpos[:] = model.qpos0
            data.qpos[:7] = pelvis
            data.qpos[7:36] = joints
            data.qpos[ball_qpos : ball_qpos + 7] = ball
            mujoco.mj_forward(model, data)
            handoff_time = float(
                source.trajectory["time"][
                    int(np.searchsorted(source.trajectory["controller_mode"], 6))
                ]
            )
            blend = float(np.clip((simulation_time - handoff_time + 0.5) / 1.0, 0.0, 1.0))
            camera.lookat[:] = (
                -1.15 * (1.0 - blend) + float(pelvis[0]) * blend,
                float(pelvis[1]) * 0.25,
                0.70,
            )
            camera.distance = 4.70 * (1.0 - blend) + 3.35 * blend
            camera.azimuth = azimuth
            camera.elevation = -10.0
            renderer.update_scene(data, camera=camera)
            mode = int(source.trajectory["controller_mode"][index])
            color = (
                (0.18, 0.64, 1.0, 0.90)
                if mode == 5
                else (1.0, 0.70, 0.16, 0.96)
                if mode == 6
                else (0.20, 1.0, 0.42, 0.96)
            )
            _append_sphere(
                mujoco,
                renderer.scene,
                np.asarray(pelvis[:3]) + np.asarray((0.0, 0.0, 0.34)),
                0.035,
                color,
            )
            frame = renderer.render().copy()
            canvas = np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)
            stream.write(np.ascontiguousarray(canvas).tobytes())


def _ffmpeg_command(
    *,
    ffmpeg: str,
    output: Path,
    fps: int,
    sources: tuple[_Source, ...],
    durations: tuple[float, ...],
) -> list[str]:
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font_option = f"fontfile={_escape_filtergraph_option(str(font))}:" if font.is_file() else ""
    title = _escape_filtergraph_option(
        "ROSClaw Growth · G1 ABSTAIN-TO-STABLE NEURAL RECOVERY"
    )
    footer = _escape_filtergraph_option(
        "BLUE APPROACH · AMBER NEURAL BRAKE · GREEN STABLE HOLD · STRICT REPLAY · SIM ONLY"
    )
    filters = [
        "drawbox=x=0:y=0:w=iw:h=118:color=0x040913@0.84:t=fill",
        "drawbox=x=0:y=h-64:w=iw:h=64:color=0x040913@0.84:t=fill",
        f"drawtext={font_option}text={title}:expansion=none:x=30:y=13:fontsize=31:fontcolor=white",
        f"drawtext={font_option}text={footer}:expansion=none:x=30:y=h-42:fontsize=18:fontcolor=0x8DD8FF",
    ]
    offset = 0.0
    for source, duration in zip(sources, durations, strict=True):
        result = source.result
        heading = _escape_filtergraph_option(
            f"SEED {int(result['planner_seed'])} · ABSTAIN THEN RECOVER · "
            f"SPEED {float(result['initial_speed_mps']):.3f} TO "
            f"{float(result['final_speed_mps']):.3f} m/s · "
            f"JOINT RMS {float(result['initial_joint_velocity_rms_rad_s']):.3f} TO "
            f"{float(result['final_joint_velocity_rms_rad_s']):.3f} rad/s"
        )
        end = offset + duration
        filters.append(
            f"drawtext={font_option}text={heading}:expansion=none:x=30:y=61:"
            f"fontsize=20:fontcolor=0x65F59A:enable='between(t,{offset:.6f},{end:.6f})'"
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
    "G1ReadinessRecoveryVideoResult",
    "render_g1_readiness_recovery_video",
]
