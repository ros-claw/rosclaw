"""Visualization-only export of coupled two-G1 relay evidence."""

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
from rosclaw.simforge.g1_hat_trick_video import (
    _append_sphere,
    _escape_filtergraph_option,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_DDS_JOINT_NAMES

_WIDTH = 640
_HEIGHT = 360


@dataclass(frozen=True)
class G1CoupledRelayVideoResult:
    output_path: str
    manifest_path: str
    video_hash: str
    evidence_report_hash: str
    source_trajectory_hash: str
    source_trajectory_digest: str
    fps: int
    frame_count: int
    duration_sec: float
    visualization_only: bool = True
    simultaneous_two_body_physics: bool = True
    shared_ball_state: bool = True
    schema_version: str = "rosclaw.g1_goalforge.coupled_relay_video.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "label_source": "strict_coupled_relay_evidence",
            "generates_task_evidence": False,
            "pixels_used_for_promotion": False,
        }


def render_g1_coupled_relay_video(
    *,
    evidence_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
) -> G1CoupledRelayVideoResult:
    """Render the complete simultaneous rollout without changing its evidence."""

    evidence = evidence_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("coupled relay video must be outside the source checkout")
    if output.suffix.lower() != ".mp4":
        raise ValueError("coupled relay video output must use .mp4")
    if not 10 <= fps <= 60:
        raise ValueError("coupled relay video fps must be in [10, 60]")
    manifest = output.with_suffix(".json")
    if output.exists() or manifest.exists():
        raise FileExistsError("coupled relay video or manifest already exists")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for coupled relay video export")

    report = json.loads(evidence.read_text(encoding="utf-8"))
    if report.get("passed") is not True or report.get("strict_replay") is not True:
        raise ValueError("coupled relay video requires passing strict-replay evidence")
    if report.get("simultaneous_two_body_physics") is not True:
        raise ValueError("evidence is not a simultaneous two-body rollout")
    trajectory_path = evidence.parent / "trajectory.npz"
    if _file_hash(trajectory_path) != report["trajectory_hash"]:
        raise ValueError("coupled relay trajectory file hash mismatch")
    trajectory = _load_coupled_trajectory(trajectory_path)
    content_digest = trajectory_digest(trajectory)
    if content_digest != report["trajectory_digest"]:
        raise ValueError("coupled relay trajectory content digest mismatch")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if qualification.body_hash != report["body_hash"]:
        raise ValueError("coupled relay video Body hash does not match evidence")

    timeline = _timeline(trajectory, fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco

        model = _coupled_model(asset_root.expanduser().resolve())
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=_HEIGHT, width=_WIDTH)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-coupled-relay-video-") as temp:
                metric_files = _write_metric_files(Path(temp), report)
                process = subprocess.Popen(
                    _ffmpeg_command(
                        ffmpeg=ffmpeg,
                        output=output,
                        fps=fps,
                        pass_time=float(report["result"]["pass_contact_time_sec"]),
                        shot_time=float(report["result"]["shot_contact_time_sec"]),
                        duration=len(timeline) / fps,
                        metric_files=metric_files,
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
                        target=np.asarray((5.0, 1.10, 1.09), dtype=np.float64),
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
                    raise RuntimeError(f"coupled relay ffmpeg failed ({code}): {stderr[-2000:]}")
        finally:
            renderer.close()
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl

    result = G1CoupledRelayVideoResult(
        output_path=str(output),
        manifest_path=str(manifest),
        video_hash=_file_hash(output),
        evidence_report_hash=_file_hash(evidence),
        source_trajectory_hash=_file_hash(trajectory_path),
        source_trajectory_digest=content_digest,
        fps=fps,
        frame_count=len(timeline),
        duration_sec=len(timeline) / fps,
    )
    manifest.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _timeline(trajectory: dict[str, np.ndarray], fps: int) -> tuple[float, ...]:
    end = float(trajectory["time"][-1])
    count = int(math.ceil(end * fps))
    return tuple(min(end, index / fps) for index in range(count))


def _load_coupled_trajectory(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        value = {name: archive[name] for name in archive.files}
    expected = {
        "time": (),
        "ball_pose": (7,),
        "passer_pelvis_pose": (7,),
        "shooter_pelvis_pose": (7,),
        "passer_joint_position": (29,),
        "shooter_joint_position": (29,),
    }
    for name, shape in expected.items():
        array = np.asarray(value.get(name))
        if array.ndim != len(shape) + 1 or array.shape[1:] != shape:
            raise ValueError(f"coupled relay trajectory {name} has invalid shape")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"coupled relay trajectory {name} is non-finite")
    time = np.asarray(value["time"], dtype=np.float64)
    if len(time) < 2 or not np.all(np.diff(time) > 0.0):
        raise ValueError("coupled relay trajectory time must be strictly increasing")
    lengths = {len(np.asarray(value[name])) for name in expected}
    if len(lengths) != 1:
        raise ValueError("coupled relay trajectory arrays do not share one timeline")
    for name in ("ball_pose", "passer_pelvis_pose", "shooter_pelvis_pose"):
        norm = np.linalg.norm(np.asarray(value[name])[:, 3:], axis=1)
        if np.any(norm <= 1e-12):
            raise ValueError(f"coupled relay trajectory {name} has a zero quaternion")
    return value


def _write_frames(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    renderer: Any,
    trajectory: dict[str, np.ndarray],
    timeline: tuple[float, ...],
    target: np.ndarray,
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
    times = np.asarray(trajectory["time"], dtype=np.float64)
    for simulation_time in timeline:
        index = int(np.argmin(np.abs(times - simulation_time)))
        data.qpos[:] = model.qpos0
        data.qpos[:7] = trajectory["shooter_pelvis_pose"][index]
        data.qpos[shooter_joint_qpos] = trajectory["shooter_joint_position"][index]
        data.qpos[passer_qpos : passer_qpos + 7] = trajectory["passer_pelvis_pose"][index]
        data.qpos[passer_joint_qpos] = trajectory["passer_joint_position"][index]
        data.qpos[ball_qpos : ball_qpos + 7] = trajectory["ball_pose"][index]
        mujoco.mj_forward(model, data)
        ball = np.asarray(trajectory["ball_pose"][index, :3], dtype=np.float64)
        if simulation_time < 7.0:
            camera.lookat[:] = (2.25, 0.0, 0.68)
            camera.distance = 6.1
        else:
            camera.lookat[:] = (2.75, 0.35, 0.78)
            camera.distance = 6.8
        camera.azimuth = 93.0
        camera.elevation = -9.0
        renderer.update_scene(data, camera=camera)
        _append_sphere(
            mujoco,
            renderer.scene,
            target,
            0.18,
            (0.18, 1.0, 0.35, 0.90),
        )
        start = max(0, index - 70)
        indices = np.linspace(start, index, min(18, index - start + 1), dtype=int)
        for trail_index, alpha in zip(
            indices,
            np.linspace(0.04, 0.52, len(indices)),
            strict=True,
        ):
            _append_sphere(
                mujoco,
                renderer.scene,
                np.asarray(trajectory["ball_pose"][trail_index, :3]),
                0.030,
                (0.20, 0.76, 1.0, float(alpha)),
            )
        _append_sphere(
            mujoco,
            renderer.scene,
            ball,
            0.004,
            (1.0, 1.0, 1.0, 0.0),
        )
        frame = renderer.render().copy()
        canvas = np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)
        stream.write(np.ascontiguousarray(canvas).tobytes())


def _write_metric_files(root: Path, report: dict[str, Any]) -> tuple[Path, ...]:
    root.mkdir(parents=True, exist_ok=True)
    result = report["result"]
    values = (
        "G1-A  PREPARES THE BACK-PASS  ·  G1-B HOLDS BALANCE",
        f"PASS CONTACT  ·  {float(result['pass_peak_ball_speed_mps']):.2f} m/s  ·  ONE SHARED BALL",
        (
            "FIRST-TIME HIGH FINISH  ·  "
            f"{float(result['shot_peak_ball_speed_mps']):.2f} m/s  ·  "
            f"CROSSING {float(result['goal_crossing_z_m']):.2f} m HIGH"
        ),
        "POST-KICK LOCOMOTION RECOVERY  ·  BOTH G1 STABLE",
    )
    paths = tuple(root / f"phase-{index}.txt" for index in range(len(values)))
    for path, value in zip(paths, values, strict=True):
        path.write_text(value, encoding="utf-8")
    return paths


def _ffmpeg_command(
    *,
    ffmpeg: str,
    output: Path,
    fps: int,
    pass_time: float,
    shot_time: float,
    duration: float,
    metric_files: tuple[Path, ...],
) -> list[str]:
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font_option = f"fontfile={_escape_filtergraph_option(str(font))}:" if font.is_file() else ""
    title = _escape_filtergraph_option("ROSClaw · COUPLED G1 PASS → HIGH FINISH")
    footer = _escape_filtergraph_option(
        "STRICT CPU MUJOCO REPLAY · SIM ONLY · PIXELS ARE NOT EVIDENCE"
    )
    filters = [
        "drawbox=x=0:y=0:w=iw:h=112:color=0x050A12@0.80:t=fill",
        "drawbox=x=0:y=h-70:w=iw:h=70:color=0x050A12@0.80:t=fill",
        f"drawtext={font_option}text={title}:expansion=none:x=34:y=18:fontsize=36:fontcolor=white",
        f"drawtext={font_option}text={footer}:expansion=none:x=34:y=h-46:fontsize=21:fontcolor=0x8DD8FF",
    ]
    intervals = (
        (0.0, pass_time),
        (pass_time, shot_time),
        (shot_time, min(duration, shot_time + 2.4)),
        (min(duration, shot_time + 2.4), duration),
    )
    for path, (start, end) in zip(metric_files, intervals, strict=True):
        filters.append(
            f"drawtext={font_option}textfile={_escape_filtergraph_option(str(path))}:"
            "expansion=none:x=34:y=68:fontsize=22:fontcolor=0x65F59A:"
            f"enable='between(t,{start:.6f},{end:.6f})'"
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


def _joint_qpos(model: Any, mujoco: Any, prefix: str) -> np.ndarray:
    return np.asarray(
        [
            model.jnt_qposadr[_id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + joint_name)]
            for joint_name in G1_DDS_JOINT_NAMES
        ],
        dtype=np.int64,
    )


def _id(model: Any, object_type: Any, name: str) -> int:
    import mujoco

    value = int(mujoco.mj_name2id(model, object_type, name))
    if value < 0:
        raise ValueError(f"coupled relay model is missing {name}")
    return value


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["G1CoupledRelayVideoResult", "render_g1_coupled_relay_video"]
