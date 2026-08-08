"""Cinematic, evidence-downstream export for the self-aware G1 showcase."""

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
from rosclaw.simforge.g1_hat_trick_video import (
    _add_targets_and_trail,
    _append_sphere,
    _escape_filtergraph_option,
    _load_trajectory,
    _sample_trajectory,
)

_WIDTH = 640
_HEIGHT = 360
_EMPTY_TRAJECTORY_DIGEST = "sha256:" + hashlib.sha256(b"").hexdigest()


@dataclass(frozen=True)
class G1SelfAwareVideoClip:
    clip_id: str
    title: str
    frame_count: int
    duration_sec: float
    source_trajectory_hash: str | None
    strict_replay: bool
    visualization_kind: str


@dataclass(frozen=True)
class G1SelfAwareShowcaseVideoResult:
    output_path: str
    manifest_path: str
    video_hash: str
    showcase_evidence_hash: str
    rejected_v2_evidence_hash: str
    self_aware_v3_evidence_hash: str
    renderer_hash: str
    fps: int
    frame_count: int
    duration_sec: float
    clips: tuple[G1SelfAwareVideoClip, ...]
    visualization_only: bool = True
    pixels_used_for_promotion: bool = False
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw.g1_goalforge.self_aware_showcase_video.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "clips": [asdict(clip) for clip in self.clips],
            "generates_task_evidence": False,
            "abstention_visualization": (
                "static qualified scene; the v3 source issued no motion and contains no trajectory"
            ),
            "holdout_visualized": False,
        }


@dataclass(frozen=True)
class _ChallengeSource:
    case_id: str
    title: str
    subtitle: str
    camera_azimuth_deg: float
    scenario: dict[str, Any]
    result: dict[str, Any]
    trajectory_hash: str
    trajectory: dict[str, np.ndarray]


@dataclass(frozen=True)
class _ContrastSource:
    old_result: dict[str, Any]
    old_trajectory_hash: str
    old_trajectory: dict[str, np.ndarray]
    abstention_result: dict[str, Any]
    abstention_belief: dict[str, Any]


def render_g1_self_aware_showcase_video(
    *,
    showcase_evidence_path: Path,
    rejected_v2_evidence_path: Path,
    self_aware_v3_evidence_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
) -> G1SelfAwareShowcaseVideoResult:
    """Render three strict challenges and one honest before/after safety contrast."""

    showcase_path = showcase_evidence_path.expanduser().resolve()
    v2_path = rejected_v2_evidence_path.expanduser().resolve()
    v3_path = self_aware_v3_evidence_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("self-aware showcase video must be outside the source checkout")
    if output.suffix.lower() != ".mp4":
        raise ValueError("self-aware showcase output must use .mp4")
    if not 10 <= fps <= 60:
        raise ValueError("self-aware showcase fps must be in [10, 60]")
    manifest = output.with_suffix(".json")
    if output.exists() or manifest.exists():
        raise FileExistsError("self-aware showcase video or manifest already exists")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for self-aware showcase export")

    showcase = json.loads(showcase_path.read_text(encoding="utf-8"))
    if showcase.get("passed") is not True or len(showcase.get("cases", ())) != 3:
        raise ValueError("self-aware showcase video requires three passing challenges")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if qualification.body_hash != showcase.get("body_hash"):
        raise ValueError("self-aware showcase Body hash does not match evidence")
    sources = tuple(_load_challenge(case, checkout) for case in showcase["cases"])
    contrast = _load_contrast(v2_path, v3_path, checkout)
    timelines = tuple(_cinematic_timeline(source, fps) for source in sources)
    contrast_timeline = _contrast_timeline(contrast, fps)
    finale_frames = 4 * fps
    durations = tuple(len(timeline) / fps for timeline in timelines)
    contrast_duration = len(contrast_timeline) / fps
    finale_duration = finale_frames / fps

    output.parent.mkdir(parents=True, exist_ok=True)
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco

        scene = qualification.asset_root / "g1_description/scene_with_ball.xml"
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        comparison_data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=_HEIGHT, width=_WIDTH)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-self-aware-video-") as temp:
                labels = _write_label_files(Path(temp), sources, contrast)
                process = subprocess.Popen(
                    _ffmpeg_command(
                        ffmpeg=ffmpeg,
                        output=output,
                        fps=fps,
                        durations=durations,
                        contrast_duration=contrast_duration,
                        finale_duration=finale_duration,
                        label_files=labels,
                    ),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                if process.stdin is None:
                    raise RuntimeError("self-aware showcase ffmpeg pipe is unavailable")
                try:
                    _write_frames(
                        mujoco=mujoco,
                        model=model,
                        data=data,
                        comparison_data=comparison_data,
                        renderer=renderer,
                        sources=sources,
                        timelines=timelines,
                        contrast=contrast,
                        contrast_timeline=contrast_timeline,
                        finale_frames=finale_frames,
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
                    raise RuntimeError(
                        f"self-aware showcase ffmpeg failed ({code}): {stderr[-2000:]}"
                    )
        finally:
            renderer.close()
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl

    clips = [
        G1SelfAwareVideoClip(
            clip_id=source.case_id,
            title=source.title,
            frame_count=len(timeline),
            duration_sec=duration,
            source_trajectory_hash=source.trajectory_hash,
            strict_replay=True,
            visualization_kind="STRICT_PHYSICS_REPLAY",
        )
        for source, timeline, duration in zip(sources, timelines, durations, strict=True)
    ]
    clips.extend(
        (
            G1SelfAwareVideoClip(
                clip_id="04-blind-vs-self-aware",
                title="BLIND ACTION vs SELF-AWARE ABSTENTION",
                frame_count=len(contrast_timeline),
                duration_sec=contrast_duration,
                source_trajectory_hash=contrast.old_trajectory_hash,
                strict_replay=True,
                visualization_kind="PHYSICS_REPLAY_VS_STATIC_NO_MOTION_RECEIPT",
            ),
            G1SelfAwareVideoClip(
                clip_id="05-scorecard",
                title="THREE CHALLENGES · THREE SAFE HITS",
                frame_count=finale_frames,
                duration_sec=finale_duration,
                source_trajectory_hash=sources[-1].trajectory_hash,
                strict_replay=True,
                visualization_kind="FINAL_VERIFIED_POSE_HOLD",
            ),
        )
    )
    result = G1SelfAwareShowcaseVideoResult(
        output_path=str(output),
        manifest_path=str(manifest),
        video_hash=_file_hash(output),
        showcase_evidence_hash=_file_hash(showcase_path),
        rejected_v2_evidence_hash=_file_hash(v2_path),
        self_aware_v3_evidence_hash=_file_hash(v3_path),
        renderer_hash=_file_hash(Path(__file__)),
        fps=fps,
        frame_count=sum(clip.frame_count for clip in clips),
        duration_sec=sum(clip.duration_sec for clip in clips),
        clips=tuple(clips),
    )
    manifest.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _load_challenge(case: dict[str, Any], checkout: Path) -> _ChallengeSource:
    if case.get("passed") is not True or case.get("strict_replay") is not True:
        raise ValueError("self-aware video source is not a strict passing challenge")
    spec = dict(case["spec"])
    scenario = dict(spec["scenario"])
    if scenario.get("partition") != "development":
        raise ValueError("self-aware video may only render declared development challenges")
    path = Path(str(case["trajectory_path"])).expanduser().resolve()
    if path == checkout or checkout in path.parents:
        raise ValueError("self-aware source trajectory must be outside the checkout")
    if _file_hash(path) != case["trajectory_hash"]:
        raise ValueError("self-aware showcase trajectory hash mismatch")
    trajectory = _load_trajectory(path)
    if trajectory_digest(trajectory) != case["trajectory_digest"]:
        raise ValueError("self-aware showcase trajectory digest mismatch")
    return _ChallengeSource(
        case_id=str(spec["case_id"]),
        title=str(spec["title"]),
        subtitle=str(spec["subtitle"]),
        camera_azimuth_deg=float(spec["camera_azimuth_deg"]),
        scenario=scenario,
        result=dict(case["result"]),
        trajectory_hash=str(case["trajectory_hash"]),
        trajectory=trajectory,
    )


def _load_contrast(v2_path: Path, v3_path: Path, checkout: Path) -> _ContrastSource:
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    v3 = json.loads(v3_path.read_text(encoding="utf-8"))
    if v2.get("decision") != "REJECTED" or v3.get("decision") != "SIM_CANDIDATE":
        raise ValueError("self-aware contrast requires rejected v2 and candidate v3 reports")
    old = next(
        (row for row in v2.get("validation", ()) if row.get("case_id") == "sealed-validation-v2-biased"),
        None,
    )
    abstention = next(
        (
            row
            for row in v3.get("validation", ())
            if row.get("case_id") == "sealed-validation-v3-unsafe-regime"
        ),
        None,
    )
    if old is None or abstention is None:
        raise ValueError("self-aware contrast source rows are missing")
    if not (
        old.get("strict_replay") is True
        and old.get("critical") is True
        and old.get("success") is False
        and old["result"].get("post_kick_fall") is True
    ):
        raise ValueError("v2 contrast is not the audited critical fall")
    if not (
        abstention.get("strict_replay") is True
        and abstention.get("abstained") is True
        and abstention.get("critical") is False
        and abstention["result"].get("physics_executed") is False
        and abstention.get("trajectory_digest") == _EMPTY_TRAJECTORY_DIGEST
    ):
        raise ValueError("v3 contrast is not the audited no-motion abstention")
    old_path = v2_path.parent / "trajectories" / "sealed-validation-v2-biased.npz"
    if old_path == checkout or checkout in old_path.parents:
        raise ValueError("v2 contrast trajectory must be outside the checkout")
    old_trajectory = _load_trajectory(old_path)
    if trajectory_digest(old_trajectory) != old["trajectory_digest"]:
        raise ValueError("v2 contrast trajectory digest mismatch")
    return _ContrastSource(
        old_result=dict(old["result"]),
        old_trajectory_hash=_file_hash(old_path),
        old_trajectory=old_trajectory,
        abstention_result=dict(abstention["result"]),
        abstention_belief=dict(abstention["regime_belief_receipt"]),
    )


def _cinematic_timeline(source: _ChallengeSource, fps: int) -> tuple[float, ...]:
    contact = float(source.result["ball_contact_time_sec"])
    end = min(float(source.trajectory["time"][-1]), contact + 7.6)
    return _segments(
        (
            (max(float(source.trajectory["time"][0]), contact - 2.1), contact - 0.45, 1.0),
            (contact - 0.45, contact + 0.85, 0.38),
            (contact + 0.85, min(end, contact + 3.2), 1.0),
            (min(end, contact + 3.2), end, 1.40),
        ),
        fps,
    )


def _contrast_timeline(source: _ContrastSource, fps: int) -> tuple[float, ...]:
    contact = float(source.old_result["ball_contact_time_sec"])
    end = min(float(source.old_trajectory["time"][-1]), contact + 7.8)
    return _segments(
        (
            (contact - 1.25, contact - 0.35, 1.0),
            (contact - 0.35, contact + 0.75, 0.45),
            (contact + 0.75, end, 0.90),
        ),
        fps,
    )


def _segments(segments: tuple[tuple[float, float, float], ...], fps: int) -> tuple[float, ...]:
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
    comparison_data: Any,
    renderer: Any,
    sources: tuple[_ChallengeSource, ...],
    timelines: tuple[tuple[float, ...], ...],
    contrast: _ContrastSource,
    contrast_timeline: tuple[float, ...],
    finale_frames: int,
    stream: BinaryIO,
) -> None:
    ball_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    for source, timeline in zip(sources, timelines, strict=True):
        for simulation_time in timeline:
            frame = _render_pose(
                mujoco=mujoco,
                model=model,
                data=data,
                renderer=renderer,
                camera=camera,
                trajectory=source.trajectory,
                scenario=source.scenario,
                simulation_time=simulation_time,
                contact_time=float(source.result["ball_contact_time_sec"]),
                camera_azimuth=source.camera_azimuth_deg,
                ball_qpos=ball_qpos,
            )
            stream.write(np.ascontiguousarray(np.repeat(np.repeat(frame, 2, 0), 2, 1)).tobytes())

    stable_source = sources[0]
    stable_time = float(stable_source.trajectory["time"][0])
    for simulation_time in contrast_timeline:
        old = _render_pose(
            mujoco=mujoco,
            model=model,
            data=comparison_data,
            renderer=renderer,
            camera=camera,
            trajectory=contrast.old_trajectory,
            scenario={"target_y_m": 0.75, "target_z_m": 0.55, "disturbance_n": 0.0},
            simulation_time=simulation_time,
            contact_time=float(contrast.old_result["ball_contact_time_sec"]),
            camera_azimuth=90.0,
            ball_qpos=ball_qpos,
        )
        safe = _render_pose(
            mujoco=mujoco,
            model=model,
            data=data,
            renderer=renderer,
            camera=camera,
            trajectory=stable_source.trajectory,
            scenario=stable_source.scenario,
            simulation_time=stable_time,
            contact_time=999.0,
            camera_azimuth=90.0,
            ball_qpos=ball_qpos,
            show_shield=True,
        )
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        canvas[180:540, :640] = old
        canvas[180:540, 640:] = safe
        stream.write(np.ascontiguousarray(canvas).tobytes())

    finale_source = sources[-1]
    finale_time = float(finale_source.trajectory["time"][-1])
    finale = _render_pose(
        mujoco=mujoco,
        model=model,
        data=data,
        renderer=renderer,
        camera=camera,
        trajectory=finale_source.trajectory,
        scenario=finale_source.scenario,
        simulation_time=finale_time,
        contact_time=float(finale_source.result["ball_contact_time_sec"]),
        camera_azimuth=96.0,
        ball_qpos=ball_qpos,
    )
    finale_canvas = np.repeat(np.repeat(finale, 2, 0), 2, 1)
    for _ in range(finale_frames):
        stream.write(np.ascontiguousarray(finale_canvas).tobytes())


def _render_pose(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    renderer: Any,
    camera: Any,
    trajectory: dict[str, np.ndarray],
    scenario: dict[str, Any],
    simulation_time: float,
    contact_time: float,
    camera_azimuth: float,
    ball_qpos: int,
    show_shield: bool = False,
) -> np.ndarray:
    index, pelvis, joints, ball = _sample_trajectory(trajectory, simulation_time)
    data.qpos[:] = model.qpos0
    data.qpos[:7] = pelvis
    data.qpos[7:36] = joints
    data.qpos[ball_qpos : ball_qpos + 7] = ball
    mujoco.mj_forward(model, data)
    if simulation_time < contact_time + 0.15:
        camera.lookat[:] = (1.45, 0.15, 0.72)
        camera.distance = 3.75
    elif simulation_time < contact_time + 2.4:
        camera.lookat[:] = (3.05, 0.48, 0.60)
        camera.distance = 6.10
    else:
        camera.lookat[:] = pelvis[:3]
        camera.lookat[2] = 0.72
        camera.distance = 3.35
    camera.azimuth = camera_azimuth
    camera.elevation = -8.0
    renderer.update_scene(data, camera=camera)
    _add_targets_and_trail(
        mujoco=mujoco,
        scene=renderer.scene,
        scenario=scenario,
        trajectory=trajectory,
        index=index,
        show_grid=False,
    )
    if float(scenario.get("disturbance_n", 0.0)) and 4.55 <= simulation_time <= 4.90:
        for offset in np.linspace(-0.58, -0.12, 7):
            _append_sphere(
                mujoco,
                renderer.scene,
                pelvis[:3] + np.asarray((0.0, offset, 0.12)),
                0.022 + (offset + 0.58) * 0.035,
                (1.0, 0.20, 0.08, 0.88),
            )
    if show_shield:
        for angle in np.linspace(0.0, 2.0 * math.pi, 18, endpoint=False):
            _append_sphere(
                mujoco,
                renderer.scene,
                pelvis[:3]
                + np.asarray((0.0, 0.52 * math.cos(angle), 0.62 + 0.52 * math.sin(angle))),
                0.022,
                (0.16, 1.0, 0.48, 0.72),
            )
    return renderer.render().copy()


def _write_label_files(
    root: Path,
    sources: tuple[_ChallengeSource, ...],
    contrast: _ContrastSource,
) -> tuple[tuple[Path, Path], ...]:
    root.mkdir(parents=True, exist_ok=True)
    labels: list[tuple[Path, Path]] = []
    for index, source in enumerate(sources, start=1):
        heading = root / f"heading-{index}.txt"
        metric = root / f"metric-{index}.txt"
        heading.write_text(f"CHALLENGE {index}/3 · {source.title}", encoding="utf-8")
        metric.write_text(
            f"{source.subtitle} · HIT {float(source.result['target_error_m']):.3f} m · "
            f"BALL {float(source.result['ball_speed_mps']):.2f} m/s · "
            f"COM {float(source.result['com_margin_min_m']):+.3f} m",
            encoding="utf-8",
        )
        labels.append((heading, metric))
    heading = root / "heading-contrast.txt"
    metric = root / "metric-contrast.txt"
    heading.write_text("BREAKTHROUGH · BLIND ACTION → SELF-AWARE ABSTENTION", encoding="utf-8")
    belief = contrast.abstention_belief
    metric.write_text(
        f"V2: BALL HIT / ROBOT FALL / JOINT LIMIT · V3: NO MOTION COMMAND · "
        f"FRICTION LB {float(belief['support_friction_lower_bound']):.3f} · "
        f"LATENCY UB {float(belief['control_latency_upper_bound_ms']):.1f} ms",
        encoding="utf-8",
    )
    labels.append((heading, metric))
    return tuple(labels)


def _ffmpeg_command(
    *,
    ffmpeg: str,
    output: Path,
    fps: int,
    durations: tuple[float, ...],
    contrast_duration: float,
    finale_duration: float,
    label_files: tuple[tuple[Path, Path], ...],
) -> list[str]:
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font_option = f"fontfile={_escape_filtergraph_option(str(font))}:" if font.is_file() else ""
    title = _escape_filtergraph_option("ROSClaw · G1 SELF-AWARE MOVING-BALL GAUNTLET")
    footer = _escape_filtergraph_option(
        "REAL CPU MUJOCO · STRICT REPLAY · SIM ONLY · DEVELOPMENT SHOWCASE"
    )
    filters = [
        "drawbox=x=0:y=0:w=iw:h=138:color=0x040913@0.86:t=fill",
        "drawbox=x=0:y=h-66:w=iw:h=66:color=0x040913@0.86:t=fill",
        f"drawtext={font_option}text={title}:expansion=none:x=32:y=14:fontsize=34:fontcolor=white",
        f"drawtext={font_option}text={footer}:expansion=none:x=32:y=h-43:fontsize=20:fontcolor=0x8DD8FF",
    ]
    offset = 0.0
    for duration, (heading, metric) in zip(
        (*durations, contrast_duration), label_files, strict=True
    ):
        end = offset + duration
        enable = f"enable='between(t,{offset:.6f},{end:.6f})'"
        filters.extend(
            (
                f"drawtext={font_option}textfile={_escape_filtergraph_option(str(heading))}:"
                f"expansion=none:x=32:y=55:fontsize=23:fontcolor=0x65F59A:{enable}",
                f"drawtext={font_option}textfile={_escape_filtergraph_option(str(metric))}:"
                f"expansion=none:x=32:y=94:fontsize=17:fontcolor=0xFFD166:{enable}",
            )
        )
        if heading.name == "heading-contrast.txt":
            filters.extend(
                (
                    f"drawtext={font_option}text={_escape_filtergraph_option('V2 · BLIND KICK')}:"
                    f"expansion=none:x=170:y=150:fontsize=22:fontcolor=0xFF766B:{enable}",
                    f"drawtext={font_option}text={_escape_filtergraph_option('V3 · SAFE REFUSAL')}:"
                    f"expansion=none:x=810:y=150:fontsize=22:fontcolor=0x65F59A:{enable}",
                )
            )
        offset = end
    finale_end = offset + finale_duration
    finale_enable = f"enable='between(t,{offset:.6f},{finale_end:.6f})'"
    filters.extend(
        (
            f"drawtext={font_option}text={_escape_filtergraph_option('3 / 3 CHALLENGES · 0 FALLS · STRICT REPLAY')}:"
            f"expansion=none:x=(w-text_w)/2:y=58:fontsize=28:fontcolor=0x65F59A:{finale_enable}",
            f"drawtext={font_option}text={_escape_filtergraph_option('THE BREAKTHROUGH: ACT WHEN CAPABLE · ABSTAIN WHEN UNSAFE')}:"
            f"expansion=none:x=(w-text_w)/2:y=98:fontsize=20:fontcolor=0xFFD166:{finale_enable}",
        )
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


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "G1SelfAwareShowcaseVideoResult",
    "G1SelfAwareVideoClip",
    "render_g1_self_aware_showcase_video",
]
