"""Evidence-bound split-screen video for G1 neural-torque adaptation.

The renderer reruns matched MuJoCo scenarios and consumes only recorded
trajectories.  It cannot promote an actor, authorize hardware, or hide failed
scenarios.  A manifest binds every panel to artifact and trajectory hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np

from rosclaw.simforge.backends.unitree_mujoco_backend import (
    G1MuJoCoBackend,
    GoalForgeEpisode,
    trajectory_digest,
)
from rosclaw.simforge.g1_hat_trick_video import _render_pose
from rosclaw.simforge.g1_neural_torque import (
    G1NeuralTorqueArtifact,
    G1NeuralTorquePolicy,
    load_g1_neural_torque_artifact,
)
from rosclaw.simforge.g1_neural_torque_validation import _pilot_scenarios
from rosclaw.simforge.g1_stability_plasticity_policy import (
    G1StabilityPlasticityGateConfig,
    G1StabilityPlasticityTorquePolicy,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import ShotParameters

_SCENE_REL = Path("g1_description/scene_with_ball.xml")
_WIDTH = 1280
_HEIGHT = 720
_PANEL_WIDTH = 640
_PANEL_HEIGHT = 360
_PANEL_TOP = 160


@dataclass(frozen=True)
class G1NeuralTorqueVideoClip:
    scenario_id: str
    scenario_commitment: str
    challenge: str
    parent_status: str
    candidate_status: str
    parent_success: bool
    candidate_success: bool
    parent_fall: bool
    candidate_fall: bool
    parent_roll_peak_rad: float
    candidate_roll_peak_rad: float
    parent_pitch_peak_rad: float
    candidate_pitch_peak_rad: float
    parent_support_slip_m: float
    candidate_support_slip_m: float
    parent_trajectory_hash: str
    candidate_trajectory_hash: str
    parent_strict_replay: bool
    candidate_strict_replay: bool
    frame_count: int
    duration_sec: float
    schema_version: str = "rosclaw.simforge.g1_neural_torque_video_clip.v1"


@dataclass(frozen=True)
class G1NeuralTorqueVideoResult:
    output_path: str
    manifest_path: str
    video_hash: str
    stable_artifact_hash: str
    parent_artifact_hash: str
    candidate_artifact_hash: str
    clips: tuple[G1NeuralTorqueVideoClip, ...]
    width: int
    height: int
    fps: int
    frame_count: int
    duration_sec: float
    strict_replay: bool
    decision: str = "REJECTED"
    rejection_reason: str = "support_slip_generalization_regression"
    evidence_domain: str = "SIM_ONLY"
    visualization_only: bool = True
    promotion_evidence_eligible: bool = False
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.simforge.g1_neural_torque_video.v1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["clips"] = [asdict(item) for item in self.clips]
        return value


def render_g1_neural_torque_comparison_video(
    *,
    asset_root: Path,
    stable_artifact_path: Path,
    parent_artifact_path: Path,
    candidate_artifact_path: Path,
    output_path: Path,
    source_checkout: Path,
    scenario_indices: tuple[int, ...] = (0, 1, 2, 3),
    fps: int = 30,
) -> G1NeuralTorqueVideoResult:
    """Render matched parent/candidate validation replays with failure labels."""

    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("neural-torque video must remain outside the checkout")
    if output.suffix.lower() != ".mp4":
        raise ValueError("neural-torque video output must use .mp4")
    manifest = output.with_suffix(".json")
    evidence_root = output.parent / f"{output.stem}-evidence"
    if output.exists() or manifest.exists() or evidence_root.exists():
        raise FileExistsError("neural-torque video output already exists")
    if not 10 <= fps <= 60:
        raise ValueError("neural-torque video fps must be in [10, 60]")
    if (
        not 2 <= len(scenario_indices) <= 7
        or len(set(scenario_indices)) != len(scenario_indices)
    ):
        raise ValueError("neural-torque video requires 2 to 7 unique scenarios")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for neural-torque video export")

    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=1)
    stable = load_g1_neural_torque_artifact(
        stable_artifact_path,
        expected_body_hash=backend.qualification.body_hash,
    )
    parent = load_g1_neural_torque_artifact(
        parent_artifact_path,
        expected_body_hash=backend.qualification.body_hash,
    )
    candidate = load_g1_neural_torque_artifact(
        candidate_artifact_path,
        expected_body_hash=backend.qualification.body_hash,
    )
    gate = G1StabilityPlasticityGateConfig(
        minimum_recovery_phase=0.20,
        minimum_pelvis_height_m=0.70,
        maximum_projected_gravity_z=-0.88,
        eligibility_warmup_steps=5,
    )
    scenarios = _pilot_scenarios()[2]
    if any(index < 0 or index >= len(scenarios) for index in scenario_indices):
        raise ValueError("neural-torque video scenario index is out of range")

    output.parent.mkdir(parents=True, exist_ok=True)
    evidence_root.mkdir(parents=True, exist_ok=False)
    sources: list[
        tuple[
            Any,
            GoalForgeEpisode,
            GoalForgeEpisode,
            float,
            float,
            G1NeuralTorqueVideoClip,
        ]
    ] = []
    parameters = ShotParameters()
    labels = {
        0: "NOMINAL LOW TARGET",
        1: "LATERAL LOW TARGET",
        2: "MASS / FRICTION SHIFT",
        3: "CALIBRATION + LATENCY SHIFT",
    }
    for clip_number, scenario_index in enumerate(scenario_indices, start=1):
        scenario = scenarios[scenario_index]
        parent_episode, parent_strict = _strict_episode(
            backend,
            scenario,
            parameters,
            stable=stable,
            plastic=parent,
            gate=gate,
        )
        candidate_episode, candidate_strict = _strict_episode(
            backend,
            scenario,
            parameters,
            stable=stable,
            plastic=candidate,
            gate=gate,
        )
        parent_path = evidence_root / f"{clip_number:02d}-{scenario.scenario_id}-parent.npz"
        candidate_path = (
            evidence_root / f"{clip_number:02d}-{scenario.scenario_id}-candidate.npz"
        )
        np.savez_compressed(parent_path, **parent_episode.trajectory)  # type: ignore[arg-type]
        np.savez_compressed(candidate_path, **candidate_episode.trajectory)  # type: ignore[arg-type]
        contact = _contact_time(parent_episode, candidate_episode)
        start = max(float(parent_episode.trajectory["time"][0]), contact - 2.0)
        end = min(
            float(parent_episode.trajectory["time"][-1]),
            float(candidate_episode.trajectory["time"][-1]),
        )
        frame_count = max(1, int(np.ceil((end - start) * fps)))
        clip = _clip_result(
            scenario=scenario,
            challenge=labels.get(scenario_index, scenario.scenario_id.upper()),
            parent=parent_episode,
            candidate=candidate_episode,
            parent_strict=parent_strict,
            candidate_strict=candidate_strict,
            frame_count=frame_count,
            fps=fps,
        )
        sources.append((scenario, parent_episode, candidate_episode, start, end, clip))

    try:
        import mujoco

        scene = asset_root.expanduser().resolve() / _SCENE_REL
        model = mujoco.MjModel.from_xml_path(str(scene))
        left_data = mujoco.MjData(model)
        right_data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=_PANEL_HEIGHT, width=_PANEL_WIDTH)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        process = subprocess.Popen(
            _ffmpeg_command(ffmpeg=ffmpeg, output=output, fps=fps, sources=sources),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if process.stdin is None:
            raise RuntimeError("neural-torque video ffmpeg pipe is unavailable")
        try:
            _write_frames(
                mujoco=mujoco,
                model=model,
                left_data=left_data,
                right_data=right_data,
                renderer=renderer,
                camera=camera,
                sources=sources,
                fps=fps,
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
            raise RuntimeError(f"neural-torque ffmpeg failed ({code}): {stderr[-2000:]}")
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl

    clips = tuple(source[-1] for source in sources)
    total_frames = sum(item.frame_count for item in clips)
    result = G1NeuralTorqueVideoResult(
        output_path=str(output),
        manifest_path=str(manifest),
        video_hash=_hash_file(output),
        stable_artifact_hash=stable.artifact_hash,
        parent_artifact_hash=parent.artifact_hash,
        candidate_artifact_hash=candidate.artifact_hash,
        clips=clips,
        width=_WIDTH,
        height=_HEIGHT,
        fps=fps,
        frame_count=total_frames,
        duration_sec=total_frames / fps,
        strict_replay=all(
            item.parent_strict_replay and item.candidate_strict_replay for item in clips
        ),
    )
    manifest.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def _strict_episode(
    backend: G1MuJoCoBackend,
    scenario: Any,
    parameters: ShotParameters,
    *,
    stable: G1NeuralTorqueArtifact,
    plastic: G1NeuralTorqueArtifact,
    gate: G1StabilityPlasticityGateConfig,
) -> tuple[GoalForgeEpisode, bool]:
    first = _episode(backend, scenario, parameters, stable=stable, plastic=plastic, gate=gate)
    replay = _episode(backend, scenario, parameters, stable=stable, plastic=plastic, gate=gate)
    strict = bool(
        first.result.summary_dict() == replay.result.summary_dict()
        and trajectory_digest(first.trajectory) == trajectory_digest(replay.trajectory)
    )
    return first, strict


def _episode(
    backend: G1MuJoCoBackend,
    scenario: Any,
    parameters: ShotParameters,
    *,
    stable: G1NeuralTorqueArtifact,
    plastic: G1NeuralTorqueArtifact,
    gate: G1StabilityPlasticityGateConfig,
) -> GoalForgeEpisode:
    policy = G1StabilityPlasticityTorquePolicy(
        G1NeuralTorquePolicy(
            stable,
            expected_body_hash=backend.qualification.body_hash,
            expected_parent_policy_hash=backend.qualification.kick_prior_hash,
        ),
        G1NeuralTorquePolicy(
            plastic,
            expected_body_hash=backend.qualification.body_hash,
            expected_parent_policy_hash=backend.qualification.kick_prior_hash,
        ),
        config=gate,
    )
    return backend.run(scenario, parameters, torque_policy=policy)


def _clip_result(
    *,
    scenario: Any,
    challenge: str,
    parent: GoalForgeEpisode,
    candidate: GoalForgeEpisode,
    parent_strict: bool,
    candidate_strict: bool,
    frame_count: int,
    fps: int,
) -> G1NeuralTorqueVideoClip:
    before = parent.result
    after = candidate.result
    return G1NeuralTorqueVideoClip(
        scenario_id=scenario.scenario_id,
        scenario_commitment=scenario.scenario_commitment,
        challenge=challenge,
        parent_status=before.status.value,
        candidate_status=after.status.value,
        parent_success=before.success,
        candidate_success=after.success,
        parent_fall=before.post_kick_fall,
        candidate_fall=after.post_kick_fall,
        parent_roll_peak_rad=before.torso_roll_peak_rad,
        candidate_roll_peak_rad=after.torso_roll_peak_rad,
        parent_pitch_peak_rad=before.torso_pitch_peak_rad,
        candidate_pitch_peak_rad=after.torso_pitch_peak_rad,
        parent_support_slip_m=before.support_foot_slip_m,
        candidate_support_slip_m=after.support_foot_slip_m,
        parent_trajectory_hash=trajectory_digest(parent.trajectory),
        candidate_trajectory_hash=trajectory_digest(candidate.trajectory),
        parent_strict_replay=parent_strict,
        candidate_strict_replay=candidate_strict,
        frame_count=frame_count,
        duration_sec=frame_count / fps,
    )


def _contact_time(parent: GoalForgeEpisode, candidate: GoalForgeEpisode) -> float:
    values = (
        parent.result.ball_contact_time_sec,
        candidate.result.ball_contact_time_sec,
    )
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return min(finite) if finite else 5.25


def _write_frames(
    *,
    mujoco: Any,
    model: Any,
    left_data: Any,
    right_data: Any,
    renderer: Any,
    camera: Any,
    sources: list[tuple[Any, GoalForgeEpisode, GoalForgeEpisode, float, float, Any]],
    fps: int,
    stream: BinaryIO,
) -> None:
    ball_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    for scenario, parent, candidate, start, end, _clip in sources:
        frame_count = max(1, int(np.ceil((end - start) * fps)))
        for simulation_time in np.linspace(start, end, frame_count, endpoint=False):
            contact = _contact_time(parent, candidate)
            left = _render_pose(
                mujoco=mujoco,
                model=model,
                data=left_data,
                renderer=renderer,
                camera=camera,
                trajectory=parent.trajectory,
                simulation_time=float(simulation_time),
                ball_qpos=ball_qpos,
                scenario=scenario.to_private_dict(),
                contact_time=contact,
                show_grid=False,
                show_push=False,
            )
            right = _render_pose(
                mujoco=mujoco,
                model=model,
                data=right_data,
                renderer=renderer,
                camera=camera,
                trajectory=candidate.trajectory,
                simulation_time=float(simulation_time),
                ball_qpos=ball_qpos,
                scenario=scenario.to_private_dict(),
                contact_time=contact,
                show_grid=False,
                show_push=False,
            )
            canvas = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
            canvas[_PANEL_TOP : _PANEL_TOP + _PANEL_HEIGHT, :_PANEL_WIDTH] = left
            canvas[_PANEL_TOP : _PANEL_TOP + _PANEL_HEIGHT, _PANEL_WIDTH:] = right
            stream.write(np.ascontiguousarray(canvas).tobytes())


def _ffmpeg_command(
    *,
    ffmpeg: str,
    output: Path,
    fps: int,
    sources: list[tuple[Any, GoalForgeEpisode, GoalForgeEpisode, float, float, Any]],
) -> list[str]:
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font_option = (
        f"fontfile={_escape_filtergraph_option(str(font))}:" if font.is_file() else ""
    )
    filters = [
        "drawbox=x=0:y=0:w=iw:h=140:color=0x050A12@0.88:t=fill",
        "drawbox=x=0:y=h-180:w=iw:h=180:color=0x050A12@0.88:t=fill",
        _drawtext(
            font_option,
            "ROSClaw G1 DATA-DRIVEN TORQUE CEREBELLUM",
            x=30,
            y=18,
            size=30,
            color="white",
        ),
        _drawtext(
            font_option,
            "PARENT · MOTIONDECODE BC",
            x=120,
            y=112,
            size=20,
            color="0xFFB45F",
        ),
        _drawtext(
            font_option,
            "ONLINE ACTOR-CRITIC",
            x=815,
            y=112,
            size=20,
            color="0x65F59A",
        ),
        _drawtext(
            font_option,
            "SIM_ONLY · VISUALIZATION · REJECTED BY SLIP GENERALIZATION GUARD",
            x=30,
            y=680,
            size=20,
            color="0xFF7070",
        ),
    ]
    offset = 0.0
    for _scenario, _parent, _candidate, _start, _end, clip in sources:
        duration = clip.duration_sec
        finish = offset + duration
        enable = f"between(t,{offset:.6f},{finish:.6f})"
        summary = (
            f"{clip.challenge} · parent {clip.parent_status} · online {clip.candidate_status}"
        )
        parent_metrics = (
            f"roll {clip.parent_roll_peak_rad:.3f} rad  pitch "
            f"{clip.parent_pitch_peak_rad:.3f}  slip {clip.parent_support_slip_m:.3f} m"
        )
        candidate_metrics = (
            f"roll {clip.candidate_roll_peak_rad:.3f} rad  pitch "
            f"{clip.candidate_pitch_peak_rad:.3f}  slip "
            f"{clip.candidate_support_slip_m:.3f} m"
        )
        filters.extend(
            (
                _drawtext(
                    font_option,
                    summary,
                    x=30,
                    y=62,
                    size=21,
                    color="0x8FD3FF",
                    enable=enable,
                ),
                _drawtext(
                    font_option,
                    parent_metrics,
                    x=45,
                    y=566,
                    size=18,
                    color="0xFFB45F",
                    enable=enable,
                ),
                _drawtext(
                    font_option,
                    candidate_metrics,
                    x=670,
                    y=566,
                    size=18,
                    color="0x65F59A",
                    enable=enable,
                ),
                _drawtext(
                    font_option,
                    "Matched sealed scenario · strict deterministic replay",
                    x=350,
                    y=620,
                    size=18,
                    color="white",
                    enable=enable,
                ),
            )
        )
        offset = finish
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
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]


def _drawtext(
    font_option: str,
    value: str,
    *,
    x: int,
    y: int,
    size: int,
    color: str,
    enable: str | None = None,
) -> str:
    result = (
        f"drawtext={font_option}text={_escape_filtergraph_option(value)}:"
        f"expansion=none:x={x}:y={y}:fontsize={size}:fontcolor={color}"
    )
    return result + (f":enable='{enable}'" if enable is not None else "")


def _escape_filtergraph_option(value: str) -> str:
    """Apply FFmpeg option-value and filtergraph escaping without a shell."""

    def escape_level(text: str, special: str) -> str:
        return "".join(("\\" + char) if char in special else char for char in text)

    return escape_level(escape_level(value, "\\':"), "\\'[],;")


def _hash_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "G1NeuralTorqueVideoClip",
    "G1NeuralTorqueVideoResult",
    "render_g1_neural_torque_comparison_video",
]
