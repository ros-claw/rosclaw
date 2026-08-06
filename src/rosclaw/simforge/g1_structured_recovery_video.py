"""Visualization-only export for a passing structured recovery campaign."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.simforge.backends.unitree_mujoco_backend import trajectory_digest
from rosclaw.simforge.g1_coupled_relay_video import _load_coupled_trajectory
from rosclaw.simforge.g1_coupled_showcase_video import (
    G1CoupledShowcaseVideoResult,
    render_g1_coupled_showcase_video,
)


@dataclass(frozen=True)
class G1StructuredRecoveryVideoResult:
    video: G1CoupledShowcaseVideoResult
    source_manifest_path: str
    growth_evidence_path: str
    candidate_hash: str
    visualization_only: bool = True
    pixels_used_for_promotion: bool = False
    schema_version: str = "rosclaw.growth.g1_structured_recovery_video.v1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["video"] = self.video.to_dict()
        value["generates_task_evidence"] = False
        return value


def render_g1_structured_recovery_video(
    *,
    evidence_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
) -> G1StructuredRecoveryVideoResult:
    """Render five candidate traces only after all eight SIM gates pass."""

    evidence = evidence_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("structured recovery video must be outside the source checkout")
    source_manifest = output.with_name(output.stem + "-source.json")
    video_manifest = output.with_suffix(".json")
    growth_manifest = output.with_name(output.stem + "-growth.json")
    if any(path.exists() for path in (source_manifest, video_manifest, growth_manifest, output)):
        raise FileExistsError("structured recovery video artifacts already exist")
    report = json.loads(evidence.read_text(encoding="utf-8"))
    if report.get("passed") is not True or report.get("status") != "SIM_GATE_PASS":
        raise ValueError("structured recovery video requires a passing SIM campaign")
    if len(report.get("cases", ())) != 8:
        raise ValueError("structured recovery video requires all eight campaign cases")
    request = json.loads((evidence.parent / "request.json").read_text(encoding="utf-8"))
    cases = [
        _visual_case(case, evidence.parent, checkout, index)
        for index, case in enumerate(report["cases"][:5])
    ]
    derived = {
        "schema_version": "rosclaw.growth.g1_structured_recovery_video_source.v1",
        "passed": all(case["passed"] for case in cases),
        "body_hash": request["body_hash"],
        "candidate_hash": report["candidate_hash"],
        "growth_evidence_path": str(evidence),
        "growth_evidence_request_hash": report["request_hash"],
        "cases": cases,
        "claims": {
            "visualization_only": True,
            "pixels_used_for_promotion": False,
            "five_of_eight_passing_cases_rendered": True,
        },
    }
    source_manifest.parent.mkdir(parents=True, exist_ok=True)
    source_manifest.write_text(
        json.dumps(derived, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        video = render_g1_coupled_showcase_video(
            evidence_path=source_manifest,
            asset_root=asset_root,
            output_path=output,
            source_checkout=source_checkout,
            fps=fps,
        )
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl
    result = G1StructuredRecoveryVideoResult(
        video=video,
        source_manifest_path=str(source_manifest),
        growth_evidence_path=str(evidence),
        candidate_hash=str(report["candidate_hash"]),
    )
    growth_manifest.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _visual_case(
    case: dict[str, Any],
    evidence_root: Path,
    checkout: Path,
    index: int,
) -> dict[str, Any]:
    if case.get("passed") is not True or case.get("candidate_strict_replay") is not True:
        raise ValueError("structured recovery video case is not a strict passing candidate")
    case_id = str(case["spec"]["case_id"])
    path = (evidence_root / f"{case_id}-candidate.npz").resolve()
    if path == checkout or checkout in path.parents:
        raise ValueError("structured recovery video source must be outside the checkout")
    trajectory = _load_coupled_trajectory(path)
    digest = trajectory_digest(trajectory)
    file_hash = _file_hash(path)
    if file_hash != case["candidate_trajectory_hash"]:
        raise ValueError("structured recovery candidate trajectory hash mismatch")
    result = case["candidate_result"]
    quality = case["candidate_quality"]
    titles = (
        "EARLY ARRIVAL · CLEAN RECOVERY",
        "SLICK PITCH · ONE TOUCH",
        "HIGH TARGET · PRECISION",
        "GRIPPY PITCH · CAPTURE STEP",
        "LATE ARRIVAL · REFLEX ROUTER",
    )
    subtitles = (
        f"10+ m/s · PATH {float(quality['post_contact_pelvis_path_length_m']):.2f} m · "
        f"SETTLE {float(quality['settling_time_sec']):.2f} s · "
        f"SLIP {float(result['shooter_post_contact_support_foot_slip_m']):.3f} m"
    )
    return {
        "schema_version": "rosclaw.g1_goalforge.coupled_showcase_case.v1",
        "passed": True,
        "strict_replay": True,
        "trajectory_path": str(path),
        "trajectory_hash": file_hash,
        "trajectory_digest": digest,
        "result": result,
        "spec": {
            "case_id": case_id,
            "title": titles[index],
            "subtitle": subtitles,
            "camera_azimuth_deg": 84.0 + 5.0 * index,
        },
    }


def _file_hash(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


__all__ = ["G1StructuredRecoveryVideoResult", "render_g1_structured_recovery_video"]
