from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw.collective.cli import dispatch_collective_argv
from rosclaw.collective.contracts import LicenseDecision, LicenseUse
from rosclaw.collective.sources.motiondecode.audit import (
    MotionQualificationLevel,
    audit_motiondecode_snapshot,
)
from rosclaw.collective.sources.motiondecode.joint_mapping import (
    MOTIONDECODE_ROOT_COLUMNS,
    mapping_from_header,
)
from rosclaw.collective.sources.motiondecode.license import snapshot_license
from rosclaw.collective.sources.motiondecode.manifest import (
    MotionDecodeRegistration,
    _stratified_sample,
    register_motiondecode_source,
    verify_registered_files,
)
from rosclaw.collective.sources.motiondecode.parser import parse_motion_csv
from rosclaw.collective.sources.motiondecode.repair import (
    MotionRepairDisposition,
    SegmentationRepairManifest,
    repair_motiondecode_snapshot,
    replay_segmentation_repair,
)
from rosclaw.collective.sources.motiondecode.taxonomy import (
    MOTIONDECODE_INDEX_COLUMNS,
    MotionFamily,
    classify_motion,
    parse_catalog,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_DDS_JOINT_NAMES

ROOT = Path(__file__).parents[2]
G1_MODEL = ROOT / "e-urdf-zoo/g1/robot.mjcf.xml"
REVISION = "c460808e801b35c683eb4cf4c8338ce61481a9bb"
OFFICIAL_MOTIONDECODE_G1_ORDER = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _header(
    joints: tuple[str, ...] = OFFICIAL_MOTIONDECODE_G1_ORDER,
) -> tuple[str, ...]:
    return MOTIONDECODE_ROOT_COLUMNS + tuple(f"dof_{name}(rad)" for name in joints)


def _dataset(
    root: Path,
    *,
    joint_value: float = 0.0,
    non_finite: bool = False,
    sample_name: str = "MD_Football_Kick_00001.csv",
) -> Path:
    metadata = root / "metadata"
    sample_dir = root / "samples/3.3.Ball_Game_Interaction/3.3.1.Football"
    metadata.mkdir(parents=True)
    sample_dir.mkdir(parents=True)
    with (metadata / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(MOTIONDECODE_INDEX_COLUMNS)
        writer.writerow(
            (
                "3.3.1",
                "Ball_Game_Interaction",
                "Football",
                "Instep_Kick",
                "csv",
                "Unitree_G1",
            )
        )
    (root / "LICENSE").write_bytes(b"")
    sample = sample_dir / sample_name
    with sample.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_header())
        for frame in range(5):
            joints = [0.0] * len(G1_DDS_JOINT_NAMES)
            joints[0] = joint_value
            if non_finite and frame == 2:
                joints[1] = float("nan")
            writer.writerow(
                [
                    0.001 * frame,
                    0.0,
                    0.78,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    *joints,
                ]
            )
    return sample


def _register(root: Path) -> MotionDecodeRegistration:
    return register_motiondecode_source(
        root,
        revision=REVISION,
        requested_use=LicenseUse.RESEARCH_NONCOMMERCIAL,
        terms_path=root / "LICENSE",
        terms_uri=(f"https://huggingface.co/datasets/CMRobot/MotionDecode/blob/{REVISION}/LICENSE"),
        families=(MotionFamily.FOOTBALL,),
        limit=40,
    )


def _append_terminal_reset(sample: Path) -> None:
    with sample.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    rows[2][0] = "0.04"
    rows[3][0] = "0.08"
    rows[4][0] = "0.12"
    rows[-1] = rows[1]
    with sample.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def test_empty_current_license_is_pending_and_cannot_be_permitted(tmp_path: Path) -> None:
    terms = tmp_path / "LICENSE"
    terms.write_bytes(b"")

    pending = snapshot_license(
        source_revision=REVISION,
        requested_use=LicenseUse.RESEARCH_NONCOMMERCIAL,
        terms_path=terms,
        terms_uri="https://example.invalid/LICENSE",
    )

    assert pending.decision is LicenseDecision.PENDING
    assert pending.terms_hash is None
    assert pending.training_permitted is False
    with pytest.raises(ValueError, match="empty or absent terms"):
        snapshot_license(
            source_revision=REVISION,
            requested_use=LicenseUse.RESEARCH_NONCOMMERCIAL,
            decision=LicenseDecision.PERMITTED,
            terms_path=terms,
            terms_uri="https://example.invalid/LICENSE",
        )


def test_nonempty_terms_are_hashed_but_permission_remains_explicit(tmp_path: Path) -> None:
    terms = tmp_path / "LICENSE"
    terms.write_text("research terms", encoding="utf-8")

    snapshot = snapshot_license(
        source_revision=REVISION,
        requested_use=LicenseUse.RESEARCH_NONCOMMERCIAL,
        decision=LicenseDecision.PENDING,
        terms_path=terms,
        terms_uri="https://example.invalid/LICENSE",
    )

    assert snapshot.terms_hash == _hash(terms)
    assert snapshot.decision is LicenseDecision.PENDING


def test_catalog_and_paths_map_to_the_four_pilot_families(tmp_path: Path) -> None:
    _dataset(tmp_path)

    rows, audit = parse_catalog(tmp_path / "metadata/index.csv")

    assert audit.schema_valid is True
    assert rows[0].family is MotionFamily.FOOTBALL
    assert (
        classify_motion("1.5.Balance_Control_Actions/Single_Leg_Standing") is MotionFamily.BALANCE
    )
    assert classify_motion("1.3.Basic_Gait_Category/Normal_Walking") is MotionFamily.GAIT
    assert (
        classify_motion("1.2.State_Transition_Category/Still_Walk_Run_Stop")
        is MotionFamily.TRANSITION_RECOVERY
    )


def test_official_36_column_header_exactly_matches_rosclaw_g1_order() -> None:
    assert OFFICIAL_MOTIONDECODE_G1_ORDER == G1_DDS_JOINT_NAMES

    mapping, time_column = mapping_from_header(_header())

    assert time_column is None
    assert mapping.exact_order is True
    assert mapping.source_joint_names == G1_DDS_JOINT_NAMES
    assert mapping.source_indices_by_target == tuple(range(29))


def test_explicit_permutation_is_reordered_without_guessing() -> None:
    source_order = tuple(reversed(G1_DDS_JOINT_NAMES))

    mapping, _ = mapping_from_header(_header(source_order))

    assert mapping.exact_order is False
    assert mapping.source_indices_by_target == tuple(reversed(range(29)))


def test_missing_joint_fails_closed() -> None:
    with pytest.raises(ValueError, match="contract mismatch"):
        mapping_from_header(_header(G1_DDS_JOINT_NAMES[:-1]))


def test_registration_is_content_addressed_and_replays(tmp_path: Path) -> None:
    sample = _dataset(tmp_path)

    registration = _register(tmp_path)
    replayed = MotionDecodeRegistration.from_dict(registration.to_dict())

    assert registration.source_registered is True
    assert registration.training_eligible is False
    assert registration.manifest.selected_sample_count == 1
    assert registration.manifest.files[1].content_hash == _hash(sample)
    assert replayed.registration_hash == registration.registration_hash
    assert replayed.manifest.source_identity.source_hash == (
        registration.manifest.source_identity.source_hash
    )


def test_bounded_pilot_is_balanced_across_families_and_leaf_skills() -> None:
    paths = [
        Path(f"samples/football/short/short_{index:03d}.csv") for index in range(20)
    ]
    paths.extend(
        Path(f"samples/football/shooting/shot_{index:03d}.csv") for index in range(20)
    )
    paths.extend(Path(f"samples/gait/walking/walk_{index:03d}.csv") for index in range(20))

    selected = _stratified_sample(paths, limit=6)

    assert selected == sorted(selected, key=lambda item: item.as_posix())
    assert sum("/gait/" in item.as_posix() for item in selected) == 3
    football = [item for item in selected if "/football/" in item.as_posix()]
    assert len(football) == 3
    assert any("/shooting/" in item.as_posix() for item in football)
    assert any("/short/" in item.as_posix() for item in football)


def test_registered_file_mutation_is_rejected_before_parse(tmp_path: Path) -> None:
    sample = _dataset(tmp_path)
    registration = _register(tmp_path)
    sample.write_text("mutated\n", encoding="utf-8")

    with pytest.raises(ValueError, match="registered file size changed"):
        verify_registered_files(registration, tmp_path)


def test_registration_rejects_oversized_motion_before_hashing(tmp_path: Path) -> None:
    sample = _dataset(tmp_path)
    with sample.open("ab") as handle:
        handle.truncate(128 * 1024 * 1024 + 1)

    with pytest.raises(ValueError, match="128 MB safety limit"):
        _register(tmp_path)


def test_parser_builds_read_only_canonical_episode_with_implicit_120hz(
    tmp_path: Path,
) -> None:
    sample = _dataset(tmp_path)
    registration = _register(tmp_path)
    record = next(
        item for item in registration.manifest.files if item.relative_path.startswith("samples/")
    )
    target_hash = "sha256:" + "1" * 64

    episode = parse_motion_csv(
        sample,
        source_manifest_hash=registration.manifest.manifest_hash,
        expected_file_hash=record.content_hash,
        target_body_hash=target_hash,
    )

    assert episode.implicit_timeline is True
    assert episode.sample_rate_hz == 120.0
    assert episode.joint_position.shape == (5, 29)
    assert episode.duration_seconds == pytest.approx(4 / 120)
    assert np.allclose(episode.root_quaternion[:, 0], 1.0)
    assert episode.joint_position.flags.writeable is False
    assert episode.ball_pose_available is False
    assert episode.action_semantics_available is False


def test_nan_is_rejected_before_it_can_reach_training(tmp_path: Path) -> None:
    sample = _dataset(tmp_path, non_finite=True)
    registration = _register(tmp_path)
    record = next(
        item for item in registration.manifest.files if item.relative_path.startswith("samples/")
    )

    with pytest.raises(ValueError, match="NaN or Inf"):
        parse_motion_csv(
            sample,
            source_manifest_hash=registration.manifest.manifest_hash,
            expected_file_hash=record.content_hash,
            target_body_hash="sha256:" + "1" * 64,
        )


def test_kinematic_audit_closes_to_unverified_capsule_not_training(
    tmp_path: Path,
) -> None:
    _dataset(tmp_path)
    registration = _register(tmp_path)

    report = audit_motiondecode_snapshot(
        registration,
        tmp_path,
        target_model_path=G1_MODEL,
    )

    assert report.kinematic_valid_count == 1
    assert report.clips[0].qualification is MotionQualificationLevel.Q1_KINEMATIC_ONLY
    assert report.clips[0].episode_summary is not None
    assert report.clips[0].episode_summary["mapping_hash"]
    assert report.experience_capsule is not None
    assert report.experience_capsule.training_eligible is False
    assert report.experience_capsule.promotion_truth_allowed is False
    assert report.training_eligible is False
    assert "LICENSE_NOT_PERMITTED" in report.training_blockers
    assert "Q3_OR_Q4_PHYSICS_QUALIFICATION_REQUIRED" in report.training_blockers
    assert report.to_dict()["eligibility"]["motion_reference_discovery"] is True
    assert report.to_dict()["eligibility"]["football_contact_training"] is False


def test_joint_limit_violation_is_q0_and_yields_no_capsule(tmp_path: Path) -> None:
    _dataset(tmp_path, joint_value=99.0)
    registration = _register(tmp_path)

    report = audit_motiondecode_snapshot(
        registration,
        tmp_path,
        target_model_path=G1_MODEL,
    )

    assert report.kinematic_valid_count == 0
    assert report.clips[0].qualification is MotionQualificationLevel.Q0_INVALID
    assert any(issue.code == "JOINT_LIMIT_ERROR" for issue in report.clips[0].issues)
    assert report.experience_capsule is None


def test_terminal_loop_closure_is_detected_as_a_discontinuity(tmp_path: Path) -> None:
    sample = _dataset(tmp_path)
    _append_terminal_reset(sample)
    registration = _register(tmp_path)

    report = audit_motiondecode_snapshot(
        registration,
        tmp_path,
        target_model_path=G1_MODEL,
    )

    assert report.clips[0].qualification is MotionQualificationLevel.Q0_INVALID
    assert any(issue.code == "ROOT_LOOP_CLOSURE_DISCONTINUITY" for issue in report.clips[0].issues)
    assert report.experience_capsule is None
    assert "NO_KINEMATICALLY_VALID_CLIPS" in report.training_blockers
    assert report.segmentation_repair_candidate_count == 1
    assert report.issue_clip_counts["ROOT_LOOP_CLOSURE_DISCONTINUITY"] == 1


def test_terminal_reset_repair_is_content_addressed_and_reaches_q1(
    tmp_path: Path,
) -> None:
    sample = _dataset(tmp_path)
    _append_terminal_reset(sample)
    registration = _register(tmp_path)
    before = audit_motiondecode_snapshot(
        registration,
        tmp_path,
        target_model_path=G1_MODEL,
    )

    report = repair_motiondecode_snapshot(
        registration,
        tmp_path,
        target_model_path=G1_MODEL,
        expected_ingest_report_hash=before.report_hash,
    )

    assert report.repaired_q1_count == 1
    assert report.q1_after_count == 1
    assert report.training_eligible is False
    assert "LICENSE_NOT_PERMITTED" in report.training_blockers
    result = report.results[0]
    assert result.disposition is MotionRepairDisposition.REPAIRED_Q1
    assert result.repair_manifest is not None
    assert result.repair_manifest.original_frame_count == 5
    assert result.repair_manifest.retained_frame_count == 4
    assert result.repair_manifest.removed_frame_count == 1
    assert result.after_audit is not None
    assert result.after_audit.kinematic_valid is True
    assert result.after_audit.episode_summary is not None
    assert (
        result.after_audit.episode_summary["derivation_manifest_hash"]
        == result.repair_manifest.manifest_hash
    )
    persisted = report.to_dict()
    assert persisted["raw_motion_persisted"] is False
    assert persisted["eligibility"]["mujoco_qualification_candidate"] is True
    assert persisted["eligibility"]["mujoco_qualification_authorized"] is False
    assert persisted["eligibility"]["motion_tracker_training"] is False

    replayed_manifest = SegmentationRepairManifest.from_dict(result.repair_manifest.to_dict())
    replayed_episode = replay_segmentation_repair(
        registration,
        tmp_path,
        target_model_path=G1_MODEL,
        manifest=replayed_manifest,
    )
    assert replayed_manifest.manifest_hash == result.repair_manifest.manifest_hash
    assert replayed_episode.time.shape == (4,)
    assert replayed_episode.joint_position.flags.writeable is False
    assert replayed_episode.derivation_manifest_hash == replayed_manifest.manifest_hash
    embedded = result.repair_manifest.to_dict()
    embedded["raw_motion_embedded"] = True
    with pytest.raises(ValueError, match="cannot embed raw motion"):
        SegmentationRepairManifest.from_dict(embedded)


def test_ambiguous_terminal_resets_are_not_automatically_trimmed(
    tmp_path: Path,
) -> None:
    sample = _dataset(tmp_path)
    with sample.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    initial = rows[1]
    sequence: list[list[str]] = []
    for root_x in (0.0, 0.06, 0.12, 0.0, 0.06, 0.12, 0.0):
        row = initial.copy()
        row[0] = str(root_x)
        sequence.append(row)
    with sample.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows([header, *sequence])
    registration = _register(tmp_path)
    before = audit_motiondecode_snapshot(
        registration,
        tmp_path,
        target_model_path=G1_MODEL,
    )

    report = repair_motiondecode_snapshot(
        registration,
        tmp_path,
        target_model_path=G1_MODEL,
        expected_ingest_report_hash=before.report_hash,
    )

    assert report.repaired_q1_count == 0
    assert report.rejected_count == 1
    assert report.results[0].disposition is MotionRepairDisposition.REJECTED
    assert report.results[0].reason_codes == ("AMBIGUOUS_TERMINAL_RESETS",)
    assert report.results[0].repair_manifest is None


def test_repair_replays_the_exact_ingest_commitment(tmp_path: Path) -> None:
    sample = _dataset(tmp_path)
    _append_terminal_reset(sample)
    registration = _register(tmp_path)

    with pytest.raises(ValueError, match="ingest report hash"):
        repair_motiondecode_snapshot(
            registration,
            tmp_path,
            target_model_path=G1_MODEL,
            expected_ingest_report_hash="sha256:" + "0" * 64,
        )


def test_non_segmentation_error_blocks_terminal_trim(tmp_path: Path) -> None:
    sample = _dataset(tmp_path, joint_value=99.0)
    _append_terminal_reset(sample)
    registration = _register(tmp_path)
    before = audit_motiondecode_snapshot(
        registration,
        tmp_path,
        target_model_path=G1_MODEL,
    )

    report = repair_motiondecode_snapshot(
        registration,
        tmp_path,
        target_model_path=G1_MODEL,
        expected_ingest_report_hash=before.report_hash,
    )

    assert report.repaired_q1_count == 0
    assert report.results[0].reason_codes == ("UNREPAIRABLE_JOINT_LIMIT_ERROR",)


def test_cli_terminal_repair_persists_only_audit_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = _dataset(tmp_path / "dataset")
    _append_terminal_reset(sample)
    registration = _register(tmp_path / "dataset")
    before = audit_motiondecode_snapshot(
        registration,
        tmp_path / "dataset",
        target_model_path=G1_MODEL,
    )
    registration_path = tmp_path / "registration.json"
    ingest_path = tmp_path / "ingest.json"
    repair_path = tmp_path / "repair.json"
    registration_path.write_text(
        json.dumps(
            {
                "schema_version": "rosclaw.collective.motiondecode_registration_artifact.v1",
                "registration": registration.to_dict(),
                "registration_hash": registration.registration_hash,
            }
        ),
        encoding="utf-8",
    )
    ingest_path.write_text(
        json.dumps(
            {
                "schema_version": "rosclaw.collective.motiondecode_ingest_artifact.v1",
                "report": before.to_dict(),
                "report_hash": before.report_hash,
            }
        ),
        encoding="utf-8",
    )

    assert (
        dispatch_collective_argv(
            [
                "collective",
                "repair",
                "motiondecode",
                "--registration",
                str(registration_path),
                "--ingest-report",
                str(ingest_path),
                "--dataset-root",
                str(tmp_path / "dataset"),
                "--target-model",
                str(G1_MODEL),
                "--output",
                str(repair_path),
                "--source-checkout",
                str(ROOT),
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["repaired_q1_count"] == 1
    assert receipt["dry_run_only"] is True
    assert receipt["raw_motion_persisted"] is False
    assert receipt["training_eligible"] is False
    artifact = json.loads(repair_path.read_text(encoding="utf-8"))
    assert artifact["report_hash"] == receipt["report_hash"]
    assert artifact["report"]["results"][0]["repair_manifest"]["raw_motion_embedded"] is False


def test_cli_rejects_tampered_ingest_commitment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = _dataset(tmp_path / "dataset")
    _append_terminal_reset(sample)
    registration = _register(tmp_path / "dataset")
    before = audit_motiondecode_snapshot(
        registration,
        tmp_path / "dataset",
        target_model_path=G1_MODEL,
    )
    registration_path = tmp_path / "registration.json"
    ingest_path = tmp_path / "ingest.json"
    registration_path.write_text(
        json.dumps(
            {
                "registration": registration.to_dict(),
                "registration_hash": registration.registration_hash,
            }
        ),
        encoding="utf-8",
    )
    ingest_path.write_text(
        json.dumps(
            {
                "report": before.to_dict(),
                "report_hash": "sha256:" + "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    assert (
        dispatch_collective_argv(
            [
                "collective",
                "repair",
                "motiondecode",
                "--registration",
                str(registration_path),
                "--ingest-report",
                str(ingest_path),
                "--dataset-root",
                str(tmp_path / "dataset"),
                "--target-model",
                str(G1_MODEL),
                "--output",
                str(tmp_path / "repair.json"),
                "--source-checkout",
                str(ROOT),
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().err)
    assert error["training_eligible"] is False
    assert "does not replay" in error["error"]


def test_cli_registration_inspection_and_ingest_are_replayable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _dataset(tmp_path / "dataset")
    registration_path = tmp_path / "evidence/registration.json"
    ingest_path = tmp_path / "evidence/ingest.json"

    assert (
        dispatch_collective_argv(
            [
                "collective",
                "source",
                "add",
                "motiondecode",
                "--dataset-root",
                str(tmp_path / "dataset"),
                "--revision",
                REVISION,
                "--terms-file",
                str(tmp_path / "dataset/LICENSE"),
                "--terms-uri",
                (f"https://huggingface.co/datasets/CMRobot/MotionDecode/blob/{REVISION}/LICENSE"),
                "--families",
                "football",
                "--limit",
                "40",
                "--output",
                str(registration_path),
                "--source-checkout",
                str(ROOT),
            ]
        )
        == 0
    )
    add_receipt = json.loads(capsys.readouterr().out)
    assert add_receipt["selected_sample_count"] == 1
    assert add_receipt["license_decision"] == "pending"
    assert add_receipt["training_eligible"] is False

    assert (
        dispatch_collective_argv(
            [
                "collective",
                "source",
                "inspect",
                "motiondecode",
                "--registration",
                str(registration_path),
            ]
        )
        == 0
    )
    inspection = json.loads(capsys.readouterr().out)
    assert inspection["registration_hash"] == add_receipt["registration_hash"]

    assert (
        dispatch_collective_argv(
            [
                "collective",
                "ingest",
                "motiondecode",
                "--registration",
                str(registration_path),
                "--dataset-root",
                str(tmp_path / "dataset"),
                "--target-model",
                str(G1_MODEL),
                "--output",
                str(ingest_path),
                "--source-checkout",
                str(ROOT),
            ]
        )
        == 0
    )
    ingest_receipt = json.loads(capsys.readouterr().out)
    assert ingest_receipt["kinematic_valid_count"] == 1
    assert ingest_receipt["experience_capsule_hash"]
    assert ingest_receipt["training_eligible"] is False
    persisted = json.loads(ingest_path.read_text(encoding="utf-8"))
    assert persisted["report_hash"] == ingest_receipt["report_hash"]


def test_cli_rejects_tampered_registration_commitment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _dataset(tmp_path / "dataset")
    registration = _register(tmp_path / "dataset")
    artifact = {
        "schema_version": "rosclaw.collective.motiondecode_registration_artifact.v1",
        "registration": registration.to_dict(),
        "registration_hash": "sha256:" + "0" * 64,
    }
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    assert (
        dispatch_collective_argv(
            [
                "collective",
                "source",
                "inspect",
                "motiondecode",
                "--registration",
                str(path),
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().err)
    assert error["training_eligible"] is False
    assert "does not replay" in error["error"]
