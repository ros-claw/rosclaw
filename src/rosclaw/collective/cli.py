"""CLI for governed external-experience registration and ingestion."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from rosclaw.collective.contracts import LicenseDecision, LicenseUse
from rosclaw.continual.services.persistence import atomic_write_json
from rosclaw.feedback.contracts import canonical_hash

if TYPE_CHECKING:
    from rosclaw.collective.sources.motiondecode.manifest import MotionDecodeRegistration
    from rosclaw.collective.sources.motiondecode.taxonomy import MotionFamily


def dispatch_collective_argv(argv: list[str]) -> int | None:
    if not argv or argv[0] != "collective":
        return None
    args = _parser().parse_args(argv[1:])
    try:
        return int(args.handler(args))
    except (OSError, RuntimeError, ValueError, KeyError, PermissionError) as exc:
        _print(
            {
                "schema_version": "rosclaw.collective.cli_error.v1",
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "training_eligible": False,
                "activation_authorized": False,
                "hardware_authorized": False,
            },
            stream=sys.stderr,
        )
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rosclaw collective",
        description="Register and audit external experience without authorizing hardware.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    source = commands.add_parser("source", help="manage content-addressed source evidence")
    source_commands = source.add_subparsers(dest="source_command", required=True)

    add = source_commands.add_parser("add", help="register an operator-managed local snapshot")
    add.add_argument("adapter", choices=["motiondecode"])
    add.add_argument("--dataset-root", type=Path, required=True)
    add.add_argument("--revision", required=True)
    add.add_argument(
        "--usage",
        choices=[item.value for item in LicenseUse],
        default=LicenseUse.RESEARCH_NONCOMMERCIAL.value,
    )
    add.add_argument(
        "--license-decision",
        choices=[item.value for item in LicenseDecision],
        default=LicenseDecision.PENDING.value,
    )
    add.add_argument("--terms-file", type=Path)
    add.add_argument("--terms-uri")
    add.add_argument("--attribution", default="ChingMu / CMRobot MotionDecode")
    add.add_argument(
        "--families",
        default="",
        help="comma-separated football,balance,gait,transition_recovery,other",
    )
    add.add_argument("--limit", type=int, default=400)
    add.add_argument("--output", type=Path, required=True)
    add.add_argument("--source-checkout", type=Path, default=Path.cwd())
    add.add_argument("--force", action="store_true")
    add.set_defaults(handler=_source_add)

    inspect = source_commands.add_parser(
        "inspect", help="replay and summarize a registration artifact"
    )
    inspect.add_argument("adapter", choices=["motiondecode"])
    inspect.add_argument("--registration", type=Path, required=True)
    inspect.set_defaults(handler=_source_inspect)

    ingest = commands.add_parser(
        "ingest",
        help="rehash and kinematically audit a registered local snapshot",
    )
    ingest.add_argument("adapter", choices=["motiondecode"])
    ingest.add_argument("--registration", type=Path, required=True)
    ingest.add_argument("--dataset-root", type=Path, required=True)
    ingest.add_argument("--target-model", type=Path, required=True)
    ingest.add_argument("--output", type=Path, required=True)
    ingest.add_argument("--source-checkout", type=Path, default=Path.cwd())
    ingest.add_argument("--force", action="store_true")
    ingest.set_defaults(handler=_ingest)

    repair = commands.add_parser(
        "repair",
        help="dry-run bounded repairs and re-audit retained motion",
    )
    repair.add_argument("adapter", choices=["motiondecode"])
    repair.add_argument("--registration", type=Path, required=True)
    repair.add_argument("--ingest-report", type=Path, required=True)
    repair.add_argument("--dataset-root", type=Path, required=True)
    repair.add_argument("--target-model", type=Path, required=True)
    repair.add_argument("--output", type=Path, required=True)
    repair.add_argument("--source-checkout", type=Path, default=Path.cwd())
    repair.add_argument("--force", action="store_true")
    repair.set_defaults(handler=_repair)

    contact = commands.add_parser(
        "contact",
        help="infer kinematic foot contact and support phases from repaired motion",
    )
    contact.add_argument("adapter", choices=["motiondecode"])
    contact.add_argument("--registration", type=Path, required=True)
    contact.add_argument("--ingest-report", type=Path, required=True)
    contact.add_argument("--repair-report", type=Path, required=True)
    contact.add_argument("--dataset-root", type=Path, required=True)
    contact.add_argument("--target-model", type=Path, required=True)
    contact.add_argument("--output", type=Path, required=True)
    contact.add_argument("--source-checkout", type=Path, default=Path.cwd())
    contact.add_argument("--force", action="store_true")
    contact.set_defaults(handler=_contact)

    qualify = commands.add_parser(
        "qualify",
        help="run strict CPU MuJoCo qualification for eligible motion references",
    )
    qualify.add_argument("adapter", choices=["motiondecode"])
    qualify.add_argument("--registration", type=Path, required=True)
    qualify.add_argument("--ingest-report", type=Path, required=True)
    qualify.add_argument("--repair-report", type=Path, required=True)
    qualify.add_argument("--contact-report", type=Path, required=True)
    qualify.add_argument("--dataset-root", type=Path, required=True)
    qualify.add_argument("--target-model", type=Path, required=True)
    qualify.add_argument("--scene", type=Path, required=True)
    qualify.add_argument("--output", type=Path, required=True)
    qualify.add_argument("--source-checkout", type=Path, default=Path.cwd())
    qualify.add_argument("--force", action="store_true")
    qualify.set_defaults(handler=_qualify)

    prior = commands.add_parser(
        "prior",
        help="build and train a self-supervised kinematic motion prior",
    )
    prior_commands = prior.add_subparsers(dest="prior_command", required=True)

    prior_build = prior_commands.add_parser(
        "build",
        help="build a bounded audited tensor pack from an ingest report",
    )
    prior_build.add_argument("adapter", choices=["motiondecode"])
    prior_build.add_argument("--registration", type=Path, required=True)
    prior_build.add_argument("--ingest-report", type=Path, required=True)
    prior_build.add_argument(
        "--repair-report",
        type=Path,
        help="optional content-addressed Q1 repair evidence to replay",
    )
    prior_build.add_argument("--dataset-root", type=Path, required=True)
    prior_build.add_argument("--target-model", type=Path, required=True)
    prior_build.add_argument(
        "--transfer-asset-root",
        type=Path,
        help="optional qualified G1 actor body to bind representation transfer",
    )
    prior_build.add_argument("--output", type=Path, required=True)
    prior_build.add_argument("--sequence-length", type=int, default=32)
    prior_build.add_argument("--maximum-windows", type=int, default=12_000)
    prior_build.add_argument("--seed", type=int, default=20260801)
    prior_build.add_argument(
        "--stratum",
        action="append",
        choices=[
            "football",
            "balance",
            "gait",
            "transition_recovery",
            "other",
        ],
    )
    prior_build.add_argument("--source-checkout", type=Path, default=Path.cwd())
    prior_build.add_argument("--force", action="store_true")
    prior_build.set_defaults(handler=_prior_build)

    prior_train = prior_commands.add_parser(
        "train",
        help="train four independent physical-GPU representation candidates",
    )
    prior_train.add_argument("adapter", choices=["motiondecode"])
    prior_train.add_argument("--pack", type=Path, required=True)
    prior_train.add_argument("--output-dir", type=Path, required=True)
    prior_train.add_argument("--epochs", type=int, default=10)
    prior_train.add_argument("--hidden-dim", type=int, default=96)
    prior_train.add_argument("--batch-size", type=int, default=256)
    prior_train.add_argument("--base-seed", type=int, default=8200)
    prior_train.add_argument("--source-checkout", type=Path, default=Path.cwd())
    prior_train.add_argument("--force", action="store_true")
    prior_train.set_defaults(handler=_prior_train)
    return parser


def _source_add(args: argparse.Namespace) -> int:
    from rosclaw.collective.sources.motiondecode.manifest import (
        register_motiondecode_source,
    )

    output = _output_path(args.output, args.source_checkout, force=args.force)
    registration = register_motiondecode_source(
        args.dataset_root,
        revision=args.revision,
        requested_use=LicenseUse(args.usage),
        license_decision=LicenseDecision(args.license_decision),
        terms_path=args.terms_file,
        terms_uri=args.terms_uri,
        attribution_text=args.attribution,
        families=_families(args.families),
        limit=args.limit,
    )
    artifact = _registration_artifact(registration)
    atomic_write_json(output, artifact)
    _print(
        {
            "schema_version": "rosclaw.collective.source_add_receipt.v1",
            "ok": registration.source_registered,
            "output": str(output),
            "registration_hash": registration.registration_hash,
            "source_manifest_hash": registration.manifest.manifest_hash,
            "catalog_schema_valid": registration.catalog_audit.schema_valid,
            "selected_sample_count": registration.manifest.selected_sample_count,
            "license_decision": registration.manifest.license_snapshot.decision.value,
            "training_eligible": registration.training_eligible,
            "training_blockers": registration.to_dict()["training_blockers"],
            "activation_authorized": False,
            "hardware_authorized": False,
        }
    )
    return 0 if registration.source_registered else 1


def _source_inspect(args: argparse.Namespace) -> int:
    registration = _read_registration(args.registration)
    _print(
        {
            "schema_version": "rosclaw.collective.source_inspection.v1",
            "ok": registration.source_registered,
            "registration_hash": registration.registration_hash,
            "source_manifest_hash": registration.manifest.manifest_hash,
            "source_identity_hash": registration.manifest.source_identity.source_hash,
            "revision": registration.manifest.revision,
            "catalog_audit": registration.catalog_audit.to_dict(),
            "selected_sample_count": registration.manifest.selected_sample_count,
            "inventory_scope": "operator_managed_local_snapshot",
            "upstream_inventory_verified": False,
            "local_discovered_sample_count": (registration.manifest.local_discovered_sample_count),
            "local_selection_complete": registration.manifest.local_selection_complete,
            "requested_families": [
                family.value for family in registration.manifest.requested_families
            ],
            "license_snapshot": registration.manifest.license_snapshot.to_dict(),
            "attribution": registration.manifest.attribution.to_dict(),
            "training_eligible": registration.training_eligible,
            "training_blockers": registration.to_dict()["training_blockers"],
            "activation_authorized": False,
            "hardware_authorized": False,
        }
    )
    return 0 if registration.source_registered else 1


def _ingest(args: argparse.Namespace) -> int:
    from rosclaw.collective.sources.motiondecode.audit import (
        audit_motiondecode_snapshot,
    )

    output = _output_path(args.output, args.source_checkout, force=args.force)
    registration = _read_registration(args.registration)
    report = audit_motiondecode_snapshot(
        registration,
        args.dataset_root,
        target_model_path=args.target_model,
    )
    artifact = {
        "schema_version": "rosclaw.collective.motiondecode_ingest_artifact.v1",
        "report": report.to_dict(),
        "report_hash": report.report_hash,
    }
    atomic_write_json(output, artifact)
    _print(
        {
            "schema_version": "rosclaw.collective.ingest_receipt.v1",
            "ok": report.kinematic_valid_count > 0,
            "output": str(output),
            "report_hash": report.report_hash,
            "source_manifest_hash": report.source_manifest_hash,
            "clip_count": len(report.clips),
            "kinematic_valid_count": report.kinematic_valid_count,
            "qualification_counts": report.qualification_counts,
            "issue_clip_counts": report.issue_clip_counts,
            "segmentation_repair_candidate_count": (report.segmentation_repair_candidate_count),
            "experience_capsule_hash": (
                report.experience_capsule.capsule_hash
                if report.experience_capsule is not None
                else None
            ),
            "training_eligible": report.training_eligible,
            "training_blockers": report.training_blockers,
            "activation_authorized": False,
            "hardware_authorized": False,
        }
    )
    return 0 if report.kinematic_valid_count > 0 else 1


def _repair(args: argparse.Namespace) -> int:
    from rosclaw.collective.sources.motiondecode.audit import (
        MotionDecodeAuditThresholds,
    )
    from rosclaw.collective.sources.motiondecode.repair import (
        repair_motiondecode_snapshot,
    )

    output = _output_path(args.output, args.source_checkout, force=args.force)
    registration = _read_registration(args.registration)
    expected_ingest_hash, thresholds = _read_ingest_commitment(
        args.ingest_report,
        MotionDecodeAuditThresholds,
    )
    report = repair_motiondecode_snapshot(
        registration,
        args.dataset_root,
        target_model_path=args.target_model,
        expected_ingest_report_hash=expected_ingest_hash,
        thresholds=thresholds,
    )
    artifact = {
        "schema_version": "rosclaw.collective.motiondecode_repair_artifact.v1",
        "report": report.to_dict(),
        "report_hash": report.report_hash,
    }
    atomic_write_json(output, artifact)
    _print(
        {
            "schema_version": "rosclaw.collective.repair_receipt.v1",
            "ok": report.q1_after_count > 0,
            "output": str(output),
            "report_hash": report.report_hash,
            "original_ingest_report_hash": report.original_ingest_report_hash,
            "detector_hash": report.detector_hash,
            "clip_count": len(report.results),
            "disposition_counts": report.disposition_counts,
            "repaired_q1_count": report.repaired_q1_count,
            "q1_after_count": report.q1_after_count,
            "rejected_count": report.rejected_count,
            "reason_clip_counts": report.reason_clip_counts,
            "quality_commitment": report.quality_commitment,
            "dry_run_only": True,
            "raw_motion_persisted": False,
            "training_eligible": report.training_eligible,
            "training_blockers": report.training_blockers,
            "activation_authorized": False,
            "hardware_authorized": False,
        }
    )
    return 0 if report.q1_after_count > 0 else 1


def _contact(args: argparse.Namespace) -> int:
    from rosclaw.collective.sources.motiondecode.audit import (
        MotionDecodeAuditThresholds,
    )
    from rosclaw.collective.sources.motiondecode.contact import (
        infer_motiondecode_contacts,
    )

    output = _output_path(args.output, args.source_checkout, force=args.force)
    registration = _read_registration(args.registration)
    expected_ingest_hash, audit_thresholds = _read_ingest_commitment(
        args.ingest_report,
        MotionDecodeAuditThresholds,
    )
    expected_repair_hash = _read_report_commitment(
        args.repair_report,
        label="repair",
    )
    report = infer_motiondecode_contacts(
        registration,
        args.dataset_root,
        target_model_path=args.target_model,
        expected_ingest_report_hash=expected_ingest_hash,
        expected_repair_report_hash=expected_repair_hash,
        audit_thresholds=audit_thresholds,
    )
    artifact = {
        "schema_version": "rosclaw.collective.motiondecode_contact_artifact.v1",
        "report": report.to_dict(),
        "report_hash": report.report_hash,
    }
    atomic_write_json(output, artifact)
    _print(
        {
            "schema_version": "rosclaw.collective.contact_receipt.v1",
            "ok": report.phase_candidate_count > 0,
            "output": str(output),
            "report_hash": report.report_hash,
            "repair_report_hash": report.repair_report_hash,
            "threshold_hash": report.thresholds.threshold_hash,
            "clip_count": len(report.clips),
            "inferred_count": report.inferred_count,
            "phase_candidate_count": report.phase_candidate_count,
            "issue_clip_counts": report.issue_clip_counts,
            "quality_commitment": report.quality_commitment,
            "frame_level_trace_persisted": False,
            "training_eligible": False,
            "training_blockers": report.training_blockers,
            "activation_authorized": False,
            "hardware_authorized": False,
        }
    )
    return 0 if report.phase_candidate_count > 0 else 1


def _qualify(args: argparse.Namespace) -> int:
    from rosclaw.collective.sources.motiondecode.audit import (
        MotionDecodeAuditThresholds,
    )
    from rosclaw.collective.sources.motiondecode.qualification import (
        qualify_motiondecode_snapshot,
    )

    output = _output_path(args.output, args.source_checkout, force=args.force)
    registration = _read_registration(args.registration)
    expected_ingest_hash, audit_thresholds = _read_ingest_commitment(
        args.ingest_report,
        MotionDecodeAuditThresholds,
    )
    expected_repair_hash = _read_report_commitment(
        args.repair_report,
        label="repair",
    )
    expected_contact_hash = _read_report_commitment(
        args.contact_report,
        label="contact",
    )
    report = qualify_motiondecode_snapshot(
        registration,
        args.dataset_root,
        target_model_path=args.target_model,
        scene_path=args.scene,
        expected_ingest_report_hash=expected_ingest_hash,
        expected_repair_report_hash=expected_repair_hash,
        expected_contact_report_hash=expected_contact_hash,
        audit_thresholds=audit_thresholds,
    )
    artifact = {
        "schema_version": "rosclaw.collective.motiondecode_qualification_artifact.v1",
        "report": report.to_dict(),
        "report_hash": report.report_hash,
    }
    atomic_write_json(output, artifact)
    _print(
        {
            "schema_version": "rosclaw.collective.qualification_receipt.v1",
            "ok": report.q3_count > 0,
            "output": str(output),
            "report_hash": report.report_hash,
            "contact_report_hash": report.contact_report_hash,
            "scene_file_hash": report.scene_file_hash,
            "compiled_scene_hash": report.compiled_scene_hash,
            "threshold_hash": report.thresholds.threshold_hash,
            "clip_count": len(report.clips),
            "status_counts": report.status_counts,
            "qualification_counts": report.qualification_counts,
            "physics_executed_count": report.physics_executed_count,
            "physics_step_count": report.physics_step_count,
            "q3_count": report.q3_count,
            "quality_commitment": report.quality_commitment,
            "training_eligible": False,
            "training_blockers": report.training_blockers,
            "promotion_truth_eligible": False,
            "activation_authorized": False,
            "hardware_authorized": False,
        }
    )
    return 0 if report.q3_count > 0 else 1


def _prior_build(args: argparse.Namespace) -> int:
    try:
        from rosclaw.collective.sources.motiondecode.motion_prior import (
            build_motion_prior_pack,
        )
    except ModuleNotFoundError as exc:
        _raise_missing_rl_extra(exc)

    output = _output_path(args.output, args.source_checkout, force=args.force)
    if output.exists() and args.force:
        output.unlink()
    sidecar = output.with_suffix(".json")
    if sidecar.exists() and args.force:
        sidecar.unlink()
    registration = _read_registration(args.registration)
    metadata = build_motion_prior_pack(
        registration=registration,
        ingest_report_path=args.ingest_report,
        repair_report_path=args.repair_report,
        dataset_root=args.dataset_root,
        model_path=args.target_model,
        transfer_asset_root=args.transfer_asset_root,
        output_path=output,
        sequence_length=args.sequence_length,
        maximum_windows=args.maximum_windows,
        seed=args.seed,
        allowed_strata=tuple(args.stratum) if args.stratum else None,
    )
    _print(
        {
            "schema_version": "rosclaw.collective.prior_build_receipt.v1",
            "ok": True,
            "output": str(output),
            "pack_hash": metadata["pack_hash"],
            "registration_hash": metadata["registration_hash"],
            "ingest_report_hash": metadata["ingest_report_hash"],
            "repair_report_hash": metadata["repair_report_hash"],
            "source_manifest_hash": metadata["source_manifest_hash"],
            "body_hash": metadata["body_hash"],
            "kinematic_body_hash": metadata["kinematic_body_hash"],
            "transfer_contract": metadata["transfer_contract"],
            "feature_count": len(metadata["feature_names"]),
            "sequence_length": metadata["sequence_length"],
            "training_windows": metadata["training_windows"],
            "validation_windows": metadata["validation_windows"],
            "source_episode_count": metadata["source_episode_count"],
            "repaired_source_episode_count": metadata["repaired_source_episode_count"],
            "allowed_strata": metadata["allowed_strata"],
            "skipped_clip_count": len(metadata["skipped_clips"]),
            "action_semantics": metadata["action_semantics"],
            "raw_data_exported": metadata["raw_data_exported"],
            "training_eligible": False,
            "activation_authorized": False,
            "hardware_authorized": False,
        }
    )
    return 0


def _prior_train(args: argparse.Namespace) -> int:
    try:
        from rosclaw.collective.sources.motiondecode.motion_prior import (
            run_four_gpu_motion_prior,
        )
    except ModuleNotFoundError as exc:
        _raise_missing_rl_extra(exc)

    output_dir = _output_path(args.output_dir, args.source_checkout, force=args.force)
    if output_dir.exists() and args.force:
        import shutil

        shutil.rmtree(output_dir)
    report = run_four_gpu_motion_prior(
        pack_path=args.pack,
        output_dir=output_dir,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        base_seed=args.base_seed,
    )
    passed = report["decision"] == "REPRESENTATION_CANDIDATE"
    _print(
        {
            "schema_version": "rosclaw.collective.prior_train_receipt.v1",
            "ok": passed,
            "output_dir": str(output_dir),
            "decision": report["decision"],
            "four_physical_gpus_exercised": report["four_physical_gpus_exercised"],
            "quality_gate": report["quality_gate"],
            "selected": report["selected"],
            "failure_count": len(report["failures"]),
            "training_eligible": False,
            "promotion_evidence_eligible": False,
            "activation_authorized": False,
            "hardware_authorized": False,
        }
    )
    return 0 if passed else 2


def _raise_missing_rl_extra(exc: ModuleNotFoundError) -> NoReturn:
    if exc.name == "torch":
        raise RuntimeError(
            "Motion-prior build/train requires the optional RL dependencies; "
            "install 'rosclaw[rl]'"
        ) from exc
    raise exc


def _families(value: str) -> tuple[MotionFamily, ...]:
    from rosclaw.collective.sources.motiondecode.taxonomy import MotionFamily

    if not value.strip():
        return ()
    names = tuple(item.strip() for item in value.split(","))
    if any(not item for item in names) or len(names) != len(set(names)):
        raise ValueError("families must contain unique non-empty names")
    try:
        return tuple(MotionFamily(item) for item in names)
    except ValueError as exc:
        raise ValueError("families contains an unknown MotionDecode family") from exc


def _registration_artifact(registration: MotionDecodeRegistration) -> dict[str, Any]:
    return {
        "schema_version": "rosclaw.collective.motiondecode_registration_artifact.v1",
        "registration": registration.to_dict(),
        "registration_hash": registration.registration_hash,
    }


def _read_registration(path: Path) -> MotionDecodeRegistration:
    from rosclaw.collective.sources.motiondecode.manifest import (
        MotionDecodeRegistration,
    )

    value = _read_object(path)
    registration_value = value.get("registration")
    if not isinstance(registration_value, dict):
        raise ValueError("registration artifact lacks a registration object")
    registration = MotionDecodeRegistration.from_dict(registration_value)
    if value.get("registration_hash") != registration.registration_hash:
        raise ValueError("registration_hash does not replay")
    return registration


def _read_ingest_commitment(
    path: Path,
    thresholds_type: Any,
) -> tuple[str, Any]:
    value = _read_object(path)
    report_value = value.get("report")
    if not isinstance(report_value, dict):
        raise ValueError("ingest artifact lacks a report object")
    expected_hash = value.get("report_hash")
    if not isinstance(expected_hash, str) or expected_hash != canonical_hash(report_value):
        raise ValueError("ingest report_hash does not replay")
    thresholds_value = report_value.get("thresholds")
    if not isinstance(thresholds_value, dict):
        raise ValueError("ingest report lacks audit thresholds")
    expected_fields = set(thresholds_type().to_dict())
    if set(thresholds_value) != expected_fields or any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for item in thresholds_value.values()
    ):
        raise ValueError("ingest report audit thresholds are invalid")
    return expected_hash, thresholds_type(**thresholds_value)


def _read_report_commitment(path: Path, *, label: str) -> str:
    value = _read_object(path)
    report_value = value.get("report")
    if not isinstance(report_value, dict):
        raise ValueError(f"{label} artifact lacks a report object")
    expected_hash = value.get("report_hash")
    if not isinstance(expected_hash, str) or expected_hash != canonical_hash(report_value):
        raise ValueError(f"{label} report_hash does not replay")
    return expected_hash


def _read_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("JSON artifact must be an object")
    return value


def _output_path(path: Path, checkout: Path, *, force: bool) -> Path:
    resolved = path.expanduser().resolve()
    source = checkout.expanduser().resolve()
    if resolved == source or source in resolved.parents:
        raise ValueError("collective evidence output must be outside the source checkout")
    if resolved.exists() and not force:
        raise FileExistsError("output already exists; pass --force to replace it")
    return resolved


def _print(value: Mapping[str, Any], *, stream: Any | None = None) -> None:
    print(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False),
        file=stream if stream is not None else sys.stdout,
    )


__all__ = ["dispatch_collective_argv"]
