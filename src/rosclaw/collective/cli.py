"""CLI for provenance-safe collective experience ingestion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NoReturn

from rosclaw.collective.sources.motiondecode.taxonomy import MotionDecodeStratum


def dispatch_collective_argv(argv: list[str]) -> int | None:
    if not argv or argv[0] != "collective":
        return None
    parser = _parser()
    args = parser.parse_args(argv[1:])
    try:
        if args.command == "source" and args.source_command == "inspect":
            return _inspect(args)
        if args.command == "ingest":
            return _ingest(args)
        if args.command == "prior" and args.prior_command == "build":
            return _build_prior(args)
        if args.command == "prior" and args.prior_command == "train":
            return _train_prior(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "rosclaw.collective.cli_error.v1",
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=True,
            )
        )
        return 2
    parser.print_help()
    return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rosclaw collective")
    commands = parser.add_subparsers(dest="command")
    source = commands.add_parser("source", help="Inspect an external experience source")
    source_commands = source.add_subparsers(dest="source_command")
    inspect = source_commands.add_parser("inspect")
    _source_arguments(inspect)
    inspect.add_argument("--json", action="store_true")

    ingest = commands.add_parser("ingest", help="Build a bounded experience pilot")
    _source_arguments(ingest)
    ingest.add_argument("--model-path", type=Path, required=True)
    ingest.add_argument("--asset-root", type=Path)
    ingest.add_argument("--output-dir", type=Path, required=True)
    ingest.add_argument("--limit", type=int, default=400)
    ingest.add_argument("--seed", type=int, default=20260801)
    ingest.add_argument("--json", action="store_true")

    prior = commands.add_parser("prior", help="Build and train a kinematic motion prior")
    prior_commands = prior.add_subparsers(dest="prior_command")
    build = prior_commands.add_parser("build", help="Build a bounded audited tensor pack")
    build.add_argument("--pilot-report", type=Path, required=True)
    build.add_argument("--dataset-root", type=Path, required=True)
    build.add_argument("--model-path", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--sequence-length", type=int, default=32)
    build.add_argument("--maximum-windows", type=int, default=12_000)
    build.add_argument("--seed", type=int, default=20260801)
    build.add_argument(
        "--stratum",
        action="append",
        choices=tuple(value.value for value in MotionDecodeStratum),
    )
    build.add_argument("--json", action="store_true")

    train = prior_commands.add_parser(
        "train", help="Train four independent physical-GPU representation candidates"
    )
    train.add_argument("--pack", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--hidden-dim", type=int, default=96)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--base-seed", type=int, default=8200)
    train.add_argument("--json", action="store_true")
    return parser


def _source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("provider", choices=("motiondecode",))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--usage",
        choices=("research", "personal-study", "noncommercial-prototype"),
        default="research",
    )


def _inspect(args: argparse.Namespace) -> int:
    from rosclaw.collective.sources.motiondecode.manifest import (
        inspect_motiondecode_source,
    )

    manifest, _ = inspect_motiondecode_source(
        args.dataset_root,
        revision=args.revision,
        requested_usage=args.usage,
    )
    value = manifest.to_dict()
    print(json.dumps(value, indent=2 if args.json else None, sort_keys=True))
    return 0


def _ingest(args: argparse.Namespace) -> int:
    from rosclaw.collective.sources.motiondecode.audit import run_motiondecode_pilot

    report = run_motiondecode_pilot(
        dataset_root=args.dataset_root,
        revision=args.revision,
        model_path=args.model_path,
        asset_root=args.asset_root,
        output_dir=args.output_dir,
        source_checkout=Path(__file__).resolve().parents[3],
        requested_usage=args.usage,
        limit=args.limit,
        seed=args.seed,
    )
    value = {
        "schema_version": report.schema_version,
        "pipeline_passed": report.pipeline_passed,
        "decision": report.decision,
        "source_manifest_hash": report.source_manifest_hash,
        "body_hash": report.body_hash,
        "selection_counts": report.selection_counts,
        "selection_shortages": report.selection_shortages,
        "aggregates": report.aggregates,
        "blockers": list(report.blockers),
        "output_dir": str(args.output_dir.expanduser().resolve()),
    }
    print(json.dumps(value, indent=2 if args.json else None, sort_keys=True))
    return 0 if report.pipeline_passed else 2


def _build_prior(args: argparse.Namespace) -> int:
    try:
        from rosclaw.collective.sources.motiondecode.motion_prior import (
            build_motion_prior_pack,
        )
    except ModuleNotFoundError as exc:
        _raise_missing_rl_extra(exc)

    metadata = build_motion_prior_pack(
        pilot_report_path=args.pilot_report,
        dataset_root=args.dataset_root,
        model_path=args.model_path,
        output_path=args.output,
        sequence_length=args.sequence_length,
        maximum_windows=args.maximum_windows,
        seed=args.seed,
        allowed_strata=tuple(args.stratum) if args.stratum else None,
    )
    value = {
        "schema_version": metadata["schema_version"],
        "pack_hash": metadata["pack_hash"],
        "body_hash": metadata["body_hash"],
        "feature_count": len(metadata["feature_names"]),
        "training_windows": metadata["training_windows"],
        "validation_windows": metadata["validation_windows"],
        "allowed_strata": metadata["allowed_strata"],
        "action_semantics": metadata["action_semantics"],
        "raw_data_exported": metadata["raw_data_exported"],
        "output": str(args.output.expanduser().resolve()),
    }
    print(json.dumps(value, indent=2 if args.json else None, sort_keys=True))
    return 0


def _train_prior(args: argparse.Namespace) -> int:
    try:
        from rosclaw.collective.sources.motiondecode.motion_prior import (
            run_four_gpu_motion_prior,
        )
    except ModuleNotFoundError as exc:
        _raise_missing_rl_extra(exc)

    report = run_four_gpu_motion_prior(
        pack_path=args.pack,
        output_dir=args.output_dir,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        base_seed=args.base_seed,
    )
    selected = report.get("selected")
    value = {
        "schema_version": report["schema_version"],
        "decision": report["decision"],
        "four_physical_gpus_exercised": report["four_physical_gpus_exercised"],
        "quality_gate": report["quality_gate"],
        "selected": selected,
        "hardware_authorized": report["hardware_authorized"],
        "output_dir": str(args.output_dir.expanduser().resolve()),
    }
    print(json.dumps(value, indent=2 if args.json else None, sort_keys=True))
    return 0 if report["decision"] == "REPRESENTATION_CANDIDATE" else 2


def _raise_missing_rl_extra(exc: ModuleNotFoundError) -> NoReturn:
    if exc.name == "torch":
        raise RuntimeError(
            "Motion-prior build/train requires the optional RL dependencies; "
            "install 'rosclaw[rl]'"
        ) from exc
    raise exc


__all__ = ["dispatch_collective_argv"]
