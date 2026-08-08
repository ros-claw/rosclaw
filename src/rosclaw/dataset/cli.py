"""CLI surface for governed dataset inspection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO

from rosclaw.dataset.contracts import FileHashMode
from rosclaw.dataset.doctor import inspect_dataset_root, write_dataset_doctor_artifacts


def dispatch_dataset_argv(argv: list[str]) -> int | None:
    if not argv or argv[0] != "dataset":
        return None
    args = _parser().parse_args(argv[1:])
    try:
        return int(args.handler(args))
    except (OSError, PermissionError, RuntimeError, ValueError) as exc:
        _print(
            {
                "schema_version": "rosclaw.dataset.cli_error.v1",
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
        prog="rosclaw dataset",
        description="Inspect external datasets without making them training or promotion truth.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="build a transfer-aware dataset inventory")
    doctor.add_argument("--root", type=Path, required=True)
    doctor.add_argument("--output-dir", type=Path, required=True)
    doctor.add_argument(
        "--hash",
        dest="hash_mode",
        choices=[value.value for value in FileHashMode],
        default=FileHashMode.METADATA.value,
    )
    doctor.add_argument(
        "--transfer-active",
        action="store_true",
        help="mark this point-in-time scan as an in-progress transfer snapshot",
    )
    doctor.add_argument("--football-match-limit", type=int, default=100)
    doctor.add_argument("--source-checkout", type=Path, default=Path.cwd())
    doctor.add_argument("--force", action="store_true")
    doctor.set_defaults(handler=_doctor)
    return parser


def _doctor(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    checkout = args.source_checkout.expanduser().resolve()
    if output_dir == checkout or checkout in output_dir.parents:
        raise ValueError("dataset evidence output must be outside the source checkout")
    expected = (
        "dataset_inventory.json",
        "dataset_quality_report.html",
        "license_manifest.json",
        "football_asset_matrix.csv",
    )
    existing = [name for name in expected if (output_dir / name).exists()]
    if existing and not args.force:
        raise ValueError(
            f"dataset doctor outputs already exist: {existing}; pass --force to replace"
        )
    report = inspect_dataset_root(
        args.root,
        hash_mode=FileHashMode(args.hash_mode),
        transfer_active=bool(args.transfer_active),
        football_match_limit=args.football_match_limit,
    )
    artifacts = write_dataset_doctor_artifacts(report, output_dir)
    _print(
        {
            "schema_version": "rosclaw.dataset.doctor_receipt.v1",
            "ok": True,
            "report_hash": report.report_hash,
            "snapshot_complete": report.snapshot_complete,
            "transfer_active": report.transfer_active,
            "dataset_states": {value.dataset_id: value.state.value for value in report.inventories},
            "artifacts": artifacts,
            "training_eligible": False,
            "promotion_truth_allowed": False,
            "activation_authorized": False,
            "hardware_authorized": False,
        }
    )
    return 0


def _print(value: dict[str, Any], *, stream: TextIO | None = None) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), file=stream or sys.stdout)


__all__ = ["dispatch_dataset_argv"]
