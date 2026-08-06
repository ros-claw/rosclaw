"""Command-line diagnostics for the asynchronous Growth control plane."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO

from rosclaw.growth.adapters import (
    triage_g1_coupled_trajectory,
    verified_coupled_evidence_context,
)
from rosclaw.growth.recovery_dataset import build_g1_recovery_dataset


def dispatch_growth_argv(argv: list[str]) -> int | None:
    if not argv or argv[0] != "growth":
        return None
    args = _parser().parse_args(argv[1:])
    try:
        return int(args.handler(args))
    except (OSError, PermissionError, RuntimeError, ValueError) as exc:
        _print(
            {
                "schema_version": "rosclaw.growth.cli_error.v1",
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "activation_authorized": False,
                "hardware_authorized": False,
            },
            stream=sys.stderr,
        )
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rosclaw growth",
        description="Triage physical evidence without entering the motor-control path.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    triage = commands.add_parser("triage", help="run PARC event and recovery triage")
    triage.add_argument("--skill", choices=("g1_football",), required=True)
    triage.add_argument("--trajectory", type=Path, required=True)
    triage.add_argument("--role", choices=("passer", "shooter"), default="shooter")
    triage.add_argument("--evidence-json", type=Path)
    triage.add_argument("--output", type=Path, required=True)
    triage.add_argument("--source-checkout", type=Path, default=Path.cwd())
    triage.add_argument("--force", action="store_true")
    triage.set_defaults(handler=_triage)
    recovery = commands.add_parser(
        "recovery-dataset",
        help="extract runtime-bound post-impact transition tuples",
    )
    recovery.add_argument("--skill", choices=("g1_football",), required=True)
    recovery.add_argument("--trajectory", type=Path, action="append", required=True)
    recovery.add_argument("--evidence-json", type=Path, required=True)
    recovery.add_argument("--output-dir", type=Path, required=True)
    recovery.add_argument("--source-checkout", type=Path, default=Path.cwd())
    recovery.set_defaults(handler=_recovery_dataset)
    train_iql = commands.add_parser(
        "train-iql",
        help="train an unevaluated conservative recovery candidate",
    )
    train_iql.add_argument("--dataset-manifest", type=Path, required=True)
    train_iql.add_argument("--output-dir", type=Path, required=True)
    train_iql.add_argument("--source-checkout", type=Path, default=Path.cwd())
    train_iql.add_argument("--steps", type=int, default=2000)
    train_iql.add_argument("--batch-size", type=int, default=256)
    train_iql.add_argument("--hidden-size", type=int, default=256)
    train_iql.add_argument("--seed", type=int, default=20260805)
    train_iql.add_argument("--device", default="cpu")
    train_iql.set_defaults(handler=_train_iql)
    evaluate_iql = commands.add_parser(
        "evaluate-iql",
        help="run a fail-fast reserved closed-loop SIM evaluation",
    )
    evaluate_iql.add_argument("--candidate", type=Path, required=True)
    evaluate_iql.add_argument("--asset-root", type=Path, required=True)
    evaluate_iql.add_argument("--output-dir", type=Path, required=True)
    evaluate_iql.add_argument("--source-checkout", type=Path, default=Path.cwd())
    evaluate_iql.set_defaults(handler=_evaluate_iql)
    evaluate_structured = commands.add_parser(
        "evaluate-structured-recovery",
        help="evaluate the frozen contact-gated recovery candidate",
    )
    evaluate_structured.add_argument("--asset-root", type=Path, required=True)
    evaluate_structured.add_argument("--output-dir", type=Path, required=True)
    evaluate_structured.add_argument("--source-checkout", type=Path, default=Path.cwd())
    evaluate_structured.set_defaults(handler=_evaluate_structured_recovery)
    return parser


def _triage(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    checkout = args.source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("Growth evidence output must be outside the source checkout")
    if output.exists() and not args.force:
        raise ValueError("Growth triage output exists; pass --force to replace")
    output.parent.mkdir(parents=True, exist_ok=True)
    context = (
        verified_coupled_evidence_context(args.evidence_json, args.trajectory)
        if args.evidence_json
        else None
    )
    report = triage_g1_coupled_trajectory(
        args.trajectory,
        role=args.role,
        evidence_context=context,
    )
    output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _print(
        {
            "schema_version": "rosclaw.growth.triage_receipt.v1",
            "ok": True,
            "report_hash": report.report_hash,
            "output": str(output),
            "failure_types": [item.primary_type for item in report.failure_signatures],
            "learner_route": report.learner_route.to_dict(),
            "promotion_ready": report.absolute_recovery.passed,
            "activation_authorized": False,
            "hardware_authorized": False,
        }
    )
    return 0


def _recovery_dataset(args: argparse.Namespace) -> int:
    receipt = build_g1_recovery_dataset(
        trajectory_paths=tuple(args.trajectory),
        evidence_path=args.evidence_json,
        output_dir=args.output_dir,
        source_checkout=args.source_checkout,
    )
    _print(receipt.to_dict())
    return 0


def _train_iql(args: argparse.Namespace) -> int:
    from rosclaw.growth.learners import IQLTrainingConfig, train_recovery_iql

    receipt = train_recovery_iql(
        dataset_manifest_path=args.dataset_manifest,
        output_dir=args.output_dir,
        source_checkout=args.source_checkout,
        config=IQLTrainingConfig(
            steps=args.steps,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            seed=args.seed,
            device=args.device,
        ),
    )
    _print(receipt.to_dict())
    return 0


def _evaluate_iql(args: argparse.Namespace) -> int:
    from rosclaw.simforge.g1_recovery_iql_evaluation import (
        evaluate_g1_recovery_iql_candidate,
    )

    result = evaluate_g1_recovery_iql_candidate(
        candidate_path=args.candidate,
        asset_root=args.asset_root,
        output_dir=args.output_dir,
        source_checkout=args.source_checkout,
    )
    _print(result.to_dict())
    return 0 if result.passed else 3


def _evaluate_structured_recovery(args: argparse.Namespace) -> int:
    from rosclaw.simforge.g1_structured_recovery_evaluation import (
        run_g1_structured_recovery_evaluation,
    )

    result = run_g1_structured_recovery_evaluation(
        asset_root=args.asset_root,
        output_dir=args.output_dir,
        source_checkout=args.source_checkout,
    )
    _print(result.to_dict())
    return 0 if result.passed else 3


def _print(value: dict[str, Any], *, stream: TextIO | None = None) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), file=stream or sys.stdout)


__all__ = ["dispatch_growth_argv"]
