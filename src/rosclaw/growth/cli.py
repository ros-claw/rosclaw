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
from rosclaw.growth.adapters.g1_free_kick import (
    triage_g1_free_kick_trajectory,
    write_g1_free_kick_triage,
)
from rosclaw.growth.approach_strike_dataset import build_g1_approach_strike_dataset
from rosclaw.growth.ballistic_contact_actor_critic import (
    derive_g1_ballistic_contact_actor_critic,
)
from rosclaw.growth.ballistic_contact_evaluation import (
    evaluate_g1_ballistic_contact_holdout,
)
from rosclaw.growth.ballistic_contact_impulse_actor import (
    derive_g1_ballistic_contact_impulse_actor,
)
from rosclaw.growth.ballistic_contact_island_gate import (
    derive_g1_ballistic_contact_island_gate,
)
from rosclaw.growth.ballistic_contact_observer import (
    derive_g1_ballistic_contact_observer,
)
from rosclaw.growth.ballistic_contact_torque_actor_critic import (
    derive_g1_ballistic_contact_torque_actor_critic,
)
from rosclaw.growth.ballistic_skill_memory import derive_g1_ballistic_skill_memory
from rosclaw.growth.contextual_phase_calibration import (
    derive_g1_contextual_phase_calibration,
)
from rosclaw.growth.football_motion_prior import derive_g1_football_motion_prior
from rosclaw.growth.football_outcome_model import derive_g1_football_outcome_model
from rosclaw.growth.motiondecode_football_skill_prior import (
    derive_motiondecode_g1_football_skill_prior,
)
from rosclaw.growth.proprioceptive_expert_router import (
    derive_g1_proprioceptive_expert_router,
)
from rosclaw.growth.proprioceptive_readiness_gate import (
    derive_g1_proprioceptive_readiness_gate,
)
from rosclaw.growth.recovery_dataset import build_g1_recovery_dataset
from rosclaw.growth.sonic_authority_calibration import (
    derive_g1_sonic_authority_calibration,
)


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
    free_kick_triage = commands.add_parser(
        "free-kick-triage",
        help="verify and segment one continuous approach-to-strike rollout",
    )
    free_kick_triage.add_argument("--trajectory", type=Path, required=True)
    free_kick_triage.add_argument("--evidence-json", type=Path, required=True)
    free_kick_triage.add_argument("--output", type=Path, required=True)
    free_kick_triage.add_argument("--source-checkout", type=Path, default=Path.cwd())
    free_kick_triage.set_defaults(handler=_free_kick_triage)
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
    approach_strike = commands.add_parser(
        "approach-strike-dataset",
        help="extract event-bound approach-to-strike transitions",
    )
    approach_strike.add_argument("--trajectory", type=Path, action="append", required=True)
    approach_strike.add_argument("--evidence-json", type=Path, action="append", required=True)
    approach_strike.add_argument("--output-dir", type=Path, required=True)
    approach_strike.add_argument("--source-checkout", type=Path, default=Path.cwd())
    approach_strike.set_defaults(handler=_approach_strike_dataset)
    sonic_calibration = commands.add_parser(
        "sonic-authority-calibration",
        help="derive replay-bound per-joint SONIC gain scales",
    )
    sonic_calibration.add_argument("--trajectory", type=Path, action="append", required=True)
    sonic_calibration.add_argument("--evidence-json", type=Path, action="append", required=True)
    sonic_calibration.add_argument("--output", type=Path, required=True)
    sonic_calibration.add_argument("--source-checkout", type=Path, default=Path.cwd())
    sonic_calibration.add_argument("--demand-quantile", type=float, default=0.995)
    sonic_calibration.add_argument("--target-demand-ratio", type=float, default=0.90)
    sonic_calibration.add_argument("--base-calibration", type=Path)
    sonic_calibration.add_argument(
        "--freeze-approach-gain",
        action="store_true",
        help="retain the replayed approach gains while learning strike/recovery authority",
    )
    sonic_calibration.add_argument(
        "--calibration-step-fraction",
        type=float,
        default=1.0,
        help="trust-region interpolation from replayed to fitted gain schedules",
    )
    sonic_calibration.set_defaults(handler=_sonic_authority_calibration)
    sonic_evaluation = commands.add_parser(
        "evaluate-sonic-authority",
        help="run a sealed paired stability evaluation of an authority calibration",
    )
    sonic_evaluation.add_argument("--baseline", type=Path, action="append", required=True)
    sonic_evaluation.add_argument("--candidate", type=Path, action="append", required=True)
    sonic_evaluation.add_argument("--baseline-calibration", type=Path, required=True)
    sonic_evaluation.add_argument("--candidate-calibration", type=Path, required=True)
    sonic_evaluation.add_argument("--output", type=Path, required=True)
    sonic_evaluation.add_argument("--source-checkout", type=Path, default=Path.cwd())
    sonic_evaluation.set_defaults(handler=_evaluate_sonic_authority)
    contextual_calibration = commands.add_parser(
        "contextual-phase-calibration",
        help="fit a replay-bound proprioceptive strike-phase selector",
    )
    contextual_calibration.add_argument(
        "--evidence-json", type=Path, action="append", required=True
    )
    contextual_calibration.add_argument("--output", type=Path, required=True)
    contextual_calibration.add_argument("--source-checkout", type=Path, default=Path.cwd())
    contextual_calibration.add_argument("--normal-phase", type=int, required=True)
    contextual_calibration.add_argument("--high-yaw-phase", type=int, required=True)
    contextual_calibration.add_argument("--holdout-seed", type=int, action="append", required=True)
    contextual_calibration.add_argument(
        "--minimum-development-improvement-m", type=float, default=0.05
    )
    contextual_calibration.add_argument("--maximum-holdout-regression-m", type=float, default=0.01)
    contextual_calibration.set_defaults(handler=_contextual_phase_calibration)
    expert_router = commands.add_parser(
        "proprioceptive-expert-router",
        help="fit a replay-bound three-expert strike router",
    )
    expert_router.add_argument("--evidence-json", type=Path, action="append", required=True)
    expert_router.add_argument("--output", type=Path, required=True)
    expert_router.add_argument("--source-checkout", type=Path, default=Path.cwd())
    expert_router.add_argument("--expert-phase", type=int, action="append", required=True)
    expert_router.add_argument("--fallback-phase", type=int, required=True)
    expert_router.add_argument("--baseline-phase", type=int, required=True)
    expert_router.add_argument("--confidence-margin", type=float, default=0.05)
    expert_router.add_argument("--maximum-centroid-distance", type=float, default=2.5)
    expert_router.add_argument("--minimum-mean-improvement-m", type=float, default=0.05)
    expert_router.set_defaults(handler=_proprioceptive_expert_router)
    evaluate_router = commands.add_parser(
        "evaluate-proprioceptive-router",
        help="evaluate a frozen router on paired sealed SIM holdout episodes",
    )
    evaluate_router.add_argument("--baseline", type=Path, action="append", required=True)
    evaluate_router.add_argument("--routed", type=Path, action="append", required=True)
    evaluate_router.add_argument("--router", type=Path, required=True)
    evaluate_router.add_argument("--output", type=Path, required=True)
    evaluate_router.add_argument("--source-checkout", type=Path, default=Path.cwd())
    evaluate_router.add_argument("--minimum-mean-improvement-m", type=float, default=0.05)
    evaluate_router.set_defaults(handler=_evaluate_proprioceptive_router)
    readiness_gate = commands.add_parser(
        "proprioceptive-readiness-gate",
        help="fit counterfactual safety support with an explicit abstention outcome",
    )
    readiness_gate.add_argument("--evidence-json", type=Path, action="append", required=True)
    readiness_gate.add_argument("--router", type=Path, required=True)
    readiness_gate.add_argument("--output", type=Path, required=True)
    readiness_gate.add_argument("--source-checkout", type=Path, default=Path.cwd())
    readiness_gate.add_argument("--neighbor-count", type=int, default=2)
    readiness_gate.add_argument("--maximum-support-distance", type=float, default=2.0)
    readiness_gate.add_argument("--minimum-attempt-coverage", type=float, default=0.50)
    readiness_gate.set_defaults(handler=_proprioceptive_readiness_gate)
    evaluate_readiness = commands.add_parser(
        "evaluate-proprioceptive-readiness",
        help="evaluate readiness decisions on sealed three-expert counterfactuals",
    )
    evaluate_readiness.add_argument("--evidence-json", type=Path, action="append", required=True)
    evaluate_readiness.add_argument("--router", type=Path, required=True)
    evaluate_readiness.add_argument("--gate", type=Path, required=True)
    evaluate_readiness.add_argument("--output", type=Path, required=True)
    evaluate_readiness.add_argument("--source-checkout", type=Path, default=Path.cwd())
    evaluate_readiness.set_defaults(handler=_evaluate_proprioceptive_readiness)
    evaluate_recovery = commands.add_parser(
        "evaluate-readiness-recovery",
        help="aggregate frozen physical recovery after readiness abstention",
    )
    evaluate_recovery.add_argument("--evidence-json", type=Path, action="append", required=True)
    evaluate_recovery.add_argument("--router", type=Path, required=True)
    evaluate_recovery.add_argument("--gate", type=Path, required=True)
    evaluate_recovery.add_argument("--output", type=Path, required=True)
    evaluate_recovery.add_argument("--source-checkout", type=Path, default=Path.cwd())
    evaluate_recovery.set_defaults(handler=_evaluate_readiness_recovery)
    football_outcome = commands.add_parser(
        "football-outcome-model",
        help="learn mandatory shot selection from paired success/failure outcomes",
    )
    football_outcome.add_argument("--evidence-json", type=Path, action="append", required=True)
    football_outcome.add_argument("--expert-phase", type=int, action="append", required=True)
    football_outcome.add_argument("--output", type=Path, required=True)
    football_outcome.add_argument("--source-checkout", type=Path, default=Path.cwd())
    football_outcome.add_argument("--minimum-precision-improvement", type=int, default=3)
    football_outcome.set_defaults(handler=_football_outcome_model)
    football_motion_prior = commands.add_parser(
        "football-motion-prior",
        help="distil a train-only, SIM_ONLY G1 contact-motion prior",
    )
    football_motion_prior.add_argument("--omnicontact-root", type=Path, required=True)
    football_motion_prior.add_argument("--joint-order-contract", type=Path, required=True)
    football_motion_prior.add_argument("--asset-root", type=Path, required=True)
    football_motion_prior.add_argument("--output", type=Path, required=True)
    football_motion_prior.add_argument("--source-checkout", type=Path, default=Path.cwd())
    football_motion_prior.add_argument("--selected-event-count", type=int, default=24)
    football_motion_prior.set_defaults(handler=_football_motion_prior)
    motiondecode_skill_prior = commands.add_parser(
        "motiondecode-football-skill-prior",
        help="distil parent-conditioned whole-body style from Q1 G1 shooting clips",
    )
    motiondecode_skill_prior.add_argument("--registration", type=Path, required=True)
    motiondecode_skill_prior.add_argument("--repair-report", type=Path, required=True)
    motiondecode_skill_prior.add_argument("--dataset-root", type=Path, required=True)
    motiondecode_skill_prior.add_argument("--target-model", type=Path, required=True)
    motiondecode_skill_prior.add_argument("--asset-root", type=Path, required=True)
    motiondecode_skill_prior.add_argument("--parent-evidence", type=Path, required=True)
    motiondecode_skill_prior.add_argument("--parent-trajectory", type=Path, required=True)
    motiondecode_skill_prior.add_argument("--output", type=Path, required=True)
    motiondecode_skill_prior.add_argument("--source-checkout", type=Path, default=Path.cwd())
    motiondecode_skill_prior.add_argument("--selected-event-count", type=int, default=16)
    motiondecode_skill_prior.add_argument(
        "--style-profile",
        choices=("parent_nearest", "lofted_drive"),
        default="parent_nearest",
    )
    motiondecode_skill_prior.set_defaults(handler=_motiondecode_football_skill_prior)
    ballistic_actor_critic = commands.add_parser(
        "ballistic-contact-actor-critic",
        help="fit a replay-stabilized SIM_ONLY critic and propose one contact action",
    )
    ballistic_actor_critic.add_argument(
        "--evidence-json", type=Path, action="append", required=True
    )
    ballistic_actor_critic.add_argument("--output", type=Path, required=True)
    ballistic_actor_critic.add_argument("--source-checkout", type=Path, default=Path.cwd())
    ballistic_actor_critic.add_argument("--trust-region-radius-rad", type=float, default=0.06)
    ballistic_actor_critic.add_argument("--ridge-regularization", type=float, default=0.02)
    ballistic_actor_critic.set_defaults(handler=_ballistic_contact_actor_critic)
    ballistic_torque_actor_critic = commands.add_parser(
        "ballistic-contact-torque-actor-critic",
        help="fit an island-bound SIM_ONLY actor over direct contact torques",
    )
    ballistic_torque_actor_critic.add_argument(
        "--evidence-json", type=Path, action="append", required=True
    )
    ballistic_torque_actor_critic.add_argument("--output", type=Path, required=True)
    ballistic_torque_actor_critic.add_argument("--source-checkout", type=Path, default=Path.cwd())
    ballistic_torque_actor_critic.add_argument("--trust-region-radius-nm", type=float, default=0.50)
    ballistic_torque_actor_critic.add_argument("--ridge-regularization", type=float, default=0.02)
    ballistic_torque_actor_critic.set_defaults(handler=_ballistic_contact_torque_actor_critic)
    ballistic_impulse_actor = commands.add_parser(
        "ballistic-contact-impulse-actor",
        help="distil strict teacher probes into a proprioceptive direct-torque actor",
    )
    ballistic_impulse_actor.add_argument(
        "--evidence-json", type=Path, action="append", required=True
    )
    ballistic_impulse_actor.add_argument("--output", type=Path, required=True)
    ballistic_impulse_actor.add_argument("--source-checkout", type=Path, default=Path.cwd())
    ballistic_impulse_actor.set_defaults(handler=_ballistic_contact_impulse_actor)
    ballistic_island_gate = commands.add_parser(
        "ballistic-contact-island-gate",
        help="learn a replay-anchored contact-event atlas before actor training",
    )
    ballistic_island_gate.add_argument("--evidence-json", type=Path, action="append", required=True)
    ballistic_island_gate.add_argument("--output", type=Path, required=True)
    ballistic_island_gate.add_argument("--source-checkout", type=Path, default=Path.cwd())
    ballistic_island_gate.set_defaults(handler=_ballistic_contact_island_gate)
    ballistic_holdout = commands.add_parser(
        "evaluate-ballistic-contact",
        help="fail closed on a frozen contact action over unseen planner seeds",
    )
    ballistic_holdout.add_argument("--evidence-json", type=Path, action="append", required=True)
    ballistic_holdout.add_argument("--output", type=Path, required=True)
    ballistic_holdout.add_argument("--source-checkout", type=Path, default=Path.cwd())
    ballistic_holdout.add_argument("--maximum-worst-error-m", type=float, default=0.75)
    ballistic_holdout.add_argument("--minimum-crossing-height-m", type=float, default=0.65)
    ballistic_holdout.add_argument("--maximum-saturation-steps", type=int, default=30)
    ballistic_holdout.set_defaults(handler=_evaluate_ballistic_contact)
    ballistic_memory = commands.add_parser(
        "ballistic-skill-memory",
        help="bind full handoff states to supported SIM_ONLY contact skills",
    )
    ballistic_memory.add_argument(
        "--skill-evidence-json", type=Path, action="append", required=True
    )
    ballistic_memory.add_argument(
        "--rejected-evidence-json", type=Path, action="append", required=True
    )
    ballistic_memory.add_argument("--output", type=Path, required=True)
    ballistic_memory.add_argument("--source-checkout", type=Path, default=Path.cwd())
    ballistic_memory.add_argument("--maximum-support-distance", type=float, default=0.35)
    ballistic_memory.add_argument("--minimum-distance-margin", type=float, default=0.05)
    ballistic_memory.set_defaults(handler=_ballistic_skill_memory)
    ballistic_observer = commands.add_parser(
        "ballistic-contact-observer",
        help="learn replay-bound contact-to-launch dynamics without motor authority",
    )
    ballistic_observer.add_argument("--evidence-json", type=Path, action="append", required=True)
    ballistic_observer.add_argument("--output", type=Path, required=True)
    ballistic_observer.add_argument("--source-checkout", type=Path, default=Path.cwd())
    ballistic_observer.add_argument("--ridge-regularization", type=float, default=0.20)
    ballistic_observer.set_defaults(handler=_ballistic_contact_observer)
    evaluate_football_outcome = commands.add_parser(
        "evaluate-football-outcome-model",
        help="evaluate mandatory shot selection on sealed counterfactual episodes",
    )
    evaluate_football_outcome.add_argument(
        "--evidence-json", type=Path, action="append", required=True
    )
    evaluate_football_outcome.add_argument("--model", type=Path, required=True)
    evaluate_football_outcome.add_argument("--output", type=Path, required=True)
    evaluate_football_outcome.add_argument("--source-checkout", type=Path, default=Path.cwd())
    evaluate_football_outcome.add_argument("--minimum-precision-improvement", type=int, default=1)
    evaluate_football_outcome.add_argument("--minimum-mean-improvement-m", type=float, default=0.02)
    evaluate_football_outcome.set_defaults(handler=_evaluate_football_outcome_model)
    evaluate_approach = commands.add_parser(
        "evaluate-approach-strike-residual",
        help="compare a bounded IQL transition residual against its baseline",
    )
    evaluate_approach.add_argument("--baseline", type=Path, required=True)
    evaluate_approach.add_argument("--candidate", type=Path, required=True)
    evaluate_approach.add_argument("--output", type=Path, required=True)
    evaluate_approach.add_argument("--source-checkout", type=Path, default=Path.cwd())
    evaluate_approach.add_argument("--minimum-effect-fraction", type=float, default=0.05)
    evaluate_approach.set_defaults(handler=_evaluate_approach_strike_residual)
    train_iql = commands.add_parser(
        "train-iql",
        help="train an unevaluated manifest-driven physical transition candidate",
    )
    train_iql.add_argument("--dataset-manifest", type=Path, required=True)
    train_iql.add_argument("--output-dir", type=Path, required=True)
    train_iql.add_argument("--source-checkout", type=Path, default=Path.cwd())
    train_iql.add_argument("--steps", type=int, default=2000)
    train_iql.add_argument("--batch-size", type=int, default=256)
    train_iql.add_argument("--hidden-size", type=int, default=256)
    train_iql.add_argument("--seed", type=int, default=20260805)
    train_iql.add_argument("--device", default="cpu")
    train_iql.add_argument(
        "--action-source",
        choices=("executed_action", "teacher_residual_action"),
        default="executed_action",
    )
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
    evaluate_residual = commands.add_parser(
        "evaluate-residual-iql",
        help="evaluate the frozen support-bound IQL recovery residual",
    )
    evaluate_residual.add_argument("--candidate", type=Path, required=True)
    evaluate_residual.add_argument("--asset-root", type=Path, required=True)
    evaluate_residual.add_argument("--output-dir", type=Path, required=True)
    evaluate_residual.add_argument("--source-checkout", type=Path, default=Path.cwd())
    evaluate_residual.set_defaults(handler=_evaluate_residual_iql)
    agentd_bridge = commands.add_parser(
        "stage-agentd-evaluation",
        help="stage measured SIM evidence in agentd without promotion",
    )
    agentd_bridge.add_argument("--evaluation", type=Path, required=True)
    agentd_bridge.add_argument("--agentd-db", type=Path, required=True)
    agentd_bridge.add_argument("--receipt", type=Path, required=True)
    agentd_bridge.add_argument("--source-checkout", type=Path, default=Path.cwd())
    agentd_bridge.set_defaults(handler=_stage_agentd_evaluation)
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


def _free_kick_triage(args: argparse.Namespace) -> int:
    report = triage_g1_free_kick_trajectory(
        trajectory_path=args.trajectory,
        evidence_path=args.evidence_json,
    )
    output = write_g1_free_kick_triage(
        report=report,
        output_path=args.output,
        source_checkout=args.source_checkout,
    )
    _print(
        {
            "schema_version": "rosclaw.growth.free_kick_triage_receipt.v1",
            "ok": True,
            "report_hash": report.report_hash,
            "output": str(output),
            "failure_types": [item.primary_type for item in report.failure_signatures],
            "learner_route": report.learner_route.to_dict(),
            "promotion_ready": report.promotion_ready,
            "activation_authorized": False,
            "hardware_authorized": False,
        }
    )
    return 0


def _approach_strike_dataset(args: argparse.Namespace) -> int:
    if len(args.trajectory) != len(args.evidence_json):
        raise ValueError("approach-strike trajectory/evidence counts must match")
    receipt = build_g1_approach_strike_dataset(
        trajectory_paths=tuple(args.trajectory),
        evidence_paths=tuple(args.evidence_json),
        output_dir=args.output_dir,
        source_checkout=args.source_checkout,
    )
    _print(receipt.to_dict())
    return 0


def _sonic_authority_calibration(args: argparse.Namespace) -> int:
    if len(args.trajectory) != len(args.evidence_json):
        raise ValueError("SONIC calibration trajectory/evidence counts must match")
    calibration = derive_g1_sonic_authority_calibration(
        trajectory_paths=tuple(args.trajectory),
        evidence_paths=tuple(args.evidence_json),
        output_path=args.output,
        source_checkout=args.source_checkout,
        demand_quantile=args.demand_quantile,
        target_demand_ratio=args.target_demand_ratio,
        base_calibration_path=args.base_calibration,
        freeze_approach_gain=args.freeze_approach_gain,
        calibration_step_fraction=args.calibration_step_fraction,
    )
    _print(calibration.to_dict())
    return 0


def _evaluate_sonic_authority(args: argparse.Namespace) -> int:
    from rosclaw.growth.sonic_authority_evaluation import (
        evaluate_g1_sonic_authority_holdout,
    )

    report = evaluate_g1_sonic_authority_holdout(
        baseline_paths=tuple(args.baseline),
        candidate_paths=tuple(args.candidate),
        baseline_calibration_path=args.baseline_calibration,
        candidate_calibration_path=args.candidate_calibration,
        output_path=args.output,
        source_checkout=args.source_checkout,
    )
    _print(report.to_dict())
    return 0 if report.accepted else 3


def _contextual_phase_calibration(args: argparse.Namespace) -> int:
    calibration = derive_g1_contextual_phase_calibration(
        evidence_paths=tuple(args.evidence_json),
        output_path=args.output,
        source_checkout=args.source_checkout,
        normal_phase_start_frame=args.normal_phase,
        high_yaw_phase_start_frame=args.high_yaw_phase,
        holdout_seeds=tuple(args.holdout_seed),
        minimum_development_improvement_m=(args.minimum_development_improvement_m),
        maximum_holdout_regression_m=args.maximum_holdout_regression_m,
    )
    _print(calibration.to_dict())
    return 0 if calibration.accepted else 3


def _proprioceptive_expert_router(args: argparse.Namespace) -> int:
    router = derive_g1_proprioceptive_expert_router(
        evidence_paths=tuple(args.evidence_json),
        output_path=args.output,
        source_checkout=args.source_checkout,
        expert_phases=tuple(args.expert_phase),
        fallback_phase=args.fallback_phase,
        baseline_phase=args.baseline_phase,
        confidence_margin=args.confidence_margin,
        maximum_centroid_distance=args.maximum_centroid_distance,
        minimum_mean_improvement_m=args.minimum_mean_improvement_m,
    )
    _print(router.to_dict())
    return 0 if router.accepted else 3


def _evaluate_proprioceptive_router(args: argparse.Namespace) -> int:
    from rosclaw.growth.proprioceptive_router_evaluation import (
        evaluate_g1_proprioceptive_router_holdout,
    )

    report = evaluate_g1_proprioceptive_router_holdout(
        baseline_paths=tuple(args.baseline),
        routed_paths=tuple(args.routed),
        router_path=args.router,
        output_path=args.output,
        source_checkout=args.source_checkout,
        minimum_mean_improvement_m=args.minimum_mean_improvement_m,
    )
    _print(report.to_dict())
    return 0 if report.accepted else 3


def _proprioceptive_readiness_gate(args: argparse.Namespace) -> int:
    gate = derive_g1_proprioceptive_readiness_gate(
        evidence_paths=tuple(args.evidence_json),
        router_path=args.router,
        output_path=args.output,
        source_checkout=args.source_checkout,
        neighbor_count=args.neighbor_count,
        maximum_support_distance=args.maximum_support_distance,
        minimum_attempt_coverage=args.minimum_attempt_coverage,
    )
    _print(gate.to_dict())
    return 0 if gate.accepted else 3


def _evaluate_proprioceptive_readiness(args: argparse.Namespace) -> int:
    from rosclaw.growth.proprioceptive_readiness_evaluation import (
        evaluate_g1_proprioceptive_readiness_holdout,
    )

    report = evaluate_g1_proprioceptive_readiness_holdout(
        evidence_paths=tuple(args.evidence_json),
        router_path=args.router,
        gate_path=args.gate,
        output_path=args.output,
        source_checkout=args.source_checkout,
    )
    _print(report.to_dict())
    return 0 if report.accepted else 3


def _evaluate_readiness_recovery(args: argparse.Namespace) -> int:
    from rosclaw.growth.readiness_recovery_evaluation import (
        evaluate_g1_readiness_recovery,
    )

    report = evaluate_g1_readiness_recovery(
        evidence_paths=tuple(args.evidence_json),
        router_path=args.router,
        gate_path=args.gate,
        output_path=args.output,
        source_checkout=args.source_checkout,
    )
    _print(report.to_dict())
    return 0 if report.accepted else 3


def _football_outcome_model(args: argparse.Namespace) -> int:
    model = derive_g1_football_outcome_model(
        evidence_paths=tuple(args.evidence_json),
        output_path=args.output,
        source_checkout=args.source_checkout,
        expert_phases=tuple(args.expert_phase),
        minimum_precision_improvement=args.minimum_precision_improvement,
    )
    _print(model.to_dict())
    return 0 if model.accepted else 3


def _football_motion_prior(args: argparse.Namespace) -> int:
    prior = derive_g1_football_motion_prior(
        omnicontact_root=args.omnicontact_root,
        joint_order_contract=args.joint_order_contract,
        asset_root=args.asset_root,
        output_path=args.output,
        source_checkout=args.source_checkout,
        selected_event_count=args.selected_event_count,
    )
    _print(prior.to_dict())
    return 0


def _motiondecode_football_skill_prior(args: argparse.Namespace) -> int:
    prior = derive_motiondecode_g1_football_skill_prior(
        registration_path=args.registration,
        repair_report_path=args.repair_report,
        dataset_root=args.dataset_root,
        target_model_path=args.target_model,
        asset_root=args.asset_root,
        parent_evidence_path=args.parent_evidence,
        parent_trajectory_path=args.parent_trajectory,
        output_path=args.output,
        source_checkout=args.source_checkout,
        selected_event_count=args.selected_event_count,
        style_profile=args.style_profile,
    )
    _print(prior.to_dict())
    return 0


def _ballistic_contact_actor_critic(args: argparse.Namespace) -> int:
    candidate = derive_g1_ballistic_contact_actor_critic(
        evidence_paths=tuple(args.evidence_json),
        output_path=args.output,
        source_checkout=args.source_checkout,
        trust_region_radius_rad=args.trust_region_radius_rad,
        ridge_regularization=args.ridge_regularization,
    )
    _print(candidate.to_dict())
    return 0 if candidate.sim_replay_recommended else 3


def _ballistic_contact_torque_actor_critic(args: argparse.Namespace) -> int:
    candidate = derive_g1_ballistic_contact_torque_actor_critic(
        evidence_paths=tuple(args.evidence_json),
        output_path=args.output,
        source_checkout=args.source_checkout,
        trust_region_radius_nm=args.trust_region_radius_nm,
        ridge_regularization=args.ridge_regularization,
    )
    _print(candidate.to_dict())
    return 0 if candidate.sim_replay_recommended else 3


def _ballistic_contact_impulse_actor(args: argparse.Namespace) -> int:
    actor = derive_g1_ballistic_contact_impulse_actor(
        evidence_paths=tuple(args.evidence_json),
        output_path=args.output,
        source_checkout=args.source_checkout,
    )
    _print(actor.to_dict())
    return 0


def _ballistic_contact_island_gate(args: argparse.Namespace) -> int:
    gate = derive_g1_ballistic_contact_island_gate(
        evidence_paths=tuple(args.evidence_json),
        output_path=args.output,
        source_checkout=args.source_checkout,
    )
    _print(gate.to_dict())
    return 0 if gate.training_ready else 3


def _evaluate_ballistic_contact(args: argparse.Namespace) -> int:
    report = evaluate_g1_ballistic_contact_holdout(
        evidence_paths=tuple(args.evidence_json),
        output_path=args.output,
        source_checkout=args.source_checkout,
        maximum_worst_error_m=args.maximum_worst_error_m,
        minimum_crossing_height_m=args.minimum_crossing_height_m,
        maximum_saturation_steps=args.maximum_saturation_steps,
    )
    _print(report.to_dict())
    return 0 if report.accepted else 3


def _ballistic_skill_memory(args: argparse.Namespace) -> int:
    memory = derive_g1_ballistic_skill_memory(
        skill_evidence_paths=tuple(args.skill_evidence_json),
        rejected_evidence_paths=tuple(args.rejected_evidence_json),
        output_path=args.output,
        source_checkout=args.source_checkout,
        maximum_support_distance=args.maximum_support_distance,
        minimum_distance_margin=args.minimum_distance_margin,
    )
    _print(memory.to_dict())
    return 0 if memory.accepted else 3


def _ballistic_contact_observer(args: argparse.Namespace) -> int:
    observer = derive_g1_ballistic_contact_observer(
        evidence_paths=tuple(args.evidence_json),
        output_path=args.output,
        source_checkout=args.source_checkout,
        ridge_regularization=args.ridge_regularization,
    )
    _print(observer.to_dict())
    return 0 if observer.training_ready else 3


def _evaluate_football_outcome_model(args: argparse.Namespace) -> int:
    from rosclaw.growth.football_outcome_evaluation import (
        evaluate_g1_football_outcome_model,
    )

    report = evaluate_g1_football_outcome_model(
        evidence_paths=tuple(args.evidence_json),
        model_path=args.model,
        output_path=args.output,
        source_checkout=args.source_checkout,
        minimum_precision_improvement=args.minimum_precision_improvement,
        minimum_mean_improvement_m=args.minimum_mean_improvement_m,
    )
    _print(report.to_dict())
    return 0 if report.accepted else 3


def _evaluate_approach_strike_residual(args: argparse.Namespace) -> int:
    from rosclaw.growth.approach_strike_evaluation import (
        evaluate_g1_approach_strike_residual,
    )

    evaluation = evaluate_g1_approach_strike_residual(
        baseline_evidence_path=args.baseline,
        candidate_evidence_path=args.candidate,
        output_path=args.output,
        source_checkout=args.source_checkout,
        minimum_effect_fraction=args.minimum_effect_fraction,
    )
    _print(evaluation.to_dict())
    return 0 if evaluation.passed else 3


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
            action_source=args.action_source,
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


def _evaluate_residual_iql(args: argparse.Namespace) -> int:
    from rosclaw.simforge.g1_recovery_residual_evaluation import (
        run_g1_residual_recovery_evaluation,
    )

    result = run_g1_residual_recovery_evaluation(
        actor_candidate_path=args.candidate,
        asset_root=args.asset_root,
        output_dir=args.output_dir,
        source_checkout=args.source_checkout,
    )
    _print(result.to_dict())
    return 0 if result.passed else 3


def _stage_agentd_evaluation(args: argparse.Namespace) -> int:
    from rosclaw.agentd.mission import MissionStore
    from rosclaw.growth.agentd_bridge import stage_growth_evaluation_candidate

    checkout = args.source_checkout.expanduser().resolve()
    database = args.agentd_db.expanduser().resolve()
    receipt_path = args.receipt.expanduser().resolve()
    for path, label in ((database, "agentd database"), (receipt_path, "bridge receipt")):
        if path == checkout or checkout in path.parents:
            raise ValueError(f"{label} must be outside the source checkout")
    if receipt_path.exists():
        raise ValueError("Growth agentd bridge receipt already exists")
    database.parent.mkdir(parents=True, exist_ok=True)
    store = MissionStore(database)
    try:
        receipt = stage_growth_evaluation_candidate(
            evaluation_path=args.evaluation,
            connection=store.connection,
            source_checkout=checkout,
        )
    finally:
        store.close()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _print(receipt.to_dict())
    return 0


def _print(value: dict[str, Any], *, stream: TextIO | None = None) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), file=stream or sys.stdout)


__all__ = ["dispatch_growth_argv"]
