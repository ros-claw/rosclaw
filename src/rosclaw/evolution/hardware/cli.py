"""``rosclaw acceptance evo-rps`` command handlers (PR-EVO-HW-1)."""

from __future__ import annotations

import argparse
import json
from typing import Any

from .orchestrator import (
    DEFAULT_CONFIG,
    OrchestratorError,
    orchestrator_for,
    phase_not_implemented,
)


def _emit(payload: Any) -> int:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0 if payload.get("ok", True) else 1


def cmd_acceptance_evo_rps_prepare(args: argparse.Namespace) -> int:
    orchestrator = orchestrator_for(getattr(args, "config", None) or DEFAULT_CONFIG)
    return _emit(orchestrator.prepare(dev_allow_mock=getattr(args, "dev_allow_mock", False)))


def cmd_acceptance_evo_rps_baseline(args: argparse.Namespace) -> int:
    orchestrator = orchestrator_for(getattr(args, "config", None) or DEFAULT_CONFIG)
    try:
        result = orchestrator.baseline(
            sessions=int(getattr(args, "sessions", 3)),
            rounds=int(getattr(args, "rounds", 40)),
            seed_start=int(getattr(args, "seed_start", 0)),
        )
    except OrchestratorError as exc:
        return _emit({"ok": False, "blocked": str(exc)})
    return _emit(result)


def cmd_acceptance_evo_rps_report(args: argparse.Namespace) -> int:
    orchestrator = orchestrator_for(getattr(args, "config", None) or DEFAULT_CONFIG)
    return _emit(orchestrator.report())


def cmd_acceptance_evo_rps_distill(args: argparse.Namespace) -> int:
    orchestrator = orchestrator_for(getattr(args, "config", None) or DEFAULT_CONFIG)
    return _emit(orchestrator.distill())


def cmd_acceptance_evo_rps_propose(args: argparse.Namespace) -> int:
    orchestrator = orchestrator_for(getattr(args, "config", None) or DEFAULT_CONFIG)
    try:
        result = orchestrator.propose(max_candidates=int(getattr(args, "max_candidates", 8)))
    except OrchestratorError as exc:
        return _emit({"ok": False, "blocked": str(exc)})
    return _emit(result)


def cmd_acceptance_evo_rps_validate(args: argparse.Namespace) -> int:
    orchestrator = orchestrator_for(getattr(args, "config", None) or DEFAULT_CONFIG)
    try:
        result = orchestrator.validate(shadow_rounds=int(getattr(args, "shadow_rounds", 12)))
    except OrchestratorError as exc:
        return _emit({"ok": False, "blocked": str(exc)})
    return _emit(result)


def cmd_acceptance_evo_rps_canary(args: argparse.Namespace) -> int:
    orchestrator = orchestrator_for(getattr(args, "config", None) or DEFAULT_CONFIG)
    try:
        result = orchestrator.canary(
            blocks=int(getattr(args, "blocks", 3)),
            rounds=int(getattr(args, "rounds", 40)),
            candidate_id=getattr(args, "candidate_id", None) or None,
        )
    except OrchestratorError as exc:
        return _emit({"ok": False, "blocked": str(exc)})
    return _emit(result)


def cmd_acceptance_evo_rps_promote(args: argparse.Namespace) -> int:
    orchestrator = orchestrator_for(getattr(args, "config", None) or DEFAULT_CONFIG)
    try:
        result = orchestrator.promote()
    except OrchestratorError as exc:
        return _emit({"ok": False, "blocked": str(exc)})
    return _emit(result)


def cmd_acceptance_evo_rps_recurrence(args: argparse.Namespace) -> int:
    orchestrator = orchestrator_for(getattr(args, "config", None) or DEFAULT_CONFIG)
    try:
        result = orchestrator.recurrence(rounds=int(getattr(args, "rounds", 40)))
    except OrchestratorError as exc:
        return _emit({"ok": False, "blocked": str(exc)})
    return _emit(result)


def cmd_acceptance_evo_rps_future(phase: str):
    def handler(args: argparse.Namespace) -> int:
        return _emit(phase_not_implemented(phase))

    return handler
