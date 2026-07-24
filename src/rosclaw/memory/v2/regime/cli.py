"""``rosclaw regime`` CLI handlers (数据库优化v4 §15).

rosclaw regime status      — current regime of the latest session
rosclaw regime explain     — why that label was assigned
rosclaw regime replay      — regime timeline + transitions over a session
rosclaw regime transitions — confirmed transitions only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .builder import CurrentRegimeBuilder
from .detector import RegimeChangeDetector
from .explain import explain_regime
from .models import load_thresholds
from .session_samples import (
    extract_samples,
    latest_session_dir,
    load_session_events,
    resolve_session_dir,
)

_DEFAULT_CONFIG = "configs/regimes/rh56_rps_v1.yaml"
_DEFAULT_DATA_ROOT = "~/.rosclaw/practice/runs/rh56_rps"


def _emit(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    sys.stdout.flush()


def _load_builder(args: argparse.Namespace) -> CurrentRegimeBuilder:
    config = getattr(args, "config", None) or _DEFAULT_CONFIG
    if Path(config).is_file():
        return CurrentRegimeBuilder(load_thresholds(config))
    return CurrentRegimeBuilder()


def _samples_for(args: argparse.Namespace, session_dir: Path):
    events = load_session_events(session_dir)
    return extract_samples(events, hand=getattr(args, "hand", "right"))


def _identity(args: argparse.Namespace, session_dir: Path) -> dict[str, Any]:
    body = getattr(args, "body_id", None) or f"rh56_{getattr(args, 'hand', 'right')}_01"
    return {
        "robot_id": getattr(args, "robot_id", None) or "rh56_rps_robot",
        "body_id": body,
        "task_id": getattr(args, "task_id", None) or "rh56_rps",
        "session_started_at": None,
    }


def cmd_regime_status(args: argparse.Namespace) -> int:
    session_dir = latest_session_dir(getattr(args, "data_root", _DEFAULT_DATA_ROOT))
    samples = _samples_for(args, session_dir)
    identity = _identity(args, session_dir)
    if samples:
        identity["session_started_at"] = samples[0].timestamp
    regime = _load_builder(args).build(samples, rounds_completed=len(samples), **identity)
    _emit(
        {
            "session_dir": str(session_dir),
            "sample_count": len(samples),
            "regime": regime.to_dict(),
        }
    )
    return 0


def cmd_regime_explain(args: argparse.Namespace) -> int:
    session_dir = latest_session_dir(getattr(args, "data_root", _DEFAULT_DATA_ROOT))
    samples = _samples_for(args, session_dir)
    identity = _identity(args, session_dir)
    if samples:
        identity["session_started_at"] = samples[0].timestamp
    regime = _load_builder(args).build(samples, rounds_completed=len(samples), **identity)
    _emit(
        {
            "session_dir": str(session_dir),
            "sample_count": len(samples),
            "explanation": explain_regime(regime),
        }
    )
    return 0


def cmd_regime_replay(args: argparse.Namespace) -> int:
    root = getattr(args, "data_root", _DEFAULT_DATA_ROOT)
    practice_id = getattr(args, "practice_id", None)
    session_dir = (
        resolve_session_dir(root, practice_id) if practice_id else latest_session_dir(root)
    )
    samples = _samples_for(args, session_dir)
    identity = _identity(args, session_dir)
    builder = _load_builder(args)
    detector = RegimeChangeDetector()

    timeline: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    # Replay the regime as it would have been computed at each round —
    # the same windowed evidence the live system would have had then.
    for index in range(1, len(samples) + 1):
        regime = builder.build(
            samples[:index],
            rounds_completed=index,
            now=samples[index - 1].timestamp,
            **identity,
        )
        transition = detector.observe(regime)
        timeline.append(
            {
                "round": index,
                "timestamp": samples[index - 1].timestamp,
                "regime_label": regime.regime_label,
                "confidence": regime.confidence,
                "temperature_c": regime.temperature_c,
            }
        )
        if transition is not None:
            transitions.append(transition.to_dict())
    output = {
        "session_dir": str(session_dir),
        "rounds": len(samples),
        "transitions": transitions,
    }
    if not getattr(args, "transitions_only", False):
        output["timeline"] = timeline
    _emit(output)
    return 0


def cmd_regime_transitions(args: argparse.Namespace) -> int:
    args.transitions_only = True
    return cmd_regime_replay(args)


def register_regime_commands(subparsers: Any) -> None:
    """Register the ``rosclaw regime`` command group."""
    parser = subparsers.add_parser("regime", help="Operating regime inspection (v4, PR-MEM-6)")
    regime_sub = parser.add_subparsers(dest="regime_command")

    def _common(p: Any) -> None:
        p.add_argument("--data-root", default=_DEFAULT_DATA_ROOT)
        p.add_argument("--config", default=_DEFAULT_CONFIG)
        p.add_argument("--hand", choices=["left", "right"], default="right")
        p.add_argument("--robot-id", default=None)
        p.add_argument("--body-id", default=None)
        p.add_argument("--task-id", default=None)

    p = regime_sub.add_parser("status", help="Current regime of the latest session")
    _common(p)
    p.set_defaults(handler=cmd_regime_status)

    p = regime_sub.add_parser("explain", help="Why the current label was assigned")
    _common(p)
    p.set_defaults(handler=cmd_regime_explain)

    p = regime_sub.add_parser("replay", help="Regime timeline + transitions over a session")
    _common(p)
    p.add_argument("--practice-id", default=None)
    p.set_defaults(handler=cmd_regime_replay)

    p = regime_sub.add_parser("transitions", help="Confirmed regime transitions only")
    _common(p)
    p.add_argument("--practice-id", default=None)
    p.set_defaults(handler=cmd_regime_transitions)
