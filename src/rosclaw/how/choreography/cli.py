"""``rosclaw how replay-patch`` — counterfactual patch replay (v4 §8.5/§15).

输出::

    原始 Timeline
    补丁后 Timeline
    违反的 Phase
    预计 Reveal Offset
    是否允许真机实验

All timing patches must pass replay before any real-machine experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rosclaw.memory.v2.regime.session_samples import (
    _payload,
    load_session_events,
    resolve_session_dir,
)

from .contract import load_contract
from .timing import RoundTiming, build_timing_model
from .validator import ChoreographyValidator

_DEFAULT_CONTRACT = "configs/choreography/rh56_rps_v1.yaml"
_DEFAULT_DATA_ROOT = "~/.rosclaw/practice/runs/rh56_rps"


def _emit(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    sys.stdout.flush()


def _rounds_from_session(session_dir: Path) -> list[RoundTiming]:
    rounds: list[RoundTiming] = []
    for event in load_session_events(session_dir):
        if event.get("event_type") != "rps.stress.round.resolved":
            continue
        payload = _payload(event)
        info = payload.get("round") or {}
        started = info.get("started_at")
        if not isinstance(started, (int, float)):
            continue
        rounds.append(
            RoundTiming(
                started_at=float(started),
                ended_at=info.get("ended_at")
                if isinstance(info.get("ended_at"), (int, float))
                else None,
                reveal_at=(
                    float(started) + float(info["latency_ms"]) / 1000.0
                    if isinstance(info.get("latency_ms"), (int, float))
                    else None
                ),
            )
        )
    rounds.sort(key=lambda r: r.started_at)
    return rounds


def cmd_how_replay_patch(args: argparse.Namespace) -> int:
    contract_path = getattr(args, "contract", None) or _DEFAULT_CONTRACT
    if Path(contract_path).is_file():
        contract = load_contract(contract_path)
    else:
        candidate = Path("configs/choreography") / f"{contract_path}.yaml"
        if not candidate.is_file():
            _emit({"ok": False, "error": f"contract not found: {contract_path}"})
            return 2
        contract = load_contract(str(candidate))

    patch_file = Path(args.patch)
    if not patch_file.is_file():
        _emit({"ok": False, "error": f"patch file not found: {patch_file}"})
        return 2
    with open(patch_file, encoding="utf-8") as handle:
        patch = json.load(handle)
    if not isinstance(patch, dict):
        _emit({"ok": False, "error": "patch JSON must be an object of parameter → value"})
        return 2

    session_dir = resolve_session_dir(
        getattr(args, "data_root", None) or _DEFAULT_DATA_ROOT, args.practice_id
    )
    rounds = _rounds_from_session(session_dir)

    current_parameters: dict[str, Any] = {}
    if getattr(args, "current_params", None):
        with open(args.current_params, encoding="utf-8") as handle:
            current_parameters = json.load(handle)

    model = build_timing_model(contract, rounds, current_parameters=current_parameters)
    validator = ChoreographyValidator(contract)
    validation = validator.validate(patch, model)

    output = {
        "contract_id": contract.contract_id,
        "session_dir": str(session_dir),
        "rounds_observed": len(rounds),
        "patch": patch,
        "original_timeline_ms": validation.original_phase_durations,
        "patched_timeline_ms": validation.patched_phase_durations,
        "expected_reveal_offset_ms": validation.expected_reveal_offset_ms,
        "reveal_window_ms": [
            contract.reveal_window_start_ms,
            contract.reveal_window_end_ms,
        ],
        "violations": validation.violations,
        "allowed_for_real_experiment": validation.allowed,
        "validation": validation.to_dict(),
    }
    _emit(output)
    return 0 if validation.allowed else 1


def register_replay_patch_command(how_subparsers: Any) -> None:
    p = how_subparsers.add_parser(
        "replay-patch",
        help="Counterfactual patch replay against a contract (v4 §8.5, PR-SAFE-2)",
    )
    p.add_argument("--practice-id", required=True)
    p.add_argument("--patch", required=True, help="patch JSON file (parameter → value)")
    p.add_argument("--contract", default=_DEFAULT_CONTRACT)
    p.add_argument("--current-params", default=None, help="current parameter JSON (optional)")
    p.add_argument("--data-root", default=None)
    p.set_defaults(how_handler=cmd_how_replay_patch)
