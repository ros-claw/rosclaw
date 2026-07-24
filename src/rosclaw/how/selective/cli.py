"""``rosclaw how decide`` / ``rosclaw how selective-metrics`` (v4 §15)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rosclaw.memory.seekdb_client import SQLiteKnowledgeStore
from rosclaw.memory.v2.regime import (
    ApplicabilityStore,
    CurrentRegimeBuilder,
    MatcherConfig,
    RegimeMatcher,
    empty_regime,
    load_thresholds,
)
from rosclaw.memory.v2.regime.cli import _DEFAULT_DATA_ROOT
from rosclaw.memory.v2.regime.session_samples import (
    extract_samples,
    latest_session_dir,
    load_session_events,
)
from rosclaw.memory.v2.runtime_retrieval import build_retrieval_facade
from rosclaw.storage.factory import StorageFactory
from rosclaw.storage.seekdb_native import SeekDBNativeStore

from .metrics import SelectiveRiskLedger
from .pipeline import SelectiveInterventionPipeline

_DEFAULT_REGIME_CONFIG = "configs/regimes/rh56_rps_v1.yaml"
_DEFAULT_CONTRACT = "configs/choreography/rh56_rps_v1.yaml"


def _timing_model_for(args: argparse.Namespace, contract: Any) -> Any | None:
    """Timing model from the latest session's rounds (real observed timing)."""
    from rosclaw.how.choreography.cli import _rounds_from_session
    from rosclaw.how.choreography.timing import build_timing_model

    data_root = getattr(args, "data_root", None) or _DEFAULT_DATA_ROOT
    try:
        session_dir = latest_session_dir(data_root)
        rounds = _rounds_from_session(session_dir)
    except (FileNotFoundError, ValueError):
        return None
    if not rounds:
        return None
    return build_timing_model(contract, rounds, current_parameters={})


def _emit(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    sys.stdout.flush()


def _regime_for(args: argparse.Namespace) -> Any:
    """Current regime from practice evidence (or honest UNKNOWN)."""
    robot_id = getattr(args, "robot_id", None) or "rh56_rps_robot"
    body_id = getattr(args, "body", None)
    thresholds = None
    config = getattr(args, "config", None) or _DEFAULT_REGIME_CONFIG
    if Path(config).is_file():
        thresholds = load_thresholds(config)
    builder = CurrentRegimeBuilder(thresholds)
    data_root = getattr(args, "data_root", None) or _DEFAULT_DATA_ROOT
    try:
        session_dir = latest_session_dir(data_root)
        samples = extract_samples(
            load_session_events(session_dir),
            hand="left" if body_id and "left" in body_id else "right",
        )
    except (FileNotFoundError, ValueError):
        samples = []
    if not samples:
        return empty_regime(
            robot_id=robot_id,
            body_id=body_id or "unknown",
            task_id=getattr(args, "task_id", None) or "rh56_rps",
        )
    return builder.build(
        samples,
        robot_id=robot_id,
        body_id=body_id or samples[-1].evidence_ref or "unknown",
        task_id=getattr(args, "task_id", None) or "rh56_rps",
        session_started_at=samples[0].timestamp,
        rounds_completed=len(samples),
    )


def _open_knowledge(args: argparse.Namespace) -> tuple[Any, str]:
    backend = getattr(args, "backend", None)
    url = getattr(args, "seekdb_url", None)
    path = getattr(args, "v2_path", None) or str(
        Path.home() / "data" / "memory" / "knowledge.sqlite"
    )
    client = StorageFactory.create_knowledge_store(
        backend=backend or ("sqlite" if not url else None),
        url=url,
        path=path,
    )
    client.connect()
    return client, url or path


def cmd_how_decide(args: argparse.Namespace) -> int:
    client, store_path = _open_knowledge(args)
    try:
        native = client if isinstance(client, SeekDBNativeStore) else None
        sqlite = client if isinstance(client, SQLiteKnowledgeStore) else None
        facade = build_retrieval_facade(native_store=native, sqlite_store=sqlite)
        matcher_config = MatcherConfig()
        config_path = getattr(args, "config", None) or _DEFAULT_REGIME_CONFIG
        if Path(config_path).is_file():
            import yaml

            with open(config_path, encoding="utf-8") as handle:
                matcher_config = MatcherConfig.from_dict(
                    (yaml.safe_load(handle) or {}).get("regime_matcher", {})
                )
        # Choreography gate: the APPLY rung needs the validator AND the
        # task's real timing model (from the same session evidence that
        # builds the regime) — never a synthetic empty model.
        choreography = None
        timing_model = None
        contract_path = getattr(args, "choreography_contract", None) or _DEFAULT_CONTRACT
        if Path(contract_path).is_file():
            from rosclaw.how.choreography import ChoreographyValidator, load_contract

            contract = load_contract(contract_path)
            choreography = ChoreographyValidator(contract)
            timing_model = _timing_model_for(args, contract)
        pipeline = SelectiveInterventionPipeline(
            facade,
            ApplicabilityStore(client),
            matcher=RegimeMatcher(matcher_config),
            choreography_validator=choreography,
            timing_model=timing_model,
        )
        regime = _regime_for(args)
        decision = pipeline.decide(
            args.failure,
            regime,
            robot_id=getattr(args, "robot_id", None),
            body_id=getattr(args, "body", None),
            joint_name=getattr(args, "joint", None),
            limit=getattr(args, "limit", 5),
        )
        # Every decision is evidence (v4 §14 observability).
        ledger = SelectiveRiskLedger(client)
        ledger.record_decision(decision, body_id=getattr(args, "body", None))

        output = {
            "action": decision.action.value,
            "reason_codes": decision.reason_codes,
            "decision": decision.to_dict(),
            "retrieval": {
                "memory_id": decision.selected_memory_id,
                "relevant": decision.selected_memory_id is not None,
            },
            "applicability": {
                "score": round(decision.applicability_score, 4),
                "matched_envelope": decision.matched_envelope_id,
            },
            "regime": {
                "label": regime.regime_label,
                "confidence": regime.confidence,
                "missing_features": regime.missing_features,
            },
            "store": store_path,
        }
    finally:
        client.disconnect()
    _emit(output)
    return 0


def cmd_how_selective_metrics(args: argparse.Namespace) -> int:
    client, _ = _open_knowledge(args)
    try:
        ledger = SelectiveRiskLedger(client)
        _emit(ledger.gate_report())
    finally:
        client.disconnect()
    return 0


def register_selective_commands(how_subparsers: Any) -> None:
    """Register selective-intervention subcommands into the how group."""
    p = how_subparsers.add_parser(
        "decide", help="Selective APPLY/SUGGEST/ABSTAIN/ESCALATE for a failure (v4, PR-HOW-3)"
    )
    p.add_argument("--failure", required=True, help="Failure symptom, e.g. joint_not_reached")
    p.add_argument("--body", required=True, help="Body instance, e.g. rh56_right_01")
    p.add_argument("--joint", default=None, help="Joint name, e.g. middle")
    p.add_argument("--robot-id", default=None)
    p.add_argument("--task-id", default=None)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--data-root", default=None, help="practice runs root for regime building")
    p.add_argument("--config", default=None, help="regime thresholds YAML")
    p.add_argument(
        "--choreography-contract",
        default=None,
        help="choreography contract YAML (enables the APPLY rung)",
    )
    p.add_argument("--v2-path", default=None, help="SQLite knowledge store path")
    p.add_argument(
        "--backend",
        choices=["sqlite", "seekdb_embedded", "seekdb_server", "mysql", "memory"],
        default=None,
    )
    p.add_argument("--seekdb-url", default=None)
    p.set_defaults(how_handler=cmd_how_decide)

    p = how_subparsers.add_parser(
        "selective-metrics",
        help="Coverage / abstention / selective harm risk + gate report (v4 §7.5)",
    )
    p.add_argument("--v2-path", default=None)
    p.add_argument(
        "--backend",
        choices=["sqlite", "seekdb_embedded", "seekdb_server", "mysql", "memory"],
        default=None,
    )
    p.add_argument("--seekdb-url", default=None)
    p.set_defaults(how_handler=cmd_how_selective_metrics)
