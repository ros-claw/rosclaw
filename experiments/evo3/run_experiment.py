"""EVO-3 experiment runner (数据库优化v4 §11).

Two modes, honest about what each can prove:

* ``replay``: drives the FULL experiment machinery over recorded practice
  sessions (real telemetry, no hardware).  It measures DECISION behavior
  per arm (coverage, abstention correctness, apply correctness vs the
  session's regime) — never counterfactual invalid-rate outcomes, which
  only the live campaign can produce.
* ``live``: prints the operator run plan for the A/B/C campaign on the
  workspace stress harness (sessions are executed by the operator, then
  scored by ``stats_analysis.promotion_report``).

Usage::

    python experiments/evo3/run_experiment.py replay \
        --data-root ~/.rosclaw/practice/runs/rh56_rps [--sessions N]
    python experiments/evo3/run_experiment.py live --protocol evo3_exp1_healthy_abstain
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arms import build_arm
from offline_validations import validate_exp3_counter_regime, validate_exp4_choreography_protection
from protocols import PROTOCOLS

from rosclaw.how.choreography import ChoreographyValidator, load_contract
from rosclaw.how.selective import SelectiveInterventionPipeline
from rosclaw.memory.seekdb_client import InMemoryKnowledgeStore
from rosclaw.memory.v2.regime import (
    ApplicabilityEnvelope,
    ApplicabilityStore,
    CurrentRegimeBuilder,
    RegimeMatcher,
    empty_regime,
)
from rosclaw.memory.v2.regime.session_samples import extract_samples, load_session_events
from rosclaw.memory.v2.runtime_retrieval import build_retrieval_facade

DEFAULT_CONTRACT = "configs/choreography/rh56_rps_v1.yaml"


def _emit(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Replay machinery
# ---------------------------------------------------------------------------


def _regime_from_samples(builder, samples, *, body_id: str, now: float):
    return builder.build(
        samples,
        robot_id="rh56_rps_robot",
        body_id=body_id,
        task_id="rh56_rps",
        session_started_at=samples[0].timestamp if samples else now,
        rounds_completed=len(samples),
        now=now,
    )


def replay_sessions(
    session_dirs: list[Path],
    *,
    arm_names: list[str],
    hand: str = "right",
) -> dict[str, Any]:
    """Replay recorded sessions through each arm's decision policy."""
    builder = CurrentRegimeBuilder()
    client = InMemoryKnowledgeStore()
    client.connect()
    applicability_store = ApplicabilityStore(client)
    facade = build_retrieval_facade(sqlite_store=client)
    validator = ChoreographyValidator(load_contract(DEFAULT_CONTRACT))
    pipeline = SelectiveInterventionPipeline(
        facade,
        applicability_store,
        matcher=RegimeMatcher(),
        choreography_validator=validator,
    )

    arm_records: dict[str, list[dict[str, Any]]] = {name: [] for name in arm_names}
    session_reports: list[dict[str, Any]] = []
    for session_dir in session_dirs:
        samples = extract_samples(load_session_events(session_dir), hand=hand)
        if not samples:
            continue
        regime = _regime_from_samples(
            builder, samples, body_id=f"rh56_{hand}_01", now=samples[-1].timestamp
        )
        # Seed the memory store with THIS session's failures (the memory
        # context the live system would have accumulated).
        from rosclaw.memory.v2.models import MemoryItem

        failures = []
        for index, sample in enumerate(samples):
            if not (sample.invalid or sample.failure):
                continue
            memory_id = f"replay_{session_dir.name}_{index}"
            client.insert(
                "memory_items",
                MemoryItem(
                    memory_id=memory_id,
                    memory_type="failure",
                    robot_id="rh56_rps_robot",
                    body_id=f"rh56_{hand}_01",
                    failure_type="joint_not_reached",
                    title=f"{session_dir.name} round {index} failure",
                    document=f"{session_dir.name} round {index} joint_not_reached",
                    outcome="failure",
                    evidence_refs=[sample.evidence_ref or f"evt_{index}"],
                ).to_record(),
            )
            applicability_store.upsert(
                ApplicabilityEnvelope(
                    memory_id=memory_id,
                    body_ids=[f"rh56_{hand}_01"],
                    task_ids=["rh56_rps"],
                    temperature_min=sample.temperature_c,
                    temperature_max=sample.temperature_c,
                    regime_labels=[regime.regime_label],
                    envelope_type="observed",
                    evidence_count=1,
                    confidence=0.5,
                )
            )
            failures.append(
                {
                    "signature": "joint_not_reached",
                    "failure_type": "joint_not_reached",
                    "body_id": f"rh56_{hand}_01",
                    "joint_name": None,
                }
            )

        invalid_count = sum(1 for s in samples if s.invalid)
        first_invalid = next((i for i, s in enumerate(samples) if s.invalid), None)
        temps = [s.temperature_c for s in samples if s.temperature_c is not None]
        for name in arm_names:
            arm = build_arm(name, pipeline=pipeline)
            decisions = [arm.respond(failure, regime) for failure in failures]
            acted = sum(1 for d in decisions if d.acted)
            abstains = sum(1 for d in decisions if d.decision_action == "ABSTAIN")
            arm_records[name].append(
                {
                    "session": session_dir.name,
                    "failures": len(failures),
                    "acted": acted,
                    "abstain": abstains,
                    "coverage": (acted / len(failures)) if failures else 0.0,
                    "abstention_rate": (abstains / len(failures)) if failures else None,
                }
            )
        session_reports.append(
            {
                "session": session_dir.name,
                "rounds": len(samples),
                "invalid": invalid_count,
                "regime_label": regime.regime_label,
                "temperature_max_c": max(temps) if temps else None,
                "first_invalid_round": first_invalid,
            }
        )
    return {
        "sessions": session_reports,
        "arm_decision_traces": arm_records,
    }


def cmd_replay(args: argparse.Namespace) -> int:
    root = Path(args.data_root).expanduser()
    sessions_root = root / "sessions"
    candidates = sorted(
        path
        for path in sessions_root.iterdir()
        if path.is_dir() and (path / "raw" / "events.jsonl").is_file()
    )
    if args.sessions:
        candidates = candidates[-args.sessions :]
    if not candidates:
        _emit({"ok": False, "error": f"no replayable sessions under {sessions_root}"})
        return 1

    result = replay_sessions(
        candidates,
        arm_names=["A_no_memory", "B_fixed_cooldown", "C_regime_aware"],
        hand=args.hand,
    )
    summary: dict[str, Any] = {
        "mode": "replay",
        "note": (
            "replay measures DECISION behavior (coverage/abstention) — "
            "counterfactual invalid-rate outcomes require the live campaign"
        ),
        "sessions_replayed": len(result["sessions"]),
        "regime_distribution": {},
        "arm_coverage": {},
    }
    for report in result["sessions"]:
        label = report["regime_label"]
        summary["regime_distribution"][label] = summary["regime_distribution"].get(label, 0) + 1
    for arm_name, traces in result["arm_decision_traces"].items():
        total_failures = sum(t["failures"] for t in traces)
        total_acted = sum(t["acted"] for t in traces)
        abstain_traces = [t["abstention_rate"] for t in traces if t["abstention_rate"] is not None]
        summary["arm_coverage"][arm_name] = {
            "failures_seen": total_failures,
            "acted": total_acted,
            "coverage": (total_acted / total_failures) if total_failures else 0.0,
            "abstention_rate_mean": (
                sum(abstain_traces) / len(abstain_traces) if abstain_traces else None
            ),
        }
    out = Path(args.out or f"/tmp/evo3_replay_{int(time.time())}.json")
    out.write_text(json.dumps({"summary": summary, "detail": result}, indent=2, ensure_ascii=False))
    _emit({**summary, "out": str(out)})
    return 0


# ---------------------------------------------------------------------------
# Live run plan
# ---------------------------------------------------------------------------


def cmd_live(args: argparse.Namespace) -> int:
    protocol = PROTOCOLS.get(args.protocol)
    if protocol is None:
        _emit(
            {"ok": False, "error": f"unknown protocol {args.protocol}", "known": sorted(PROTOCOLS)}
        )
        return 2
    plan = {
        "protocol": protocol.experiment_id,
        "title": protocol.title,
        "arms": list(protocol.arms),
        "sessions_per_arm": protocol.sessions_per_arm,
        "rounds_per_session": protocol.rounds_per_session,
        "hand_balance": protocol.hand_balance,
        "randomize_order": protocol.randomize_order,
        "trigger": protocol.trigger,
        "safety_limits": protocol.safety_limits,
        "expected": protocol.expected,
        "operator_steps": [
            "1. 按 hand_balance 准备左右手各半的会话计划（顺序用 --seed 随机化）",
            "2. 每臂 N 会话：A=无记忆基线, B=固定冷却 5s, C=选择性管线",
            "3. 会话事件落入 practice runs 后: 提取 rounds → SessionRecord",
            "4. stats_analysis.promotion_report(records, arm_a, arm_b) 出晋升报告",
            "5. 全程不得超过 safety_limits 的温度/电流阈值",
        ],
        "note": "本命令只出计划，不驱动硬件；真机执行由操作员按计划运行",
    }
    _emit(plan)
    return 0


def cmd_validate_offline(args: argparse.Namespace) -> int:
    """Run exp3 + exp4 offline proofs (no hardware)."""
    from rosclaw.memory.v2.regime import RegimeLabel

    validator = ChoreographyValidator(load_contract(DEFAULT_CONTRACT))
    exp4 = validate_exp4_choreography_protection(validator)

    # Exp 3 needs a pipeline whose store contains the thermal memory.
    client = InMemoryKnowledgeStore()
    client.connect()
    from rosclaw.memory.v2.models import MemoryItem

    client.insert(
        "memory_items",
        MemoryItem(
            memory_id="mem_hot_slowdown",
            memory_type="failure",
            robot_id="rh56_rps_robot",
            body_id="rh56_right_01",
            joint_name="middle",
            failure_type="joint_not_reached",
            title="热退化 middle 不到位（56–58°C 两小时会话）",
            document="两小时 56–58°C 会话 middle joint_not_reached，记录为减速+延时",
            outcome="failure",
            evidence_refs=["run1"],
            metadata={"recovery_hint": "减速并增加延时"},
        ).to_record(),
    )
    store = ApplicabilityStore(client)
    store.upsert(
        ApplicabilityEnvelope(
            memory_id="mem_hot_slowdown",
            body_ids=["rh56_right_01"],
            task_ids=["rh56_rps"],
            joints=["middle"],
            temperature_min=55.0,
            temperature_max=60.0,
            elapsed_sec_min=3600.0,
            elapsed_sec_max=7200.0,
            regime_labels=[RegimeLabel.THERMAL_TRACKING_DEGRADATION.value],
            envelope_type="validated",
            evidence_count=4,
            success_count=3,
            confidence=0.85,
        )
    )
    store.upsert(
        ApplicabilityEnvelope(
            memory_id="mem_hot_slowdown",
            body_ids=["rh56_right_01"],
            task_ids=["rh56_rps"],
            regime_labels=[RegimeLabel.COLD_HEALTHY.value, RegimeLabel.WARM_STABLE.value],
            envelope_type="contraindicated",
            reason="breaks_reveal_timing",
            evidence_refs=["run1_patch_proofs"],
            confidence=0.9,
        )
    )
    facade = build_retrieval_facade(sqlite_store=client)
    pipeline = SelectiveInterventionPipeline(facade, store, choreography_validator=validator)
    healthy = empty_regime(robot_id="rh56_rps_robot", body_id="rh56_right_01", task_id="rh56_rps")
    healthy.regime_label = RegimeLabel.COLD_HEALTHY.value
    healthy.temperature_c = 49.0
    healthy.temperature_slope_c_per_min = 0.01
    healthy.session_elapsed_sec = 600.0
    healthy.cumulative_action_count = 80
    healthy.recent_invalid_rate = 0.02
    healthy.confidence = 0.85
    exp3 = validate_exp3_counter_regime(
        pipeline,
        healthy_regime=healthy,
        failure_signature="middle joint_not_reached",
        body_id="rh56_right_01",
        joint_name="middle",
    )
    _emit(
        {
            "exp3_counter_regime": exp3,
            "exp4_choreography_protection": exp4,
            "passed": exp3["passed"] and exp4["passed"],
        }
    )
    return 0 if (exp3["passed"] and exp4["passed"]) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("replay", help="replay recorded sessions through the arms")
    p.add_argument("--data-root", default="~/.rosclaw/practice/runs/rh56_rps")
    p.add_argument("--sessions", type=int, default=None)
    p.add_argument("--hand", choices=["left", "right"], default="right")
    p.add_argument("--out", default=None)
    p.set_defaults(handler=cmd_replay)
    p = sub.add_parser("live", help="print the operator run plan for a protocol")
    p.add_argument("--protocol", required=True, choices=sorted(PROTOCOLS))
    p.set_defaults(handler=cmd_live)
    p = sub.add_parser("validate-offline", help="exp3 + exp4 proofs (no hardware)")
    p.set_defaults(handler=cmd_validate_offline)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
