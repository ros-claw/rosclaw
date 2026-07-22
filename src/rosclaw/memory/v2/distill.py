"""Memory 2.0 practice distillation pipeline (§5.6).

    Practice Session
        ↓ EpisodeExtractor
        ↓ FailureSegmenter
        ↓ BodyStateSummarizer
        ↓ InterventionExtractor
        ↓ SkillEvidenceExtractor
        ↓ MemoryCandidateBuilder
        ↓ MemoryWriteGate
        ↓ MemoryRepository

Every extractor is a pure function over the session's event list so it can be
unit-tested and replayed without side effects.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rosclaw.memory.v2.gate import MemoryWriteGate
from rosclaw.memory.v2.models import MemoryItem, MemoryType
from rosclaw.memory.v2.repository import MemoryRepository

logger = logging.getLogger("rosclaw.memory.v2.distill")


@dataclass
class SessionContext:
    """Identity context shared by all extractors."""

    practice_id: str
    session_id: str | None = None
    episode_id: str | None = None
    robot_id: str = "unknown"
    body_id: str | None = None
    task_id: str | None = None
    skill_id: str | None = None


@dataclass
class DistillResult:
    """Outcome of one distillation run."""

    practice_id: str
    candidates: int = 0
    stored: list[str] = field(default_factory=list)
    merged: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    ignored: int = 0
    quarantined: int = 0
    decisions: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Event loading
# ---------------------------------------------------------------------------


def load_session_events(session_dir: str | Path) -> tuple[SessionContext, list[dict[str, Any]]]:
    """Load events.jsonl and derive the session context from the first event."""
    events_path = Path(session_dir) / "raw" / "events.jsonl"
    events: list[dict[str, Any]] = []
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not events:
        raise ValueError(f"no events found in {events_path}")
    first = events[0]
    context = SessionContext(
        practice_id=first.get("practice_id", "unknown"),
        session_id=first.get("session_id"),
        episode_id=first.get("episode_id"),
        robot_id=first.get("robot_id", "unknown"),
        body_id=first.get("body_id"),
        task_id=first.get("task_id"),
        skill_id=first.get("skill_id"),
    )
    return context, events


def _event_time(event: dict[str, Any]) -> float:
    """Evidence time of an event in epoch seconds (fallback: now)."""
    ts_ns = event.get("timestamp_ns")
    if isinstance(ts_ns, (int, float)) and ts_ns > 0:
        return ts_ns / 1_000_000_000.0
    return time.time()


# ---------------------------------------------------------------------------
# Extractors (pure functions)
# ---------------------------------------------------------------------------


def extract_episode_memory(
    context: SessionContext, events: list[dict[str, Any]]
) -> list[MemoryItem]:
    """One episodic memory per session: what happened, outcome, scale."""
    stop_events = [e for e in events if e.get("event_type") == "practice.session_stopped"]
    summary_events = [e for e in events if e.get("event_type", "").endswith("session.summary")]
    outcome = None
    if stop_events:
        outcome = (stop_events[-1].get("payload") or {}).get("outcome")
    gesture_events = [e for e in events if e.get("event_type", "").endswith("gesture.executed")]
    rounds = [e for e in events if e.get("event_type", "").endswith("round.resolved")]
    first_ts = events[0].get("timestamp_utc", "")
    last_ts = events[-1].get("timestamp_utc", "")

    doc_parts = [
        f"Session {context.practice_id} on robot {context.robot_id}: "
        f"{len(events)} events, {len(gesture_events)} gestures, {len(rounds)} rounds."
    ]
    if summary_events:
        payload = summary_events[-1].get("payload") or {}
        summary_text = payload.get("summary") or payload.get("text")
        if summary_text:
            doc_parts.append(str(summary_text))
    evidence = [e.get("event_id") for e in (stop_events or events[-1:]) if e.get("event_id")]
    return [
        MemoryItem(
            memory_type=MemoryType.EPISODIC.value,
            robot_id=context.robot_id,
            body_id=context.body_id,
            practice_id=context.practice_id,
            session_id=context.session_id,
            episode_id=context.episode_id,
            task_id=context.task_id,
            skill_id=context.skill_id,
            title=f"Episode {context.practice_id}",
            document="\n".join(doc_parts),
            outcome=(outcome or "unknown").lower() if outcome else None,
            evidence_refs=[eid for eid in evidence if eid],
            tags=["episode", "distilled"],
            metadata={"window": [first_ts, last_ts], "event_count": len(events)},
            event_time=_event_time(events[-1]),
        )
    ]


def extract_failure_memories(
    context: SessionContext, events: list[dict[str, Any]]
) -> list[MemoryItem]:
    """Failure memory for every verified-failed gesture and explicit error."""
    memories: list[MemoryItem] = []
    for event in events:
        event_type = event.get("event_type", "")
        payload = event.get("payload") or {}
        event_id = event.get("event_id")
        if event_type.endswith("gesture.executed"):
            if payload.get("command_success") and payload.get("verified") is not False:
                continue
            reason = payload.get("failure_reason") or "unverified"
            hand = payload.get("hand", "unknown")
            gesture = payload.get("gesture_name", "unknown")
            telemetry = payload.get("telemetry_summary") or {}
            memories.append(
                MemoryItem(
                    memory_type=MemoryType.FAILURE.value,
                    robot_id=context.robot_id,
                    body_id=context.body_id or hand,
                    practice_id=context.practice_id,
                    session_id=context.session_id,
                    episode_id=context.episode_id,
                    task_id=context.task_id,
                    skill_id=gesture,
                    title=f"{hand} {gesture} failed: {reason}",
                    document=(
                        f"Gesture {gesture} on {hand} hand failed ({reason}). "
                        f"telemetry: peak_current={telemetry.get('current_peak')}mA "
                        f"peak_force={telemetry.get('force_peak')} "
                        f"max_temp={telemetry.get('temperature_max')}°C."
                    ),
                    outcome="failure",
                    confidence=0.9,
                    importance=0.7,
                    evidence_refs=[event_id] if event_id else [],
                    tags=["failure", "gesture", hand, gesture],
                    event_time=_event_time(event),
                )
            )
        elif event_type in {"runtime.error", "serial.fault", "camera.wedge"}:
            memories.append(
                MemoryItem(
                    memory_type=MemoryType.FAILURE.value,
                    robot_id=context.robot_id,
                    body_id=context.body_id,
                    practice_id=context.practice_id,
                    session_id=context.session_id,
                    episode_id=context.episode_id,
                    task_id=context.task_id,
                    title=f"{event_type}: {str(payload)[:80]}",
                    document=f"{event_type} occurred: {json.dumps(payload, default=str)[:400]}",
                    outcome="failure",
                    confidence=0.85,
                    importance=0.8,
                    evidence_refs=[event_id] if event_id else [],
                    tags=["failure", event_type],
                    event_time=_event_time(event),
                )
            )
    return memories


def extract_body_memories(
    context: SessionContext, events: list[dict[str, Any]]
) -> list[MemoryItem]:
    """Body memory from telemetry trends (thermal drift, error states)."""
    telemetry = [e for e in events if e.get("event_type") == "health_check"]
    if len(telemetry) < 2:
        return []
    memories: list[MemoryItem] = []
    for side in ("left", "right"):
        temps = []
        error_events = []
        for event in telemetry:
            payload = event.get("payload") or {}
            side_payload = payload.get(side) or {}
            summary = side_payload.get("summary") or {}
            temp_max = summary.get("temperature_max")
            if temp_max:
                temps.append((payload.get("runtime_s", 0.0), temp_max, event.get("event_id")))
            errors = side_payload.get("error") or {}
            if any(v for v in errors.values() if isinstance(v, int)):
                error_events.append(event.get("event_id"))
        if len(temps) >= 2:
            (t0, temp0, first_id), (t1, temp1, last_id) = temps[0], temps[-1]
            rise = temp1 - temp0
            if abs(rise) >= 3 and t1 > t0:
                rate = rise / max((t1 - t0) / 60.0, 1e-6)  # °C per minute
                memories.append(
                    MemoryItem(
                        memory_type=MemoryType.BODY.value,
                        robot_id=context.robot_id,
                        body_id=context.body_id or side,
                        practice_id=context.practice_id,
                        session_id=context.session_id,
                        episode_id=context.episode_id,
                        task_id=context.task_id,
                        title=f"{side} hand thermal drift {temp0}→{temp1}°C over {(t1 - t0) / 60:.0f}min",
                        document=(
                            f"{side} hand max temperature moved from {temp0}°C to {temp1}°C "
                            f"over {(t1 - t0) / 60:.1f} minutes ({rate:+.2f}°C/min)."
                        ),
                        confidence=0.8,
                        importance=min(0.5 + abs(rise) / 50.0, 0.9),
                        evidence_refs=[eid for eid in (first_id, last_id) if eid],
                        tags=["body", "thermal", side],
                        event_time=_event_time(telemetry[-1]),
                    )
                )
        if error_events:
            memories.append(
                MemoryItem(
                    memory_type=MemoryType.BODY.value,
                    robot_id=context.robot_id,
                    body_id=context.body_id or side,
                    practice_id=context.practice_id,
                    session_id=context.session_id,
                    episode_id=context.episode_id,
                    task_id=context.task_id,
                    title=f"{side} hand reported {len(error_events)} error-state health checks",
                    document=(
                        f"{side} hand error register was non-zero in "
                        f"{len(error_events)} health checks during {context.practice_id}."
                    ),
                    confidence=0.85,
                    importance=0.7,
                    evidence_refs=[eid for eid in error_events[:5] if eid],
                    tags=["body", "error_register", side],
                    event_time=_event_time(telemetry[-1]),
                )
            )
    return memories


def extract_intervention_memories(
    context: SessionContext, events: list[dict[str, Any]]
) -> list[MemoryItem]:
    """Intervention memory from HOW recovery events."""
    memories: list[MemoryItem] = []
    for event in events:
        event_type = event.get("event_type", "")
        if not any(token in event_type for token in ("recovery", "intervention", "heuristic")):
            continue
        payload = event.get("payload") or {}
        event_id = event.get("event_id")
        success = payload.get("success", payload.get("command_success"))
        outcome = "unknown" if success is None else ("success" if success else "failure")
        memories.append(
            MemoryItem(
                memory_type=MemoryType.INTERVENTION.value,
                robot_id=context.robot_id,
                body_id=context.body_id,
                practice_id=context.practice_id,
                session_id=context.session_id,
                episode_id=context.episode_id,
                task_id=context.task_id,
                title=f"{event_type} ({outcome})",
                document=(
                    f"Intervention {event_type} executed with outcome "
                    f"{outcome}: "
                    f"{json.dumps(payload, default=str)[:300]}"
                ),
                outcome=outcome,
                confidence=0.85,
                importance=0.8 if success is not False else 0.9,
                evidence_refs=[event_id] if event_id else [],
                tags=["intervention", event_type],
                event_time=_event_time(event),
            )
        )
    return memories


def extract_skill_memories(
    context: SessionContext, events: list[dict[str, Any]]
) -> list[MemoryItem]:
    """Skill memory: per-gesture success rates across the session."""
    stats: dict[str, dict[str, int]] = {}
    evidence: dict[str, list[str]] = {}
    for event in events:
        if not event.get("event_type", "").endswith("gesture.executed"):
            continue
        payload = event.get("payload") or {}
        gesture = payload.get("gesture_name")
        if not gesture:
            continue
        ok = bool(payload.get("command_success") and payload.get("verified") is not False)
        bucket = stats.setdefault(gesture, {"success": 0, "failure": 0})
        bucket["success" if ok else "failure"] += 1
        if event.get("event_id"):
            evidence.setdefault(gesture, []).append(event["event_id"])
    memories: list[MemoryItem] = []
    for gesture, counts in stats.items():
        total = counts["success"] + counts["failure"]
        if total < 3:
            continue
        rate = counts["success"] / total
        memories.append(
            MemoryItem(
                memory_type=MemoryType.SKILL.value,
                robot_id=context.robot_id,
                body_id=context.body_id,
                practice_id=context.practice_id,
                session_id=context.session_id,
                episode_id=context.episode_id,
                task_id=context.task_id,
                skill_id=gesture,
                title=f"{gesture}: {counts['success']}/{total} ({rate:.0%}) verified",
                document=(
                    f"Gesture {gesture} succeeded {counts['success']}/{total} "
                    f"({rate:.1%}) in session {context.practice_id}."
                ),
                outcome="success" if rate >= 0.8 else "partial",
                confidence=min(0.5 + total / 50.0, 0.95),
                importance=0.5 + abs(rate - 0.5),
                evidence_refs=evidence.get(gesture, [])[:5],
                tags=["skill", gesture],
                event_time=max(
                    (
                        _event_time(e)
                        for e in events
                        if (e.get("payload") or {}).get("gesture_name") == gesture
                    ),
                    default=time.time(),
                ),
            )
        )
    return memories


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def build_candidates(context: SessionContext, events: list[dict[str, Any]]) -> list[MemoryItem]:
    """Run every extractor and collect candidates (MemoryCandidateBuilder).

    When a task adapter matches (数据库优化v3 §4), it OWNS failure and
    body-pattern semantics for its task (implicit failures, observed
    correlations) and decorates the episode memory with a quality
    distribution (verified-rate, not a blanket SUCCESS).  Generic
    extractors still cover episode, intervention, and skill evidence.
    """
    from rosclaw.memory.v2.adapters import adapter_for

    adapter = adapter_for(context, events)
    episode_items = extract_episode_memory(context, events)
    if adapter is not None:
        quality = adapter.build_episode_quality(context, events)
        if quality and episode_items:
            episode_items[0].outcome = quality.get("outcome", episode_items[0].outcome)
            episode_items[0].metadata = {
                **episode_items[0].metadata,
                "quality": quality.get("quality", {}),
            }
        failures = adapter.extract_failures(context, events)
        body = adapter.extract_body_patterns(context, events)
    else:
        failures = extract_failure_memories(context, events)
        body = extract_body_memories(context, events)

    candidates: list[MemoryItem] = []
    candidates.extend(episode_items)
    candidates.extend(failures)
    candidates.extend(body)
    candidates.extend(extract_intervention_memories(context, events))
    candidates.extend(extract_skill_memories(context, events))
    return candidates


def distill_events(
    context: SessionContext,
    events: list[dict[str, Any]],
    *,
    gate: MemoryWriteGate,
    repository: MemoryRepository,
) -> DistillResult:
    """Distill one session's events through the gate into the repository.

    Idempotent: re-distilling the same session produces the same content
    hashes, which dedup at the gate (IGNORE/UPDATE) and repository layers.
    """
    result = DistillResult(practice_id=context.practice_id)
    candidates = build_candidates(context, events)
    result.candidates = len(candidates)
    for candidate in candidates:
        decision = gate.evaluate(candidate)
        result.decisions.append(
            {
                "title": candidate.title,
                "memory_type": candidate.memory_type,
                "decision": decision.decision,
                "reason": decision.reason,
            }
        )
        if decision.decision == "STORE":
            result.stored.append(repository.store(candidate))
        elif decision.decision == "MERGE" and decision.target_memory_id:
            if repository.merge_into(decision.target_memory_id, candidate):
                result.merged.append(decision.target_memory_id)
        elif decision.decision == "UPDATE" and decision.target_memory_id:
            result.updated.append(repository.supersede(decision.target_memory_id, candidate))
        elif decision.decision == "QUARANTINE":
            result.quarantined += 1
            candidate.status = "quarantined"
            repository.store(candidate)
        else:
            result.ignored += 1
    return result


def distill_session_dir(
    session_dir: str | Path,
    *,
    gate: MemoryWriteGate,
    repository: MemoryRepository,
) -> DistillResult:
    """Load a practice session directory and distill it."""
    context, events = load_session_events(session_dir)
    return distill_events(context, events, gate=gate, repository=repository)


def iter_session_dirs(data_root: str | Path) -> Iterable[Path]:
    """Yield prac_* session directories under a practice data root."""
    sessions_root = Path(data_root) / "sessions"
    if not sessions_root.exists():
        return
    for entry in sorted(sessions_root.iterdir()):
        if entry.is_dir() and entry.name.startswith("prac_"):
            yield entry
