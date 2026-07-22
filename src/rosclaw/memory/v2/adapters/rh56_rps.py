"""RH56 RPS stress-protocol distillation adapter (数据库优化v3 §4.2).

Handles the event shapes the generic distiller cannot:

* ``rps.stress.round.resolved`` with ``result=invalid`` — an IMPLICIT
  failure (no explicit failure_event is emitted for these rounds);
* gesture-level verification failures inside rounds that still resolved
  VALID — invisible at round granularity, real at gesture granularity;
* episode quality as a verified-rate distribution with a configurable
  threshold (PARTIAL_SUCCESS, not a blanket SUCCESS);
* temperature rise as observed statistics with an explicit
  ``causal_status`` — correlation, never a claimed thermal limit.

Linkage keys, in priority order: round_id -> time window.  Joint-level
attribution is only filled when the session actually recorded it;
current RPS sessions never log the failing joint, and the adapter
leaves ``joint_name=None`` rather than inventing one.
"""

from __future__ import annotations

import json
from typing import Any

from rosclaw.memory.v2.document import MultilingualMemoryDocumentBuilder
from rosclaw.memory.v2.models import MemoryItem, MemoryType

from .base import TaskDistillationAdapter

_RPS_EVENT_PREFIXES = ("rps.stress.", "rps.gesture.", "rps.telemetry")


def _payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {}
    return payload or {}


def _round_key(event: dict[str, Any]) -> str | None:
    payload = _payload(event)
    rnd = payload.get("round_id")
    if rnd:
        return str(rnd)
    inner = payload.get("round")
    if isinstance(inner, dict) and inner.get("round_id"):
        return str(inner["round_id"])
    return None


class Rh56RpsAdapter(TaskDistillationAdapter):
    """Adapter for the RH56 rock-paper-scissors stress protocol."""

    task_ids = {"rh56_rps"}

    def __init__(
        self,
        *,
        success_min_verified_rate: float = 0.98,
        partial_min_verified_rate: float = 0.80,
    ) -> None:
        # Task-declared thresholds (数据库优化v3 §5.1 — core never
        # hardcodes them; the adapter owns the task's quality bar).
        self.success_min = success_min_verified_rate
        self.partial_min = partial_min_verified_rate
        self._docs = MultilingualMemoryDocumentBuilder()

    # ------------------------------------------------------------------
    def matches_events(self, events: list[dict[str, Any]]) -> bool:
        return any(str(e.get("event_type", "")).startswith(_RPS_EVENT_PREFIXES) for e in events)

    # ------------------------------------------------------------------
    def extract_failures(self, context: Any, events: list[dict[str, Any]]) -> list[MemoryItem]:
        # Round windows for the TIME-WINDOW linkage fallback
        # (数据库优化v3 §4.2 priority: round_id -> action_id -> trace_id
        # -> time window).  Current rps.gesture.executed events carry NO
        # round_id, so windows built from round.started -> round.resolved
        # are the real linkage on production sessions.
        windows = _round_windows(events)
        gestures_by_round = _gestures_by_round(events, windows)
        health_checks: list[dict[str, Any]] = []
        for event in events:
            etype = str(event.get("event_type", ""))
            if etype == "health_check":
                health_checks.append(event)

        memories: list[MemoryItem] = []
        invalid_rounds: set[str] = set()
        for event in events:
            if event.get("event_type") != "rps.stress.round.resolved":
                continue
            payload = _payload(event)
            rnd = payload.get("round") if isinstance(payload.get("round"), dict) else payload
            if rnd.get("result") != "invalid":
                continue
            round_id = _round_key(event) or "unknown"
            invalid_rounds.add(round_id)
            linked = gestures_by_round.get(round_id, [])
            failing = [
                e
                for e in linked
                if _payload(e).get("verified") is False
                or _payload(e).get("command_success") is False
            ]
            if failing:
                # One failure item per failing gesture in the invalid
                # round — every one of them is a real failure (数据库
                # 优化v3 §18: all joint_not_reached must form memory).
                for gesture_event in failing:
                    gp = _payload(gesture_event)
                    hand = gp.get("hand")
                    memories.append(
                        self._failure_item(
                            context,
                            hand=hand,
                            gesture_name=gp.get("gesture_name"),
                            joint_name=gp.get("joint_name"),
                            failure_reason=(
                                gp.get("failure_reason")
                                or rnd.get("robot_gesture_failure_reason")
                                or rnd.get("failure_reason")
                                or "unverified"
                            ),
                            round_id=round_id,
                            round_index=_round_index(round_id),
                            temperature=_temperature_near(health_checks, gesture_event, hand),
                            evidence=[event.get("event_id"), gesture_event.get("event_id")],
                            source_event=gesture_event,
                        )
                    )
            else:
                # No linked gesture telemetry — the invalid round itself
                # is the failure, attribution left honest (hand from the
                # resolved payload only, never invented).
                hand, gesture_name, joint_name = _gesture_facts(linked, rnd)
                memories.append(
                    self._failure_item(
                        context,
                        hand=hand,
                        gesture_name=gesture_name,
                        joint_name=joint_name,
                        failure_reason=(
                            rnd.get("robot_gesture_failure_reason")
                            or rnd.get("failure_reason")
                            or "unverified"
                        ),
                        round_id=round_id,
                        round_index=_round_index(round_id),
                        temperature=_temperature_near(health_checks, event, hand),
                        evidence=[event.get("event_id")],
                        source_event=event,
                    )
                )

        # Gesture-level failures NOT already covered by an invalid-round
        # item — including rounds that still resolved VALID and gestures
        # executed BETWEEN rounds (e.g. the inter-round ready pose,
        # which has no round window at all).
        covered_ids = {eid for item in memories for eid in item.evidence_refs}
        for round_id, gesture_events in gestures_by_round.items():
            if round_id in invalid_rounds:
                continue  # already covered by the round-level failure
            for event in gesture_events:
                payload = _payload(event)
                if not _gesture_failed(payload):
                    continue
                if event.get("event_id") in covered_ids:
                    continue
                hand = payload.get("hand")
                memories.append(
                    self._failure_item(
                        context,
                        hand=hand,
                        gesture_name=payload.get("gesture_name"),
                        joint_name=payload.get("joint_name"),
                        failure_reason=payload.get("failure_reason") or "unverified",
                        round_id=round_id,
                        round_index=_round_index(round_id),
                        temperature=_temperature_near(health_checks, event, hand),
                        evidence=[event.get("event_id")],
                        source_event=event,
                    )
                )
        # Between-round gesture failures (no window at all).
        for event in events:
            if event.get("event_type") != "rps.gesture.executed":
                continue
            payload = _payload(event)
            if not _gesture_failed(payload):
                continue
            if event.get("event_id") in covered_ids:
                continue
            key = _round_key(event) or _window_for(event, windows)
            if key is not None:
                continue  # handled above (windowed gestures)
            hand = payload.get("hand")
            memories.append(
                self._failure_item(
                    context,
                    hand=hand,
                    gesture_name=payload.get("gesture_name"),
                    joint_name=payload.get("joint_name"),
                    failure_reason=payload.get("failure_reason") or "unverified",
                    round_id="between_rounds",
                    round_index=None,
                    temperature=_temperature_near(health_checks, event, hand),
                    evidence=[event.get("event_id")],
                    source_event=event,
                )
            )
        return memories

    def _failure_item(
        self,
        context: Any,
        *,
        hand: str | None,
        gesture_name: str | None,
        joint_name: str | None,
        failure_reason: str,
        round_id: str,
        round_index: int | None,
        temperature: float | None,
        evidence: list[Any],
        source_event: dict[str, Any],
    ) -> MemoryItem:
        doc = self._docs.build_failure(
            hand=hand or "unknown",
            joint=joint_name,
            gesture=gesture_name,
            failure_type=failure_reason,
            round_index=round_index,
            temperature_c=temperature,
        )
        return MemoryItem(
            memory_type=MemoryType.FAILURE.value,
            robot_id=context.robot_id,
            body_id=_body_for_hand(context, hand),
            practice_id=context.practice_id,
            session_id=context.session_id,
            episode_id=context.episode_id,
            task_id=context.task_id,
            skill_id=context.skill_id,
            failure_type=failure_reason,
            joint_name=joint_name,
            gesture_name=gesture_name,
            outcome="failure",
            title=(f"{hand or '?'} {gesture_name or '?'} failed at {round_id}: {failure_reason}"),
            document=doc.combined,
            confidence=0.9,
            importance=0.75,
            evidence_refs=[eid for eid in evidence if eid],
            tags=["failure", "rps", hand or "unknown", gesture_name or "unknown"],
            metadata={
                "round_id": round_id,
                "round_index": round_index,
                "temperature_c": temperature,
                "joint_attribution": ("recorded" if joint_name else "not_recorded_in_session"),
                "aliases": doc.aliases,
                "exact_terms": doc.exact_terms,
            },
            event_time=_event_time(source_event),
        )

    # ------------------------------------------------------------------
    def extract_body_patterns(self, context: Any, events: list[dict[str, Any]]) -> list[MemoryItem]:
        health = [e for e in events if e.get("event_type") == "health_check"]
        if len(health) < 2:
            return []
        invalid_temps = _invalid_temperatures(events)
        memories: list[MemoryItem] = []
        for side in ("left", "right"):
            series: list[tuple[float, float, str | None]] = []
            for event in health:
                payload = _payload(event)
                temp = _side_max_temp(payload.get(side) or {})
                if temp is not None:
                    series.append(
                        (
                            float(payload.get("runtime_s") or 0.0),
                            temp,
                            event.get("event_id"),
                        )
                    )
            if len(series) < 2:
                continue
            t0, temp0, first_id = series[0]
            t1, temp1, last_id = series[-1]
            temps = [t for _, t, _ in series]
            delta = temp1 - temp0
            minutes = max((t1 - t0) / 60.0, 1e-6)
            first_fail_temp = invalid_temps.get(side)
            correlation = (
                "observed_correlation"
                if first_fail_temp is not None and abs(delta) >= 3.0
                else "insufficient_data"
            )
            zh = (
                f"{side} 手在 {minutes:.0f} 分钟内由 {temp0:.0f}°C 变为 {temp1:.0f}°C"
                f"（观测区间 {min(temps):.0f}–{max(temps):.0f}°C，速率 {delta / minutes:+.2f}°C/min）。"
            )
            en = (
                f"{side} hand moved from {temp0:.0f}°C to {temp1:.0f}°C over "
                f"{minutes:.0f} min (observed range {min(temps):.0f}-{max(temps):.0f}°C, "
                f"rate {delta / minutes:+.2f}°C/min)."
            )
            if first_fail_temp is not None:
                zh += f" 首次 invalid 出现在 {first_fail_temp:.0f}°C。"
                en += f" First invalid round at {first_fail_temp:.0f}°C."
            zh += " 这是观测相关性，不是温度阈值结论。"
            en += " This is an observed correlation, NOT a thermal limit."
            memories.append(
                MemoryItem(
                    memory_type=MemoryType.BODY.value,
                    robot_id=context.robot_id,
                    body_id=_body_for_hand(context, side),
                    practice_id=context.practice_id,
                    session_id=context.session_id,
                    episode_id=context.episode_id,
                    task_id=context.task_id,
                    title=f"{side} hand thermal observation {temp0:.0f}→{temp1:.0f}°C over {minutes:.0f}min",
                    document=f"[ZH]\n{zh}\n\n[EN]\n{en}",
                    confidence=0.8,
                    importance=min(0.5 + abs(delta) / 50.0, 0.9),
                    evidence_refs=[eid for eid in (first_id, last_id) if eid],
                    tags=["body", "thermal", side],
                    metadata={
                        # 数据库优化v3 §5.2: observation statistics, never
                        # a "thermal_limits" claim.
                        "observed_temperature_min": min(temps),
                        "observed_temperature_max": max(temps),
                        "temperature_delta": delta,
                        "temperature_rise_rate_per_min": delta / minutes,
                        "first_failure_temperature": first_fail_temp,
                        "causal_status": correlation,
                    },
                    event_time=_event_time(health[-1]),
                )
            )
        return memories

    # ------------------------------------------------------------------
    def extract_skill_evidence(
        self, context: Any, events: list[dict[str, Any]]
    ) -> list[MemoryItem]:
        # The generic per-gesture verified-rate extractor already covers
        # RPS gestures; the adapter adds nothing here.
        return []

    # ------------------------------------------------------------------
    def build_episode_quality(self, context: Any, events: list[dict[str, Any]]) -> dict[str, Any]:
        rounds = [e for e in events if e.get("event_type") == "rps.stress.round.resolved"]
        if not rounds:
            return {}
        total = len(rounds)
        distribution: dict[str, int] = {}
        invalid = 0
        first_degradation: int | None = None
        for event in rounds:
            payload = _payload(event)
            rnd = payload.get("round") if isinstance(payload.get("round"), dict) else payload
            if rnd.get("result") == "invalid":
                invalid += 1
                reason = (
                    rnd.get("robot_gesture_failure_reason")
                    or rnd.get("failure_reason")
                    or "unverified"
                )
                distribution[reason] = distribution.get(reason, 0) + 1
                idx = _round_index(_round_key(event) or "")
                if idx is not None and (first_degradation is None or idx < first_degradation):
                    first_degradation = idx
        verified = total - invalid
        rate = verified / total if total else 0.0
        if rate >= self.success_min:
            outcome = "success"
        elif rate >= self.partial_min:
            outcome = "partial_success"
        else:
            outcome = "failure"
        return {
            "outcome": outcome,
            "quality": {
                "total_rounds": total,
                "verified_rounds": verified,
                "invalid_rounds": invalid,
                "verified_rate": round(rate, 4),
                "failure_distribution": distribution,
                "first_degradation_round": first_degradation,
                "thresholds": {
                    "success_min_verified_rate": self.success_min,
                    "partial_min_verified_rate": self.partial_min,
                },
            },
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _round_index(round_id: str) -> int | None:
    digits = "".join(ch for ch in round_id if ch.isdigit())
    return int(digits) if digits else None


def _gesture_failed(payload: dict[str, Any]) -> bool:
    """A missing status is unknown, not evidence of a failure."""
    return payload.get("verified") is False or payload.get("command_success") is False


def _round_windows(events: list[dict[str, Any]]) -> list[tuple[float, float, str]]:
    windows: list[tuple[float, float, str]] = []
    started: dict[str, float] = {}
    for event in events:
        event_type = str(event.get("event_type", ""))
        round_id = _round_key(event)
        if event_type == "rps.stress.round.started" and round_id:
            started[round_id] = _event_time(event)
        elif event_type == "rps.stress.round.resolved" and round_id in started:
            windows.append((started[round_id], _event_time(event), round_id))
    return windows


def _window_for(event: dict[str, Any], windows: list[tuple[float, float, str]]) -> str | None:
    timestamp = _event_time(event)
    for start, end, round_id in windows:
        if start <= timestamp <= end:
            return round_id
    return None


def _gestures_by_round(
    events: list[dict[str, Any]],
    windows: list[tuple[float, float, str]],
) -> dict[str, list[dict[str, Any]]]:
    linked: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("event_type") != "rps.gesture.executed":
            continue
        round_id = _round_key(event) or _window_for(event, windows)
        if round_id:
            linked.setdefault(round_id, []).append(event)
    return linked


def _gesture_facts(
    linked: list[dict[str, Any]], resolved: dict[str, Any]
) -> tuple[str | None, str | None, str | None]:
    hand = None
    gesture = None
    joint = None
    for event in linked:
        payload = _payload(event)
        if payload.get("verified") is False or payload.get("command_success") is False:
            hand = payload.get("hand") or hand
            gesture = payload.get("gesture_name") or gesture
            joint = payload.get("joint_name") or joint
    if hand is None:
        hand = resolved.get("hand")
    if gesture is None:
        gesture = resolved.get("robot_choice") or resolved.get("gesture_name")
    return hand, gesture, joint


def _side_max_temp(side_payload: dict[str, Any]) -> float | None:
    summary = side_payload.get("summary") or {}
    temp = summary.get("temperature_max")
    if isinstance(temp, (int, float)):
        return float(temp)
    temps = side_payload.get("temperature_c")
    if isinstance(temps, dict) and temps:
        values = [float(v) for v in temps.values() if isinstance(v, (int, float))]
        return max(values) if values else None
    return None


def _temperature_near(
    health_checks: list[dict[str, Any]], event: dict[str, Any], hand: str | None
) -> float | None:
    if not health_checks or hand not in ("left", "right"):
        return None
    target = _event_time(event)
    best: tuple[float, float] | None = None
    for hc in health_checks:
        payload = _payload(hc)
        temp = _side_max_temp(payload.get(hand) or {})
        if temp is None:
            continue
        distance = abs(_event_time(hc) - target)
        if best is None or distance < best[0]:
            best = (distance, temp)
    return best[1] if best else None


def _invalid_temperatures(events: list[dict[str, Any]]) -> dict[str, float]:
    """First invalid-round temperature per side (observation only)."""
    out: dict[str, float] = {}
    health = [e for e in events if e.get("event_type") == "health_check"]
    gestures_by_round = _gestures_by_round(events, _round_windows(events))
    for event in events:
        if event.get("event_type") != "rps.stress.round.resolved":
            continue
        payload = _payload(event)
        rnd = payload.get("round") if isinstance(payload.get("round"), dict) else payload
        if rnd.get("result") != "invalid":
            continue
        round_id = _round_key(event)
        hand, _, _ = _gesture_facts(gestures_by_round.get(round_id or "", []), rnd)
        if hand not in ("left", "right") or hand in out:
            continue
        temp = _temperature_near(health, event, hand)
        if temp is not None:
            out[hand] = temp
    return out


def _body_for_hand(context: Any, hand: str | None) -> str | None:
    if hand == "left":
        return "rh56_left_01"
    if hand == "right":
        return "rh56_right_01"
    return context.body_id


def _event_time(event: dict[str, Any]) -> float:
    ns = event.get("timestamp_ns")
    if isinstance(ns, (int, float)) and ns:
        return float(ns) / 1e9
    raw = event.get("timestamp_utc") or ""
    try:
        from datetime import datetime

        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return 0.0
