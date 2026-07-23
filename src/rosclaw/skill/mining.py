"""Heuristic practice-episode mining for ROSClaw skill candidates."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rosclaw.skill.evidence import write_mining_report
from rosclaw.skill.hash import validate_candidate_id
from rosclaw.skill.models import (
    LineageCandidate,
    LineageYaml,
    MiningReport,
    SkillPackage,
)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class CanonicalEvent:
    episode_id: str
    timestamp: float
    event_type: str
    payload: dict[str, Any]
    confidence: float = 1.0


@dataclass
class PracticeEpisode:
    episode_id: str
    task: str
    robot: str | None
    outcome: str
    events: list[CanonicalEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PracticeQuery:
    task: str
    robot: str | None = None
    min_episodes: int = 10
    include_failures: bool = True


# ---------------------------------------------------------------------------
# Practice loader
# ---------------------------------------------------------------------------


EVENT_TYPE_ALIASES = {
    "start": "task_start",
    "begin": "task_start",
    "detect": "object_detected",
    "perception": "perception_update",
    "state": "state_snapshot",
    "sandbox": "sandbox_decision",
    "plan": "plan",
    "approach": "approach",
    "contact": "contact",
    "kick": "action_command",
    "action": "action_command",
    "failure": "failure_event",
    "fail": "failure_event",
    "how": "how_intervention",
    "memory": "memory_write",
    "result": "outcome",
}


def normalize_event_type(raw: str) -> str:
    return EVENT_TYPE_ALIASES.get(raw.lower(), raw)


def load_episodes(source_dir: Path, query: PracticeQuery) -> list[PracticeEpisode]:
    episodes: list[PracticeEpisode] = []
    source_dir = Path(source_dir).expanduser().resolve()
    if not source_dir.exists():
        return episodes

    for path in sorted(source_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        if data.get("task") != query.task:
            continue
        robot = data.get("robot")
        if query.robot and robot != query.robot:
            continue
        outcome = data.get("outcome", "unknown")
        if outcome in ("failure", "failed"):
            outcome = "failure"
        elif outcome in ("success", "succeeded"):
            outcome = "success"
        elif "recover" in outcome:
            outcome = "failure_recovered"

        raw_events = data.get("events", [])
        events = []
        for idx, ev in enumerate(raw_events):
            if not isinstance(ev, dict):
                continue
            events.append(
                CanonicalEvent(
                    episode_id=data.get("episode_id", path.stem),
                    timestamp=float(ev.get("t", idx)),
                    event_type=normalize_event_type(ev.get("type", "unknown")),
                    payload=ev.get("payload", {}),
                )
            )
        episodes.append(
            PracticeEpisode(
                episode_id=data.get("episode_id", path.stem),
                task=query.task,
                robot=robot,
                outcome=outcome,
                events=events,
                metadata={k: v for k, v in data.items() if k not in ("events",)},
            )
        )
    return episodes


# ---------------------------------------------------------------------------
# Mining logic
# ---------------------------------------------------------------------------


def _next_candidate_id(pkg: SkillPackage) -> str:
    prefix = "candidate"
    existing: list[int] = []
    for params_file in (pkg.root / "policies" / "params").glob("candidate_*.yaml"):
        m = re.match(r"candidate_(\d+)", params_file.stem)
        if m:
            existing.append(int(m.group(1)))
    if pkg.lineage:
        for c in pkg.lineage.candidates:
            m = re.match(r"candidate_(\d+)", c.id)
            if m:
                existing.append(int(m.group(1)))
    next_id = max(existing, default=0) + 1
    return f"{prefix}_{next_id:04d}"


def _segment_episode(episode: PracticeEpisode) -> list[dict[str, Any]]:
    phases = []
    current: dict[str, Any] = {"phase": "observe", "events": []}
    for ev in episode.events:
        if ev.event_type == "task_start" and not current["events"]:
            current["phase"] = "observe"
        elif ev.event_type == "object_detected":
            if current["events"]:
                phases.append(current)
            current = {"phase": "plan", "events": [ev]}
        elif ev.event_type in ("approach",):
            if current["events"]:
                phases.append(current)
            current = {"phase": "approach", "events": [ev]}
        elif ev.event_type == "sandbox_decision":
            if current["events"]:
                phases.append(current)
            current = {"phase": "precheck", "events": [ev]}
        elif ev.event_type == "action_command":
            if current["events"]:
                phases.append(current)
            current = {"phase": "execute", "events": [ev]}
        elif ev.event_type == "outcome":
            if current["events"]:
                phases.append(current)
            current = {"phase": "verify", "events": [ev]}
        elif ev.event_type == "failure_event":
            if current["events"]:
                phases.append(current)
            current = {"phase": "recover", "events": [ev]}
        else:
            current["events"].append(ev)
    if current["events"]:
        phases.append(current)
    return phases


def _extract_params(success_episodes: list[PracticeEpisode]) -> dict[str, Any]:
    """Median-ish parameter extraction from successful episodes."""
    approach_speeds = []
    target_distances = []
    kick_strengths = []
    for ep in success_episodes:
        for ev in ep.events:
            p = ev.payload
            if "params" in p:
                params = p["params"]
                if isinstance(params, dict):
                    if "max_speed_mps" in params:
                        approach_speeds.append(params["max_speed_mps"])
                    if "target_distance_m" in params:
                        target_distances.append(params["target_distance_m"])
                    if "kick_strength" in params:
                        kick_strengths.append(params["kick_strength"])

    def _median(values: list[float]) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        n = len(s)
        if n % 2:
            return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2.0

    return {
        "approach": {
            "target_distance_m": round(_median(target_distances) or 0.32, 2),
            "max_speed_mps": round(_median(approach_speeds) or 0.25, 2),
            "heading_tolerance_deg": 8,
        },
        "kick": {
            "leg": "right",
            "strength": round(_median(kick_strengths) or 0.45, 2),
            "swing_duration_s": 0.28,
        },
        "stability": {
            "stabilize_duration_s": 0.45,
        },
        "retry": {
            "max_attempts": 2,
        },
    }


def _failure_recovery_patterns(episodes: list[PracticeEpisode]) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    failure_events = []
    for ep in episodes:
        if ep.outcome not in ("failure", "failure_recovered"):
            continue
        for ev in ep.events:
            if ev.event_type == "failure_event":
                failure_events.append(ev.payload)
    if not failure_events:
        return patterns
    # Aggregate by failure type.
    by_type: dict[str, list[dict[str, Any]]] = {}
    for payload in failure_events:
        key = payload.get("type", "unknown")
        by_type.setdefault(key, []).append(payload)
    for key, items in by_type.items():
        patterns.append(
            {
                "failure": key,
                "evidence_count": len(items),
                "patch": {},
                "retry_success_rate": 0.0,
            }
        )
    return patterns


def mine_skill_candidate(
    pkg: SkillPackage,
    source_dir: Path,
    candidate_id: str | None = None,
) -> MiningReport:
    if pkg.dojo is None:
        raise RuntimeError("dojo.yaml not loaded")
    query = PracticeQuery(
        task=pkg.dojo.practice_sources.default_query.get("task", pkg.name),
        robot=pkg.dojo.practice_sources.default_query.get("robot")
        or pkg.eurdf_compat.compatible_robots[0].robot
        if pkg.eurdf_compat and pkg.eurdf_compat.compatible_robots
        else None,
        min_episodes=pkg.dojo.mining.min_episodes,
        include_failures=pkg.dojo.mining.include_failure_recovery,
    )

    episodes = load_episodes(source_dir, query)
    if candidate_id is None:
        candidate_id = _next_candidate_id(pkg)
    candidate_id = validate_candidate_id(candidate_id)

    success_episodes = [e for e in episodes if e.outcome == "success"]
    failure_episodes = [e for e in episodes if e.outcome in ("failure", "failure_recovered")]

    params = _extract_params(success_episodes)
    recovery_patterns = _failure_recovery_patterns(episodes)

    params_yaml = {
        "source": {"mined_from": [e.episode_id for e in success_episodes[:5]]},
        "params": params,
        "confidence": dict.fromkeys(("approach", "kick"), 0.75),
        "failure_recovery_patterns": recovery_patterns,
    }

    params_path = pkg.root / "policies" / "params" / f"{candidate_id}.yaml"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(
        yaml.safe_dump(params_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    # Generate candidate behavior tree.
    bt_path = pkg.root / f"behavior_tree.{candidate_id}.xml"
    default_bt = (pkg.root / "behavior_tree.xml").read_text(encoding="utf-8")
    bt_path.write_text(default_bt, encoding="utf-8")

    # Update lineage.
    if pkg.lineage is None:
        pkg.lineage = LineageYaml()
    pkg.lineage.candidates.append(
        LineageCandidate(
            id=candidate_id,
            source="practice_mining",
            status="candidate",
            eval_report=f"evidence/reports/{candidate_id}_eval.json",
        )
    )
    # Update skill.yaml candidate_id.
    if pkg.skill is not None:
        pkg.skill.metadata.candidate_id = candidate_id
        pkg.skill.metadata.stage = "candidate"
        pkg.skill.status.promotion_state = "candidate"
        pkg.skill.evidence.latest_eval_report = f"evidence/reports/{candidate_id}_eval.json"
    pkg.write_skill_yaml()
    pkg.write_lineage_yaml()

    score = _score_candidate(success_episodes, failure_episodes)
    report = MiningReport(
        candidate_id=candidate_id,
        source_episodes=[e.episode_id for e in episodes],
        score=score,
        metrics={
            "episodes": len(episodes),
            "success": len(success_episodes),
            "failure": len(failure_episodes),
            "recovery_patterns": len(recovery_patterns),
        },
        generated_files=[
            str(params_path.relative_to(pkg.root)),
            str(bt_path.relative_to(pkg.root)),
        ],
    )
    write_mining_report(pkg.root, report)
    return report


def _score_candidate(success: list[Any], failure: list[Any]) -> float:
    total = len(success) + len(failure)
    if total == 0:
        return 0.0
    success_rate = len(success) / total
    # Heuristic score.
    return round(
        0.3 * success_rate
        + 0.25 * success_rate
        + 0.2 * 0.9
        + 0.15 * min(len(failure) / max(total, 1), 1.0)
        + 0.1 * 0.8,
        2,
    )
