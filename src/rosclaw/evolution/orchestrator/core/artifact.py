"""Artifact — 进化报告和冠军卡片."""

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class EvolutionReport:
    id: str
    task_id: str
    summary: str = ""
    proposals_created: int = 0
    experiments_run: int = 0
    champions_promoted: int = 0
    deadends_registered: int = 0
    baseline_metrics: dict = field(default_factory=dict)
    best_candidate_metrics: dict = field(default_factory=dict)
    improvement_delta: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "summary": self.summary,
            "proposals_created": self.proposals_created,
            "experiments_run": self.experiments_run,
            "champions_promoted": self.champions_promoted,
            "deadends_registered": self.deadends_registered,
            "baseline_metrics": self.baseline_metrics,
            "best_candidate_metrics": self.best_candidate_metrics,
            "improvement_delta": self.improvement_delta,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvolutionReport":
        return cls(
            id=d["id"],
            task_id=d["task_id"],
            summary=d.get("summary", ""),
            proposals_created=d.get("proposals_created", 0),
            experiments_run=d.get("experiments_run", 0),
            champions_promoted=d.get("champions_promoted", 0),
            deadends_registered=d.get("deadends_registered", 0),
            baseline_metrics=d.get("baseline_metrics", {}),
            best_candidate_metrics=d.get("best_candidate_metrics", {}),
            improvement_delta=d.get("improvement_delta", {}),
            created_at=d.get("created_at", ""),
        )


@dataclass
class ChampionCard:
    skill_id: str
    previous_champion: str = ""
    promotion_date: str = ""
    improvement: dict = field(default_factory=dict)
    accepted_changes: list[dict] = field(default_factory=list)
    validated_on: list[str] = field(default_factory=list)
    known_limitations: list[str] = field(default_factory=list)
    rollback_to: str = ""

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "previous_champion": self.previous_champion,
            "promotion_date": self.promotion_date,
            "improvement": self.improvement,
            "accepted_changes": self.accepted_changes,
            "validated_on": self.validated_on,
            "known_limitations": self.known_limitations,
            "rollback_to": self.rollback_to,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChampionCard":
        return cls(
            skill_id=d["skill_id"],
            previous_champion=d.get("previous_champion", ""),
            promotion_date=d.get("promotion_date", ""),
            improvement=d.get("improvement", {}),
            accepted_changes=d.get("accepted_changes", []),
            validated_on=d.get("validated_on", []),
            known_limitations=d.get("known_limitations", []),
            rollback_to=d.get("rollback_to", ""),
        )

    def to_markdown(self) -> str:
        lines = [
            "# Champion Skill Card",
            "",
            f"Skill: {self.skill_id}",
            f"Previous Champion: {self.previous_champion}",
            f"Promotion Date: {self.promotion_date}",
            "",
            "## Improvement",
        ]
        for k, v in self.improvement.items():
            lines.append(f"- {k}: {v}")
        lines.extend(["", "## Accepted Changes"])
        for change in self.accepted_changes:
            lines.append(f"- {change}")
        lines.extend(["", "## Validated On"])
        for v in self.validated_on:
            lines.append(f"- {v}")
        lines.extend(["", "## Known Limitations"])
        for lim in self.known_limitations:
            lines.append(f"- {lim}")
        lines.extend(["", "## Rollback", f"rollback_to: {self.rollback_to}"])
        return "\n".join(lines)
