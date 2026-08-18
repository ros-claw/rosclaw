"""Event schemas for rosclaw-auto Event Bus integration."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal


@dataclass
class EventEnvelope:
    """Unified event envelope used across ROSClaw modules."""

    event_id: str
    event_type: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    trace_id: str = ""
    run_id: str = ""
    task_id: str = ""
    robot_id: str = ""
    skill_id: str = ""
    source: str = "rosclaw-auto"
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "robot_id": self.robot_id,
            "skill_id": self.skill_id,
            "source": self.source,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EventEnvelope":
        return cls(
            event_id=d["event_id"],
            event_type=d["event_type"],
            timestamp=d.get("timestamp", ""),
            trace_id=d.get("trace_id", ""),
            run_id=d.get("run_id", ""),
            task_id=d.get("task_id", ""),
            robot_id=d.get("robot_id", ""),
            skill_id=d.get("skill_id", ""),
            source=d.get("source", "rosclaw-auto"),
            payload=d.get("payload", {}),
        )


@dataclass
class PraxisFailedEvent(EventEnvelope):
    """Emitted by rosclaw-practice when a skill execution fails."""

    event_type: str = "rosclaw.practice.failed"
    failure_mode: str = ""
    phase: str = ""
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    evidence: dict = field(default_factory=dict)
    praxis_uri: str = ""

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["payload"].update(
            {
                "failure_mode": self.failure_mode,
                "phase": self.phase,
                "severity": self.severity,
                "evidence": self.evidence,
                "praxis_uri": self.praxis_uri,
            }
        )
        return d


@dataclass
class BenchmarkCompletedEvent(EventEnvelope):
    """Emitted by rosclaw-darwin when a benchmark run completes."""

    event_type: str = "rosclaw.darwin.benchmark.completed"
    benchmark_id: str = ""
    task_id: str = ""
    skill_id: str = ""
    metrics: dict = field(default_factory=dict)
    regression_detected: bool = False

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["payload"].update(
            {
                "benchmark_id": self.benchmark_id,
                "task_id": self.task_id,
                "skill_id": self.skill_id,
                "metrics": self.metrics,
                "regression_detected": self.regression_detected,
            }
        )
        return d


@dataclass
class AutoProposalCreatedEvent(EventEnvelope):
    """Published by rosclaw-auto when a new proposal is generated."""

    event_type: str = "rosclaw.auto.proposal.created"
    proposal_id: str = ""
    task_id: str = ""
    target_skill_id: str = ""
    hypothesis_statement: str = ""


@dataclass
class ChampionPromotedEvent(EventEnvelope):
    """Published by rosclaw-auto when a skill is promoted to champion."""

    event_type: str = "rosclaw.auto.champion.promoted"
    champion_id: str = ""
    skill_id: str = ""
    task_id: str = ""
    level: str = ""
    metrics: dict = field(default_factory=dict)


@dataclass
class DeadEndRegisteredEvent(EventEnvelope):
    """Published by rosclaw-auto when a dead-end is registered."""

    event_type: str = "rosclaw.auto.deadend.registered"
    deadend_id: str = ""
    task_id: str = ""
    direction: str = ""
    rejection_reason: str = ""
