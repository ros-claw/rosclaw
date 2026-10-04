"""Augment L2/L3; never replace Body truth or grant action permissions."""

from dataclasses import replace
from datetime import UTC, datetime

from rosclaw.agentd.context.sources import CapabilityInfo
from rosclaw.connectors.ros.diagnosis import diagnose
from rosclaw.connectors.ros.intelligence import RosSystemModel
from rosclaw.connectors.ros.resolver import resolve_capabilities
from rosclaw.contracts.common import content_hash


def compile_agent_summary(model: RosSystemModel, *, now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    diagnosis = diagnose(model, now=now)
    caps = resolve_capabilities(model, now=now)
    lines = [
        "ROS runtime observations (not Body truth or authorization)",
        f"snapshot_id={model.snapshot_id} snapshot_hash={model.snapshot_hash}",
        f"captured_at={model.captured_at.isoformat()} generated_at={now.isoformat()} age_ms={model.age_ms(now):.0f}",
        f"environment={model.environment.get('ros_generation')} distro={model.environment.get('distro')}",
        f"health={diagnosis['status']} unknown_checks={','.join(diagnosis['unknown_checks'])}",
    ]
    lines.extend(f"{c['semantic_id']}={c['status']}" for c in caps)
    lines.extend(
        f"issue={i['issue_code']} severity={i['severity']}" for i in diagnosis["issues"][:20]
    )
    lines.append("Physical effects require request_action → rosclawd → verified receipt.")
    return "\n".join(lines)


class RosSelfAugmentingSource:
    def __init__(self, base, model: RosSystemModel, *, now: datetime | None = None):
        self.base, self.model = base, model
        self.now = now

    def get_self(self, body_id):
        facts = self.base.get_self(body_id)
        if facts is None or body_id != self.model.robot_id:
            return facts
        return replace(
            facts,
            self_snapshot_hash=content_hash(
                "selfsnap", [facts.self_snapshot_hash, self.model.snapshot_hash]
            ),
            observed_at=min(facts.observed_at, self.model.captured_at),
            summary=facts.summary + "\n" + compile_agent_summary(self.model, now=self.now),
            health="DEGRADED"
            if diagnose(self.model, now=self.now)["status"] != "HEALTHY"
            else facts.health,
        )


class RosCapabilitySourceAdapter:
    def __init__(self, base, model: RosSystemModel, *, now: datetime | None = None):
        self.base, self.model = base, model
        self.now = now

    def list_capabilities(self, query: str, limit: int) -> list[CapabilityInfo]:
        infos = list(self.base.list_capabilities(query, limit))
        for capability in resolve_capabilities(self.model, now=self.now):
            semantic, status = capability["semantic_id"], capability["status"]
            if status == "MISSING":
                continue
            physical = semantic.startswith(("navigation.", "coverage.execute", "cleaning."))
            infos.append(
                CapabilityInfo(
                    name=semantic,
                    kind="physical_action" if physical else "observation",
                    summary=f"ROS readiness={status}; snapshot={self.model.snapshot_id}; execution via request_action",
                    permission="operator_only"
                    if physical and status == "AVAILABLE"
                    else "unknown"
                    if status == "UNKNOWN"
                    else "denied"
                    if status == "BLOCKED"
                    else "granted",
                    priority=50,
                )
            )
        return infos


def augment_sources(sources, model: RosSystemModel, *, now: datetime | None = None):
    """Opt-in runtime wiring, preserving the frozen context bundle schema."""
    return replace(
        sources,
        self_source=RosSelfAugmentingSource(sources.self_source, model, now=now),
        capabilities=RosCapabilitySourceAdapter(sources.capabilities, model, now=now),
    )
