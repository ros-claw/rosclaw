"""ROSClaw Event Topics — Standardized EventBus topic namespace.

All EventBus topics used across ROSClaw are defined here as constants.
This ensures:
  1. No hardcoded topic strings scattered across modules
  2. Clear topic hierarchy: rosclaw.<module>.<event>
  3. Easy refactoring and discovery
  4. IDE autocomplete support

Migration guide:
  OLD                    → NEW (v1.0 standard)
  agent.command          → rosclaw.agent.command
  skill.execution.start  → rosclaw.skill.execution.start
  skill.execution.complete → rosclaw.skill.execution.complete
  praxis.completed       → rosclaw.praxis.completed
  praxis.failed          → rosclaw.praxis.failed
  praxis.recorded        → rosclaw.praxis.recorded
  firewall.action_blocked → rosclaw.sandbox.action.blocked
  safety.violation       → rosclaw.safety.violation
  agent.response         → rosclaw.agent.response
  agent.capability.request → rosclaw.agent.capability.request
  robot.emergency_stop   → rosclaw.robot.emergency_stop
  memory.experience.stored → rosclaw.memory.experience.stored
  rosclaw.memory.write.completed → rosclaw.memory.write.completed
  heuristic.recovery_suggested → rosclaw.how.recovery_hint.generated
  heuristic.recovery_executed → rosclaw.how.recovery_executed
  runtime.status         → rosclaw.runtime.status
"""


class EventTopics:
    """Canonical EventBus topic constants for ROSClaw v1.0."""

    # ── Runtime Lifecycle ──
    RUNTIME_STARTED = "rosclaw.runtime.started"
    RUNTIME_STOPPED = "rosclaw.runtime.stopped"
    RUNTIME_STATUS = "rosclaw.runtime.status"

    # ── Agent ──
    AGENT_COMMAND = "rosclaw.agent.command"
    AGENT_RESPONSE = "rosclaw.agent.response"
    AGENT_CAPABILITY_REQUEST = "rosclaw.agent.capability.request"
    AGENT_DECISION_MADE = "rosclaw.agent.decision.made"

    # ── Skill Execution ──
    SKILL_EXECUTION_START = "rosclaw.skill.execution.start"
    SKILL_EXECUTION_COMPLETE = "rosclaw.skill.execution.complete"

    # ── Provider ──
    PROVIDER_INFERENCE_REQUESTED = "rosclaw.provider.inference.requested"
    PROVIDER_INFERENCE_COMPLETED = "rosclaw.provider.inference.completed"
    PROVIDER_REGISTERED = "rosclaw.provider.registered"
    PROVIDER_HEALTH_CHANGED = "rosclaw.provider.health_changed"

    # ── Unified Trace ──
    TRACE_SPAN_STARTED = "rosclaw.trace.span.started"
    TRACE_SPAN_COMPLETED = "rosclaw.trace.span.completed"
    TRACE_SPAN_FAILED = "rosclaw.trace.span.failed"

    # ── Sandbox / Firewall ──
    SANDBOX_EPISODE_STARTED = "rosclaw.sandbox.episode.started"
    SANDBOX_EPISODE_FINISHED = "rosclaw.sandbox.episode.finished"
    SANDBOX_ACTION_ALLOWED = "rosclaw.sandbox.action.allowed"
    SANDBOX_ACTION_BLOCKED = "rosclaw.sandbox.action.blocked"

    # ── Practice / Praxis ──
    PRAXIS_COMPLETED = "rosclaw.praxis.completed"
    PRAXIS_FAILED = "rosclaw.praxis.failed"
    PRAXIS_RECORDED = "rosclaw.praxis.recorded"
    PRACTICE_EVENT_CREATED = "rosclaw.practice.event.created"

    # ── Safety ──
    SAFETY_VIOLATION = "rosclaw.safety.violation"
    ROBOT_EMERGENCY_STOP = "rosclaw.robot.emergency_stop"

    # ── Memory ──
    MEMORY_EXPERIENCE_STORED = "rosclaw.memory.experience.stored"
    MEMORY_WRITE_COMPLETED = "rosclaw.memory.write.completed"

    # ── How / Recovery ──
    HOW_RECOVERY_HINT_GENERATED = "rosclaw.how.recovery_hint.generated"
    HOW_RECOVERY_EXECUTED = "rosclaw.how.recovery_executed"

    # ── Critic ──
    CRITIC_SUCCESS_DETECTED = "rosclaw.critic.success.detected"
    CRITIC_JUDGMENT = "rosclaw.critic.judgment"

    # ── Dashboard ──
    DASHBOARD_TRACE_UPDATED = "rosclaw.dashboard.trace.updated"

    # ── Auto / Research loop ──
    AUTO_PROPOSAL_CREATED = "rosclaw.auto.proposal.created"
    AUTO_CHAMPION_PROMOTED = "rosclaw.auto.champion.promoted"
    AUTO_EXPERIMENT_COMPLETED = "rosclaw.auto.experiment.completed"
    AUTO_DEADEND_REGISTERED = "rosclaw.auto.deadend.registered"

    # ── How evidence ──
    HOW_EVIDENCE_GENERATED = "rosclaw.how.evidence.generated"

    # ── Telemetry ──
    ROBOT_TELEMETRY = "rosclaw.robot.telemetry"
    ROBOT_JOINT_STATES = "rosclaw.robot.joint_states"

    # ── Sense / BodySense ──
    SENSE_STATE_UPDATED = "rosclaw.sense.state.updated"
    SENSE_BODY_UPDATED = "rosclaw.sense.body.updated"
    SENSE_EVENT_DETECTED = "rosclaw.sense.event.detected"
    SENSE_READINESS_UPDATED = "rosclaw.sense.readiness.updated"
    SENSE_CAPABILITY_BLOCKED = "rosclaw.sense.capability.blocked"
    SENSE_CAPABILITY_DEGRADED = "rosclaw.sense.capability.degraded"


# ── Backward compatibility mapping ──
# OLD_TOPIC → NEW_TOPIC
_TOPIC_COMPAT: dict[str, str] = {
    "agent.command": EventTopics.AGENT_COMMAND,
    "skill.execution.start": EventTopics.SKILL_EXECUTION_START,
    "skill.execution.complete": EventTopics.SKILL_EXECUTION_COMPLETE,
    "praxis.completed": EventTopics.PRAXIS_COMPLETED,
    "praxis.failed": EventTopics.PRAXIS_FAILED,
    "praxis.recorded": EventTopics.PRAXIS_RECORDED,
    "firewall.action_blocked": EventTopics.SANDBOX_ACTION_BLOCKED,
    "safety.violation": EventTopics.SAFETY_VIOLATION,
    "agent.response": EventTopics.AGENT_RESPONSE,
    "agent.capability.request": EventTopics.AGENT_CAPABILITY_REQUEST,
    "robot.emergency_stop": EventTopics.ROBOT_EMERGENCY_STOP,
    "memory.experience.stored": EventTopics.MEMORY_EXPERIENCE_STORED,
    "rosclaw.memory.write.completed": EventTopics.MEMORY_WRITE_COMPLETED,
    "heuristic.recovery_suggested": EventTopics.HOW_RECOVERY_HINT_GENERATED,
    "heuristic.recovery_executed": EventTopics.HOW_RECOVERY_EXECUTED,
    "runtime.status": EventTopics.RUNTIME_STATUS,
}


def normalize_topic(topic: str) -> str:
    """Normalize a topic string to the v1.0 standard namespace.

    If the topic is an old/legacy name, returns the new canonical name.
    If already canonical (starts with ``rosclaw.``), returns as-is.
    Unknown topics are returned unchanged.

    Example:
        >>> normalize_topic("agent.command")
        'rosclaw.agent.command'
        >>> normalize_topic("rosclaw.praxis.completed")
        'rosclaw.praxis.completed'
        >>> normalize_topic("custom.topic")
        'custom.topic'
    """
    if topic.startswith("rosclaw."):
        return topic
    return _TOPIC_COMPAT.get(topic, topic)


def list_deprecated_topics() -> list[str]:
    """Return all deprecated topic names that have canonical replacements."""
    return list(_TOPIC_COMPAT.keys())
