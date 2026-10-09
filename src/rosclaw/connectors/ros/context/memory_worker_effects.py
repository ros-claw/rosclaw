"""Refuse host process escape routes in opt-in historical Memory experiments.

Native container mounts do not confine operations executed by a host agentd.
Use its already resolved canonical effect; do not create another classifier.
This restriction applies equally to M0/M1/M2 and grants no new authority.
"""

import hashlib
import json

from rosclaw.contracts.agent.capability import EffectClassV1

HOST_OPERATION_READS = frozenset({"rosclaw_process_status", "rosclaw_process_output"})


def _declaration(raw):
    agent = raw.get("agent", {}) if isinstance(raw, dict) else {}
    expert = agent.get("ros_expert", {}) if isinstance(agent, dict) else {}
    return expert.get("memory_intervention") if isinstance(expert, dict) else None


def memory_worker_cache_binding(raw, request):
    """Bind experimental cached replies to exact policy, session and arguments.

    A pre-experiment response or another tool's idempotency key must not expose
    a host process result before the new effect gate runs. No cached response
    is executed again; incompatible replies are refused, never overwritten.
    """
    declaration = _declaration(raw)
    if declaration is None:
        return None
    value = {
        "declaration": declaration,
        "tool_name": request.tool_name,
        "arguments": request.arguments,
        "session": request.pi_session_id,
        "mission": request.mission_id,
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def memory_worker_host_operation_refused(raw, *, tool_name, effect_class):
    """Configured but malformed intervention still cannot opt out of isolation.

    No source retrieval, database access, task admission or operation lookup.
    The caller remains responsible for all existing session/lease/effect checks.
    """
    configured = _declaration(raw) is not None
    return configured and (
        effect_class == EffectClassV1.HOST_PROCESS.value or tool_name in HOST_OPERATION_READS
    )
