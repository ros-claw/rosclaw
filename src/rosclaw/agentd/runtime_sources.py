"""Production source adapters for the ContextCompiler (PR-NA-020 wiring).

SIM sources are honest: the sim body summary says it is simulated, the sim
self source increments its sequence per read. If a real body is linked via
``rosclaw.body.BodyResolver``, ``ResolverBodySource`` exposes it (fail
closed on any resolver error).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rosclaw.agentd.context.sources import (
    BodyFacts,
    CapabilityInfo,
    ConsentFacts,
    MemoryItem,
    OrgFacts,
    SelfFacts,
)
from rosclaw.contracts.common import content_hash


class SimBodySource:
    """Deterministic simulated body, explicitly marked as simulation."""

    def __init__(self, body_id: str, summary: str | None = None) -> None:
        self._body_id = body_id
        self._summary = summary or (
            f"SIMULATED body {body_id} (evidence_class=simulated; "
            "not usable as REAL physical proof)"
        )
        self._hash = content_hash("body", {"body_id": body_id, "kind": "sim", "v": 1})

    def get_body(self, body_id: str) -> BodyFacts | None:
        if body_id != self._body_id:
            return None
        return BodyFacts(
            body_id=self._body_id,
            effective_body_hash=self._hash,
            summary=self._summary,
            calibrated=True,
        )

    @property
    def body_hash(self) -> str:
        return self._hash


class ResolverBodySource:
    """Real body via rosclaw.body.BodyResolver. Fail closed on errors."""

    def __init__(self, *, workspace: Path | None = None, body_id: str | None = None) -> None:
        self._resolver = None
        self._body_id = body_id
        try:
            from rosclaw.body.resolver import BodyResolver

            self._resolver = BodyResolver(workspace=workspace, body_id=body_id)
        except Exception:  # noqa: BLE001 - absence means "no real body"
            self._resolver = None

    def get_body(self, body_id: str) -> BodyFacts | None:
        if self._resolver is None:
            return None
        if self._body_id is not None and body_id != self._body_id:
            return None
        try:
            effective = self._resolver.get_effective_body()
            body_hash = effective.compute_hash()
            if effective.effective_body_hash and effective.effective_body_hash != body_hash:
                return None
            summary = f"EffectiveBody {body_id} (hash {body_hash[:18]}…)"
            return BodyFacts(
                body_id=body_id,
                effective_body_hash=body_hash,
                summary=summary,
                calibrated=True,
            )
        except Exception:  # noqa: BLE001 - resolver failure must fail closed
            return None


class DaemonSelfSource:
    """Fresh control-plane Self facts for a real body.

    This adapter deliberately reports only daemon-observed runtime health. It
    does not fabricate pose, sensor, or task evidence when those observations
    are unavailable.
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self._sequence = 0

    def get_self(self, body_id: str) -> SelfFacts | None:
        try:
            status = self._client.get_runtime_status()
        except Exception:  # noqa: BLE001 - daemon loss fails context compilation closed
            return None
        self._sequence += 1
        running = bool(status.get("running")) and status.get("runtime_state") == "RUNNING"
        recovery = bool((status.get("recovery") or {}).get("required"))
        estop = bool(status.get("emergency_stop_latched"))
        health = "OK" if running and not recovery and not estop else "DEGRADED"
        public = {
            "body_id": body_id,
            "daemon_instance_id": status.get("daemon_instance_id"),
            "runtime_state": status.get("runtime_state"),
            "supervision_state": status.get("supervision_state"),
            "emergency_stop_latched": estop,
            "recovery_required": recovery,
            "robot_pack": (status.get("robot_pack") or {}).get("pack_ref"),
            "registered_executors": status.get("registered_executors") or [],
            "sequence": self._sequence,
        }
        return SelfFacts(
            self_snapshot_hash=content_hash("selfsnap", public),
            sequence=self._sequence,
            observed_at=datetime.now(UTC),
            health=health,
            summary=(
                f"rosclawd runtime={public['runtime_state']} "
                f"supervision={public['supervision_state']} health={health}; "
                "pose and sensor evidence not asserted by this source"
            ),
        )


class SimSelfSource:
    def __init__(self) -> None:
        self._sequence = 0

    def get_self(self, body_id: str) -> SelfFacts | None:
        self._sequence += 1
        return SelfFacts(
            self_snapshot_hash=content_hash(
                "selfsnap", {"body_id": body_id, "seq": self._sequence}
            ),
            sequence=self._sequence,
            observed_at=datetime.now(UTC),
            health="OK",
            summary=f"SIMULATED self state seq={self._sequence} health=OK",
        )


class StaticCapabilitySource:
    def __init__(self, names: list[str]) -> None:
        self._names = sorted(names)

    def list_capabilities(self, query: str, limit: int) -> list[CapabilityInfo]:
        return [
            CapabilityInfo(name=n, kind="tool", summary=f"builtin tool {n}")
            for n in self._names[: limit * 2]
        ]


class CatalogCapabilitySource:
    """Capability layer backed by the PR-05 ToolCatalog (dynamic, honest).

    PHYSICAL_ACTION tools appear with permission="operator_only" so the model
    knows the capability exists but can never call it directly — it must go
    through REQUEST_APPROVAL → Operator grant → REQUEST_ACTION.
    """

    def __init__(self, catalog) -> None:  # ToolCatalog (avoid import cycle)
        self._catalog = catalog

    def list_capabilities(self, query: str, limit: int) -> list[CapabilityInfo]:
        from rosclaw.contracts.agent.tool import ExecutionClass

        infos: list[CapabilityInfo] = []
        for d in self._catalog.list()[: limit * 2]:
            if d.execution_class is ExecutionClass.PHYSICAL_ACTION:
                infos.append(
                    CapabilityInfo(
                        name=d.tool_id,
                        kind="physical_action",
                        summary=(
                            f"{d.description or d.tool_id} [PHYSICAL ACTION — never "
                            "directly callable; requires Operator exact-action grant]"
                        ),
                        permission="operator_only",
                    )
                )
            else:
                quarantined = self._catalog.quarantine_reason(d.tool_id) is not None
                infos.append(
                    CapabilityInfo(
                        name=d.tool_id,
                        kind="tool",
                        summary=d.description or f"tool {d.tool_id}",
                        permission="denied" if quarantined else "granted",
                    )
                )
        return infos


class EmptyMemorySource:
    def retrieve(self, query: str, limit: int) -> list[MemoryItem]:
        return []


class NullOrgSource:
    def get_org(self) -> OrgFacts:
        return OrgFacts()


class ConfigConsentSource:
    """Public consent facts from config. Grants arrive with Operator Broker."""

    def __init__(self, allowed_risk_tiers: tuple[str, ...] = ("LOW",)) -> None:
        self._tiers = allowed_risk_tiers
        self._policy_hash = content_hash(
            "pol", {"policy": "default_sim_only", "tiers": list(allowed_risk_tiers)}
        )

    @property
    def policy_hash(self) -> str:
        return self._policy_hash

    def get_consent(self, mission_id: str) -> ConsentFacts | None:
        return ConsentFacts(
            policy_hash=self._policy_hash,
            public_scope_summary=(
                "default policy: SIMULATION only, EXACT_ACTION authorization, "
                f"allowed_risk_tiers={list(self._tiers)}"
            ),
            allowed_risk_tiers=self._tiers,
        )
