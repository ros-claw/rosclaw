"""ROS Connector - ROS Capability Provider.

Provider implementation that integrates with ROSClaw's ProviderRegistry and
CapabilityRouter. It exposes discovered ROS capabilities as ROSClaw capabilities
and routes execution through safety checks and stop guards.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rosclaw.connectors.ros.compiler import (
    CapabilityManifest,
    CapabilityManifestCompiler,
    RosCapability,
    RosCapabilityRisk,
    RosInterface,
    SafetyContract,
    SafetyContractCompiler,
    SandboxDecision,
)
from rosclaw.connectors.ros.discovery import RosGraphDiscovery
from rosclaw.connectors.ros.discovery.graph import RosGraphSnapshot
from rosclaw.connectors.ros.transport import (
    MockTransport,
    RosbridgeEndpoint,
    RosbridgeTransport,
    RosTransportResult,
)
from rosclaw.provider.core.errors import CapabilityNotSupportedError, GuardBlockedError
from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.provider import Provider
from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse

try:
    from rosclaw.core.event_bus import Event, EventPriority
except ImportError:
    Event = None
    EventPriority = None

logger = logging.getLogger("rosclaw.connectors.ros.provider.ros_capability_provider")


@dataclass
class RosCapabilityResult:
    """Result of executing a ROS capability."""

    ok: bool
    capability_id: str
    ros_interface: str
    ros_kind: str
    started_at: str
    ended_at: str
    duration_ms: float
    result: dict[str, Any] | None = None
    error: str | None = None
    sandbox_decision: SandboxDecision | None = None
    practice_trace_id: str | None = None
    safety_events: list[dict] | None = None
    raw_ros_response: dict[str, Any] | None = None

    def __post_init__(self):
        if self.safety_events is None:
            self.safety_events = []


class RosCapabilityProvider(Provider):
    """ROSClaw provider that discovers and executes ROS capabilities safely."""

    name = "ros_capability_provider"
    version = "0.1.0"
    capabilities: list[str] = []

    def __init__(self, manifest: ProviderManifest):
        super().__init__(manifest)
        self._transport: Any | None = None
        self._manifest: CapabilityManifest | None = None
        self._contract: SafetyContract | None = None
        self._robot_id = manifest.extra.get("robot_id", "unknown")
        runtime = manifest.runtime
        if isinstance(runtime, dict):
            self._endpoint_url = runtime.get("endpoint") or "ws://127.0.0.1:9090"
        else:
            self._endpoint_url = getattr(runtime, "endpoint", None) or "ws://127.0.0.1:9090"
        self._robot_spec_path = manifest.extra.get("robot_spec_path")
        self._auto_discover = manifest.extra.get("auto_discover", True)
        self._dry_run = manifest.extra.get("dry_run", False)
        self._event_bus = manifest.extra.get("event_bus")

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------
    def _publish_event(self, topic: str, payload: dict[str, Any], priority: Any | None = None) -> None:
        """Publish an event to the EventBus if available."""
        if self._event_bus is None or Event is None:
            return
        try:
            self._event_bus.publish(Event(
                topic=topic,
                payload=payload,
                source="ros_capability_provider",
                priority=priority or (EventPriority.NORMAL if EventPriority else None),
            ))
        except Exception as exc:
            logger.debug("Failed to publish event %s: %s", topic, exc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def load(self) -> None:
        self._transport = self._create_transport()
        robot_spec = self._load_robot_spec()

        if self._auto_discover and not self._dry_run:
            try:
                discovery = RosGraphDiscovery(self._transport)
                snapshot = discovery.discover()
                self._manifest = CapabilityManifestCompiler(
                    robot_id=self._robot_id,
                    robot_spec=robot_spec,
                ).compile(snapshot)
                self._contract = SafetyContractCompiler().compile(self._manifest)
                self.capabilities = [cap.id for cap in self._manifest.capabilities]
            except Exception as exc:
                logger.warning("ROS auto-discovery failed: %s", exc)
                # Fall back to empty manifest so provider still registers.
                self._manifest = CapabilityManifest(robot_id=self._robot_id)
                self._contract = SafetyContract(robot_id=self._robot_id)
        else:
            # Use static manifest if provided, else empty.
            manifest_path = self.manifest.extra.get("manifest_path")
            if manifest_path and Path(manifest_path).exists():
                self._manifest = self._load_static_manifest(manifest_path)
                self._contract = SafetyContractCompiler().compile(self._manifest)
                self.capabilities = [cap.id for cap in self._manifest.capabilities]
            else:
                self._manifest = CapabilityManifest(robot_id=self._robot_id)
                self._contract = SafetyContract(robot_id=self._robot_id)

        # Optional Phase 9 integrations: seed capabilities into KNOW and
        # ROS-specific recovery rules into HOW.
        self._seed_ros_knowledge()
        self._seed_ros_recovery_rules()

    def _seed_ros_knowledge(self) -> None:
        """Publish discovered capabilities to the knowledge graph."""
        knowledge_interface = self.manifest.extra.get("knowledge_interface")
        if knowledge_interface is None or not self.capabilities:
            return
        try:
            from rosclaw.connectors.ros.know.ros_knowledge_seed import seed_ros_capabilities
            seed_ros_capabilities(knowledge_interface, self._robot_id, self.capabilities)
        except Exception as exc:
            logger.debug("ROS knowledge seeding failed: %s", exc)

    def _seed_ros_recovery_rules(self) -> None:
        """Seed ROS-specific recovery rules into the heuristic engine."""
        seekdb_client = self.manifest.extra.get("seekdb_client")
        if seekdb_client is None:
            return
        try:
            from rosclaw.connectors.ros.how.ros_recovery_rules import seed_ros_recovery_rules
            seed_ros_recovery_rules(seekdb_client)
        except Exception as exc:
            logger.debug("ROS recovery rule seeding failed: %s", exc)

    async def unload(self) -> None:
        if self._transport is not None:
            import contextlib

            with contextlib.suppress(Exception):
                self._transport.close()
            self._transport = None

    async def health(self) -> dict[str, Any]:
        healthy = False
        error = ""
        if self._transport is not None:
            try:
                result = self._transport.call_service("/rosapi/topics", {})
                healthy = result.ok
            except Exception as exc:
                error = str(exc)
        else:
            error = "transport not initialized"
        return {
            "ok": healthy,
            "provider": self.name,
            "version": self.version,
            "capabilities": self.capabilities,
            "load_error": error,
        }

    # ------------------------------------------------------------------
    # Provider infer
    # ------------------------------------------------------------------
    async def infer(self, request: ProviderRequest) -> ProviderResponse:
        capability_id = request.capability
        args = dict(request.inputs)
        context = dict(request.context)
        dry_run = context.get("dry_run", False) or self._dry_run

        started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        t0 = time.monotonic()

        try:
            self._ensure_capability_supported(capability_id)
            result = self._execute_capability(
                capability_id=capability_id,
                args=args,
                dry_run=dry_run,
                context=context,
            )
        except GuardBlockedError as exc:
            return ProviderResponse(
                request_id=request.request_id,
                provider=self.name,
                capability=capability_id,
                status="blocked",
                errors=[str(exc)],
                latency_ms=int((time.monotonic() - t0) * 1000),
                trace={"guard_checks": exc.checks, "recommended_action": exc.recommended_action},
            )
        except CapabilityNotSupportedError as exc:
            return ProviderResponse(
                request_id=request.request_id,
                provider=self.name,
                capability=capability_id,
                status="failed",
                errors=[str(exc)],
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
        except Exception as exc:
            return ProviderResponse(
                request_id=request.request_id,
                provider=self.name,
                capability=capability_id,
                status="failed",
                errors=[str(exc)],
                latency_ms=int((time.monotonic() - t0) * 1000),
            )

        latency_ms = int((time.monotonic() - t0) * 1000)
        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability=capability_id,
            status="ok" if result.ok else "failed",
            result=result.result or {},
            errors=[result.error] if result.error else [],
            latency_ms=latency_ms,
            trace={
                "ros_interface": result.ros_interface,
                "ros_kind": result.ros_kind,
                "sandbox_decision": result.sandbox_decision.to_dict() if result.sandbox_decision else None,
                "practice_trace_id": result.practice_trace_id,
                "started_at": started_at,
                "ended_at": result.ended_at,
            },
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def _execute_capability(
        self,
        capability_id: str,
        args: dict[str, Any],
        dry_run: bool,
        context: dict[str, Any],
    ) -> RosCapabilityResult:
        started_at = datetime.now(UTC).isoformat()
        t0 = time.monotonic()

        if self._manifest is None or self._contract is None:
            return RosCapabilityResult(
                ok=False,
                capability_id=capability_id,
                ros_interface="",
                ros_kind="",
                started_at=started_at,
                ended_at=datetime.now(UTC).isoformat(),
                duration_ms=0.0,
                error="Manifest or safety contract not loaded",
            )

        cap = self._manifest.get_capability(capability_id)
        if cap is None:
            raise CapabilityNotSupportedError(
                f"Capability '{capability_id}' not found in ROS manifest",
                provider=self.name,
            )

        trace_id = f"ros_{uuid.uuid4().hex[:12]}"

        # 1. Safety evaluation.
        compiler = SafetyContractCompiler()
        decision = compiler.evaluate(self._contract, capability_id, args)

        if decision.decision == "BLOCK":
            self._emit_practice_event(
                capability_id=capability_id,
                cap=cap,
                args=args,
                decision=decision,
                success=False,
                error="Blocked by safety contract",
                trace_id=trace_id,
            )
            self._publish_event(
                "firewall.action_blocked",
                {
                    "episode_id": trace_id,
                    "request_id": trace_id,
                    "capability_id": capability_id,
                    "robot_id": self._robot_id,
                    "reason": f"Safety contract blocked {capability_id}",
                    "violations": [{"description": v} for v in decision.violated_constraints],
                },
                priority=EventPriority.HIGH if EventPriority else None,
            )
            raise GuardBlockedError(
                message=f"ROS capability '{capability_id}' blocked by safety contract",
                provider=self.name,
                checks=[{"check": v} for v in decision.violated_constraints],
                recommended_action="replan",
            )

        if decision.decision == "MODIFY" and decision.modified_args is not None:
            args = decision.modified_args

        # 2. Execute against ROS transport.
        raw_response: RosTransportResult | None = None
        exec_error: str | None = None

        try:
            if dry_run:
                raw_response = RosTransportResult(ok=True, data={"dry_run": True})
            else:
                raw_response = self._invoke_ros_interface(cap, args)
            ok = raw_response.ok if raw_response else False
            if raw_response and raw_response.error:
                exec_error = raw_response.error
                ok = False
        except Exception as exc:
            ok = False
            exec_error = str(exc)
            raw_response = None

        ended_at = datetime.now(UTC).isoformat()
        duration_ms = (time.monotonic() - t0) * 1000

        result_data: dict[str, Any] = {
            "capability_id": capability_id,
            "args": args,
            "ok": ok,
        }
        if raw_response and raw_response.data:
            result_data["ros_response"] = raw_response.data

        # 3. Stop guard for velocity commands: send zero command after duration.
        stop_guard_triggered = False
        if ok and cap.risk.requires_stop_guard and not dry_run:
            stop_guard_triggered = self._run_stop_guard(cap)
            result_data["stop_guard_triggered"] = stop_guard_triggered

        # 4. Practice capture.
        success = ok and exec_error is None
        self._emit_practice_event(
            capability_id=capability_id,
            cap=cap,
            args=args,
            decision=decision,
            success=success,
            error=exec_error,
            trace_id=trace_id,
            raw_response=raw_response.data if raw_response else None,
        )

        # 5. Memory / KNOW / HOW integration events.
        self._publish_event(
            "rosclaw.runtime.execution.completed",
            {
                "episode_id": trace_id,
                "request_id": trace_id,
                "robot_id": self._robot_id,
                "capability_id": capability_id,
                "ros_name": cap.interface.name,
                "ros_kind": cap.interface.ros_kind,
                "status": "ok" if success else "failed",
                "duration_ms": duration_ms,
                "result": result_data,
                "error": exec_error,
            },
            priority=EventPriority.NORMAL if EventPriority else None,
        )
        if not success:
            self._publish_event(
                "rosclaw.runtime.execution.failed",
                {
                    "episode_id": trace_id,
                    "request_id": trace_id,
                    "robot_id": self._robot_id,
                    "capability_id": capability_id,
                    "error_type": exec_error or "unknown",
                    "error": exec_error,
                    "context": {"ros_name": cap.interface.name, "ros_kind": cap.interface.ros_kind},
                },
                priority=EventPriority.HIGH if EventPriority else None,
            )
            self._publish_event(
                "rosclaw.how.recovery_hint.requested",
                {
                    "episode_id": trace_id,
                    "request_id": trace_id,
                    "robot_id": self._robot_id,
                    "failure_type": exec_error or "ros_execution_failed",
                    "context": {
                        "capability_id": capability_id,
                        "ros_name": cap.interface.name,
                        "ros_kind": cap.interface.ros_kind,
                        "args": args,
                    },
                },
                priority=EventPriority.HIGH if EventPriority else None,
            )

        return RosCapabilityResult(
            ok=ok and exec_error is None,
            capability_id=capability_id,
            ros_interface=cap.interface.name,
            ros_kind=cap.interface.ros_kind,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            result=result_data,
            error=exec_error,
            sandbox_decision=decision,
            practice_trace_id=trace_id,
            raw_ros_response=raw_response.data if raw_response else None,
        )

    def _invoke_ros_interface(self, cap: RosCapability, args: dict[str, Any]) -> RosTransportResult:
        """Dispatch to transport based on ROS interface kind."""
        if self._transport is None:
            return RosTransportResult(ok=False, error="transport not initialized")

        if cap.interface.ros_kind == "topic":
            return self._transport.publish(cap.interface.name, args)
        if cap.interface.ros_kind == "service":
            return self._transport.call_service(
                service=cap.interface.name,
                args=args,
                service_type=cap.interface.msg_type or None,
            )
        if cap.interface.ros_kind == "action":
            # Actions require topic-level goal publishing via rosbridge.
            goal_topic = f"{cap.interface.name}/goal"
            return self._transport.publish(goal_topic, args)
        return RosTransportResult(ok=False, error=f"Unsupported ros_kind: {cap.interface.ros_kind}")

    def _run_stop_guard(self, cap: RosCapability) -> bool:
        """Send a zero command to a command topic to stop motion."""
        if cap.interface.ros_kind != "topic":
            return False
        if "cmd_vel" in cap.interface.name.lower():
            zero = {
                "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
            }
            result = self._transport.publish(cap.interface.name, zero)
            return result.ok
        return False

    # ------------------------------------------------------------------
    # Discovery helper exposed as a capability
    # ------------------------------------------------------------------
    def discover(self) -> RosGraphSnapshot:
        """Run ROS graph discovery and return the raw snapshot."""
        if self._transport is None:
            self._transport = self._create_transport()
        discovery = RosGraphDiscovery(self._transport)
        return discovery.discover()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _create_transport(self):
        if self._dry_run:
            return MockTransport(dry_run=True)
        endpoint = RosbridgeEndpoint.from_url(self._endpoint_url)
        return RosbridgeTransport(endpoint=endpoint)

    def _load_robot_spec(self) -> dict[str, Any]:
        if not self._robot_spec_path:
            return {}
        path = Path(self._robot_spec_path)
        if not path.exists():
            return {}
        try:
            import yaml
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning("Failed to load robot spec %s: %s", path, exc)
            return {}

    def _load_static_manifest(self, path: str) -> CapabilityManifest:
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # Minimal reconstruction; for full fidelity use dataclass from_dict.
        manifest = CapabilityManifest(
            robot_id=data.get("robot_id", self._robot_id),
            endpoint=data.get("endpoint", {}),
            ros=data.get("ros", {}),
        )
        for cap_data in data.get("capabilities", []):
            manifest.capabilities.append(self._capability_from_dict(cap_data))
        return manifest

    @staticmethod
    def _capability_from_dict(data: dict[str, Any]) -> RosCapability:
        interface_data = data.get("interface", {})
        interface = RosInterface(
            ros_kind=interface_data.get("ros_kind", ""),
            name=interface_data.get("name", ""),
            msg_type=interface_data.get("type", ""),
        )
        risk_data = data.get("risk", {})
        risk = RosCapabilityRisk(
            level=risk_data.get("level", "low"),
            read_only=risk_data.get("read_only", True),
            destructive=risk_data.get("destructive", False),
            requires_sandbox=risk_data.get("requires_sandbox", False),
            requires_runtime_guard=risk_data.get("requires_runtime_guard", False),
            requires_stop_guard=risk_data.get("requires_stop_guard", False),
            max_duration_sec=risk_data.get("max_duration_sec"),
            max_rate_hz=risk_data.get("max_rate_hz"),
        )
        return RosCapability(
            id=data.get("id", ""),
            kind=data.get("kind", ""),
            interface=interface,
            schema=data.get("schema", {}),
            risk=risk,
            safety=data.get("safety", {}),
            practice=data.get("practice", {}),
            preferred=data.get("preferred", True),
            enabled=data.get("enabled", True),
            reason=data.get("reason", ""),
        )

    def _emit_practice_event(
        self,
        capability_id: str,
        cap: RosCapability,
        args: dict[str, Any],
        decision: SandboxDecision,
        success: bool,
        error: str | None,
        trace_id: str | None = None,
        raw_response: dict[str, Any] | None = None,
    ) -> None:
        if self._event_bus is None:
            return
        try:
            from rosclaw.core.event_bus import Event
            topic = "rosclaw.practice.event.created" if success else "rosclaw.sandbox.episode.failed"
            self._event_bus.publish(Event(
                topic=topic,
                payload={
                    "event_type": "RosExecutionSucceededEvent" if success else "RosExecutionFailedEvent",
                    "trace_id": trace_id,
                    "robot_id": self._robot_id,
                    "capability_id": capability_id,
                    "ros_kind": cap.interface.ros_kind,
                    "ros_name": cap.interface.name,
                    "ros_type": cap.interface.msg_type,
                    "args": args,
                    "sandbox_decision": decision.to_dict() if hasattr(decision, "to_dict") else {},
                    "result": {"ok": success, "raw": raw_response},
                    "error": error,
                },
                source="ros_capability_provider",
            ))
        except Exception as exc:
            logger.debug("Failed to emit practice event: %s", exc)
