"""ROS Connector - MCP tools.

Registers safe ROS MCP tools. We intentionally do NOT expose raw
``publish_once`` or ``call_any_service`` tools to Agents.
"""

from __future__ import annotations

import logging
from typing import Any

from rosclaw.connectors.ros.compiler import (
    CapabilityManifestCompiler,
    SafetyContractCompiler,
)
from rosclaw.connectors.ros.discovery import RosApiResolver, RosGraphDiscovery
from rosclaw.connectors.ros.transport import RosbridgeEndpoint, RosbridgeTransport

logger = logging.getLogger("rosclaw.connectors.ros.mcp.tools")


def register_ros_tools(mcp, runtime: Any | None = None) -> None:
    """Register ROS MCP tools with a FastMCP-like server object.

    The ``mcp`` object must support ``@mcp.tool()`` decorator semantics.
    """

    @mcp.tool(
        description=(
            "Ping a rosbridge endpoint and return connectivity info.\n"
            "Example: ros_ping(endpoint='ws://127.0.0.1:9090')"
        ),
    )
    def ros_ping(endpoint: str = "ws://127.0.0.1:9090") -> dict:
        """Check if rosbridge is reachable and detect ROS version."""
        try:
            ep = RosbridgeEndpoint.from_url(endpoint)
            transport = RosbridgeTransport(endpoint=ep, max_retries=0)
            resolver = RosApiResolver(transport)
            profile = resolver.resolve()
            transport.close()
            return {
                "ok": True,
                "endpoint": endpoint,
                "ros_version": profile.version.value,
                "distro": profile.distro,
                "service_prefix": profile.service_prefix,
            }
        except Exception as exc:
            return {"ok": False, "endpoint": endpoint, "error": str(exc)}

    @mcp.tool(
        description=(
            "Discover the ROS graph at an endpoint and return a snapshot summary.\n"
            "Example: ros_discover(endpoint='ws://127.0.0.1:9090', robot_id='turtlesim')"
        ),
    )
    def ros_discover(endpoint: str = "ws://127.0.0.1:9090", robot_id: str = "unknown") -> dict:
        """Discover topics, services, actions, nodes, and parameters."""
        try:
            ep = RosbridgeEndpoint.from_url(endpoint)
            transport = RosbridgeTransport(endpoint=ep)
            discovery = RosGraphDiscovery(transport)
            snapshot = discovery.discover()
            transport.close()
            return {
                "ok": True,
                "endpoint": snapshot.endpoint,
                "ros_version": snapshot.ros_version,
                "distro": snapshot.distro,
                "topics": [{"name": t.name, "type": t.msg_type, "risk": t.risk_hint} for t in snapshot.topics],
                "services": [{"name": s.name, "type": s.srv_type, "risk": s.risk_hint} for s in snapshot.services],
                "actions": [{"name": a.name, "type": a.action_type, "risk": a.risk_hint} for a in snapshot.actions],
                "nodes": [n["name"] for n in snapshot.nodes],
                "params": snapshot.params,
            }
        except Exception as exc:
            return {"ok": False, "endpoint": endpoint, "error": str(exc)}

    @mcp.tool(
        description=(
            "Compile a discovered ROS graph into a CapabilityManifest.\n"
            "Requires a prior ros_discover result or endpoint.\n"
            "Example: ros_compile_manifest(endpoint='ws://127.0.0.1:9090', robot_id='turtlesim')"
        ),
    )
    def ros_compile_manifest(
        endpoint: str = "ws://127.0.0.1:9090",
        robot_id: str = "unknown",
    ) -> dict:
        """Compile capabilities from the ROS graph."""
        try:
            ep = RosbridgeEndpoint.from_url(endpoint)
            transport = RosbridgeTransport(endpoint=ep)
            discovery = RosGraphDiscovery(transport)
            snapshot = discovery.discover()
            manifest = CapabilityManifestCompiler(robot_id=robot_id).compile(snapshot)
            transport.close()
            return {
                "ok": True,
                "robot_id": robot_id,
                "capabilities": [cap.id for cap in manifest.capabilities],
                "manifest": manifest.to_dict(),
            }
        except Exception as exc:
            return {"ok": False, "endpoint": endpoint, "error": str(exc)}

    @mcp.tool(
        description=(
            "List compiled ROS capabilities for a robot.\n"
            "Example: ros_list_capabilities(robot_id='turtlesim')"
        ),
    )
    def ros_list_capabilities(robot_id: str = "unknown") -> dict:
        """List capabilities available through the registered ROS provider."""
        if runtime is None:
            return {"ok": False, "error": "Runtime not available"}
        registry = getattr(runtime, "provider_registry", None)
        if registry is None:
            return {"ok": False, "error": "Provider registry not available"}
        try:
            provider = registry.get("ros_capability_provider")
            manifest = getattr(provider, "_manifest", None)
            if manifest is None:
                return {"ok": False, "error": "ROS provider manifest not loaded"}
            return {
                "ok": True,
                "robot_id": robot_id,
                "capabilities": [
                    {"id": cap.id, "kind": cap.kind, "risk": cap.risk.level, "enabled": cap.enabled}
                    for cap in manifest.capabilities
                ],
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @mcp.tool(
        description=(
            "Inspect a specific ROS capability.\n"
            "Example: ros_inspect_capability(capability_id='turtlesim.base.velocity_command')"
        ),
    )
    def ros_inspect_capability(capability_id: str) -> dict:
        if runtime is None:
            return {"ok": False, "error": "Runtime not available"}
        registry = getattr(runtime, "provider_registry", None)
        if registry is None:
            return {"ok": False, "error": "Provider registry not available"}
        try:
            provider = registry.get("ros_capability_provider")
            manifest = getattr(provider, "_manifest", None)
            cap = manifest.get_capability(capability_id) if manifest else None
            if cap is None:
                return {"ok": False, "error": f"Capability '{capability_id}' not found"}
            return {"ok": True, "capability": cap.to_dict()}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @mcp.tool(
        description=(
            "Validate capability arguments against the safety contract (dry run).\n"
            "Example: ros_validate_capability(capability_id='turtlesim.base.velocity_command', args={...})"
        ),
    )
    def ros_validate_capability(capability_id: str, args: dict) -> dict:
        if runtime is None:
            return {"ok": False, "error": "Runtime not available"}
        registry = getattr(runtime, "provider_registry", None)
        if registry is None:
            return {"ok": False, "error": "Provider registry not available"}
        try:
            provider = registry.get("ros_capability_provider")
            contract = getattr(provider, "_contract", None)
            if contract is None:
                return {"ok": False, "error": "Safety contract not loaded"}
            compiler = SafetyContractCompiler()
            decision = compiler.evaluate(contract, capability_id, args)
            return {
                "ok": decision.decision in ("ALLOW", "MODIFY"),
                "decision": decision.decision,
                "risk_score": decision.risk_score,
                "reason": decision.reason,
                "violated_constraints": decision.violated_constraints,
                "modified_args": decision.modified_args,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @mcp.tool(
        description=(
            "Execute a ROS capability through the safety-gated provider.\n"
            "Example: ros_execute_capability(capability_id='turtlesim.base.velocity_command', args={...})"
        ),
    )
    def ros_execute_capability(
        capability_id: str,
        args: dict,
        dry_run: bool = False,
    ) -> dict:
        if runtime is None:
            return {"ok": False, "error": "Runtime not available"}
        registry = getattr(runtime, "provider_registry", None)
        if registry is None:
            return {"ok": False, "error": "Provider registry not available"}
        try:
            import asyncio

            from rosclaw.provider.core.request import ProviderRequest

            provider = registry.get("ros_capability_provider")
            request = ProviderRequest(
                request_id=f"mcp_ros_{capability_id}",
                capability=capability_id,
                inputs=args,
                context={"dry_run": dry_run},
            )
            response = asyncio.run(provider.infer(request))
            return {
                "ok": response.is_ok,
                "status": response.status,
                "result": response.result,
                "errors": response.errors,
                "latency_ms": response.latency_ms,
                "trace": response.trace,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @mcp.tool(
        description=(
            "Emergency stop a robot by sending zero velocity and disabling active commands.\n"
            "Example: ros_emergency_stop(robot_id='turtlesim')"
        ),
    )
    def ros_emergency_stop(robot_id: str = "unknown") -> dict:
        if runtime is None:
            return {"ok": False, "error": "Runtime not available"}
        try:
            from rosclaw.core.event_bus import Event
            event_bus = getattr(runtime, "event_bus", None)
            if event_bus is not None:
                event_bus.publish(Event(
                    topic="robot.emergency_stop",
                    payload={"reason": f"MCP emergency stop for {robot_id}"},
                    source="ros_mcp_tools",
                ))
            return {"ok": True, "robot_id": robot_id, "action": "emergency_stop_triggered"}
        except Exception as exc:
            return {"ok": False, "robot_id": robot_id, "error": str(exc)}
