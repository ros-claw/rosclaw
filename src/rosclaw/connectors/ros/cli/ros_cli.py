"""ROS Connector - ROS CLI commands.

Safe, human-facing CLI for the ROS bridge. We never expose raw
``publish_once`` / ``call_any_service`` primitives here; execution goes
through the safety-gated ``RosCapabilityProvider``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from rosclaw.connectors.ros.compiler import (
    CapabilityManifest,
    CapabilityManifestCompiler,
    SafetyContractCompiler,
)
from rosclaw.connectors.ros.discovery import RosGraphDiscovery
from rosclaw.connectors.ros.provider import RosCapabilityProvider
from rosclaw.connectors.ros.transport import RosbridgeEndpoint, RosbridgeTransport
from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.request import ProviderRequest


def _load_robot_spec(robot_id: str) -> dict[str, Any] | None:
    """Load an embodiment card YAML for the given robot id if it exists."""
    specs_dir = Path(__file__).parent.parent / "specs"
    for name in (robot_id, robot_id.replace("_", "-")):
        for ext in (".yaml", ".yml"):
            path = specs_dir / f"{name}{ext}"
            if path.exists():
                import yaml

                with open(path, encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
    return None


def _load_json_or_yaml(path: Path) -> dict[str, Any]:
    """Load a JSON or YAML file."""
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text) or {}
    return json.loads(text)


def _load_manifest(path: str | Path) -> CapabilityManifest:
    """Load a CapabilityManifest from disk."""
    data = _load_json_or_yaml(Path(path))
    return CapabilityManifest.from_dict(data)


def _load_graph(path: str | Path) -> Any:
    """Load a RosGraphSnapshot from disk."""
    from rosclaw.connectors.ros.discovery.graph import RosGraphSnapshot

    data = _load_json_or_yaml(Path(path))
    return RosGraphSnapshot.from_dict(data)


def _make_provider(
    endpoint: str,
    robot_id: str,
    dry_run: bool = False,
    auto_discover: bool = True,
) -> RosCapabilityProvider:
    """Build a RosCapabilityProvider from CLI args."""
    robot_spec = _load_robot_spec(robot_id)
    manifest = ProviderManifest(
        name="ros_capability_provider",
        version="0.1.0",
        type="ros",
        runtime={"endpoint": endpoint},
        extra={
            "robot_id": robot_id,
            "dry_run": dry_run,
            "auto_discover": auto_discover,
            "robot_spec_path": None,
        },
    )
    if robot_spec:
        # The provider's _load_robot_spec currently reads from a path.
        # We temporarily write the loaded dict to a sidecar path so the
        # provider can consume it without changing its internal API.
        import tempfile

        import yaml

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as tmp:
            yaml.safe_dump(robot_spec, tmp)
        manifest.extra["robot_spec_path"] = tmp.name

    return RosCapabilityProvider(manifest)


def _maybe_json(args: argparse.Namespace, payload: dict[str, Any]) -> int:
    """Print result as JSON if requested, else pretty-print."""
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, default=str))
    else:
        _pretty_print(payload)
    return 0 if payload.get("ok", True) else 1


def _pretty_print(payload: dict[str, Any]) -> None:
    """Human-readable rendering of a structured ROS tool result."""
    print("=" * 60)
    print(f"ROS Command Result — {payload.get('action', 'ros')}")
    print("=" * 60)
    for key, value in payload.items():
        if key == "ok":
            icon = "✅" if value else "❌"
            print(f"{icon} Status: {'OK' if value else 'FAILED'}")
        elif key in ("error", "reason") and value:
            print(f"  {key.capitalize()}: {value}")
        elif isinstance(value, (dict, list)):
            print(f"  {key}:")
            print(json.dumps(value, indent=4, default=str))
        else:
            print(f"  {key}: {value}")
    print("=" * 60)


# ------------------------------------------------------------------
# Subcommand handlers
# ------------------------------------------------------------------

def cmd_ros_ping(args: argparse.Namespace) -> int:
    """Ping a rosbridge endpoint."""
    endpoint = args.endpoint
    try:
        ep = RosbridgeEndpoint.from_url(endpoint)
        transport = RosbridgeTransport(endpoint=ep, max_retries=0)
        # Minimal connectivity check via rosapi/topics.
        result = transport.call_service("/rosapi/topics", {})
        transport.close()
        return _maybe_json(
            args,
            {
                "ok": result.ok,
                "action": "ping",
                "endpoint": endpoint,
                "message": "rosbridge reachable" if result.ok else "rosbridge unreachable",
                "error": result.error if not result.ok else None,
            },
        )
    except Exception as exc:
        return _maybe_json(
            args,
            {"ok": False, "action": "ping", "endpoint": endpoint, "error": str(exc)},
        )


def cmd_ros_discover(args: argparse.Namespace) -> int:
    """Discover the ROS graph and print a snapshot summary."""
    endpoint = args.endpoint
    robot_id = args.robot_id
    try:
        ep = RosbridgeEndpoint.from_url(endpoint)
        transport = RosbridgeTransport(endpoint=ep)
        discovery = RosGraphDiscovery(transport)
        snapshot = discovery.discover()
        transport.close()
        payload = {
            "ok": True,
            "action": "discover",
            "endpoint": endpoint,
            "robot_id": robot_id,
            "ros_version": snapshot.ros_version,
            "distro": snapshot.distro,
            "topics": [
                {"name": t.name, "type": t.msg_type, "risk": t.risk_hint}
                for t in snapshot.topics
            ],
            "services": [
                {"name": s.name, "type": s.srv_type, "risk": s.risk_hint}
                for s in snapshot.services
            ],
            "actions": [
                {"name": a.name, "type": a.action_type, "risk": a.risk_hint}
                for a in snapshot.actions
            ],
            "nodes": [n["name"] for n in snapshot.nodes],
            "params": snapshot.params,
        }
        if getattr(args, "out", None):
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(snapshot.to_dict(), indent=2, default=str), encoding="utf-8")
            payload["output_path"] = str(out_path.resolve())
        return _maybe_json(args, payload)
    except Exception as exc:
        return _maybe_json(
            args,
            {"ok": False, "action": "discover", "endpoint": endpoint, "error": str(exc)},
        )


def cmd_ros_compile(args: argparse.Namespace) -> int:
    """Compile a discovered ROS graph into a CapabilityManifest."""
    robot_id = args.robot_id
    graph_path = getattr(args, "graph", None)
    try:
        if graph_path and Path(graph_path).exists():
            snapshot = _load_graph(graph_path)
        else:
            endpoint = args.endpoint
            ep = RosbridgeEndpoint.from_url(endpoint)
            transport = RosbridgeTransport(endpoint=ep)
            discovery = RosGraphDiscovery(transport)
            snapshot = discovery.discover()
            transport.close()
        robot_spec = _load_robot_spec(robot_id) or {}
        manifest = CapabilityManifestCompiler(
            robot_id=robot_id,
            robot_spec=robot_spec,
        ).compile(snapshot)

        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(manifest.to_dict(), indent=2, default=str), encoding="utf-8")

        return _maybe_json(
            args,
            {
                "ok": True,
                "action": "compile",
                "endpoint": getattr(args, "endpoint", None),
                "robot_id": robot_id,
                "capabilities": [cap.id for cap in manifest.capabilities],
                "manifest": manifest.to_dict(),
                "output_path": str(Path(args.output).resolve()) if args.output else None,
            },
        )
    except Exception as exc:
        return _maybe_json(
            args,
            {"ok": False, "action": "compile", "endpoint": getattr(args, "endpoint", None), "error": str(exc)},
        )


def _get_provider_manifest(args: argparse.Namespace) -> CapabilityManifest | None:
    """Return the compiled manifest from CLI flags, on-disk file, provider, or live discovery."""
    manifest_path = getattr(args, "manifest", None)
    if manifest_path and Path(manifest_path).exists():
        return _load_manifest(manifest_path)
    provider = getattr(args, "_ros_provider", None)
    if provider is not None:
        manifest = getattr(provider, "_manifest", None)
        if manifest is not None:
            return manifest
    # Fallback: compile live from endpoint.
    robot_id = getattr(args, "robot_id", "unknown")
    endpoint = getattr(args, "endpoint", "ws://127.0.0.1:9090")
    try:
        ep = RosbridgeEndpoint.from_url(endpoint)
        transport = RosbridgeTransport(endpoint=ep)
        discovery = RosGraphDiscovery(transport)
        snapshot = discovery.discover()
        robot_spec = _load_robot_spec(robot_id) or {}
        manifest = CapabilityManifestCompiler(
            robot_id=robot_id,
            robot_spec=robot_spec,
        ).compile(snapshot)
        transport.close()
        return manifest
    except Exception:
        return None


def cmd_ros_list_capabilities(args: argparse.Namespace) -> int:
    """List compiled ROS capabilities."""
    manifest = _get_provider_manifest(args)
    if manifest is None:
        return _maybe_json(
            args,
            {"ok": False, "action": "list_capabilities", "error": "No manifest loaded and live discovery failed."},
        )

    return _maybe_json(
        args,
        {
            "ok": True,
            "action": "list_capabilities",
            "robot_id": manifest.robot_id,
            "capabilities": [
                {"id": cap.id, "kind": cap.kind, "risk": cap.risk.level, "enabled": cap.enabled}
                for cap in manifest.capabilities
            ],
        },
    )


def cmd_ros_inspect_capability(args: argparse.Namespace) -> int:
    """Inspect a single capability."""
    capability_id = args.capability_id
    manifest = _get_provider_manifest(args)
    if manifest is None:
        return _maybe_json(
            args,
            {
                "ok": False,
                "action": "inspect_capability",
                "capability_id": capability_id,
                "error": "No manifest loaded. Run 'rosclaw ros compile' first.",
            },
        )
    cap = manifest.get_capability(capability_id)
    if cap is None:
        return _maybe_json(
            args,
            {
                "ok": False,
                "action": "inspect_capability",
                "capability_id": capability_id,
                "error": f"Capability '{capability_id}' not found",
            },
        )
    return _maybe_json(
        args,
        {
            "ok": True,
            "action": "inspect_capability",
            "capability": cap.to_dict(),
        },
    )


def cmd_ros_validate_capability(args: argparse.Namespace) -> int:
    """Validate capability arguments against the safety contract."""
    capability_id = args.capability_id
    try:
        arguments = json.loads(args.args) if isinstance(args.args, str) else args.args
    except json.JSONDecodeError as exc:
        return _maybe_json(
            args,
            {
                "ok": False,
                "action": "validate_capability",
                "capability_id": capability_id,
                "error": f"Invalid JSON args: {exc}",
            },
        )

    manifest = _get_provider_manifest(args)
    if manifest is None:
        return _maybe_json(
            args,
            {
                "ok": False,
                "action": "validate_capability",
                "capability_id": capability_id,
                "error": "No manifest loaded. Run 'rosclaw ros compile' first.",
            },
        )

    contract = SafetyContractCompiler().compile(manifest)
    compiler = SafetyContractCompiler()
    decision = compiler.evaluate(contract, capability_id, arguments)
    return _maybe_json(
        args,
        {
            "ok": decision.decision in ("ALLOW", "MODIFY"),
            "action": "validate_capability",
            "capability_id": capability_id,
            "decision": decision.decision,
            "risk_score": decision.risk_score,
            "reason": decision.reason,
            "violated_constraints": decision.violated_constraints,
            "modified_args": decision.modified_args,
        },
    )


def cmd_ros_execute_capability(args: argparse.Namespace) -> int:
    """Execute a capability through the safety-gated provider."""
    capability_id = args.capability_id
    endpoint = args.endpoint
    robot_id = args.robot_id
    try:
        arguments = json.loads(args.args) if isinstance(args.args, str) else args.args
    except json.JSONDecodeError as exc:
        return _maybe_json(
            args,
            {
                "ok": False,
                "action": "execute_capability",
                "capability_id": capability_id,
                "error": f"Invalid JSON args: {exc}",
            },
        )

    try:
        import asyncio

        provider = _make_provider(
            endpoint=endpoint,
            robot_id=robot_id,
            dry_run=args.dry_run,
            auto_discover=True,
        )
        asyncio.run(provider.load())
        request = ProviderRequest(
            request_id=f"cli_ros_{capability_id}",
            capability=capability_id,
            inputs=arguments,
            context={"dry_run": args.dry_run},
        )
        response = asyncio.run(provider.infer(request))
        asyncio.run(provider.unload())
        return _maybe_json(
            args,
            {
                "ok": response.is_ok,
                "action": "execute_capability",
                "capability_id": capability_id,
                "status": response.status,
                "result": response.result,
                "errors": response.errors,
                "latency_ms": response.latency_ms,
                "trace": response.trace,
            },
        )
    except Exception as exc:
        return _maybe_json(
            args,
            {
                "ok": False,
                "action": "execute_capability",
                "capability_id": capability_id,
                "error": str(exc),
            },
        )


def cmd_ros_emergency_stop(args: argparse.Namespace) -> int:
    """Emergency stop: send zero velocity on all discovered command topics."""
    endpoint = args.endpoint
    robot_id = args.robot_id
    try:
        ep = RosbridgeEndpoint.from_url(endpoint)
        transport = RosbridgeTransport(endpoint=ep)
        zero_twist = {
            "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
        }

        # Discover current topics and their types from rosapi.
        topics_types: dict[str, str] = {}
        try:
            result = transport.call_service("/rosapi/topics", {})
            if result.ok and isinstance(result.data, dict):
                values = result.data.get("values", result.data)
                if isinstance(values, dict):
                    topics = values.get("topics", [])
                    types = values.get("types", [])
                    topics_types = dict(zip(topics, types, strict=False))
        except Exception:
            pass

        # Always cover the canonical topics, plus any other cmd_vel topics.
        command_topics = {"/cmd_vel", "/turtle1/cmd_vel", f"/{robot_id}/cmd_vel"}
        command_topics.update({t for t in topics_types if t.endswith("/cmd_vel")})

        # Determine the correct Twist type for this ROS distribution.
        twist_type: str | None = None
        for topic in command_topics:
            if topic in topics_types:
                twist_type = topics_types[topic]
                break
        if twist_type is None:
            try:
                version_result = transport.call_service("/rosapi/get_ros_version", {})
                if version_result.ok and isinstance(version_result.data, dict):
                    values = version_result.data.get("values", version_result.data)
                    twist_type = (
                        "geometry_msgs/msg/Twist"
                        if (isinstance(values, dict) and values.get("version") == 2)
                        else "geometry_msgs/Twist"
                    )
            except Exception:
                pass
        if twist_type is None:
            twist_type = "geometry_msgs/Twist"

        published_any = False
        is_ros2 = "/msg/" in (twist_type or "")
        for topic in sorted(command_topics):
            try:
                msg_type = topics_types.get(topic, twist_type)
                adv = transport.send({"op": "advertise", "topic": topic, "type": msg_type})
                if adv.ok:
                    if is_ros2:
                        # ROS2 needs a moment for publisher-subscriber matching.
                        transport.receive(timeout_sec=0.3)
                    else:
                        # ROS1 advertise does not return a status. Give rosbridge
                        # and the subscriber a little longer to register the new
                        # publisher before sending the stop command.
                        time.sleep(0.3)
                    # Publish zero velocity for ~0.5 s. A sustained burst ensures
                    # the stop command is delivered on both ROS1 and ROS2 before
                    # we unadvertise.
                    for _ in range(10):
                        pub = transport.publish(topic, zero_twist)
                        if pub.ok:
                            published_any = True
                        time.sleep(0.05)
                    transport.send({"op": "unadvertise", "topic": topic})
            except Exception:
                pass
        transport.close()

        if not published_any:
            return _maybe_json(
                args,
                {
                    "ok": False,
                    "action": "emergency_stop",
                    "robot_id": robot_id,
                    "endpoint": endpoint,
                    "error": "Could not publish zero velocity to any command topic",
                },
            )
        return _maybe_json(
            args,
            {
                "ok": True,
                "action": "emergency_stop",
                "robot_id": robot_id,
                "endpoint": endpoint,
                "message": "Zero velocity commands published to command topics",
            },
        )
    except Exception as exc:
        return _maybe_json(
            args,
            {
                "ok": False,
                "action": "emergency_stop",
                "robot_id": robot_id,
                "endpoint": endpoint,
                "error": str(exc),
            },
        )


def cmd_doctor_ros(args: argparse.Namespace | None = None) -> int:
    """Check the rosbridge-based ROS connector without importing rclpy."""
    endpoint = getattr(args, "endpoint", "ws://127.0.0.1:9090") if args else "ws://127.0.0.1:9090"
    checks = []

    try:
        ep = RosbridgeEndpoint.from_url(endpoint)
        transport = RosbridgeTransport(endpoint=ep, max_retries=0)
        result = transport.call_service("/rosapi/topics", {})
        transport.close()
        checks.append(("rosbridge connectivity", f"{endpoint}", result.ok))
        if not result.ok:
            checks.append(("rosbridge error", result.error or "unknown", False))
    except Exception as exc:
        checks.append(("rosbridge connectivity", f"{endpoint}", False))
        checks.append(("rosbridge error", str(exc), False))

    # Ensure core connector modules are importable.
    for mod_path in (
        "rosclaw.connectors.ros.transport",
        "rosclaw.connectors.ros.discovery",
        "rosclaw.connectors.ros.compiler",
        "rosclaw.connectors.ros.provider",
    ):
        try:
            __import__(mod_path)
            checks.append((f"import {mod_path}", "OK", True))
        except Exception as exc:
            checks.append((f"import {mod_path}", f"FAIL: {exc}", False))

    print("=" * 60)
    print("ROSClaw Doctor — ROS Connector (rosbridge, no rclpy)")
    print("=" * 60)
    for name, value, ok in checks:
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name:<32} {value}")
    print("=" * 60)

    all_ok = all(ok for _, _, ok in checks)
    if all_ok:
        print("  ROS connector profile: READY")
        return 0
    print("  ROS connector profile: NOT READY")
    return 1


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------

def add_ros_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register the ``ros`` subcommand tree on the main parser."""
    ros_parser = subparsers.add_parser("ros", help="ROS bridge commands")
    ros_subparsers = ros_parser.add_subparsers(dest="ros_command")

    # Common endpoint / robot flags for most subcommands.
    def _add_common(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--endpoint", default="ws://127.0.0.1:9090", help="rosbridge endpoint")
        parser.add_argument("--robot-id", default="unknown", help="Robot identifier")
        parser.add_argument("--json", action="store_true", help="Output as JSON")

    # ping
    ping_parser = ros_subparsers.add_parser("ping", help="Ping rosbridge endpoint")
    _add_common(ping_parser)

    # discover
    discover_parser = ros_subparsers.add_parser("discover", help="Discover ROS graph")
    _add_common(discover_parser)
    discover_parser.add_argument("--out", default=None, help="Write graph snapshot to file")

    # compile
    compile_parser = ros_subparsers.add_parser("compile", help="Compile CapabilityManifest from ROS graph")
    _add_common(compile_parser)
    compile_parser.add_argument("--output", "-o", default=None, help="Write manifest to file")
    compile_parser.add_argument("--graph", default=None, help="Use a saved graph snapshot instead of live discovery")

    # list-capabilities
    list_parser = ros_subparsers.add_parser("list-capabilities", help="List compiled capabilities")
    _add_common(list_parser)
    list_parser.add_argument("--manifest", default=None, help="Load manifest from file")

    # inspect-capability
    inspect_parser = ros_subparsers.add_parser("inspect-capability", help="Inspect a capability")
    _add_common(inspect_parser)
    inspect_parser.add_argument("--manifest", default=None, help="Load manifest from file")
    inspect_parser.add_argument("capability_id", help="Capability id")

    # validate-capability
    validate_parser = ros_subparsers.add_parser("validate-capability", help="Validate args against safety contract")
    _add_common(validate_parser)
    validate_parser.add_argument("--manifest", default=None, help="Load manifest from file")
    validate_parser.add_argument("capability_id", help="Capability id")
    validate_parser.add_argument("--args", default="{}", help="JSON arguments")

    # execute-capability
    execute_parser = ros_subparsers.add_parser("execute-capability", help="Execute a capability")
    _add_common(execute_parser)
    execute_parser.add_argument("--manifest", default=None, help="Load manifest from file")
    execute_parser.add_argument("capability_id", help="Capability id")
    execute_parser.add_argument("--args", default="{}", help="JSON arguments")
    execute_parser.add_argument("--dry-run", action="store_true", help="Dry run only")

    # emergency-stop
    stop_parser = ros_subparsers.add_parser("emergency-stop", help="Send emergency zero velocity")
    _add_common(stop_parser)

    return ros_parser


# ------------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------------

def dispatch_ros_command(args: argparse.Namespace) -> int:
    """Dispatch the selected ``ros`` subcommand."""
    cmd = getattr(args, "ros_command", None)
    if cmd == "ping":
        return cmd_ros_ping(args)
    elif cmd == "discover":
        return cmd_ros_discover(args)
    elif cmd == "compile":
        return cmd_ros_compile(args)
    elif cmd == "list-capabilities":
        return cmd_ros_list_capabilities(args)
    elif cmd == "inspect-capability":
        return cmd_ros_inspect_capability(args)
    elif cmd == "validate-capability":
        return cmd_ros_validate_capability(args)
    elif cmd == "execute-capability":
        return cmd_ros_execute_capability(args)
    elif cmd == "emergency-stop":
        return cmd_ros_emergency_stop(args)
    return 1
