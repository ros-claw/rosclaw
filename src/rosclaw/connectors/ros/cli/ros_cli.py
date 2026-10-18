"""ROS connector CLI for discovery, validation, and explicit dry runs.

Physical capability execution is blocked until it enters through rosclawd.
The emergency-stop command is a daemon client and reports request dispatch
separately from physical stop verification.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
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
from rosclaw.daemon.client import DaemonClient, DaemonClientError
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

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
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
                {"name": t.name, "type": t.msg_type, "risk": t.risk_hint} for t in snapshot.topics
            ],
            "services": [
                {"name": s.name, "type": s.srv_type, "risk": s.risk_hint} for s in snapshot.services
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
            out_path.write_text(
                json.dumps(snapshot.to_dict(), indent=2, default=str), encoding="utf-8"
            )
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
            {
                "ok": False,
                "action": "compile",
                "endpoint": getattr(args, "endpoint", None),
                "error": str(exc),
            },
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
            {
                "ok": False,
                "action": "list_capabilities",
                "error": "No manifest loaded and live discovery failed.",
            },
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
    """Dry-run a capability or report that runtime-gated execution is required."""
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
    """Request emergency stop through rosclawd."""
    robot_id = args.robot_id
    daemon = getattr(args, "_daemon_client", None) or DaemonClient()
    try:
        receipt = daemon.emergency_stop(
            f"ROS connector CLI emergency stop for {robot_id}",
            source="rosclaw.ros_cli",
        )
        payload = receipt if isinstance(receipt, dict) else receipt.to_dict()
        return _maybe_json(
            args,
            {
                **payload,
                "ok": bool(
                    payload.get("stopped", False) and payload.get("physical_stop_observed", False)
                ),
                "action": "emergency_stop",
                "robot_id": robot_id,
            },
        )
    except DaemonClientError as exc:
        return _maybe_json(
            args,
            {
                "ok": False,
                "action": "emergency_stop",
                "robot_id": robot_id,
                "error_code": exc.code,
                "error": exc.message,
                "request_dispatched": False,
                "stopped": False,
                "message": "Activate the certified physical E-stop immediately.",
            },
        )
    except Exception as exc:
        return _maybe_json(
            args,
            {
                "ok": False,
                "action": "emergency_stop",
                "robot_id": robot_id,
                "error_code": "ROSCLAWD_REQUEST_FAILED",
                "error": str(exc),
                "request_dispatched": False,
                "stopped": False,
                "message": "Activate the certified physical E-stop immediately.",
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
    compile_parser = ros_subparsers.add_parser(
        "compile", help="Compile CapabilityManifest from ROS graph"
    )
    _add_common(compile_parser)
    compile_parser.add_argument("--output", "-o", default=None, help="Write manifest to file")
    compile_parser.add_argument(
        "--graph", default=None, help="Use a saved graph snapshot instead of live discovery"
    )

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
    validate_parser = ros_subparsers.add_parser(
        "validate-capability", help="Validate args against safety contract"
    )
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
    stop_parser = ros_subparsers.add_parser(
        "emergency-stop",
        help="Request emergency stop through rosclawd",
    )
    _add_common(stop_parser)

    for command in (
        "inspect-system",
        "diagnose",
        "discover-body",
        "review-navigation",
        "compile-navigation-body",
        "resolve",
        "context",
        "mission",
        "performance",
    ):
        expert_parser = ros_subparsers.add_parser(command, help=f"ROS Expert Harness: {command}")
        _add_common(expert_parser)
        expert_parser.add_argument(
            "--snapshot",
            required=command == "compile-navigation-body",
            help="Replay a sealed RosSystemModel JSON",
        )
        expert_parser.add_argument("--graph", help="Saved RosGraphSnapshot JSON")
        expert_parser.add_argument("--native", help="Native probe JSON capture")
        expert_parser.add_argument(
            "--asset-originals",
            help="Explicit SHA-pinned collision originals for reference snapshots",
        )
        expert_parser.add_argument("--body", help="Body binding JSON (never inferred from graph)")
        expert_parser.add_argument(
            "--deep", action="store_true", help="Require native sidecar observations"
        )
        if command == "compile-navigation-body":
            expert_parser.set_defaults(output=None)
        else:
            expert_parser.add_argument("--output", help="Write derived artifact")
        if command in {"discover-body", "review-navigation", "compile-navigation-body"}:
            expert_parser.add_argument(
                "--urdf",
                required=True,
                help="Expanded URDF file matching the observed robot_description",
            )
        if command == "discover-body":
            expert_parser.add_argument(
                "--interfaces", help="JSON mapping of roles to explicit ROS names"
            )
        if command in {"review-navigation", "compile-navigation-body"}:
            expert_parser.add_argument(
                "--requirements",
                required=True,
                help="Explicit navigation observation requirements JSON",
            )
        if command == "compile-navigation-body":
            expert_parser.add_argument(
                "--policy", required=True, help="Source-bound SIM fixture policy JSON"
            )
            expert_parser.add_argument(
                "--workspace",
                dest="navigation_workspace",
                required=True,
                help="New simulator-owned Body workspace directory",
            )
        if command == "diagnose":
            expert_parser.add_argument(
                "--profile",
                default="all",
                choices=["all", "navigation", "tf", "qos", "sensors", "time"],
            )
        if command in {"resolve", "mission"}:
            expert_parser.add_argument("--task", required=True)
        if command == "mission":
            expert_parser.add_argument("operation", choices=["plan"])
            expert_parser.add_argument("--mission-id", default="ros_coverage_mission")

    for command in ("diagnostic-plan", "diagnostic-assess"):
        diagnostic_parser = ros_subparsers.add_parser(
            command, help="Offline source-bound read-only diagnosis; never executes probes"
        )
        diagnostic_parser.add_argument("--json", action="store_true")
        diagnostic_parser.add_argument(
            "--operational-graph", required=True, help="Original sealed operational graph JSON"
        )
        diagnostic_parser.add_argument(
            "--output", help="New derived artifact; existing files are never overwritten"
        )
        if command == "diagnostic-plan":
            diagnostic_parser.add_argument("--issue-id", action="append", dest="issue_ids")
            diagnostic_parser.add_argument("--budget-sec", type=int, default=20)
            diagnostic_parser.add_argument("--max-probes", type=int, default=8)
        else:
            diagnostic_parser.add_argument(
                "--experiment",
                required=True,
                help="Sealed experiment JSON or diagnostic-plan response",
            )
            diagnostic_parser.add_argument("--followup-graph", required=True)
            diagnostic_parser.add_argument(
                "--reviewed-at",
                required=True,
                help="Explicit aware assessment time; replay remains historical",
            )

    for command in ("intervention-plan", "experiment-result"):
        experiment_parser = ros_subparsers.add_parser(
            command, help="Offline SIM staging and original-reference contracts; no execution"
        )
        experiment_parser.add_argument("--json", action="store_true")
        experiment_parser.add_argument("--output", help="Exclusive new derived artifact")
        if command == "intervention-plan":
            experiment_parser.add_argument("--operational-graph", required=True)
            experiment_parser.add_argument("--snapshot", required=True)
            experiment_parser.add_argument(
                "--asset-originals", help="Explicit SHA-pinned collision originals"
            )
            experiment_parser.add_argument("--staging-nav2", required=True)
            experiment_parser.add_argument("--staging-bringup", required=True)
            experiment_parser.add_argument("--proposal", required=True)
        else:
            experiment_parser.add_argument("--plan", required=True)
            experiment_parser.add_argument("--phases", required=True)

    for command in ("application-plan", "application-staging"):
        app_parser = ros_subparsers.add_parser(
            command, help="Offline existing ROS2 component reuse; no install or execution"
        )
        for flag in (
            "operational-graph",
            "snapshot",
            "urdf",
            "requirements",
            "component-manifests",
            "request",
        ):
            app_parser.add_argument("--" + flag, required=True)
        if command == "application-staging":
            app_parser.add_argument("--plan", required=True)
            app_parser.add_argument("--artifacts", required=True)
        app_parser.add_argument("--asset-originals", help="Explicit SHA-pinned collision originals")
        app_parser.add_argument("--output", help="Exclusive new offline JSON artifact")
        app_parser.add_argument("--json", action="store_true")

    bootstrap_parser = ros_subparsers.add_parser(
        "application-bootstrap", help="Offline driver-only Nav2 proposal; no install or execution"
    )
    for flag in (
        "operational-graph",
        "snapshot",
        "urdf",
        "driver-bringup",
        "software-template",
        "component-manifests",
        "request",
    ):
        bootstrap_parser.add_argument("--" + flag, required=True)
    bootstrap_parser.add_argument(
        "--asset-originals", help="Explicit SHA-pinned collision originals"
    )
    bootstrap_parser.add_argument("--output", help="Exclusive new offline JSON artifact")
    bootstrap_parser.add_argument("--json", action="store_true")

    mesh_parser = ros_subparsers.add_parser(
        "inspect-collision-assets",
        help="Offline original STL facts; no live source or Body admission",
    )
    mesh_parser.add_argument("--urdf", required=True)
    mesh_parser.add_argument("--base-frame", required=True)
    mesh_parser.add_argument(
        "--asset-originals", required=True, help="Bounded JSON URI/SHA/base64 originals"
    )
    mesh_parser.add_argument("--output", help="Exclusive new offline artifact")
    mesh_parser.add_argument("--json", action="store_true")

    case_parser = ros_subparsers.add_parser(
        "engineering-case", help="Read-only engineering dossier from a sealed offline snapshot"
    )
    case_parser.add_argument("--json", action="store_true")
    case_parser.add_argument("--snapshot", required=True, help="Sealed RosSystemModel JSON")
    case_parser.add_argument("--urdf", required=True, help="Original expanded URDF")
    case_parser.add_argument("--asset-originals", help="Explicit SHA-pinned collision originals")
    case_parser.add_argument(
        "--requirements", required=True, help="Original navigation requirements JSON"
    )
    case_parser.add_argument(
        "--reviewed-at", required=True, help="Explicit timezone-aware original review clock"
    )
    case_parser.add_argument(
        "--source-review", help="Historical source review JSON, never live admission"
    )
    case_parser.add_argument("--expected-run-id")
    case_parser.add_argument("--expected-body-hash")
    case_parser.add_argument(
        "--summary-only", action="store_true", help="Return compact task card and binding hashes"
    )
    case_parser.add_argument(
        "--output", help="Create a NEW derived JSON artifact; never overwrite inputs"
    )

    case_parser.add_argument(
        "--plan-staging",
        action="store_true",
        help="Deterministic conditional proposal only; never applies or executes",
    )
    case_parser.add_argument(
        "--staging-nav2", help="Original concrete Nav2 YAML for an explicit staging plan"
    )
    case_parser.add_argument(
        "--staging-bringup",
        help="Original bringup JSON; no endpoint ownership authority is accepted",
    )

    workspace_parser = ros_subparsers.add_parser(
        "inspect-navigation-workspace", help="Read-only compiled navigation artifact integrity"
    )
    _add_common(workspace_parser)
    workspace_parser.add_argument(
        "--workspace", dest="navigation_workspace", required=True, help="Compiled fixture workspace"
    )
    workspace_parser.add_argument(
        "--expected-manifest-hash", help="Separately pinned manifest hash"
    )
    workspace_parser.add_argument("--expected-body-hash", help="Separately pinned Body hash")

    return ros_parser


# ------------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------------


def dispatch_ros_command(args: argparse.Namespace) -> int:
    """Dispatch the selected ``ros`` subcommand."""
    cmd = getattr(args, "ros_command", None)
    if cmd == "inspect-collision-assets":
        return cmd_ros_collision_assets(args)
    if cmd == "application-bootstrap":
        return cmd_ros_application_bootstrap(args)
    if cmd in {"application-plan", "application-staging"}:
        return cmd_ros_application(args)
    if cmd in {"intervention-plan", "experiment-result"}:
        return cmd_ros_counterfactual(args)
    if cmd in {"diagnostic-plan", "diagnostic-assess"}:
        return cmd_ros_diagnostic(args)
    if cmd == "engineering-case":
        return cmd_ros_engineering_case(args)
    if cmd == "inspect-navigation-workspace":
        from rosclaw.connectors.ros.context.navigation_workspace_review import (
            inspect_navigation_body_workspace,
        )

        try:
            result = inspect_navigation_body_workspace(
                args.navigation_workspace,
                expected_manifest_hash=args.expected_manifest_hash,
                expected_body_hash=args.expected_body_hash,
            )
            return _maybe_json(args, {"ok": True, "action": cmd, **result})
        except (ValueError, OSError) as exc:
            return _maybe_json(args, {"ok": False, "action": cmd, "error": str(exc)})
    if cmd in {
        "inspect-system",
        "diagnose",
        "discover-body",
        "review-navigation",
        "compile-navigation-body",
        "resolve",
        "context",
        "mission",
        "performance",
    }:
        return cmd_ros_expert(args)
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


def _read_discovery_urdf(path: str, description: str = "URDF") -> bytes:
    """Bound a regular expanded-URDF input before any live discovery read."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= 5_000_000:
            raise ValueError(f"bounded nonempty regular {description} file required")
        raw = stream.read(5_000_001)
        if not 0 < len(raw) <= 5_000_000:
            raise ValueError(f"{description} input size outside bounded limit")
        return raw


def cmd_ros_diagnostic(args: argparse.Namespace) -> int:
    """Offline proposal and evidence assessment; no live discovery or staging."""
    import hashlib
    from datetime import datetime

    from rosclaw.connectors.ros.engineering.diagnostic_experiment import RosDiagnosticExperimentV1
    from rosclaw.connectors.ros.engineering.probe_planner import (
        assess_diagnostic_followup,
        plan_diagnostic_probes,
    )
    from rosclaw.connectors.ros.intelligence.graph_contracts import RosOperationalGraphV1

    bindings = {}

    def read(path, label):
        raw = _read_discovery_urdf(path, label)
        bindings[label] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    try:
        graph = RosOperationalGraphV1.from_dict(read(args.operational_graph, "operational_graph"))
        if args.ros_command == "diagnostic-plan":
            value = plan_diagnostic_probes(
                graph,
                issue_ids=args.issue_ids,
                budget_sec=args.budget_sec,
                max_probes=args.max_probes,
            )
            field = "experiment"
        else:
            original = read(args.experiment, "experiment")
            experiment = RosDiagnosticExperimentV1.from_dict(original.get("experiment", original))
            after = RosOperationalGraphV1.from_dict(read(args.followup_graph, "followup_graph"))
            value = assess_diagnostic_followup(
                experiment, graph, after, now=datetime.fromisoformat(args.reviewed_at)
            )
            field = "assessment"
        result = {
            "ok": True,
            "action": args.ros_command,
            field: value.model_dump(mode="json"),
            "input_file_sha256": bindings,
            "current_authority": "NONE",
            "current_l1": "NOT_EVALUATED",
            "scope": "OFFLINE_OBSERVATIONAL_DIAGNOSIS_NOT_PROBE_EXECUTION_OR_CURRENT_ADMISSION",
        }
        if args.output:
            with Path(args.output).open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(result, indent=2) + "\n")
        return _maybe_json(args, result)
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
        return _maybe_json(args, {"ok": False, "action": args.ros_command, "error": str(exc)})


def cmd_ros_collision_assets(args: argparse.Namespace) -> int:
    """Read exact bounded local originals only; no URI resolution or ROS discovery."""
    import hashlib

    from rosclaw.connectors.ros.context.geometry import derive_collision_envelope

    try:
        urdf = _read_discovery_urdf(args.urdf, "original mesh-bearing URDF")
        originals = _read_discovery_urdf(args.asset_originals, "collision asset originals")
        report = derive_collision_envelope(
            urdf, base_frame=args.base_frame, mesh_originals=json.loads(originals)
        )
        result = {
            "ok": True,
            "action": args.ros_command,
            "collision_envelope": report,
            "input_file_sha256": {
                "urdf": hashlib.sha256(urdf).hexdigest(),
                "asset_originals": hashlib.sha256(originals).hexdigest(),
            },
            "execution_authority": "NONE",
            "current_l1": "NOT_EVALUATED",
            "physical_loaded_geometry": "NOT_VERIFIED",
            "next_gate": "source-owner and actual loaded mesh resource identity/bytes/scale/pose proof; immutable Body and fresh L1 remain required",
        }
        if args.output:
            with Path(args.output).open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(result, indent=2) + "\n")
        return _maybe_json(args, result)
    except (ValueError, TypeError, KeyError, AttributeError, OSError, RecursionError) as error:
        return _maybe_json(args, {"ok": False, "action": args.ros_command, "error": str(error)})


def _bind_model_originals(model, args, *, read=None):
    path = getattr(args, "asset_originals", None)
    if path is None:
        return None
    from rosclaw.connectors.ros.context.collision_assets import bind_prepared_collision_originals

    raw = read(path, "asset_originals") if read else _read_discovery_urdf(path, "asset originals")
    return bind_prepared_collision_originals(model.observations, raw)


def cmd_ros_application_bootstrap(args: argparse.Namespace) -> int:
    """Bounded originals only; missing Nav2 is a target, never motion authority."""
    import hashlib

    from rosclaw.connectors.ros.engineering.application_bootstrap import compile_driver_bootstrap
    from rosclaw.connectors.ros.intelligence import RosSystemModel
    from rosclaw.connectors.ros.intelligence.graph_contracts import RosOperationalGraphV1

    pins = {}

    def read(path, label):
        raw = _read_discovery_urdf(path, label)
        pins[label] = hashlib.sha256(raw).hexdigest()
        return raw

    try:
        graph = RosOperationalGraphV1.from_dict(
            json.loads(read(args.operational_graph, "operational_graph"))
        )
        model = RosSystemModel.from_dict(json.loads(read(args.snapshot, "snapshot")))
        _bind_model_originals(model, args, read=read)
        urdf = read(args.urdf, "source_urdf")
        bringup = json.loads(read(args.driver_bringup, "driver_bringup"))
        template = json.loads(read(args.software_template, "software_template"))
        manifests = json.loads(read(args.component_manifests, "component_manifests"))
        request = json.loads(read(args.request, "request"))
        plan = compile_driver_bootstrap(graph, model, urdf, bringup, template, manifests, request)
        result = {
            "ok": True,
            "action": args.ros_command,
            "application_bootstrap": plan,
            "input_file_sha256": pins,
            "current_authority": "NONE",
            "current_l1": "NOT_EVALUATED",
            "scope": "OFFLINE_DRIVER_ONLY_SIM_PROPOSAL_NOT_DEPLOYMENT_OR_PHYSICAL_ACCEPTANCE",
        }
        if args.output:
            with Path(args.output).open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(result, indent=2) + "\n")
        return _maybe_json(args, result)
    except (ValueError, TypeError, KeyError, AttributeError, OSError, RecursionError) as error:
        return _maybe_json(args, {"ok": False, "action": args.ros_command, "error": str(error)})


def cmd_ros_application(args: argparse.Namespace) -> int:
    """Bounded source artifacts only; no discovery, installation or actions."""
    import hashlib

    from rosclaw.connectors.ros.engineering.application_plan import (
        RosApplicationPlanV1,
        compile_application_plan,
    )
    from rosclaw.connectors.ros.engineering.application_staging import compile_application_staging
    from rosclaw.connectors.ros.intelligence import RosSystemModel
    from rosclaw.connectors.ros.intelligence.graph_contracts import RosOperationalGraphV1

    pins = {}

    def read(path, label):
        raw = _read_discovery_urdf(path, label)
        pins[label] = hashlib.sha256(raw).hexdigest()
        return raw

    try:
        graph = RosOperationalGraphV1.from_dict(
            json.loads(read(args.operational_graph, "operational_graph"))
        )
        model = RosSystemModel.from_dict(json.loads(read(args.snapshot, "snapshot")))
        _bind_model_originals(model, args, read=read)
        urdf = read(args.urdf, "source_urdf")
        requirements = json.loads(read(args.requirements, "requirements"))
        manifests = json.loads(read(args.component_manifests, "component_manifests"))
        request = json.loads(read(args.request, "request"))
        plan = compile_application_plan(graph, model, urdf, requirements, manifests, request)
        value = plan.model_dump(mode="json")
        field = "application_plan"
        if args.ros_command == "application-staging":
            document = json.loads(read(args.plan, "plan"))
            original = RosApplicationPlanV1.from_dict(document.get("application_plan", document))
            if original.plan_hash != plan.plan_hash:
                raise ValueError("supplied application plan differs from unchanged original inputs")
            artifacts = json.loads(read(args.artifacts, "artifacts"))
            value = compile_application_staging(
                original, graph, model, urdf, requirements, manifests, artifacts
            )
            field = "application_staging"
        result = {
            "ok": True,
            "action": args.ros_command,
            field: value,
            "input_file_sha256": pins,
            "current_authority": "NONE",
            "current_l1": "NOT_EVALUATED",
            "scope": "OFFLINE_EXISTING_ROS2_COMPONENT_REUSE_NOT_INSTALLATION_OR_PHYSICAL_ACCEPTANCE",
        }
        if args.output:
            with Path(args.output).open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(result, indent=2) + "\n")
        return _maybe_json(args, result)
    except (ValueError, TypeError, KeyError, AttributeError, OSError, RecursionError) as error:
        return _maybe_json(args, {"ok": False, "action": args.ros_command, "error": str(error)})


def cmd_ros_counterfactual(args: argparse.Namespace) -> int:
    """Bounded artifact input/output only; never discovers ROS or starts a World."""
    import hashlib

    from rosclaw.connectors.ros.engineering.experiment_result import bind_experiment_result
    from rosclaw.connectors.ros.engineering.intervention_plan import (
        RosInterventionPlanV1,
        compile_intervention_plan,
    )
    from rosclaw.connectors.ros.engineering.staging_plan import parse_staging_documents
    from rosclaw.connectors.ros.intelligence import RosSystemModel
    from rosclaw.connectors.ros.intelligence.graph_contracts import RosOperationalGraphV1

    bindings = {}

    def read(path, label):
        raw = _read_discovery_urdf(path, label)
        bindings[label] = hashlib.sha256(raw).hexdigest()
        return raw

    try:
        if args.ros_command == "intervention-plan":
            graph = RosOperationalGraphV1.from_dict(
                json.loads(read(args.operational_graph, "operational_graph"))
            )
            model = RosSystemModel.from_dict(json.loads(read(args.snapshot, "snapshot")))
            _bind_model_originals(model, args, read=read)
            staging = parse_staging_documents(
                read(args.staging_nav2, "staging_nav2"),
                read(args.staging_bringup, "staging_bringup"),
            )
            value = compile_intervention_plan(
                graph, model, staging, json.loads(read(args.proposal, "proposal"))
            )
            field = "intervention_plan"
        else:
            document = json.loads(read(args.plan, "plan"))
            plan = RosInterventionPlanV1.from_dict(document.get("intervention_plan", document))
            phases = json.loads(read(args.phases, "phases"))
            if type(phases) is not list or len(phases) != 3:
                raise ValueError("exactly three explicit original phase records required")
            value = bind_experiment_result(plan, phases)
            field = "experiment_result"
        result = {
            "ok": True,
            "action": args.ros_command,
            field: value.model_dump(mode="json"),
            "input_file_sha256": bindings,
            "current_authority": "NONE",
            "current_l1": "NOT_EVALUATED",
            "scope": "OFFLINE_SIM_STAGING_ORIGINAL_REFERENCES_NOT_EXECUTION_OR_PHYSICAL_VERIFICATION",
        }
        if args.output:
            with Path(args.output).open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(result, indent=2) + "\n")
        return _maybe_json(args, result)
    except (ValueError, TypeError, KeyError, AttributeError, OSError, RecursionError) as error:
        return _maybe_json(args, {"ok": False, "action": args.ros_command, "error": str(error)})


def cmd_ros_expert(args: argparse.Namespace) -> int:
    from rosclaw.connectors.ros.context import compile_agent_summary
    from rosclaw.connectors.ros.diagnosis import diagnose
    from rosclaw.connectors.ros.expert import inspect_system, load_system
    from rosclaw.connectors.ros.intelligence import build_system_model
    from rosclaw.connectors.ros.mission import compile_mission
    from rosclaw.connectors.ros.resolver import resolve_capabilities, resolve_task

    try:
        urdf = (
            _read_discovery_urdf(args.urdf)
            if args.ros_command in {"discover-body", "review-navigation", "compile-navigation-body"}
            else None
        )
        interfaces = (
            json.loads(_read_discovery_urdf(args.interfaces, "interface requirements"))
            if getattr(args, "interfaces", None)
            else None
        )
        requirements = (
            json.loads(_read_discovery_urdf(args.requirements, "navigation requirements"))
            if args.ros_command in {"review-navigation", "compile-navigation-body"}
            else None
        )
        policy = (
            json.loads(_read_discovery_urdf(args.policy, "navigation fixture policy"))
            if args.ros_command == "compile-navigation-body"
            else None
        )
        body = _load_json_or_yaml(Path(args.body)) if args.body else None
        if args.snapshot:
            model = load_system(_load_json_or_yaml(Path(args.snapshot)))
        elif args.graph:
            native = _load_json_or_yaml(Path(args.native)) if args.native else None
            if args.deep and not native:
                raise ValueError("deep replay requires --native")
            model = build_system_model(
                _load_graph(args.graph), robot_id=args.robot_id, body=body, native=native
            )
        else:
            if args.native:
                raise ValueError("--native requires --graph")
            model = inspect_system(
                endpoint=args.endpoint, robot_id=args.robot_id, deep=args.deep, body=body
            )
        _bind_model_originals(model, args)
        command = args.ros_command
        if command == "inspect-system":
            result = {"system": model.to_dict(), "capabilities": resolve_capabilities(model)}
        elif command == "performance":
            from rosclaw.connectors.ros.intelligence.performance import performance_graph

            result = performance_graph(model).model_dump(mode="json")
        elif command == "diagnose":
            result = diagnose(model, profile=args.profile)
        elif command == "discover-body":
            from rosclaw.connectors.ros.context.discovery import discover_body_candidate

            assert urdf is not None
            result = discover_body_candidate(model, urdf, interface_names=interfaces)
        elif command == "review-navigation":
            from rosclaw.connectors.ros.context.navigation_review import review_navigation_candidate

            assert urdf is not None
            result = review_navigation_candidate(model, urdf, requirements)
        elif command == "compile-navigation-body":
            from rosclaw.connectors.ros.context.navigation_workspace import (
                compile_navigation_body_workspace,
            )

            assert urdf is not None
            result = compile_navigation_body_workspace(
                args.navigation_workspace, model, urdf, requirements, policy=policy
            )
        elif command == "resolve":
            result = resolve_task(model, args.task)
        elif command == "mission":
            result = compile_mission(model, args.task, mission_id=args.mission_id).model_dump(
                mode="json"
            )
        else:
            result = {"summary": compile_agent_summary(model)}
        if args.output:
            artifact = (
                result["summary"]
                if command == "context"
                else json.dumps(result.get("system", result), indent=2)
            )
            Path(args.output).write_text(artifact + "\n", encoding="utf-8")
        if command == "performance" and not args.json:
            result = {
                "snapshot_id": result["snapshot_id"],
                "graph_hash": result["graph_hash"],
                "node_count": len(result["nodes"]),
                "edge_count": len(result["edges"]),
                "unknown_backend_count": sum(
                    n["buffer_backend"] == "UNKNOWN" for n in result["nodes"]
                ),
                "findings": result["findings"][:20],
                "validation_checks": result["validation_checks"],
                "optimization_verified": result["optimization_verified"],
                "output_path": args.output,
            }
        return _maybe_json(args, {"ok": True, "action": command, **result})
    except (ValueError, OSError, ConnectionError, TimeoutError) as exc:
        return _maybe_json(args, {"ok": False, "action": args.ros_command, "error": str(exc)})


def cmd_ros_engineering_case(args: argparse.Namespace) -> int:
    """Offline-only file replay. Never discovers ROS, binds Body or starts a World."""
    import hashlib
    from datetime import datetime

    from rosclaw.connectors.ros.engineering import compile_engineering_case, summarize_case
    from rosclaw.connectors.ros.expert import load_system

    try:
        if not getattr(args, "snapshot", None) or not getattr(args, "reviewed_at", None):
            raise ValueError("sealed snapshot and explicit aware review clock required")
        reviewed_at = datetime.fromisoformat(args.reviewed_at)
        if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
            raise ValueError("timezone-aware original review clock required")
        raw_snapshot = _read_discovery_urdf(args.snapshot, "sealed system model")
        model = load_system(json.loads(raw_snapshot))
        _bind_model_originals(model, args)
        urdf = _read_discovery_urdf(args.urdf)
        raw_requirements = _read_discovery_urdf(args.requirements, "navigation requirements")
        requirements = json.loads(raw_requirements)
        if not isinstance(requirements, dict):
            raise ValueError("navigation requirements must be a JSON object")
        review_path = getattr(args, "source_review", None)
        raw_source_review = (
            _read_discovery_urdf(review_path, "historical source review") if review_path else None
        )
        source_review = json.loads(raw_source_review) if raw_source_review is not None else None
        if source_review is not None and not isinstance(source_review, dict):
            raise ValueError("historical source review must be a JSON object")
        expected_run = getattr(args, "expected_run_id", None)
        expected_body = getattr(args, "expected_body_hash", None)
        if (expected_run or expected_body) and source_review is None:
            raise ValueError("expected run/Body checks require the original source review")
        case = compile_engineering_case(
            model,
            urdf,
            requirements,
            source_ref="sealed-model:" + model.snapshot_hash,
            now=reviewed_at,
            evidence_domain="SEALED_REPLAY",
            source_review=source_review,
            expected_run_id=expected_run,
            expected_body_hash=expected_body,
        )
        result = {
            "ok": True,
            "action": "engineering-case",
            "scope": "SEALED_OFFLINE_REPLAY_NOT_CURRENT_STATE_OR_ADMISSION",
            "reviewed_at": reviewed_at.isoformat(),
            "case_hash": case.case_hash,
            "snapshot_hash": case.snapshot_hash,
            "original_urdf_sha256": case.source_urdf_sha256,
            "input_file_sha256": {
                "snapshot": hashlib.sha256(raw_snapshot).hexdigest(),
                "URDF": hashlib.sha256(urdf).hexdigest(),
                "requirements": hashlib.sha256(raw_requirements).hexdigest(),
            },
            "current_authority": "NONE",
            "current_l1": "NOT_EVALUATED",
            "source_review_scope": "CALLER_SUPPLIED_HISTORICAL_INPUT_NOT_INDEPENDENT_CURRENT_GATE",
            "summary": summarize_case(case),
        }
        if raw_source_review is not None:
            result["input_file_sha256"]["source_review"] = hashlib.sha256(
                raw_source_review
            ).hexdigest()
        if not getattr(args, "summary_only", False):
            result["case"] = case.model_dump(mode="json")
        plan_requested = getattr(args, "plan_staging", False)
        nav2_path = getattr(args, "staging_nav2", None)
        bringup_path = getattr(args, "staging_bringup", None)
        if plan_requested or nav2_path or bringup_path:
            if not plan_requested or not nav2_path or not bringup_path:
                raise ValueError(
                    "--plan-staging requires both original --staging-nav2 and --staging-bringup"
                )
            from rosclaw.connectors.ros.engineering.staging_plan import (
                parse_staging_documents,
                plan_staging_repairs,
            )

            raw_nav2 = _read_discovery_urdf(nav2_path, "original Nav2 staging")
            raw_bringup = _read_discovery_urdf(bringup_path, "original bringup staging")
            staging = parse_staging_documents(raw_nav2, raw_bringup)
            result["staging_plan"] = plan_staging_repairs(case, model, urdf, requirements, staging)
            result["input_file_sha256"].update(
                {
                    "staging_nav2": hashlib.sha256(raw_nav2).hexdigest(),
                    "staging_bringup": hashlib.sha256(raw_bringup).hexdigest(),
                }
            )
        if getattr(args, "output", None):
            with Path(args.output).open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(result, indent=2) + "\n")
        return _maybe_json(args, result)
    except (ValueError, TypeError, OSError) as exc:
        return _maybe_json(args, {"ok": False, "action": "engineering-case", "error": str(exc)})
