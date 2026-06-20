"""rosclaw body CLI subcommands."""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import tarfile
import uuid
from datetime import UTC
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import yaml

from rosclaw.body.compiler import compute_checksum
from rosclaw.body.diff import BodyDiffer
from rosclaw.body.notes import MaintenanceLog
from rosclaw.body.query import BodyQueryEngine
from rosclaw.body.registry import BodyRegistryError, BodyRegistryManager
from rosclaw.body.renderer import EmbodimentRenderer
from rosclaw.body.resolver import BodyNotLinkedError, BodyResolver
from rosclaw.body.ros_introspection import RosIntrospectionError, introspect_ros
from rosclaw.body.schema import (
    BodyYaml,
    CalibrationYaml,
    MaintenanceEvent,
    SkillCompatibilityReport,
)
from rosclaw.body.validator import BodyValidator
from rosclaw.body.validators import parse_set_expression, validate_update_path
from rosclaw.connectors.ros.transport.base import RosbridgeEndpoint
from rosclaw.eurdf.registry import RobotRegistry
from rosclaw.memory.interface import MemoryInterface


def add_body_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    """Register the `rosclaw body` subcommand tree."""
    body_parser = subparsers.add_parser("body", help="Body / embodiment commands")
    body_subparsers = body_parser.add_subparsers(dest="body_command")

    # init
    init_parser = body_subparsers.add_parser("init", help="Initialize a new body instance from an e-URDF profile")
    init_parser.add_argument("--robot", required=True, help="Robot profile ID (e.g., unitree-g1)")
    init_parser.add_argument("--profile", default="default", help="Alias for --robot; ignored if --robot given")
    init_parser.add_argument("--name", default=None, help="Body instance ID")
    init_parser.add_argument("--workspace", default=None, help="ROSClaw workspace")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing body link")
    init_parser.add_argument("--no-alias", action="store_true", help="Skip creating BODY.md alias")
    init_parser.add_argument("--render", action="store_true", default=True, help="Render EMBODIMENT.md (default)")
    init_parser.add_argument("--validate", action="store_true", help="Run validation after init")
    _add_body_arg(init_parser)

    # create
    create_parser = body_subparsers.add_parser("create", help="Create a new body instance from an e-URDF profile")
    create_parser.add_argument("--robot", required=True, help="Robot profile ID (e.g., unitree-g1)")
    create_parser.add_argument("--name", required=True, help="Body instance ID")
    create_parser.add_argument("--nickname", default=None, help="Human-readable nickname")
    create_parser.add_argument("--workspace", default=None, help="ROSClaw workspace")
    create_parser.add_argument("--force", action="store_true", help="Overwrite an existing body with the same ID")
    create_parser.add_argument("--no-alias", action="store_true", help="Skip creating BODY.md alias")

    # switch
    switch_parser = body_subparsers.add_parser("switch", help="Switch the active body")
    switch_parser.add_argument("body_id", help="Body ID to activate")
    switch_parser.add_argument("--workspace", default=None, help="ROSClaw workspace")

    # remove
    remove_parser = body_subparsers.add_parser("remove", help="Remove a body instance")
    remove_parser.add_argument("body_id", help="Body ID to remove")
    remove_parser.add_argument("--workspace", default=None, help="ROSClaw workspace")
    remove_parser.add_argument("--archive", action="store_true", help="Archive body data instead of deleting")

    # validate
    validate_parser = body_subparsers.add_parser("validate", help="Validate body workspace")
    validate_parser.add_argument("--json", action="store_true", help="Output JSON report")
    validate_parser.add_argument("--workspace", default=None, help="ROSClaw workspace")
    _add_body_arg(validate_parser)

    # render
    render_parser = body_subparsers.add_parser("render", help="Force re-render of EMBODIMENT.md and summaries")
    render_parser.add_argument("--workspace", default=None, help="ROSClaw workspace")
    _add_body_arg(render_parser)

    # show
    show_parser = body_subparsers.add_parser("show", help="Show body summary")
    show_parser.add_argument("--agent", action="store_true", help="Output agent-readable summary")
    show_parser.add_argument("--workspace", default=None, help="ROSClaw workspace")
    _add_body_arg(show_parser)

    # state
    state_parser = body_subparsers.add_parser("state", help="Print unified body state")
    state_parser.add_argument("--json", action="store_true", help="Output JSON")
    state_parser.add_argument("--workspace", default=None, help="ROSClaw workspace")
    _add_body_arg(state_parser)

    # query
    query_parser = body_subparsers.add_parser("query", help="Ask a question about the body")
    query_parser.add_argument("question", help="Question string")
    query_parser.add_argument("--workspace", default=None, help="ROSClaw workspace")
    query_parser.add_argument("--json", action="store_true", help="Output JSON")
    _add_body_arg(query_parser)

    # fault
    fault_parser = body_subparsers.add_parser("fault", help="Manage known faults")
    fault_subparsers = fault_parser.add_subparsers(dest="fault_command")
    fault_add = fault_subparsers.add_parser("add", help="Add a known fault")
    fault_add.add_argument("--component", required=True, help="Affected component")
    fault_add.add_argument("--severity", required=True, choices=["low", "medium", "high", "critical"], help="Fault severity")
    fault_add.add_argument("--summary", required=True, help="Fault summary")
    fault_add.add_argument("--workspace", default=None, help="ROSClaw workspace")
    _add_body_arg(fault_add)
    fault_resolve = fault_subparsers.add_parser("resolve", help="Resolve a known fault")
    fault_resolve.add_argument("fault_id", help="Fault ID to resolve")
    fault_resolve.add_argument("--summary", default="", help="Resolution summary")
    fault_resolve.add_argument("--workspace", default=None, help="ROSClaw workspace")
    _add_body_arg(fault_resolve)

    # maintenance
    maint_parser = body_subparsers.add_parser("maintenance", help="Add maintenance event")
    maint_subparsers = maint_parser.add_subparsers(dest="maint_command")
    maint_add = maint_subparsers.add_parser("add", help="Add a maintenance event")
    maint_add.add_argument("--type", default="maintenance", dest="maint_type", choices=["maintenance", "repair", "inspection", "incident", "safety"], help="Event type")
    maint_add.add_argument("--component", required=True, help="Affected component")
    maint_add.add_argument("--summary", required=True, help="Event summary")
    maint_add.add_argument("--workspace", default=None, help="ROSClaw workspace")
    _add_body_arg(maint_add)

    # calibration update
    cal_parser = body_subparsers.add_parser("calibration", help="Calibration commands")
    cal_subparsers = cal_parser.add_subparsers(dest="cal_command")
    cal_update = cal_subparsers.add_parser("update", help="Update calibration from a YAML file")
    cal_update.add_argument("--file", required=True, help="Path to calibration YAML file")
    cal_update.add_argument("--workspace", default=None, help="ROSClaw workspace")
    _add_body_arg(cal_update)

    # retrofit
    retro_parser = body_subparsers.add_parser("retrofit", help="Record a hardware retrofit")
    retro_subparsers = retro_parser.add_subparsers(dest="retro_command")
    retro_add = retro_subparsers.add_parser("add", help="Record a retrofit")
    retro_add.add_argument("--component", required=True, help="Installed component ID")
    retro_add.add_argument("--type", required=True, choices=["sensor_install", "tool_install", "actuator_swap", "structural_mod", "other"], help="Retrofit type")
    retro_add.add_argument("--summary", required=True, help="Summary of the change")
    retro_add.add_argument("--workspace", default=None, help="ROSClaw workspace")
    _add_body_arg(retro_add)

    # capability
    cap_parser = body_subparsers.add_parser("capability", help="Manage capabilities")
    cap_subparsers = cap_parser.add_subparsers(dest="cap_command")
    cap_disable = cap_subparsers.add_parser("disable", help="Disable a capability")
    cap_disable.add_argument("capability_id", help="Capability ID")
    cap_disable.add_argument("--reason", required=True, help="Reason")
    cap_disable.add_argument("--workspace", default=None, help="ROSClaw workspace")
    _add_body_arg(cap_disable)
    cap_degrade = cap_subparsers.add_parser("degrade", help="Degrade a capability")
    cap_degrade.add_argument("capability_id", help="Capability ID")
    cap_degrade.add_argument("--mode", default="sim_only", choices=["slow", "sim_only", "restricted_workspace", "human_supervised"], help="Degradation mode")
    cap_degrade.add_argument("--reason", required=True, help="Reason")
    cap_degrade.add_argument("--workspace", default=None, help="ROSClaw workspace")
    _add_body_arg(cap_degrade)
    cap_enable = cap_subparsers.add_parser("enable", help="Enable a capability")
    cap_enable.add_argument("capability_id", help="Capability ID")
    cap_enable.add_argument("--after-validation", default=None, help="Validation run ID or evidence")
    cap_enable.add_argument("--workspace", default=None, help="ROSClaw workspace")
    _add_body_arg(cap_enable)

    # link-eurdf
    link_parser = body_subparsers.add_parser("link-eurdf", help="Link current body to an e-URDF profile")
    link_parser.add_argument("profile_id", help="e-URDF profile ID (e.g., unitree-g1)")
    link_parser.add_argument("--version", default="latest", help="Profile version")
    link_parser.add_argument("--instance-id", default=None, help="Body instance ID")
    link_parser.add_argument("--nickname", default=None, help="Human-readable nickname")
    link_parser.add_argument("--workspace", default=None, help="ROSClaw workspace")
    link_parser.add_argument("--force", action="store_true", help="Overwrite existing body link")
    link_parser.add_argument("--mode", default="copy", choices=["copy", "lock-only"], help="Link mode")
    _add_body_arg(link_parser)

    # inspect
    inspect_parser = body_subparsers.add_parser("inspect", help="Inspect current body state")
    inspect_parser.add_argument("--json", action="store_true", help="Output as JSON")
    inspect_parser.add_argument("--agent", action="store_true", help="Output agent-readable summary")
    inspect_parser.add_argument("--source-trace", action="store_true", help="Show source trace")
    inspect_parser.add_argument("--capabilities", action="store_true", help="Show only capabilities")
    inspect_parser.add_argument("--components", action="store_true", help="Show only components")
    inspect_parser.add_argument("--skills", action="store_true", help="Show skill compatibility")
    _add_body_arg(inspect_parser)

    # diff
    diff_parser = body_subparsers.add_parser("diff", help="Compare body states")
    diff_parser.add_argument("--against", default="eurdf", help="Compare against eurdf, snapshot:<name>, or live")
    diff_parser.add_argument("--format", default="text", choices=["text", "json", "patch"], help="Output format")
    diff_parser.add_argument("--only", default=None, help="Filter by category")
    _add_body_arg(diff_parser)

    # update-state
    update_parser = body_subparsers.add_parser("update-state", help="Update body instance state")
    update_parser.add_argument("--set", action="append", default=[], help="Set key=value")
    update_parser.add_argument("--unset", action="append", default=[], help="Unset key")
    update_parser.add_argument("--enable-capability", action="append", default=[], help="Enable capability")
    update_parser.add_argument("--disable-capability", action="append", default=[], help="Disable capability")
    update_parser.add_argument("--component-status", action="append", default=[], help="component_id=status")
    update_parser.add_argument("--reason", default="", help="Reason for the change")
    update_parser.add_argument("--source", default="human", help="Source of the update")
    update_parser.add_argument("--dry-run", action="store_true", help="Do not persist changes")
    update_parser.add_argument("--no-skill-check", action="store_true", help="Skip skill compatibility recheck")
    update_parser.add_argument("--from-ros", action="store_true", help="Introspect live ROS graph and update runtime_state")
    update_parser.add_argument("--ros-endpoint", default=None, help="Rosbridge endpoint URL (ws://host:port); default ws://127.0.0.1:9090")
    _add_body_arg(update_parser)

    # note
    note_parser = body_subparsers.add_parser("note", help="Add a maintenance/incident note")
    note_parser.add_argument("message", help="Note message")
    note_parser.add_argument("--type", default="note", choices=["note", "maintenance", "calibration", "incident", "repair", "inspection", "safety"], help="Note type")
    note_parser.add_argument("--severity", default="info", choices=["info", "warning", "critical"], help="Severity")
    note_parser.add_argument("--affects", default="", help="Comma-separated affected components/capabilities")
    note_parser.add_argument("--tags", default="", help="Comma-separated tags")
    note_parser.add_argument("--author", default="human", help="Author")
    _add_body_arg(note_parser)

    # history
    history_parser = body_subparsers.add_parser("history", help="List body snapshots")
    history_parser.add_argument("--json", action="store_true", help="Output JSON")
    history_parser.add_argument("--workspace", default=None, help="ROSClaw workspace")
    _add_body_arg(history_parser)

    # export
    export_parser = body_subparsers.add_parser("export", help="Export body directory as an archive")
    export_parser.add_argument("dest", help="Destination file or directory")
    export_parser.add_argument("--format", default="zip", choices=["zip", "tar"], help="Archive format")
    export_parser.add_argument("--workspace", default=None, help="ROSClaw workspace")
    _add_body_arg(export_parser)

    return body_parser


def _add_body_arg(parser: argparse.ArgumentParser) -> None:
    """Add the --body selector to a leaf body subcommand."""
    parser.add_argument("--body", default=None, help="Target body ID (defaults to current body)")


def _resolver_for(args: argparse.Namespace) -> BodyResolver:
    """Create a BodyResolver honoring --workspace and --body."""
    workspace = getattr(args, "workspace", None)
    body_id = getattr(args, "body", None)
    return BodyResolver(
        workspace=Path(workspace) if workspace else None,
        body_id=body_id if body_id else None,
    )


def dispatch_body_command(args: argparse.Namespace) -> int:
    """Route body subcommands."""
    command = getattr(args, "body_command", None)
    if command == "init":
        return cmd_body_init(args)
    if command == "create":
        return cmd_body_create(args)
    if command == "switch":
        return cmd_body_switch(args)
    if command == "remove":
        return cmd_body_remove(args)
    if command == "validate":
        return cmd_body_validate(args)
    if command == "render":
        return cmd_body_render(args)
    if command == "show":
        return cmd_body_show(args)
    if command == "state":
        return cmd_body_state(args)
    if command == "query":
        return cmd_body_query(args)
    if command == "link-eurdf":
        return cmd_body_link_eurdf(args)
    if command == "inspect":
        return cmd_body_inspect(args)
    if command == "diff":
        return cmd_body_diff(args)
    if command == "update-state":
        return cmd_body_update_state(args)
    if command == "note":
        return cmd_body_note(args)
    if command == "history":
        return cmd_body_history(args)
    if command == "export":
        return cmd_body_export(args)
    if command == "fault":
        return cmd_body_fault(args)
    if command == "maintenance":
        return cmd_body_maintenance(args)
    if command == "calibration":
        return cmd_body_calibration(args)
    if command == "retrofit":
        return cmd_body_retrofit(args)
    if command == "capability":
        return cmd_body_capability(args)
    print("[ROSClaw] body: no subcommand given. Use: init, create, switch, remove, validate, render, show, state, query, link-eurdf, inspect, diff, update-state, note, history, export, fault, maintenance, calibration, retrofit, capability")
    return 1


def _resolve_profile_alias(profile_id: str) -> str:
    """Map common user-facing profile IDs to zoo directory names."""
    aliases = {
        "unitree-g1": "g1",
        "unitree-go2": "unitree_go2",
        "franka-panda": "franka_panda",
        "fetch": "fetch_robot",
    }
    return aliases.get(profile_id, profile_id)


def cmd_body_init(args: argparse.Namespace) -> int:
    """Initialize a body: wrapper around link-eurdf with --robot semantics."""
    # Normalize --robot / --profile
    robot_id = args.robot or args.profile
    if not robot_id:
        print("[ROSClaw] --robot is required.")
        return 1

    proxy = argparse.Namespace(
        profile_id=robot_id,
        version="latest",
        instance_id=args.name,
        nickname=args.name,
        workspace=args.workspace,
        force=args.force,
        mode="copy",
    )
    return cmd_body_link_eurdf(proxy)


def cmd_body_create(args: argparse.Namespace) -> int:
    """Create a new body instance and link it to an e-URDF profile."""
    workspace = Path(args.workspace) if args.workspace else None
    ws = workspace or Path.home() / ".rosclaw"
    manager = BodyRegistryManager(ws)

    body_id = args.name.strip().lower()
    nickname = args.nickname or body_id

    try:
        manager.create_body(
            body_id=body_id,
            profile_id=args.robot,
            nickname=nickname,
            force=args.force,
        )
    except BodyRegistryError as exc:
        print(f"[ROSClaw] {exc}")
        return 1

    proxy = argparse.Namespace(
        profile_id=args.robot,
        version="latest",
        instance_id=body_id,
        nickname=nickname,
        workspace=args.workspace,
        force=True,
        mode="copy",
    )
    return cmd_body_link_eurdf(proxy)


def cmd_body_switch(args: argparse.Namespace) -> int:
    """Switch the active body."""
    workspace = Path(args.workspace) if args.workspace else None
    ws = workspace or Path.home() / ".rosclaw"
    manager = BodyRegistryManager(ws)
    try:
        manager.set_current_body_id(args.body_id)
    except BodyRegistryError as exc:
        print(f"[ROSClaw] {exc}")
        return 1
    print(f"Switched to body: {args.body_id}")
    return 0


def cmd_body_remove(args: argparse.Namespace) -> int:
    """Remove a body instance, optionally archiving its data."""
    workspace = Path(args.workspace) if args.workspace else None
    ws = workspace or Path.home() / ".rosclaw"
    manager = BodyRegistryManager(ws)
    try:
        removal = manager.remove_body(args.body_id, archive=args.archive)
    except BodyRegistryError as exc:
        print(f"[ROSClaw] {exc}")
        return 1
    if removal.archived and removal.archive_path:
        print(f"Removed body '{args.body_id}' and archived data to {removal.archive_path}")
    else:
        print(f"Removed body '{args.body_id}'")
    return 0


def cmd_body_validate(args: argparse.Namespace) -> int:
    """Run full body validation and print report."""
    try:
        resolver = _resolver_for(args)
    except BodyRegistryError as exc:
        print(f"[ROSClaw] {exc}")
        return 1
    validator = BodyValidator(resolver)
    report = validator.validate_all()
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print("=" * 60)
        print("ROSClaw Body Validation")
        print("=" * 60)
        print(f"Result: {report.result}")
        print("")
        for check in report.checks:
            status_marker = {"pass": "✓", "warn": "!", "fail": "✗", "block": "⊘"}.get(check.status, "?")
            print(f"  [{status_marker}] {check.check_id}: {check.message}")
        print("")
        print("Summary:")
        for status, count in sorted(report.summary.items()):
            print(f"  {status}: {count}")
        print("=" * 60)
    return 0 if report.result in ("PASS", "PASS_WITH_WARNINGS") else 1


def cmd_body_render(args: argparse.Namespace) -> int:
    """Force re-render EMBODIMENT.md and generated summaries."""
    try:
        resolver = _resolver_for(args)
    except BodyRegistryError as exc:
        print(f"[ROSClaw] {exc}")
        return 1
    if not resolver.is_linked():
        print("[ROSClaw] No body linked. Run: rosclaw body init --robot <profile_id>")
        return 1
    try:
        effective, report = resolver.refresh_all_artifacts(reason="manual render")
    except Exception as exc:
        print(f"[ROSClaw] Render failed: {exc}")
        return 1
    print("Rendered body artifacts.")
    print(f"  {resolver.embodiment_md_path}")
    print(f"  {resolver.body_md_path}")
    print(f"  {resolver.generated_dir}")
    print(f"Effective body hash: {effective.effective_body_hash}")
    print(f"Generation: {effective.generation}")
    return 0


def cmd_body_show(args: argparse.Namespace) -> int:
    """Show body summary; --agent prints the agent-readable summary JSON."""
    try:
        resolver = _resolver_for(args)
    except BodyRegistryError as exc:
        print(f"[ROSClaw] {exc}")
        return 1
    if not resolver.is_linked():
        print("[ROSClaw] No body linked. Run: rosclaw body init --robot <profile_id>")
        return 1

    if args.agent:
        agent_path = resolver.generated_dir / "embodiment.agent.json"
        if agent_path.exists():
            print(agent_path.read_text(encoding="utf-8"))
            return 0

    body = resolver.get_effective_body()
    body_yaml = resolver.get_current_body_yaml()
    identity = body_yaml.get_identity()
    print("=" * 60)
    print("ROSClaw Body Show")
    print("=" * 60)
    print(f"Instance:   {body.body_instance_id}")
    print(f"Nickname:   {identity.get('nickname') or 'N/A'}")
    print(f"Model:      {identity.get('robot_model') or 'N/A'}")
    print(f"Profile:    {body_yaml.model_ref.get('profile_id', 'N/A')}")
    print(f"Hash:       {body.effective_body_hash}")
    print(f"Generation: {body.generation}")
    print(f"Safety:     {body_yaml.get_safety_status()}")
    print("\nCapabilities:")
    print(f"  enabled:  {body.capabilities.get('enabled', [])}")
    print(f"  degraded: {body.capabilities.get('degraded', [])}")
    print(f"  blocked:  {body.capabilities.get('blocked', [])}")
    print("=" * 60)
    return 0


def cmd_body_state(args: argparse.Namespace) -> int:
    """Print unified body/calibration/maintenance state."""
    try:
        resolver = _resolver_for(args)
    except BodyRegistryError as exc:
        print(f"[ROSClaw] {exc}")
        return 1
    if not resolver.is_linked():
        print("[ROSClaw] No body linked. Run: rosclaw body init --robot <profile_id>")
        return 1

    body = resolver.get_effective_body()
    body_yaml = resolver.get_current_body_yaml()
    calibration = resolver.get_calibration()
    maintenance = resolver.get_maintenance_events()
    identity = body_yaml.get_identity()

    open_faults = [f.get("id") for f in body.known_faults if f.get("status") == "open"]
    state = {
        "schema": "rosclaw.body_state.v1",
        "generated_at": _utc_now(),
        "robot_instance_id": body.body_instance_id,
        "robot_model": identity.get("robot_model") or body_yaml.body_instance.get("robot_model", "unknown"),
        "safety_status": body_yaml.get_safety_status(),
        "calibration_status": calibration.overall_status(),
        "capabilities": body.capabilities,
        "forbidden_capabilities": body.forbidden_capabilities,
        "open_faults": open_faults,
        "agent_policy": {
            "physical_execution_requires_sandbox": True,
            "direct_real_robot_execution_allowed": False,
            "human_approval_required_for_high_risk": True,
        },
        "generation": body.generation,
        "effective_body_hash": body.effective_body_hash,
        "maintenance_event_count": len(maintenance),
    }
    if args.json:
        print(json.dumps(state, indent=2, default=str))
    else:
        print("=" * 60)
        print("ROSClaw Body State")
        print("=" * 60)
        for key, value in state.items():
            print(f"  {key}: {value}")
        print("=" * 60)
    return 0


def cmd_body_query(args: argparse.Namespace) -> int:
    """Answer a natural-language question about the body."""
    try:
        resolver = _resolver_for(args)
    except BodyRegistryError as exc:
        print(f"[ROSClaw] {exc}")
        return 1
    if not resolver.is_linked():
        print("[ROSClaw] No body linked. Run: rosclaw body init --robot <profile_id>")
        return 1

    body = resolver.get_effective_body()
    body_yaml = resolver.get_current_body_yaml()
    calibration = resolver.get_calibration()
    maintenance = resolver.get_maintenance_events()

    engine = BodyQueryEngine(body, body_yaml, calibration, maintenance)
    result = engine.answer(args.question)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        print(result.answer)
        if result.actionable_policy:
            print("")
            print("Policy:")
            for line in result.actionable_policy:
                print(f"  - {line}")
    return 0


def cmd_body_fault(args: argparse.Namespace) -> int:
    """Add or resolve a fault."""
    try:
        resolver = _resolver_for(args)
    except BodyRegistryError as exc:
        print(f"[ROSClaw] {exc}")
        return 1
    if not resolver.is_linked():
        print("[ROSClaw] No body linked. Run: rosclaw body init --robot <profile_id>")
        return 1

    log = MaintenanceLog(resolver.maintenance_log_path)
    body_yaml = resolver.get_current_body_yaml()
    body_id = body_yaml.body_instance.get("id", "unknown")

    if args.fault_command == "add":
        import uuid
        fault_id = f"fault-{uuid.uuid4().hex[:8]}"
        event = log.write_fault_event(
            body_instance_id=body_id,
            component=args.component,
            severity=args.severity,
            summary=args.summary,
            fault_id=fault_id,
        )
        resolver.recompile_effective_body()
        print(f"Added fault {fault_id} on {args.component} ({args.severity})")
        print(f"Event: {event.event_id}")
        return 0

    if args.fault_command == "resolve":
        fault_id = args.fault_id
        # Try to find the original component from prior fault events.
        component = "unknown"
        for event in log.read_events(type_filter="fault"):
            if event.result.get("fault_id") == fault_id:
                component = event.component or (event.affects[0] if event.affects else "unknown")
                break
        event = log.write_resolution_event(
            body_instance_id=body_id,
            fault_id=fault_id,
            component=component,
            summary=args.summary or f"Resolved fault {fault_id}",
        )
        resolver.recompile_effective_body()
        print(f"Resolved fault {fault_id}")
        print(f"Event: {event.event_id}")
        return 0

    print("[ROSClaw] fault: expected add or resolve")
    return 1


def cmd_body_maintenance(args: argparse.Namespace) -> int:
    """Add a generic maintenance event."""
    try:
        resolver = _resolver_for(args)
    except BodyRegistryError as exc:
        print(f"[ROSClaw] {exc}")
        return 1
    if not resolver.is_linked():
        print("[ROSClaw] No body linked. Run: rosclaw body init --robot <profile_id>")
        return 1

    body_yaml = resolver.get_current_body_yaml()
    body_id = body_yaml.body_instance.get("id", "unknown")
    log = MaintenanceLog(resolver.maintenance_log_path)
    event = log.write_update_event(
        body_instance_id=body_id,
        change_summary=args.summary,
        affects=[args.component],
        author="human",
        reason=f"type={args.maint_type}",
        event_type=args.maint_type,
    )
    resolver.recompile_effective_body()
    print(f"Added {args.maint_type} event for {args.component}")
    print(f"Event: {event.event_id}")
    return 0


def cmd_body_calibration(args: argparse.Namespace) -> int:
    """Update calibration from a YAML file."""
    try:
        resolver = _resolver_for(args)
    except BodyRegistryError as exc:
        print(f"[ROSClaw] {exc}")
        return 1
    if not resolver.is_linked():
        print("[ROSClaw] No body linked. Run: rosclaw body init --robot <profile_id>")
        return 1

    src = Path(args.file)
    if not src.exists():
        print(f"[ROSClaw] Calibration file not found: {src}")
        return 1

    shutil.copy2(src, resolver.calibration_yaml_path)

    body_yaml = resolver.get_current_body_yaml()
    body_id = body_yaml.body_instance.get("id", "unknown")
    log = MaintenanceLog(resolver.maintenance_log_path)
    event = log.write_calibration_event(
        body_instance_id=body_id,
        summary=f"Calibration updated from {src.name}",
        before={"source": "previous"},
        after={"source": str(src)},
    )
    resolver.recompile_effective_body()
    print(f"Calibration updated from {src}")
    print(f"Event: {event.event_id}")
    return 0


def cmd_body_retrofit(args: argparse.Namespace) -> int:
    """Record a hardware retrofit."""
    try:
        resolver = _resolver_for(args)
    except BodyRegistryError as exc:
        print(f"[ROSClaw] {exc}")
        return 1
    if not resolver.is_linked():
        print("[ROSClaw] No body linked. Run: rosclaw body init --robot <profile_id>")
        return 1

    body_yaml = resolver.get_current_body_yaml()
    body_id = body_yaml.body_instance.get("id", "unknown")
    log = MaintenanceLog(resolver.maintenance_log_path)
    event = log.write_retrofit_event(
        body_instance_id=body_id,
        component=args.component,
        retrofit_type=args.type,
        summary=args.summary,
    )

    # Mark component installed in body.yaml if not already present.
    components = body_yaml.installed_components
    category = "sensors" if args.type == "sensor_install" else "actuators" if args.type == "actuator_swap" else "tools"
    components.setdefault(category, {})[args.component] = {"installed": True, "status": "available", "notes": [args.summary]}
    resolver.update_body_yaml({"installed_components": components})

    resolver.recompile_effective_body()
    print(f"Recorded retrofit {args.type} for {args.component}")
    print(f"Event: {event.event_id}")
    return 0


def cmd_body_capability(args: argparse.Namespace) -> int:
    """Disable, degrade, or enable a capability."""
    try:
        resolver = _resolver_for(args)
    except BodyRegistryError as exc:
        print(f"[ROSClaw] {exc}")
        return 1
    if not resolver.is_linked():
        print("[ROSClaw] No body linked. Run: rosclaw body init --robot <profile_id>")
        return 1

    body_yaml = resolver.get_current_body_yaml()
    body_id = body_yaml.body_instance.get("id", "unknown")
    cap_id = args.capability_id
    caps = body_yaml.capabilities
    enabled = set(caps.get("enabled", []))
    disabled = set(caps.get("disabled", []))
    degraded = set(caps.get("degraded", []))

    log = MaintenanceLog(resolver.maintenance_log_path)
    if args.cap_command == "disable":
        enabled.discard(cap_id)
        degraded.discard(cap_id)
        disabled.add(cap_id)
        action = "disable"
    elif args.cap_command == "degrade":
        enabled.discard(cap_id)
        disabled.discard(cap_id)
        degraded.add(cap_id)
        action = "degrade"
    elif args.cap_command == "enable":
        disabled.discard(cap_id)
        degraded.discard(cap_id)
        enabled.add(cap_id)
        action = "enable"
    else:
        print("[ROSClaw] capability: expected disable, degrade, or enable")
        return 1

    reason = getattr(args, "reason", "")
    after_validation = getattr(args, "after_validation", None)
    if after_validation:
        reason = f"{reason} (validation: {after_validation})" if reason else f"validation: {after_validation}"

    event = log.write_capability_event(
        body_instance_id=body_id,
        capability=cap_id,
        action=action,
        reason=reason or action,
    )

    resolver.update_body_yaml({
        "capabilities": {
            "enabled": sorted(enabled),
            "disabled": sorted(disabled),
            "degraded": sorted(degraded),
        }
    })
    resolver.recompile_effective_body()
    print(f"Capability {cap_id} {action}d")
    print(f"Event: {event.event_id}")
    return 0


def cmd_body_link_eurdf(args: argparse.Namespace) -> int:
    """Bind current body instance to an e-URDF profile."""
    original_profile_id = args.profile_id
    profile_id = _resolve_profile_alias(args.profile_id)
    version = args.version if args.version != "latest" else "1.0.0"
    try:
        resolver = _resolver_for(args)
    except BodyRegistryError as exc:
        print(f"[ROSClaw] {exc}")
        return 1

    if resolver.is_linked() and not args.force:
        print("[ROSClaw] Body already linked. Use --force to overwrite.")
        return 1

    # Resolve profile via existing RobotRegistry
    registry = RobotRegistry()
    profile = registry.get(profile_id)
    if profile is None:
        print(f"[ROSClaw] e-URDF profile not found: {args.profile_id}")
        available = registry.list_available()
        if available:
            print(f"[ROSClaw] Available: {', '.join(available)}")
        return 1

    eurdf = profile  # RobotCompleteProfile
    from rosclaw.body.schema import EurdfProfile
    normalized = EurdfProfile.from_robot_complete_profile(eurdf)

    instance_id = args.instance_id or f"body-{profile_id}-001"
    nickname = args.nickname or instance_id
    now = _utc_now()

    resolver.ensure_body_dir()

    # Write normalized profile reference
    with open(resolver.eurdf_profile_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(normalized.to_dict(), f, sort_keys=False, allow_unicode=True)

    # Compute checksum of the written profile file
    checksum = compute_checksum(resolver.eurdf_profile_path)

    # Write lock file
    lock = {
        "schema_version": "rosclaw.eurdf_lock.v1",
        "profile_id": profile_id,
        "profile_version": version,
        "uri": f"rosclaw://eurdf/{profile_id}@{version}",
        "source": "builtin",
        "checksum": checksum,
        "locked_at": now,
    }
    with open(resolver.eurdf_lock_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(lock, f, sort_keys=False, allow_unicode=True)

    # Write body.yaml
    body_yaml = BodyYaml(
        body_instance={
            "id": instance_id,
            "nickname": nickname,
            "robot_model": args.profile_id,
            "serial_number": "UNKNOWN",
            "owner": "local",
            "deployment_site": "lab",
            "created_at": now,
            "updated_at": now,
        },
        model_ref={
            "eurdf_uri": f"rosclaw://eurdf/{args.profile_id}@{version}",
            "profile_id": args.profile_id,
            "profile_version": version,
            "profile_checksum": checksum,
            "lock_file": "refs/eurdf.lock",
        },
        calibration={
            "file": "calibration.yaml",
            "checksum": "sha256:uninitialized",
            "last_calibrated_at": None,
            "status": "factory_default",
        },
        maintenance={
            "log_file": "maintenance.log",
            "last_event_at": None,
            "safety_relevant_open_items": [],
        },
        installed_components=_default_installed_components(normalized),
        capabilities={"enabled": [], "disabled": [], "degraded": []},
        prohibited_capabilities=[],
        safety_overrides={},
        runtime_state={
            "battery_percent": None,
            "last_seen_at": None,
            "health": "unknown",
            "online": False,
        },
        fingerprint={
            "effective_body_hash": None,
            "last_compiled_at": None,
            "last_skill_check_at": None,
        },
        compatibility_summary={
            "compatible_skills": 0,
            "degraded_skills": 0,
            "blocked_skills": 0,
            "unknown_skills": 0,
        },
    )
    with open(resolver.body_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(body_yaml.to_dict(), f, sort_keys=False, allow_unicode=True)

    # Write calibration.yaml
    calibration = CalibrationYaml(
        body_instance_id=instance_id,
        model_ref=f"rosclaw://eurdf/{profile_id}@{version}",
        validation={"status": "factory_default", "last_validated_at": None, "errors": [], "warnings": []},
    )
    with open(resolver.calibration_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(calibration.to_dict(), f, sort_keys=False, allow_unicode=True)

    # Write initial maintenance log
    MaintenanceLog(resolver.maintenance_log_path).write_init_event(instance_id, f"rosclaw://eurdf/{profile_id}@{version}")

    # Compile effective body, check skills, render EMBODIMENT.md
    try:
        effective, report = resolver.refresh_all_artifacts(reason="link-eurdf")
    except Exception as exc:
        print(f"[ROSClaw] Warning: failed to refresh artifacts: {exc}")
        effective = resolver.recompile_effective_body()
        report = SkillCompatibilityReport(body_instance_id=instance_id, effective_body_hash=effective.effective_body_hash)

    # Snapshot
    resolver.create_snapshot(effective)

    print("Linked e-URDF profile:")
    print(f"  profile: {original_profile_id}@{version}")
    print(f"  uri: rosclaw://eurdf/{original_profile_id}@{version}")
    print(f"  checksum: {checksum}")
    print("")
    print("Created/updated:")
    print(f"  {resolver.body_yaml_path}")
    print(f"  {resolver.calibration_yaml_path}")
    print(f"  {resolver.maintenance_log_path}")
    print(f"  {resolver.eurdf_lock_path}")
    print(f"  {resolver.effective_body_path}")
    print(f"  {resolver.embodiment_md_path}")
    print("")
    print(f"Effective body hash:\n  {effective.effective_body_hash}")
    print("")
    print("Skill compatibility:")
    for status, count in report.summary.items():
        print(f"  {status}: {count}")
    print("")
    print("Next:\n  rosclaw body inspect")
    return 0


def _default_installed_components(eurdf: Any) -> dict[str, Any]:
    """Build default installed_components from e-URDF sensors/actuators."""
    sensors = {}
    for sensor in eurdf.sensors:
        name = sensor.get("name")
        if name:
            sensors[name] = {"installed": True, "status": "available", "provider_ref": None, "notes": []}
    actuators = {}
    for actuator in eurdf.actuators:
        name = actuator.get("name")
        if name:
            actuators[name] = {"installed": True, "status": "available", "notes": []}
    return {"sensors": sensors, "actuators": actuators}


def cmd_body_inspect(args: argparse.Namespace) -> int:
    """Show current effective body state."""
    try:
        resolver = _resolver_for(args)
        body = resolver.get_effective_body()
        body_yaml = resolver.get_current_body_yaml()
    except BodyRegistryError as exc:
        print(f"[ROSClaw] {exc}")
        return 1
    except BodyNotLinkedError as exc:
        print(f"[ROSClaw] {exc}")
        return 1

    if args.json:
        print(json.dumps(body.to_dict(), indent=2, default=str))
        return 0

    print("=" * 60)
    print("ROSClaw Body Inspect")
    print("=" * 60)
    print("\nInstance:")
    print(f"  id: {body.body_instance_id}")
    print(f"  nickname: {body_yaml.body_instance.get('nickname', 'N/A')}")
    print(f"  model: {body_yaml.body_instance.get('robot_model', 'N/A')}")
    print("\ne-URDF:")
    print(f"  uri: {body.eurdf_uri}")
    print("\nEffective Body:")
    print(f"  hash: {body.effective_body_hash}")
    print(f"  compiled_at: {body.compiled_at}")
    print("\nHealth:")
    print(f"  online: {body_yaml.runtime_state.get('online', False)}")
    print(f"  runtime health: {body_yaml.runtime_state.get('health', 'unknown')}")
    print(f"  calibration: {body_yaml.calibration.get('status', 'unknown')}")
    print("\nCapabilities:")
    print(f"  enabled: {body.capabilities.get('enabled', [])}")
    print(f"  degraded: {body.capabilities.get('degraded', [])}")
    print(f"  blocked: {body.capabilities.get('blocked', [])}")
    print("\nSafety:")
    for key, value in sorted(body.safety.get('safety_limits', {}).items()):
        print(f"  {key}: {value}")

    if args.skills:
        report = resolver.get_skill_compatibility()
        print("\nSkill compatibility:")
        for status, count in report.summary.items():
            print(f"  {status}: {count}")

    if args.components:
        print("\nSensors:")
        for name, sensor in sorted(body.sensors.items()):
            print(f"  {name}: {sensor.get('status', 'unknown')}")
        print("\nActuators:")
        for name, actuator in sorted(body.actuators.items()):
            print(f"  {name}: {actuator.get('status', 'unknown')}")

    if args.source_trace:
        print("\nSource trace:")
        for key, value in sorted(body.source_trace.items()):
            print(f"  {key}: {value}")

    print("=" * 60)
    return 0


def cmd_body_diff(args: argparse.Namespace) -> int:
    """Show differences against e-URDF base or a snapshot."""
    try:
        resolver = _resolver_for(args)
        effective = resolver.get_effective_body()
    except BodyRegistryError as exc:
        print(f"[ROSClaw] {exc}")
        return 1
    except BodyNotLinkedError as exc:
        print(f"[ROSClaw] {exc}")
        return 1

    against = args.against
    differ = BodyDiffer()
    if against == "eurdf":
        eurdf = resolver.get_current_eurdf_profile()
        diff = differ.diff_against_eurdf(effective, eurdf)
    elif against.startswith("snapshot:"):
        snapshot_name = against.split(":", 1)[1]
        snapshot_path = resolver.snapshots_dir / snapshot_name
        diff = differ.diff_against_snapshot(effective, snapshot_path)
    elif against == "live":
        print("[ROSClaw] --against live not implemented in P0")
        return 1
    else:
        print(f"[ROSClaw] Unknown diff target: {against}")
        return 1

    if args.only:
        diff.changes = [c for c in diff.changes if c.category == args.only]

    if args.format == "json":
        print(json.dumps(diff.to_dict(), indent=2, default=str))
        return 0

    print("=" * 60)
    print("ROSClaw Body Diff")
    print("=" * 60)
    print(f"Base: {against}")
    print("Current: rosclaw://body/current/effective")
    print(f"Hash: {effective.effective_body_hash}")
    print("")
    if not diff.changes:
        print("No changes detected.")
    else:
        print("Changed fields:")
        for change in diff.changes:
            print(f"  [{change.category.upper()}] {change.path}")
            print(f"    old: {change.old}")
            print(f"    new: {change.new}")
            if change.reason:
                print(f"    reason: {change.reason}")
            print(f"    requires_skill_recheck: {change.requires_skill_recheck}")
    print("")
    print("Summary:")
    for category, count in sorted(diff.summary.items()):
        print(f"  {category}: {count}")
    print(f"  requires_skill_recheck: {diff.requires_skill_recheck}")
    print("=" * 60)
    return 0


def _record_body_change_events(
    robot_id: str,
    old_hash: str,
    new_hash: str,
    reason: str,
    report: SkillCompatibilityReport | None,
) -> None:
    """Persist body_change and skill_compatibility_change events to memory."""
    try:
        memory = MemoryInterface(robot_id=robot_id)
        memory._client.connect()
    except Exception:
        return

    with contextlib.suppress(Exception):
        memory.store_experience(
            event_id=f"body-change-{uuid.uuid4().hex}",
            event_type="body_change",
            instruction=f"Body state changed: {reason}",
            outcome="success",
            metadata={
                "effective_body_hash_before": old_hash,
                "effective_body_hash_after": new_hash,
                "reason": reason,
            },
        )

    if report is None:
        return
    with contextlib.suppress(Exception):
        memory.store_experience(
            event_id=f"skill-compat-{uuid.uuid4().hex}",
            event_type="skill_compatibility_change",
            instruction="Skill compatibility rechecked after body change",
            outcome="success",
            metadata={
                "effective_body_hash_before": old_hash,
                "effective_body_hash_after": new_hash,
                "skills": {key: result.status for key, result in report.skills.items()},
                "summary": report.summary,
            },
        )


def cmd_body_update_state(args: argparse.Namespace) -> int:
    """Apply a patch to body.yaml and recompile artifacts."""
    try:
        resolver = _resolver_for(args)
    except BodyRegistryError as exc:
        print(f"[ROSClaw] {exc}")
        return 1
    if not resolver.is_linked():
        print("[ROSClaw] No body linked. Run: rosclaw body link-eurdf <profile_id>")
        return 1

    patch: dict[str, Any] = {}
    affects: list[str] = []

    for expr in args.set:
        try:
            key, value = parse_set_expression(expr)
        except ValueError as exc:
            print(f"[ROSClaw] Invalid --set: {exc}")
            return 1
        ok, reason = validate_update_path(key)
        if not ok:
            print(f"[ROSClaw] Cannot update {key}: {reason}")
            return 1
        patch[key] = value
        affects.append(key)

    for cap in args.disable_capability:
        patch["capabilities.disabled"] = patch.get("capabilities.disabled", []) + [{"capability": cap, "reason": args.reason or "disabled via CLI", "source": args.source}]
        affects.append(cap)

    for cap in args.enable_capability:
        # Remove from disabled if present; simple approach
        patch["capabilities.enabled"] = patch.get("capabilities.enabled", []) + [cap]
        affects.append(cap)

    for expr in args.component_status:
        if "=" not in expr:
            print(f"[ROSClaw] Invalid --component-status (expected id=status): {expr}")
            return 1
        comp_id, status = expr.split("=", 1)
        # Try sensors first, then actuators
        for category in ("sensors", "actuators"):
            key = f"installed_components.{category}.{comp_id}.status"
            ok, reason = validate_update_path(key)
            if ok:
                patch[key] = status
                affects.append(comp_id)
                break

    if args.from_ros:
        source = args.source if args.source != "human" else "ros"
        endpoint = RosbridgeEndpoint.from_url(args.ros_endpoint) if args.ros_endpoint else None
        try:
            snapshot, runtime_state = introspect_ros(endpoint)
        except RosIntrospectionError as exc:
            print(f"[ROSClaw] {exc}")
            return 1
        for key, value in runtime_state.items():
            patch[f"runtime_state.{key}"] = value
        affects.append("runtime_state")
        if args.dry_run:
            print("[ROSClaw] Live ROS snapshot:")
            print(json.dumps(snapshot.to_dict(), indent=2, default=str))
            print("")
        args.source = source

    if not patch:
        print("[ROSClaw] No updates specified.")
        return 1

    # Load old effective body for diff
    old_effective = resolver.get_effective_body()
    old_hash = old_effective.effective_body_hash

    if args.dry_run:
        print("[ROSClaw] Dry run. Would apply patch:")
        for key, value in patch.items():
            print(f"  {key}: {value}")
        return 0

    # Apply patch
    resolver.update_body_yaml(patch)

    # Recompile
    new_effective = resolver.recompile_effective_body()
    new_hash = new_effective.effective_body_hash

    # Diff
    differ = BodyDiffer()
    diff = differ.diff_effective_bodies(old_effective, new_effective)

    # Maintenance log
    MaintenanceLog(resolver.maintenance_log_path).write_update_event(
        body_instance_id=new_effective.body_instance_id,
        change_summary="; ".join(f"{k}={v}" for k, v in patch.items()),
        affects=affects,
        author=args.source,
        reason=args.reason,
    )

    # Skill recheck
    report = None
    if diff.requires_skill_recheck and not args.no_skill_check:
        affected_ids = set(diff.affected_ids)
        effective, report = resolver.refresh_all_artifacts(
            reason=args.reason, affected_only=affected_ids or None
        )
    else:
        # Still render EMBODIMENT.md
        body_yaml = resolver.get_current_body_yaml()
        maintenance = resolver.get_maintenance_events()
        report = resolver.get_skill_compatibility()
        renderer = EmbodimentRenderer()
        existing_md = resolver.embodiment_md_path.read_text(encoding="utf-8") if resolver.embodiment_md_path.exists() else None
        if existing_md:
            md = renderer.render_into_existing(existing_md, new_effective, body_yaml, report, maintenance)
        else:
            md = renderer.render(new_effective, body_yaml, report, maintenance)
        resolver.embodiment_md_path.write_text(md, encoding="utf-8")

    # Snapshot
    _record_body_change_events(
        robot_id=new_effective.body_instance_id,
        old_hash=old_hash,
        new_hash=new_hash,
        reason=args.reason or "body update",
        report=report,
    )
    resolver.create_snapshot(new_effective)

    print("Updated body state.")
    print(f"Effective body hash:  old: {old_hash}  new: {new_hash}")
    print(f"Change category:  {', '.join(diff.affected_categories) if diff.affected_categories else 'none'}")
    print(f"Skill compatibility recheck:  triggered: {'yes' if diff.requires_skill_recheck and not args.no_skill_check else 'no'}")
    if report:
        print(f"  affected skills: {list(report.skills.keys())}")
    print("\nUpdated:")
    print(f"  {resolver.body_yaml_path}")
    print(f"  {resolver.effective_body_path}")
    print(f"  {resolver.skill_compatibility_path}")
    print(f"  {resolver.embodiment_md_path}")
    print(f"  {resolver.maintenance_log_path}")
    return 0


def cmd_body_note(args: argparse.Namespace) -> int:
    """Append a maintenance note and optionally trigger skill recheck."""
    try:
        resolver = _resolver_for(args)
    except BodyRegistryError as exc:
        print(f"[ROSClaw] {exc}")
        return 1
    if not resolver.is_linked():
        print("[ROSClaw] No body linked. Run: rosclaw body link-eurdf <profile_id>")
        return 1

    affects = [a.strip() for a in args.affects.split(",") if a.strip()]
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    recheck_keywords = {"sensor", "camera", "actuator", "arm", "leg", "motor", "gripper", "capability", "safety", "calibration"}
    requires_recheck = (
        args.type in ("incident", "repair", "safety", "calibration")
        or any(any(kw in a.lower() for kw in recheck_keywords) for a in affects)
    )

    body_yaml = resolver.get_current_body_yaml()
    old_hash = resolver.get_effective_body_hash()
    event = MaintenanceEvent(
        ts=_utc_now(),
        type=args.type,
        severity=args.severity,
        author=args.author,
        body_instance_id=body_yaml.body_instance.get("id", ""),
        message=args.message,
        affects=affects,
        tags=tags,
        requires_skill_recheck=requires_recheck,
    )
    MaintenanceLog(resolver.maintenance_log_path).append(event)

    if requires_recheck:
        affected_ids = set(affects)
        if args.type == "calibration":
            affected_ids.add("calibration")
        if args.type == "safety":
            affected_ids.add("safety")
        effective, report = resolver.refresh_all_artifacts(
            reason=args.message, affected_only=affected_ids or None
        )
    else:
        effective = resolver.recompile_effective_body()
        report = resolver.get_skill_compatibility()
        maintenance = resolver.get_maintenance_events()
        renderer = EmbodimentRenderer()
        existing_md = resolver.embodiment_md_path.read_text(encoding="utf-8") if resolver.embodiment_md_path.exists() else None
        if existing_md:
            md = renderer.render_into_existing(existing_md, effective, body_yaml, report, maintenance)
        else:
            md = renderer.render(effective, body_yaml, report, maintenance)
        resolver.embodiment_md_path.write_text(md, encoding="utf-8")

    _record_body_change_events(
        robot_id=effective.body_instance_id or body_yaml.body_instance.get("id", ""),
        old_hash=old_hash,
        new_hash=effective.effective_body_hash,
        reason=args.message or f"{args.type} note",
        report=report,
    )

    print("Added body note.")
    print(f"Type:  {args.type}")
    print(f"Severity:  {args.severity}")
    print(f"Affects:  - {chr(10).join(affects) if affects else 'none'}")
    print(f"Skill recheck:  triggered: {'yes' if requires_recheck else 'no'}")
    print("\nUpdated:")
    print(f"  {resolver.maintenance_log_path}")
    print(f"  {resolver.skill_compatibility_path}")
    print(f"  {resolver.embodiment_md_path}")
    return 0


def cmd_body_history(args: argparse.Namespace) -> int:
    """List body snapshots."""
    try:
        resolver = _resolver_for(args)
    except BodyRegistryError as exc:
        print(f"[ROSClaw] {exc}")
        return 1
    if not resolver.is_linked():
        print("[ROSClaw] No body linked. Run: rosclaw body link-eurdf <profile_id>")
        return 1

    snaps: list[dict[str, Any]] = []
    for snap_path in sorted(resolver.snapshots_dir.glob("body-*.yaml")):
        fingerprint_path = snap_path.with_suffix(".fingerprint")
        if fingerprint_path.exists():
            hash_value = fingerprint_path.read_text(encoding="utf-8").strip()
        else:
            hash_value = ""
        stat = snap_path.stat()
        snaps.append({
            "timestamp": snap_path.stem.replace("body-", ""),
            "hash": hash_value,
            "snapshot": snap_path.name,
            "size": stat.st_size,
        })

    if args.json:
        print(json.dumps(snaps, indent=2, default=str))
        return 0

    print("=" * 60)
    print("ROSClaw Body History")
    print("=" * 60)
    if not snaps:
        print("No snapshots yet.")
        print("Snapshots are created by body init, update-state, and note.")
    else:
        try:
            current_hash = resolver.get_effective_body_hash()
        except Exception:
            current_hash = ""
        print(f"Current hash: {current_hash}")
        print(f"Snapshots: {len(snaps)}")
        print("")
        for snap in snaps:
            print(f"  {snap['snapshot']}  hash={snap['hash'][:16]}...  size={snap['size']} bytes")
    print("=" * 60)
    return 0


def cmd_body_export(args: argparse.Namespace) -> int:
    """Export the current body directory as a zip or tar archive."""
    try:
        resolver = _resolver_for(args)
    except BodyRegistryError as exc:
        print(f"[ROSClaw] {exc}")
        return 1
    if not resolver.is_linked():
        print("[ROSClaw] No body linked.")
        return 1

    dest = Path(args.dest)
    fmt = args.format

    body_id = resolver.body_id
    if dest.is_dir() or args.dest.endswith("/"):
        dest.mkdir(parents=True, exist_ok=True)
        archive_path = dest / f"{body_id}.{fmt}"
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        archive_path = dest
        if not archive_path.suffix:
            archive_path = archive_path.with_suffix(f".{fmt}")

    body_dir = resolver.body_dir
    if fmt == "zip":
        with ZipFile(archive_path, "w") as zf:
            for path in body_dir.rglob("*"):
                if path.is_file():
                    arcname = f"{body_id}/{path.relative_to(body_dir)}"
                    zf.write(path, arcname)
    else:
        with tarfile.open(archive_path, "w") as tf:
            for path in body_dir.rglob("*"):
                if path.is_file():
                    arcname = f"{body_id}/{path.relative_to(body_dir)}"
                    tf.add(path, arcname)

    print(f"Exported body '{body_id}' to {archive_path}")
    return 0


def _utc_now() -> str:
    from datetime import datetime
    return datetime.now(UTC).isoformat()
