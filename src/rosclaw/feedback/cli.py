"""Feedback and telemetry CLI commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from rosclaw.firstboot.workspace import resolve_home

from .config import TelemetryConfig
from .consent import ConsentManager
from .export import FeedbackExporter
from .installation import InstallationManager
from .store import count_events, directory_size_mb, event_file_for_date
from .telemetry_client import TelemetryClient
from .upload import FeedbackUploader


def add_feedback_subparser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[name-defined]
    """Register the `rosclaw feedback` command tree."""
    feedback_parser = subparsers.add_parser(
        "feedback",
        help="Feedback, telemetry, and privacy controls",
    )
    feedback_subparsers = feedback_parser.add_subparsers(dest="feedback_command")

    # status
    status_parser = feedback_subparsers.add_parser(
        "status", help="Show feedback and telemetry status"
    )
    status_parser.add_argument("--workspace", default=None, help="ROSClaw workspace path")

    # telemetry
    telemetry_parser = feedback_subparsers.add_parser(
        "telemetry", help="Manage product telemetry"
    )
    telemetry_parser.add_argument("--workspace", default=None, help="ROSClaw workspace path")
    telemetry_subparsers = telemetry_parser.add_subparsers(dest="telemetry_command")

    for name, help_text in [
        ("status", "Show telemetry status"),
        ("on", "Enable product telemetry"),
        ("off", "Disable product telemetry"),
        ("ping", "Send a test telemetry ping"),
        ("reset-id", "Rotate the anonymous installation ID"),
    ]:
        p = telemetry_subparsers.add_parser(name, help=help_text)
        p.add_argument("--workspace", default=None, help="ROSClaw workspace path")

    # consent
    consent_parser = feedback_subparsers.add_parser(
        "consent", help="Manage upload consent"
    )
    consent_parser.add_argument("--workspace", default=None, help="ROSClaw workspace path")
    consent_parser.add_argument("--diagnostics", action="store_true", help="Enable diagnostic upload consent")
    consent_parser.add_argument("--rich-feedback", action="store_true", help="Enable rich feedback upload consent")
    consent_parser.add_argument("--revoke-diagnostics", action="store_true", help="Revoke diagnostic upload consent")
    consent_parser.add_argument("--revoke-all", action="store_true", help="Revoke all upload consent")
    consent_parser.add_argument("--show", action="store_true", help="Show current consent")

    # export
    export_parser = feedback_subparsers.add_parser(
        "export", help="Export local feedback events to a bundle"
    )
    export_parser.add_argument("--workspace", default=None, help="ROSClaw workspace path")
    export_parser.add_argument("--days", type=int, default=30, help="Number of days to include")
    export_parser.add_argument("--redact", action="store_true", help="Redact sensitive fields")
    export_parser.add_argument("--output", default=None, help="Output tar.gz path")

    # upload
    upload_parser = feedback_subparsers.add_parser(
        "upload", help="Upload a redacted feedback bundle"
    )
    upload_parser.add_argument("--workspace", default=None, help="ROSClaw workspace path")
    upload_parser.add_argument("--redact", action="store_true", required=True, help="Redact before upload")
    upload_parser.add_argument("--days", type=int, default=30, help="Number of days to include")
    upload_parser.add_argument("--dry-run", action="store_true", help="Prepare bundle without uploading")
    upload_parser.add_argument("--include-media", action="store_true", help="Include local media files")


def dispatch_feedback_command(args: argparse.Namespace) -> int:
    """Dispatch to the appropriate feedback subcommand."""
    command = getattr(args, "feedback_command", None)
    if command == "status":
        return cmd_feedback_status(args)
    if command == "telemetry":
        return cmd_feedback_telemetry(args)
    if command == "consent":
        return cmd_feedback_consent(args)
    if command == "export":
        return cmd_feedback_export(args)
    if command == "upload":
        return cmd_feedback_upload(args)

    print("Usage: rosclaw feedback {status|telemetry|consent|export|upload}")
    return 1


def cmd_feedback_status(args: argparse.Namespace) -> int:
    home = resolve_home(getattr(args, "workspace", None))
    install_mgr = InstallationManager(home)
    telemetry_cfg = TelemetryConfig.load(home)
    consent = ConsentManager(home).show()

    anonymous_id = install_mgr.get_anonymous_installation_id()
    telemetry_enabled = bool(
        telemetry_cfg.mode.get("enabled", True)
        and telemetry_cfg.mode.get("product_telemetry", True)
    )

    telemetry_count = count_events(event_file_for_date(home, "telemetry"))
    feedback_count = count_events(event_file_for_date(home, "feedback"))
    crash_count = len(list((home / "feedback" / "crashes").glob("crash_*.json")))
    local_size = directory_size_mb(home / "telemetry") + directory_size_mb(home / "feedback")

    print("ROSClaw Feedback & Telemetry Status\n")
    print("Product Telemetry:")
    print(f"  enabled: {telemetry_enabled}")
    print(f"  heartbeat: {telemetry_cfg.product_telemetry.get('heartbeat', True)}")
    last_hb = _last_heartbeat(home)
    print(f"  last_heartbeat: {last_hb or 'never'}")
    print(f"  anonymous_installation_id: {anonymous_id or 'not initialized'}\n")

    print("Diagnostics:")
    print(f"  enabled: {consent.diagnostics}")
    print(f"  upload: {consent.diagnostics}")
    print("  redact: True\n")

    print("Rich Feedback:")
    print("  manual_upload_only: True")
    print("  raw_prompt_upload: False")
    print(f"  raw_media_upload: {consent.rich_feedback and False}")
    print("  raw_mcap_upload: False\n")

    print("Local Store:")
    print(f"  telemetry_events: {telemetry_count}")
    print(f"  feedback_events: {feedback_count}")
    print(f"  crashes: {crash_count}")
    print(f"  local_size: {local_size} MB\n")

    print("Commands:")
    print("  disable telemetry:")
    print("    rosclaw feedback telemetry off\n")
    print("  export local feedback:")
    print("    rosclaw feedback export --days 7\n")
    print("  upload redacted feedback:")
    print("    rosclaw feedback upload --redact")
    return 0


def cmd_feedback_telemetry(args: argparse.Namespace) -> int:
    home = resolve_home(getattr(args, "workspace", None))
    subcommand = getattr(args, "telemetry_command", "status")
    install_mgr = InstallationManager(home)

    if subcommand == "status":
        cfg = TelemetryConfig.load(home)
        enabled = bool(cfg.mode.get("enabled", True) and cfg.mode.get("product_telemetry", True))
        print(f"Product telemetry is {'enabled' if enabled else 'disabled'}.")
        return 0

    if subcommand == "on":
        cfg = TelemetryConfig.load(home)
        cfg.mode["enabled"] = True
        cfg.mode["product_telemetry"] = True
        cfg.product_telemetry["enabled"] = True
        cfg.save(home)
        install_mgr.set_telemetry_enabled(True)
        print("Product telemetry is now enabled.")
        return 0

    if subcommand == "off":
        cfg = TelemetryConfig.load(home)
        cfg.mode["enabled"] = False
        cfg.mode["product_telemetry"] = False
        cfg.product_telemetry["enabled"] = False
        cfg.save(home)
        install_mgr.set_telemetry_enabled(False)
        print("Product telemetry is now disabled.")
        print("\nROSClaw will no longer send:")
        print("- heartbeat")
        print("- command usage")
        print("- module usage")
        print("- version distribution events")
        print("\nLocal feedback storage is unchanged.")
        return 0

    if subcommand == "ping":
        result = TelemetryClient(home).ping()
        if result.get("ok"):
            print("Telemetry ping succeeded.")
            print(f"endpoint: {TelemetryConfig.load(home).upload.get('endpoint')}")
            print(f"request_id: {result.get('request_id', 'n/a')}")
            return 0
        print("Telemetry ping failed, but ROSClaw will continue to work locally.")
        print(f"reason: {result.get('error', 'unknown')}")
        return 0

    if subcommand == "reset-id":
        install_mgr.reset_installation_id()
        new_anon = install_mgr.get_anonymous_installation_id()
        print("Installation ID rotated.")
        print(f"New anonymous installation ID: {new_anon}")
        return 0

    print("Usage: rosclaw feedback telemetry {status|on|off|ping|reset-id}")
    return 1


def cmd_feedback_consent(args: argparse.Namespace) -> int:
    home = resolve_home(getattr(args, "workspace", None))
    mgr = ConsentManager(home)

    if args.revoke_all:
        state = mgr.revoke_all()
    elif args.revoke_diagnostics:
        state = mgr.set_diagnostics(False)
    elif args.diagnostics:
        state = mgr.set_diagnostics(True)
    elif args.rich_feedback:
        state = mgr.set_rich_feedback(True)
    else:
        state = mgr.show()

    if getattr(args, "show", False) or not any([
        args.diagnostics, args.rich_feedback, args.revoke_diagnostics, args.revoke_all
    ]):
        print("ROSClaw Feedback Consent\n")
        print(f"Product telemetry: {'enabled' if state.product_telemetry else 'disabled'}")
        print(f"Diagnostics upload: {'enabled' if state.diagnostics else 'disabled'}")
        print(f"Rich feedback upload: {'enabled' if state.rich_feedback else 'disabled'}")
        print(f"Updated at: {state.updated_at}")
        return 0

    print("Consent updated.")
    print(f"Diagnostics: {'enabled' if state.diagnostics else 'disabled'}")
    print(f"Rich feedback: {'enabled' if state.rich_feedback else 'disabled'}")
    return 0


def cmd_feedback_export(args: argparse.Namespace) -> int:
    home = resolve_home(getattr(args, "workspace", None))
    output = FeedbackExporter(home).export(
        days=args.days,
        redact=args.redact,
        output_path=args.output,
    )
    print(f"Exported feedback bundle to: {output}")
    return 0


def cmd_feedback_upload(args: argparse.Namespace) -> int:
    home = resolve_home(getattr(args, "workspace", None))
    result = FeedbackUploader(home).upload(
        redact=args.redact,
        days=args.days,
        dry_run=args.dry_run,
        include_media=args.include_media,
    )
    if not result.get("ok"):
        print(f"Upload failed: {result.get('error')}")
        if result.get("message"):
            print(result["message"])
        return 1

    if result.get("dry_run"):
        print(f"Dry-run prepared bundle: {result.get('bundle_path')}")
        return 0

    print("Feedback upload accepted.")
    if result.get("request_id"):
        print(f"request_id: {result['request_id']}")
    return 0


def _last_heartbeat(home: Path) -> str | None:
    path = home / "telemetry" / "heartbeat" / "last_heartbeat.json"
    if not path.exists():
        return None
    try:
        data = __import__("json").loads(path.read_text(encoding="utf-8"))
        return data.get("timestamp")
    except Exception:
        return None


__all__ = ["add_feedback_subparser", "dispatch_feedback_command"]
