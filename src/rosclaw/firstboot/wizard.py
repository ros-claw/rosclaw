"""ROSClaw First Boot interactive and non-interactive wizard."""

from __future__ import annotations

import argparse
import contextlib
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rosclaw.feedback.directories import ensure_feedback_dirs
from rosclaw.feedback.installation import InstallationManager
from rosclaw.feedback.telemetry_client import TelemetryClient

from .config import FirstbootConfig, generate_rosclaw_yaml
from .doctor import FirstbootDoctor
from .mcp import generate_mcp_config
from .telemetry import generate_feedback_yaml, generate_telemetry_yaml
from .workspace import (
    init_workspace,
    is_workspace_initialized,
    load_install_state,
    resolve_home,
    save_install_state,
)

WELCOME_MESSAGE = """Welcome to ROSClaw First Boot

ROSClaw grounds AI agents into the physical world through:
  • local runtime
  • e-URDF embodiment profiles
  • sandbox / firewall validation
  • practice capture
  • memory
  • MCP agent interface
  • self-evolving skills

No robot will be moved during first boot.
"""

SAFETY_NOTICE = """
╔══════════════════════════════════════════════════════════════╗
║  Real Robot Safety Notice                                    ║
║                                                              ║
║  • ROSClaw will not connect to or command any real robot     ║
║    during first boot.                                        ║
║  • All real-robot capabilities require explicit opt-in.      ║
║  • Keep emergency stop systems engaged when testing.         ║
╚══════════════════════════════════════════════════════════════╝
"""


# ------------------------------------------------------------------
# Interactive prompts
# ------------------------------------------------------------------


def _is_tty() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def ask_choice(prompt: str, choices: list[str], default: str) -> str:
    """Ask user to select one of several choices."""
    if not _is_tty():
        return default

    print(prompt)
    for idx, item in enumerate(choices, 1):
        marker = " [default]" if item == default else ""
        print(f"  {idx}. {item}{marker}")

    try:
        value = input(f"Select [{default}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return default

    if not value:
        return default

    if value.isdigit():
        idx = int(value) - 1
        if 0 <= idx < len(choices):
            return choices[idx]

    if value in choices:
        return value

    print(f"Invalid choice. Using default: {default}")
    return default


def ask_yes_no(prompt: str, default: bool) -> bool:
    """Ask a yes/no question."""
    if not _is_tty():
        return default

    suffix = " [Y/n]" if default else " [y/N]"
    try:
        value = input(f"{prompt}{suffix}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return default

    if not value:
        return default
    return value in ("y", "yes")


def ask_text(prompt: str, default: str) -> str:
    """Ask for free text input."""
    if not _is_tty():
        return default

    try:
        value = input(f"{prompt} [{default}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return default

    return value if value else default


# ------------------------------------------------------------------
# Interactive flow
# ------------------------------------------------------------------


def _interactive_workspace(home: Path) -> Path:
    print("Where should ROSClaw store local data?")
    print(f"  [default: {home}]")
    print("  1. ~/.rosclaw")
    print("  2. ./.rosclaw")
    print("  3. custom path")
    choice = ask_choice("Select:", ["~/.rosclaw", "./.rosclaw", "custom"], "~/.rosclaw")
    if choice == "./.rosclaw":
        return Path(".rosclaw").resolve()
    if choice == "custom":
        custom = ask_text("Enter path", str(home))
        return Path(custom).expanduser().resolve()
    return home


def _interactive_profile(default: str = "offline") -> str:
    return ask_choice(
        "Choose your default operating mode:",
        ["offline", "cloud", "hybrid"],
        default,
    )


def _interactive_use_cases() -> dict[str, bool]:
    print("\nWhat do you want to use ROSClaw for first?")
    defaults = {
        "sandbox": True,
        "ros2": False,
        "mcp": True,
        "practice": False,
        "memory": False,
        "auto": False,
        "real_robot": False,
    }
    if not _is_tty():
        return defaults

    choices = [
        ("sandbox", "Simulation sandbox"),
        ("ros2", "ROS 2 robot connection"),
        ("mcp", "MCP Agent / Claude Code integration"),
        ("practice", "Practice capture"),
        ("memory", "Memory experiments"),
        ("auto", "Auto evolution / Darwin benchmark"),
        ("real_robot", "Real robot safety precheck"),
    ]

    selected: dict[str, bool] = {}
    for key, label in choices:
        selected[key] = ask_yes_no(label, defaults[key])
    return selected


def _interactive_robot() -> str:
    robots = ["sim_ur5e", "turtlebot", "unitree_go2", "unitree_g1", "custom / later"]
    return ask_choice("Choose a default robot profile:", robots, "sim_ur5e")


def _interactive_safety() -> str:
    return ask_choice(
        "Choose safety level:",
        ["strict", "moderate", "relaxed"],
        "strict",
    )


def _interactive_telemetry() -> bool:
    print(
        "\nHelp improve ROSClaw?\n\n"
        "ROSClaw can send lightweight anonymous product telemetry:\n"
        "  • install success\n"
        "  • firstboot completion\n"
        "  • version\n"
        "  • OS / Python / ROS environment\n"
        "  • command success/failure summary\n"
        "  • module usage\n"
        "  • daily active heartbeat\n\n"
        "Never sent by default:\n"
        "  • prompts\n"
        "  • logs\n"
        "  • camera/video/audio\n"
        "  • local file paths\n"
        "  • API keys\n"
        "  • robot serial numbers\n"
        "  • MCAP/raw traces\n\n"
        "You can disable this anytime:\n"
        "  rosclaw feedback telemetry off"
    )
    return ask_yes_no("Enable lightweight product telemetry", True)


def _interactive_diagnostics() -> bool:
    print(
        "\nAllow ROSClaw to upload redacted diagnostic summaries when errors occur?\n"
        "This may include:\n"
        "  • crash type\n"
        "  • failure type\n"
        "  • sandbox block reason\n"
        "  • provider latency bucket\n\n"
        "This does NOT include full stacktraces, prompts, logs, or paths."
    )
    return ask_yes_no("Enable redacted diagnostics", False)


def _interactive_rich_feedback() -> bool:
    print(
        "\nRich feedback bundles (logs, media, MCAP) are only uploaded manually.\n"
        "Keep manual-only mode enabled for maximum privacy."
    )
    return ask_yes_no("Enable manual rich feedback upload", False)


def _print_summary(
    home: Path,
    profile: str,
    robot: str,
    safety: str,
    telemetry_enabled: bool,
    diagnostics_enabled: bool,
    rich_feedback_enabled: bool,
    enable_mcp: bool,
    use_cases: dict[str, bool],
) -> None:
    print("\n" + "=" * 62)
    print(" First Boot Summary")
    print("=" * 62)
    print(f"Workspace:       {home}")
    print(f"Profile:         {profile}")
    print(f"Robot:           {robot}")
    print(f"Safety:          {safety}")
    print(f"Telemetry:       {'enabled' if telemetry_enabled else 'disabled'}")
    print(f"Diagnostics:     {'enabled' if diagnostics_enabled else 'disabled'}")
    print(f"Rich feedback:   {'enabled' if rich_feedback_enabled else 'disabled'}")
    print(f"MCP config:      {'enabled' if enable_mcp else 'disabled'}")
    print("Use cases:")
    for key, val in use_cases.items():
        print(f"  • {key}: {'yes' if val else 'no'}")
    print("=" * 62)


def run_firstboot_interactive(args: argparse.Namespace) -> int:
    """Run the interactive firstboot wizard."""
    print(WELCOME_MESSAGE)

    home = resolve_home(getattr(args, "workspace", None))
    home = _interactive_workspace(home)

    if is_workspace_initialized(home) and not args.force:
        print(f"\nExisting ROSClaw workspace detected: {home}")
        print("A backup of your current rosclaw.yaml will be created before merging.")
        if not ask_yes_no("Continue and merge configuration", True):
            print("First boot cancelled. Existing configuration kept.")
            return 0

    profile = _interactive_profile(getattr(args, "profile", "offline"))
    use_cases = _interactive_use_cases()
    robot = _interactive_robot() if not getattr(args, "robot", None) else args.robot
    safety = _interactive_safety() if not getattr(args, "safety", None) else args.safety

    print("\nProvider setup: skipped by default. Set API keys via environment variables later.")

    enable_mcp = use_cases.get("mcp", True)
    if enable_mcp:
        enable_mcp = ask_yes_no("Generate MCP config sample", True)

    telemetry_enabled = _interactive_telemetry()
    diagnostics_enabled = _interactive_diagnostics()
    rich_feedback_enabled = _interactive_rich_feedback()

    _print_summary(
        home=home,
        profile=profile,
        robot=robot,
        safety=safety,
        telemetry_enabled=telemetry_enabled,
        diagnostics_enabled=diagnostics_enabled,
        rich_feedback_enabled=rich_feedback_enabled,
        enable_mcp=enable_mcp,
        use_cases=use_cases,
    )

    if not ask_yes_no("Apply these settings", True):
        print("\nFirst boot cancelled. No changes were made.")
        return 0

    if use_cases.get("real_robot", False):
        safety_path = home / "config" / "REAL_ROBOT_SAFETY.md"
        safety_path.write_text(SAFETY_NOTICE, encoding="utf-8")
        print(f"\n{safety_path} written.")

    return _write_firstboot(
        home=home,
        profile=profile,
        robot=robot,
        safety=safety,
        telemetry_enabled=telemetry_enabled,
        diagnostics_enabled=diagnostics_enabled,
        rich_feedback_enabled=rich_feedback_enabled,
        enable_mcp=enable_mcp,
        use_cases=use_cases,
        force=args.force,
        json_output=args.json,
        dev=args.dev,
    )


# ------------------------------------------------------------------
# Non-interactive flow
# ------------------------------------------------------------------


def run_firstboot_noninteractive(args: argparse.Namespace) -> int:
    """Run firstboot with CLI arguments only."""
    home = resolve_home(getattr(args, "workspace", None))

    profile = getattr(args, "profile", "offline")
    robot = getattr(args, "robot", "sim_ur5e")
    safety = getattr(args, "safety", "strict")

    telemetry_enabled = not args.no_telemetry if getattr(args, "no_telemetry", False) else True
    diagnostics_enabled = getattr(args, "diagnostics", False) and not getattr(args, "no_diagnostics", False)
    rich_feedback_enabled = getattr(args, "rich_feedback", False) and not getattr(args, "no_rich_feedback", False)
    enable_mcp = not args.disable_mcp if getattr(args, "disable_mcp", False) else True

    use_cases = {
        "sandbox": True,
        "ros2": getattr(args, "enable_ros2", False),
        "mcp": enable_mcp,
        "practice": getattr(args, "enable_practice", False),
        "memory": getattr(args, "enable_memory", False),
        "auto": getattr(args, "enable_auto", False),
        "real_robot": False,
    }

    return _write_firstboot(
        home=home,
        profile=profile,
        robot=robot,
        safety=safety,
        telemetry_enabled=telemetry_enabled,
        diagnostics_enabled=diagnostics_enabled,
        rich_feedback_enabled=rich_feedback_enabled,
        enable_mcp=enable_mcp,
        use_cases=use_cases,
        force=args.force,
        json_output=args.json,
        dev=args.dev,
    )


# ------------------------------------------------------------------
# Shared write logic
# ------------------------------------------------------------------


def _write_firstboot(
    home: Path,
    profile: str,
    robot: str,
    safety: str,
    telemetry_enabled: bool,
    diagnostics_enabled: bool,
    rich_feedback_enabled: bool,
    enable_mcp: bool,
    use_cases: dict[str, bool],
    force: bool,
    json_output: bool,
    dev: bool,
) -> int:
    """Create workspace, write configs, run doctor, print results."""
    _ = dev  # reserved for future dev-mode path handling

    init_workspace(home, force=force)
    ensure_feedback_dirs(home)

    config = FirstbootConfig(
        workspace={"home": str(home), "profile": profile},
        runtime={"robot_id": robot, "safety_level": safety},
    )
    config.apply_profile(profile)

    if use_cases.get("sandbox", True):
        config.sandbox["enabled"] = True
    if use_cases.get("ros2", False):
        config.runtime["ros2_enabled"] = True
    if use_cases.get("practice", False):
        config.practice["enabled"] = True
    if use_cases.get("memory", False):
        config.memory["enabled"] = True
    if use_cases.get("auto", False):
        config.auto["enabled"] = True
        config.darwin["enabled"] = True

    config.telemetry["enabled"] = telemetry_enabled
    config.telemetry["anonymous_install_ping"] = telemetry_enabled
    config.mcp["enabled"] = enable_mcp

    generate_rosclaw_yaml(home, config)
    generate_telemetry_yaml(home, telemetry_enabled)
    generate_feedback_yaml(home)
    if enable_mcp:
        generate_mcp_config(home)

    install_state = load_install_state(home) or {}
    install_channel = install_state.get("install_channel", "stable")
    install_state["firstboot_completed"] = True
    install_state["firstboot_profile"] = profile
    install_state["firstboot_robot"] = robot
    install_state["firstboot_safety"] = safety
    install_state["firstboot_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    save_install_state(home, install_state)

    installation = InstallationManager(home).ensure_installation(
        install_channel=install_channel,
        telemetry_enabled=telemetry_enabled,
        diagnostics_enabled=diagnostics_enabled,
        rich_feedback_enabled=rich_feedback_enabled,
    )

    client = TelemetryClient(home)
    if telemetry_enabled:
        with contextlib.suppress(Exception):
            client.record_event(event_type="firstboot_started")
        if installation.is_new:
            with contextlib.suppress(Exception):
                client.record_event(
                    event_type="install_completed",
                    payload={"install_channel": install_channel},
                )

    doctor = FirstbootDoctor(home)
    result = doctor.run_full(fix=False, json_output=json_output)

    if telemetry_enabled and not json_output:
        with contextlib.suppress(Exception):
            client.record_event(
                event_type="firstboot_completed",
                payload={
                    "profile": profile,
                    "robot": robot,
                    "safety": safety,
                    "doctor_status": result.status.value.lower(),
                    "enabled_modules": [k for k, v in use_cases.items() if v],
                },
            )

    if not json_output:
        _print_success(home, profile, result.status.value, result.checks)

    return result.exit_code


def _print_success(home: Path, profile: str, status: str, checks: list[Any]) -> None:
    warnings = [c for c in checks if getattr(c, "status", None) in ("WARN", "FAIL")]

    print("\n" + "=" * 62)
    print(" ROSClaw First Boot Complete")
    print("=" * 62)
    print(f"Profile:    {profile}")
    print(f"Workspace:  {home}")
    print(f"Config:     {home / 'config' / 'rosclaw.yaml'}")
    print(f"Runtime:    {status}")
    if warnings:
        print("\nWarnings:")
        for c in warnings:
            print(f"  • {getattr(c, 'name', c.id)}: {getattr(c, 'message', '')}")
    print("\nNext steps:")
    print("  1. Check status")
    print("     rosclaw status")
    print("\n  2. List robot profiles")
    print("     rosclaw robot list")
    print("\n  3. Run a sandbox demo")
    print("     rosclaw sandbox run --robot sim_ur5e --world tabletop --task reach")
    print("\n  4. Start MCP server")
    print("     rosclaw hub start")
    print("\n  5. View doctor report")
    print("     rosclaw doctor --full")
    print("=" * 62)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def run_firstboot(args: argparse.Namespace) -> int:
    """Dispatch to interactive or non-interactive firstboot."""
    yes_mode = args.yes or not _is_tty()
    if yes_mode:
        return run_firstboot_noninteractive(args)
    return run_firstboot_interactive(args)


__all__ = [
    "ask_choice",
    "ask_yes_no",
    "ask_text",
    "run_firstboot",
    "run_firstboot_interactive",
    "run_firstboot_noninteractive",
]
