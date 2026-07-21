"""Builtin runner for the ``realsense_capture_rgbd`` skill.

This handler is executed by ``SkillExecutor``.  It never imports
``pyrealsense2`` directly; it talks to the bound RealSense MCP server over
stdio and writes the resulting artifacts to the requested output directory.
"""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("rosclaw.skill.builtins.realsense_capture_rgbd")


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _load_body(
    params: dict[str, Any],
) -> tuple[Path | None, str | None, dict[str, Any], dict[str, Any]]:
    """Resolve body workspace, id, body yaml, and eurdf profile from parameters."""
    from rosclaw.body.resolver import BodyResolver
    from rosclaw.firstboot.workspace import resolve_home

    workspace = params.get("workspace")
    home = resolve_home(str(workspace) if workspace else None)
    body_id = params.get("body_id") or params.get("body")
    resolver: BodyResolver | None = None
    if body_id:
        try:
            resolver = BodyResolver(workspace=home, body_id=body_id)
        except Exception:
            resolver = None
    if resolver is None:
        resolver = BodyResolver(workspace=home)
    if not resolver.is_linked():
        return home, body_id, {}, {}
    try:
        body = resolver.get_current_body_yaml().to_dict()
    except Exception:
        body = {}
    profile: dict[str, Any] = {}
    try:
        import yaml

        profile_text = resolver.eurdf_profile_path.read_text(encoding="utf-8")
        profile = yaml.safe_load(profile_text) or {}
    except Exception:
        profile = {}
    return home, body_id, body, profile


def _is_perception_only(body: dict[str, Any], profile: dict[str, Any] | None = None) -> bool:
    """Check whether the active body is a perception-only camera rig.

    The body yaml may not carry the perception-only marker if it was generated
    from an older body service; fall back to the linked e-URDF profile.
    """
    safety = body.get("safety", {})
    if isinstance(safety, dict):
        env = safety.get("environment", {})
        if env.get("perception_only") or env.get("no_actuation"):
            return True
        limits = safety.get("safety_limits", {})
        if limits.get("perception_only") or limits.get("no_actuation"):
            return True
    meta = body.get("metadata", {})
    if meta.get("perception_only") or meta.get("no_actuation"):
        return True
    forbidden = body.get("forbidden_capabilities", []) or []
    if "actuation" in forbidden or "motion" in forbidden:
        return True
    if profile:
        identity = profile.get("identity", {})
        if identity.get("robot_class") in (
            "perception_only_camera",
            "realsense_d405",
            "realsense_d435i",
        ):
            return True
        profile_safety = profile.get("safety", {})
        limits = profile_safety.get("safety_limits", {}) if isinstance(profile_safety, dict) else {}
        if limits.get("perception_only") or limits.get("no_actuation"):
            return True
    return False


def _discover_realsense_mcp(
    home: Path | None,
    *,
    required_server_name: str | None = None,
) -> tuple[str, str] | tuple[None, None]:
    """Return (server_name, tool_name) for an installed RealSense MCP."""
    from rosclaw.mcp.onboarding.installed import InstalledRegistry
    from rosclaw.mcp.onboarding.stdio_client import list_server_tools

    registry = InstalledRegistry(home=home) if home else InstalledRegistry()
    candidates = []
    for rec in registry.list():
        if required_server_name is not None and rec.server_name != required_server_name:
            continue
        name = rec.server_name.lower()
        if "realsense" in name or "librealsense" in name:
            candidates.append(("prefer", rec.server_name))
        elif "d405" in name or "d435" in name:
            candidates.append(("fallback", rec.server_name))

    # Prefer librealsense-mcp, then realsense-ros-mcp, then any realsense.
    order = {"librealsense-mcp": 0, "realsense-ros-mcp": 1}
    candidates.sort(key=lambda x: (order.get(x[1].lower(), 99), x[1]))

    for _prio, server_name in candidates:
        try:
            tools = list_server_tools(server_name, home=home, timeout=20.0)
            tool_names = {t.get("name") for t in tools}
            if "capture_aligned_rgbd" in tool_names:
                return server_name, "capture_aligned_rgbd"
            if "capture_color_image" in tool_names:
                return server_name, "capture_color_image"
            if "capture_frames" in tool_names:
                return server_name, "capture_frames"
        except Exception as exc:
            logger.warning("Health check failed for %s: %s", server_name, exc)
            continue
    return None, None


def _resolve_serial(
    body: dict[str, Any],
    params: dict[str, Any],
    home: Path | None = None,
    profile: dict[str, Any] | None = None,
) -> str | None:
    """Pick a RealSense serial number.

    The explicit ``--serial`` parameter wins, then the linked body's
    ``serial_number``.  If the body has no serial (or it is ``UNKNOWN``), ask
    the MCP to enumerate devices and pick one that matches the e-URDF profile
    when possible.  This avoids grabbing the wrong camera on a multi-RealSense
    rig (e.g. D435I when the body is a D405).
    """
    serial = params.get("serial")
    if serial:
        return serial
    serial = body.get("body_instance", {}).get("serial_number")
    if serial and serial != "UNKNOWN":
        return serial

    # Ask the MCP to enumerate devices and match against the profile.
    from rosclaw.firstboot.workspace import resolve_home
    from rosclaw.mcp.onboarding.installed import InstalledRegistry
    from rosclaw.mcp.onboarding.stdio_client import call_server_tool

    resolved_home = home or resolve_home(None)
    registry = InstalledRegistry(home=resolved_home)
    for rec in registry.list():
        if "realsense" not in rec.server_name.lower():
            continue
        try:
            result = call_server_tool(
                rec.server_name, "list_devices", {}, home=resolved_home, timeout=20.0
            )
            content = result.get("content", [])
            text = content[0].get("text", "{}") if content else "{}"
            data = json.loads(text)
            devices = data.get("devices", [])
            if not devices:
                continue
            # Prefer a device whose product line matches the e-URDF profile.
            matched = [d for d in devices if _device_matches_profile(d, profile)]
            chosen = matched[0] if matched else devices[0]
            return str(chosen.get("serial", chosen.get("serial_number")))
        except Exception:
            continue
    return None


def _device_matches_profile(device: dict[str, Any], profile: dict[str, Any] | None) -> bool:
    """Return True if a RealSense device matches the e-URDF profile."""
    if not profile:
        return False
    name = (device.get("name") or "").lower()
    identity = profile.get("identity", {}) if isinstance(profile, dict) else {}
    robot_class = (identity.get("robot_class") or profile.get("profile_id") or "").lower()
    if not robot_class:
        return False
    if "d405" in robot_class and "d405" in name:
        return True
    if "d435i" in robot_class and "d435i" in name:
        return True
    if "d435" in robot_class and ("d435" in name or "d435i" in name):
        return True
    if "dual" in robot_class:
        return "realsense" in name
    return False


def _copy_artifact(src: str, dst_dir: Path) -> Path | None:
    """Copy an artifact file into the output directory if it exists."""
    if not src:
        return None
    src_path = Path(src)
    if not src_path.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src_path.name
    if src_path.resolve() == dst.resolve():
        return dst
    try:
        shutil.copy2(src_path, dst)
    except shutil.SameFileError:
        return dst
    return dst


def run(params: dict[str, Any]) -> dict[str, Any]:
    """Execute the RealSense RGB-D capture skill."""
    from rosclaw.mcp.onboarding.stdio_client import McpServerSession, McpStdioError

    t0 = time.time()
    home, body_id, body, profile = _load_body(params)

    if body_id and not _is_perception_only(body, profile):
        return {
            "status": "blocked",
            "reason": "Skill requires a perception-only RealSense body",
            "body_id": body_id,
        }

    output_dir = Path(params.get("output_dir") or params.get("output") or "./capture")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    required_server_name = params.get("server_name")
    if required_server_name is not None and not isinstance(required_server_name, str):
        return {
            "status": "error",
            "reason": "Required RealSense MCP server name must be a string",
            "body_id": body_id,
            "artifacts": {},
        }
    server_name, tool_name = _discover_realsense_mcp(
        home,
        required_server_name=required_server_name,
    )
    if not server_name:
        if required_server_name:
            reason = (
                f"Required RealSense MCP server {required_server_name!r} is not installed, "
                "healthy, or compatible"
            )
        else:
            reason = (
                "No RealSense MCP server installed or healthy. Run: rosclaw mcp install "
                "--from-git https://github.com/ros-claw/librealsense-mcp"
            )
        return {
            "status": "error",
            "reason": reason,
            "body_id": body_id,
            "artifacts": {},
        }

    serial = _resolve_serial(body, params, home=home, profile=profile)
    if not serial:
        return {
            "status": "error",
            "reason": "Could not determine RealSense serial number",
            "body_id": body_id,
            "server_name": server_name,
        }

    color_path = str(output_dir / "color.png")
    depth_path = str(output_dir / "depth.png")

    try:
        # Ensure the device pipeline is active before capturing. Both calls must
        # happen in the same stdio session because pipeline state lives in the
        # MCP server process.
        from rosclaw.mcp.onboarding.stdio_client import McpServerSession

        with McpServerSession(server_name, home=home, start_timeout=15.0) as session:
            start_result = session.call("start_pipeline", {"serial": serial}, timeout=60.0)
            if (
                isinstance(start_result, dict)
                and start_result.get("error")
                and "already" not in str(start_result.get("error", "")).lower()
            ):
                return {
                    "status": "error",
                    "reason": f"Failed to start pipeline: {start_result['error']}",
                    "server_name": server_name,
                    "body_id": body_id,
                    "mcp_result": start_result,
                }

            if tool_name == "capture_aligned_rgbd":
                arguments = {
                    "serial": serial,
                    "color_path": color_path,
                    "depth_path": depth_path,
                }
            elif tool_name == "capture_color_image":
                arguments = {"serial": serial, "save_path": color_path}
            else:
                arguments = {"serial": serial, "align_depth": True}

            raw = session.call(tool_name, arguments, timeout=60.0)

        content = raw.get("content", [])
        text = content[0].get("text", "{}") if content else "{}"
        mcp_result = json.loads(text) if isinstance(text, str) else text
    except McpStdioError as exc:
        return {
            "status": "error",
            "reason": f"MCP tool call failed: {exc}",
            "server_name": server_name,
            "tool": tool_name,
            "body_id": body_id,
        }
    except Exception as exc:
        return {
            "status": "error",
            "reason": f"Unexpected error calling {tool_name}: {exc}",
            "server_name": server_name,
            "body_id": body_id,
        }

    if isinstance(mcp_result, dict) and mcp_result.get("error"):
        return {
            "status": "error",
            "reason": mcp_result["error"],
            "server_name": server_name,
            "tool": tool_name,
            "body_id": body_id,
            "mcp_result": mcp_result,
        }

    # Collect artifacts from MCP result and copy them into the output dir.
    artifacts: dict[str, str | None] = {}
    for key in ("color_path", "depth_path", "save_path", "color", "depth"):
        val = mcp_result.get(key) if isinstance(mcp_result, dict) else None
        if val:
            copied = _copy_artifact(str(val), output_dir)
            artifacts[key] = str(copied) if copied else str(val)

    # Ensure expected artifacts exist even if the server returned absolute paths.
    for label, src in [("color", color_path), ("depth", depth_path)]:
        if label not in artifacts:
            artifacts[label] = src if Path(src).exists() else None

    latency_ms = round((time.time() - t0) * 1000, 2)
    usb_mode = "unknown"
    degraded = False
    if isinstance(mcp_result, dict):
        usb_mode = mcp_result.get("usb_mode") or mcp_result.get("usb_type") or usb_mode
        if mcp_result.get("degraded") or "USB2" in str(usb_mode).upper():
            degraded = True

    result = {
        "status": "success",
        "skill": "realsense_capture_rgbd",
        "body_id": body_id,
        "server_name": server_name,
        "tool": tool_name,
        "serial": serial,
        "artifacts": artifacts,
        "metrics": {
            "latency_ms": latency_ms,
            "usb_mode": usb_mode,
            "degraded": degraded,
        },
        "mcp_result": mcp_result,
        "output_dir": str(output_dir),
        "timestamp": _utc_now(),
    }

    # Persist a machine-readable summary next to the images.
    summary_path = output_dir / "capture_result.json"
    with contextlib.suppress(OSError):
        summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    return result
