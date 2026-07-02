"""Runtime handlers for camera/perception skills."""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any

from rosclaw.runtime.plugin import runtime_handler

logger = logging.getLogger("rosclaw.runtime.handlers.camera")


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _camera_name_from_body(params: dict[str, Any]) -> str:
    """Resolve the RealSense camera namespace from the linked body.

    Priority:
      1. Explicit ``camera_name`` / ``camera`` parameter.
      2. ROS2 topic prefix declared in the linked e-URDF profile.
      3. Body instance id heuristic (d435i / d405 / dual).
    """
    camera = params.get("camera_name") or params.get("camera")
    if camera:
        return camera

    try:
        from rosclaw.body.resolver import BodyResolver
        from rosclaw.firstboot.workspace import resolve_home

        home = resolve_home(params.get("workspace"))
        body_id_arg = params.get("body_id")
        resolver = BodyResolver(workspace=home, body_id=body_id_arg)
        if resolver.is_linked():
            body = resolver.get_current_body_yaml().to_dict()
            body_id = body.get("body_instance", {}).get("id", "")
            # Heuristic fallback.
            if "d435i" in body_id.lower():
                return "d435i"
            if "d405" in body_id.lower():
                return "d405"
            # Try to read the camera namespace from the linked profile sensors.
            profile_path = resolver.eurdf_profile_path
            if profile_path.exists():
                import yaml

                profile = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
                for sensor in profile.get("sensors", []):
                    topic = sensor.get("topic", "")
                    m = re.match(r"/camera/([^/]+)/", topic)
                    if m:
                        return m.group(1)
    except Exception as exc:
        logger.warning("Failed to resolve camera name from body: %s", exc)

    return "d435i"


def _discover_realsense_ros_mcp(home: Path | None) -> str | None:
    """Return the installed realsense-ros-mcp server name if healthy."""
    from rosclaw.mcp.onboarding.installed import InstalledRegistry
    from rosclaw.mcp.onboarding.stdio_client import list_server_tools

    registry = InstalledRegistry(home=home) if home else InstalledRegistry()
    for rec in registry.list():
        name = rec.server_name.lower()
        if "realsense-ros" in name:
            try:
                tools = list_server_tools(rec.server_name, home=home, timeout=20.0)
                tool_names = {t.get("name") for t in tools}
                if "capture_rgbd" in tool_names:
                    return rec.server_name
            except Exception as exc:
                logger.warning("Health check failed for %s: %s", rec.server_name, exc)
                continue
    return None


def _copy_artifact(src: str, dst_dir: Path) -> Path | None:
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


@runtime_handler("realsense_capture_rgbd")
def _handle_realsense_capture_rgbd(params: dict[str, Any]) -> dict[str, Any]:
    """Runtime handler for the RealSense RGB-D capture skill.

    Captures an aligned color/depth frame pair through the realsense-ros-mcp
    server and returns artifact paths. All execution metadata is published as
    runtime events by ``SkillExecutor``.
    """
    from rosclaw.firstboot.workspace import resolve_home

    t0 = time.time()
    home = resolve_home(params.get("workspace"))
    camera_name = _camera_name_from_body(params)

    output_dir = Path(
        params.get("output_dir")
        or params.get("output")
        or str(home / "captures" / "realsense_capture_rgbd" / _utc_now().replace(":", "-"))
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    server_name = _discover_realsense_ros_mcp(home)
    if not server_name:
        return {
            "status": "error",
            "skill": "realsense_capture_rgbd",
            "reason": (
                "No realsense-ros-mcp server installed or healthy. "
                "Install with: rosclaw mcp install --from-git https://github.com/ros-claw/realsense-ros-mcp"
            ),
            "camera_name": camera_name,
        }

    color_path = str(output_dir / "color.png")
    depth_path = str(output_dir / "depth.png")

    try:
        from rosclaw.mcp.onboarding.stdio_client import call_server_tool

        raw = call_server_tool(
            server_name,
            "capture_rgbd",
            {
                "camera_name": camera_name,
                "color_path": color_path,
                "depth_path": depth_path,
                "aligned": True,
                "timeout_sec": params.get("timeout_sec", 10.0),
            },
            home=home,
            timeout=30.0,
        )
    except Exception as exc:
        logger.exception("realsense-ros-mcp capture_rgbd failed")
        return {
            "status": "error",
            "skill": "realsense_capture_rgbd",
            "reason": f"MCP call failed: {exc}",
            "camera_name": camera_name,
        }

    # The MCP server wraps the result in MCP content; prefer structured result.
    payload: dict[str, Any] = {}
    structured = raw.get("structuredContent") or {}
    if isinstance(structured, dict) and structured.get("result"):
        try:
            payload = json.loads(structured["result"])
        except Exception:
            payload = {}
    if not payload and raw.get("content"):
        text = raw["content"][0].get("text", "{}") if raw["content"] else "{}"
        try:
            payload = json.loads(text)
        except Exception:
            payload = {}

    if not payload.get("success"):
        return {
            "status": "error",
            "skill": "realsense_capture_rgbd",
            "reason": payload.get("error") or "capture_rgbd returned failure",
            "camera_name": camera_name,
            "mcp_result": payload,
        }

    color_info = payload.get("color", {})
    depth_info = payload.get("depth", {})

    # Copy artifacts into the requested output directory if the server wrote
    # them elsewhere.
    color_dst = _copy_artifact(color_info.get("path"), output_dir)
    depth_dst = _copy_artifact(depth_info.get("path"), output_dir)

    color_artifact = str(color_dst or color_info.get("path", color_path))
    depth_artifact = str(depth_dst or depth_info.get("path", depth_path))
    return {
        "status": "success",
        "skill": "realsense_capture_rgbd",
        "camera_name": camera_name,
        "frames": {
            "color": color_artifact,
            "depth": depth_artifact,
        },
        "artifacts": {
            "color": color_artifact,
            "depth": depth_artifact,
        },
        "topics": {
            "color": color_info.get("topic"),
            "depth": depth_info.get("topic"),
        },
        "resolution": {
            "width": color_info.get("width"),
            "height": color_info.get("height"),
        },
        "encoding": {
            "color": color_info.get("encoding"),
            "depth": depth_info.get("encoding"),
        },
        "duration_sec": time.time() - t0,
        "source": "runtime_handler",
    }


@runtime_handler("scene_risk_scan")
def _handle_scene_risk_scan(params: dict[str, Any]) -> dict[str, Any]:
    """Runtime handler for the scene risk scan skill.

    In a full implementation this would call ``PhysicalReasoner.reason()`` with
    the latest camera frame. The runtime handler returns a safe stub so the
    skill dispatch path can be tested without a live reasoning endpoint.
    """
    return {
        "status": "success",
        "skill": "scene_risk_scan",
        "scene": params.get("scene", "unknown"),
        "risk_score": 0.0,
        "physical_risks": [],
        "requires_guard": True,
        "source": "runtime_handler",
    }
