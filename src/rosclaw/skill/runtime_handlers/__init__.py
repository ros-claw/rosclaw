"""Skill runtime handlers for RealSense skills.

These are small Python functions invoked by ``SkillRunner`` for built-in
RealSense skills.  They call the MCP ``run_tool`` helpers directly (no
subprocess) so that ``rosclaw run skill`` works on the local machine.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from rosclaw.body.resolver import BodyResolver
from rosclaw.firstboot.workspace import resolve_home
from rosclaw.mcp.servers import run_tool as _mcp_run_tool
from rosclaw.provider.cli_call import call_provider


def _resolve_body(body: str | None) -> str:
    """Return a body instance id, falling back to the linked body."""
    if body:
        return body
    resolver = BodyResolver(resolve_home())
    if resolver.is_linked():
        return resolver.get_current_body_id() or "default"
    return "default"


def _camera_name_from_body(body_id: str) -> str:
    """Map a body instance id to the RealSense camera namespace."""
    if "d435i" in body_id.lower():
        return "d435i"
    if "d405" in body_id.lower():
        return "d405"
    return "d405"


def realsense_capture_rgbd(
    body: str | None,
    output_dir: str | None = None,
    duration_sec: float = 5.0,
) -> dict[str, Any]:
    """Capture aligned RGB-D pair and return paths."""
    body_id = _resolve_body(body)
    camera_name = _camera_name_from_body(body_id)
    out = output_dir or "/tmp"
    Path(out).mkdir(parents=True, exist_ok=True)
    result = _mcp_run_tool("capture_rgbd_pair", output_dir=out, camera_name=camera_name)
    result["body"] = body_id
    result["duration_sec"] = duration_sec
    return result


def realsense_depth_health_check(body: str | None, duration_sec: float = 30.0) -> dict[str, Any]:
    """Sample depth frames and compute statistics."""
    body_id = _resolve_body(body)
    camera_name = _camera_name_from_body(body_id)
    # For now, a single-frame check; future iterations can loop.
    result = _mcp_run_tool("check_depth_validity", camera_name=camera_name)
    result["body"] = body_id
    result["duration_sec"] = duration_sec
    return result


def realsense_imu_check(body: str | None, duration_sec: float = 5.0) -> dict[str, Any]:
    """Sample IMU and return a reading."""
    body_id = _resolve_body(body)
    camera_name = _camera_name_from_body(body_id)
    result = _mcp_run_tool("get_imu_sample", camera_name=camera_name)
    result["body"] = body_id
    result["duration_sec"] = duration_sec
    return result


def _camera_info_dict(msg) -> dict[str, Any]:
    """Serialize a sensor_msgs/CameraInfo message."""
    return {
        "width": int(getattr(msg, "width", 0)),
        "height": int(getattr(msg, "height", 0)),
        "distortion_model": str(getattr(msg, "distortion_model", "")),
        "d": list(getattr(msg, "d", [])),
        "k": list(getattr(msg, "k", [])),
        "r": list(getattr(msg, "r", [])),
        "p": list(getattr(msg, "p", [])),
    }


def realsense_camera_info_check(body: str | None, duration_sec: float = 5.0) -> dict[str, Any]:
    """Validate camera_info topics and intrinsic parameters."""
    body_id = _resolve_body(body)
    camera_name = _camera_name_from_body(body_id)
    try:
        import rclpy
        from rosclaw.mcp.servers import _subscribe_once
    except Exception as exc:  # noqa: BLE001
        return {"body": body_id, "error": f"rclpy not available: {exc}", "valid": False}

    color_topic = f"/camera/{camera_name}/color/camera_info"
    depth_topic = f"/camera/{camera_name}/depth/camera_info"
    color_msg = _subscribe_once(color_topic, "sensor_msgs/CameraInfo", timeout_sec=5.0)
    depth_msg = _subscribe_once(depth_topic, "sensor_msgs/CameraInfo", timeout_sec=5.0)
    if color_msg is None or depth_msg is None:
        return {
            "body": body_id,
            "error": f"No camera_info received ({'color missing' if color_msg is None else ''} {'depth missing' if depth_msg is None else ''}).",
            "valid": False,
        }

    color_info = _camera_info_dict(color_msg)
    depth_info = _camera_info_dict(depth_msg)
    errors: list[str] = []
    for name, info in [("color", color_info), ("depth", depth_info)]:
        k = info.get("k", [])
        if len(k) < 9 or k[0] <= 0 or k[4] <= 0:
            errors.append(f"{name} intrinsics missing or invalid focal length")
        if len(k) >= 9 and (k[2] <= 0 or k[2] >= info["width"] or k[5] <= 0 or k[5] >= info["height"]):
            errors.append(f"{name} principal point outside image")
        if info["width"] <= 0 or info["height"] <= 0:
            errors.append(f"{name} resolution invalid")

    return {
        "body": body_id,
        "duration_sec": duration_sec,
        "color": color_info,
        "depth": depth_info,
        "valid": len(errors) == 0,
        "errors": errors if errors else None,
    }


def realsense_capture_frame(body: str | None, output_path: str | None = None, duration_sec: float = 5.0) -> dict[str, Any]:
    """Capture a single color frame."""
    body_id = _resolve_body(body)
    camera_name = _camera_name_from_body(body_id)
    out = output_path or "/tmp/capture_frame.jpg"
    result = _mcp_run_tool("capture_rgb_frame", output_path=out, camera_name=camera_name)
    result["body"] = body_id
    result["duration_sec"] = duration_sec
    return result


def scene_risk_scan(body: str | None, provider: str | None = None, image_path: str | None = None) -> dict[str, Any]:
    """Run a scene risk scan using a VLM provider and a real RGB frame."""
    body_id = _resolve_body(body)
    # Capture an RGB frame if no image path is provided
    if not image_path:
        camera_name = _camera_name_from_body(body_id)
        cap = _mcp_run_tool("capture_rgb_frame", output_path="/tmp/scene_risk_scan.jpg", camera_name=camera_name)
        image_path = cap.get("path")
    if not image_path or not Path(image_path).exists():
        return {
            "body": body_id,
            "provider": provider,
            "image_path": image_path,
            "error": "No image available for risk scan",
            "status": "failed",
        }

    # If a provider is specified, call it through the canonical provider call helper.
    if provider:
        try:
            result = asyncio.run(
                call_provider(
                    provider_id=provider,
                    capability="vlm.risk_assessment",
                    image_path=image_path,
                    robot_id=body_id,
                    task_id="scene_risk_scan",
                )
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "body": body_id,
                "provider": provider,
                "image_path": image_path,
                "error": str(exc),
                "status": "failed",
            }
        return {
            "body": body_id,
            "provider": provider,
            "image_path": image_path,
            "request_id": result.get("request_id"),
            "latency_ms": result.get("latency_ms"),
            "scene": result.get("scene"),
            "risks": result.get("risks", []),
            "risk_score": result.get("normalized_risk"),
            "executable": result.get("executable"),
            "requires_guard": result.get("requires_guard"),
            "input_frame_uri": result.get("input_frame_uri"),
            "status": result.get("status", "ok"),
        }

    if not image_path:
        return {
            "body": body_id,
            "provider": provider,
            "image_path": image_path,
            "error": "No image available for risk scan and no provider configured",
            "status": "failed",
        }

    return {
        "body": body_id,
        "provider": provider,
        "image_path": image_path,
        "risks": [],
        "status": "ok",
    }
