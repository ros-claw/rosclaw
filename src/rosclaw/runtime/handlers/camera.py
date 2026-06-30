"""Runtime handlers for camera/perception skills."""

from __future__ import annotations

from typing import Any

from rosclaw.runtime.plugin import runtime_handler


@runtime_handler("realsense_capture_rgbd")
def _handle_realsense_capture_rgbd(params: dict[str, Any]) -> dict[str, Any]:
    """Runtime handler for the RealSense RGB-D capture skill."""
    return {
        "status": "success",
        "skill": "realsense_capture_rgbd",
        "frames": {
            "color": params.get("color_path") or "rosclaw://camera/latest/color",
            "depth": params.get("depth_path") or "rosclaw://camera/latest/depth",
        },
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
