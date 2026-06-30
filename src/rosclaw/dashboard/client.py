"""Lightweight client for pushing RealSense metrics to the ROSClaw dashboard.

Used by MCP servers and practice adapters so the `/realsense` page reflects
live captures without manual HTTP posts.
"""
from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request
from typing import Any


_DASHBOARD_URL = "http://localhost:8765"


def _post_json(path: str, payload: dict[str, Any]) -> None:
    """Fire-and-forget POST to the dashboard; failures are silently ignored."""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{_DASHBOARD_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            resp.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        pass


def set_realsense_online(camera_key: str, online: bool, info: dict[str, Any] | None = None) -> None:
    """Notify the dashboard that a RealSense camera is online/offline."""
    payload = {"camera": camera_key, "online": online}
    if info:
        payload["info"] = info
    _post_json("/metrics/realsense/record", payload)


def record_realsense_frame(
    camera_key: str,
    frame_type: str,
    path: str,
    latency_ms: float | None = None,
    drop_count: int | None = None,
) -> None:
    """Notify the dashboard that a RealSense frame was captured."""
    payload = {
        "camera": camera_key,
        "online": True,
        "frame_path": path,
        "frame_type": frame_type,
    }
    if latency_ms is not None:
        payload["latency_ms"] = latency_ms
    if drop_count is not None:
        payload["drop_count"] = drop_count
    _post_json("/metrics/realsense/record", payload)


def record_realsense_frame_async(
    camera_key: str,
    frame_type: str,
    path: str,
    latency_ms: float | None = None,
    drop_count: int | None = None,
) -> None:
    """Non-blocking variant of ``record_realsense_frame`` for hot paths."""
    threading.Thread(
        target=record_realsense_frame,
        args=(camera_key, frame_type, path, latency_ms, drop_count),
        daemon=True,
    ).start()


# Avoid circular import: json is only needed here.
import json  # noqa: E402
