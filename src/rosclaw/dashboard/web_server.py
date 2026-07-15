"""FastAPI wrapper for DashboardServer — provides HTTP API + WebSocket streaming."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

from rosclaw.practice.storage.layout import PracticeLayout
from rosclaw.runtime import RuntimeBus, RuntimeQueryAPI

from .firstboot import FIRSTBOOT_PAGE_HTML, build_firstboot_command, preview_firstboot_config
from .metrics import DashboardMetrics
from .server import DashboardServer

_BODY_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ROSClaw Body</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; margin: 2rem; background: #0f172a; color: #e2e8f0; }
    h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
    .card { background: #1e293b; border-radius: 0.5rem; padding: 1rem; margin: 1rem 0; }
    .muted { color: #94a3b8; }
    .status { font-weight: bold; }
    ul { padding-left: 1.2rem; }
    pre { background: #0f172a; padding: 0.75rem; border-radius: 0.25rem; overflow-x: auto; }
  </style>
</head>
<body>
  <h1>ROSClaw Body Dashboard</h1>
  <div id="current" class="card">Loading...</div>
  <div id="bodies" class="card">
    <h2>Registry</h2>
    <ul id="bodies-list"></ul>
  </div>
  <div id="compatibility" class="card">
    <h2>Compatibility</h2>
    <pre id="compatibility-json">No compatibility data yet.</pre>
  </div>

  <script>
    async function fetchBody() {
      try {
        const res = await fetch('/api/body');
        const data = await res.json();
        document.getElementById('current').innerHTML =
          `<div>Current body: <span class="status">${data.current || 'none'}</span></div>` +
          `<div class="muted">Workspace: ${data.workspace || ''}</div>`;
        const list = document.getElementById('bodies-list');
        list.innerHTML = (data.bodies || []).map(b =>
          `<li><strong>${b.body_id}</strong> — ${b.nickname || b.body_id} (${b.profile_id || 'unknown'})</li>`
        ).join('');
        document.getElementById('compatibility-json').textContent =
          JSON.stringify(data.compatibility || {}, null, 2);
      } catch (err) {
        document.getElementById('current').innerHTML = `<div class="muted">Error: ${err.message}</div>`;
      }
    }
    fetchBody();
    setInterval(fetchBody, 5000);

    const ws = new WebSocket(`ws://${location.host}/ws`);
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === 'snapshot' && msg.data && msg.data.body) {
        document.getElementById('compatibility-json').textContent =
          JSON.stringify(msg.data.body.compatibility || {}, null, 2);
      }
    };
    ws.onopen = () => ws.send(JSON.stringify({type: 'ping'}));
  </script>
</body>
</html>
"""


_REALSENSE_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ROSClaw RealSense</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; margin: 2rem; background: #0f172a; color: #e2e8f0; }
    h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
    .card { background: #1e293b; border-radius: 0.5rem; padding: 1rem; margin: 1rem 0; }
    .muted { color: #94a3b8; }
    .status { font-weight: bold; }
    .empty { color: #94a3b8; font-style: italic; }
    pre { background: #0f172a; padding: 0.75rem; border-radius: 0.25rem; overflow-x: auto; }
    img { max-width: 100%; border-radius: 0.25rem; margin-top: 0.5rem; }
  </style>
</head>
<body>
  <h1>ROSClaw RealSense Dashboard</h1>
  <div id="status" class="card">Loading status...</div>
  <div id="latest" class="card">
    <h2>Latest Frame</h2>
    <div id="frame-container" class="muted">Loading...</div>
  </div>

  <script>
    async function fetchStatus() {
      try {
        const res = await fetch('/api/realsense/status');
        const data = await res.json();
        document.getElementById('status').innerHTML =
          `<div>Profile exists: <span class="status">${data.profile_exists}</span></div>` +
          `<div>Body linked: <span class="status">${data.body_linked}</span></div>` +
          `<div>Current body: <span class="status">${data.body_id || 'none'}</span></div>`;
      } catch (err) {
        document.getElementById('status').innerHTML = `<div class="muted">Error: ${err.message}</div>`;
      }
    }
    async function fetchLatestFrame() {
      try {
        const res = await fetch('/api/realsense/latest-frame');
        const data = await res.json();
        const container = document.getElementById('frame-container');
        if (data.found) {
          container.innerHTML =
            `<div class="muted">Episode: ${data.practice_id}</div>` +
            `<img src="${data.frame_url}" alt="latest frame" />`;
        } else {
          container.innerHTML =
            `<div class="empty">${data.message}</div>` +
            `<pre>${data.command}</pre>`;
        }
      } catch (err) {
        document.getElementById('frame-container').innerHTML = `<div class="muted">Error: ${err.message}</div>`;
      }
    }
    fetchStatus();
    fetchLatestFrame();
    setInterval(fetchStatus, 5000);
    setInterval(fetchLatestFrame, 5000);
  </script>
</body>
</html>
"""


def _practice_data_root(query_root: str | None = None) -> Path:
    """Resolve the practice data root from query param, env, or default."""
    if query_root:
        return Path(query_root)
    return Path(os.environ.get("ROSCLAW_PRACTICE_DATA_ROOT", "/data/rosclaw/practice"))


def _list_episodes(data_root: Path) -> list[dict[str, Any]]:
    """Return practice episode summaries from disk, newest first."""
    layout = PracticeLayout(data_root)
    sessions_dir = layout.sessions_dir
    if not sessions_dir.exists():
        return []

    episodes: list[dict[str, Any]] = []
    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue
        episode_path = session_dir / "episode.json"
        if not episode_path.exists():
            continue
        try:
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        episodes.append(
            {
                "practice_id": episode.get("practice_id", session_dir.name),
                "robot_id": episode.get("robot_id"),
                "robot_type": episode.get("robot_type"),
                "outcome": episode.get("outcome", "UNKNOWN"),
                "event_count": episode.get("event_count", 0),
                "start_time": episode.get("start_time"),
                "session_dir": str(session_dir),
            }
        )

    episodes.sort(key=lambda e: e.get("start_time") or "", reverse=True)
    return episodes


def _episode_dir(data_root: Path, episode_id: str) -> Path:
    """Resolve an episode session directory from id or direct path."""
    layout = PracticeLayout(data_root)
    session_dir = layout.session_dir(episode_id)
    if session_dir.exists():
        return session_dir
    direct = Path(episode_id)
    if direct.exists() and direct.is_dir():
        return direct
    raise HTTPException(status_code=404, detail=f"Episode not found: {episode_id}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Safely read a JSONL file."""
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read {path}: {exc}") from exc


async def _latest_frame_info(
    query_api: RuntimeQueryAPI | None = None,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Find the most recently recorded RealSense RGB frame.

    Uses the RuntimeQueryAPI when available; otherwise scans disk.
    """
    command = (
        "rosclaw practice run --robot d405_lab_01 "
        "--skill realsense_capture_rgbd --provider cosmos-reason2-lan "
        "--output-root ./episode"
    )

    if query_api is not None:
        ev = query_api.latest("camera.rgbd_frame")
        if ev is not None:
            payload = ev.payload or {}
            rgb_ref = payload.get("rgb_ref")
            if rgb_ref:
                episode_id = (
                    ev.metadata.get("trace_id") or ev.metadata.get("episode_id") or "unknown"
                )
                return {
                    "found": True,
                    "practice_id": episode_id,
                    "rgb_ref": rgb_ref,
                    "frame_url": f"/api/artifacts/{episode_id}/{rgb_ref}",
                }

    layout = PracticeLayout(data_root or _practice_data_root())
    for ep in _list_episodes(layout.data_root):
        timeline = _read_jsonl(layout.timeline_jsonl_path(ep["practice_id"]))
        for ev in reversed(timeline):
            if ev.get("source") == "camera" and ev.get("event_type") == "rgbd_frame":
                payload = ev.get("payload", {})
                rgb_ref = payload.get("rgb_ref")
                if rgb_ref:
                    return {
                        "found": True,
                        "practice_id": ep["practice_id"],
                        "rgb_ref": rgb_ref,
                        "frame_url": f"/api/artifacts/{ep['practice_id']}/{rgb_ref}",
                    }
    return {
        "found": False,
        "message": "No RealSense frames recorded yet.",
        "command": command,
    }


def _list_realsense_frames(
    query_api: RuntimeQueryAPI | None = None,
    data_root: Path | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Return the most recent RealSense RGB-D frame events across episodes."""
    command = (
        "rosclaw practice run --robot d405_lab_01 "
        "--skill realsense_capture_rgbd --provider cosmos-reason2-lan "
        "--output-root ./episode"
    )

    if query_api is not None:
        events = query_api.latest_n("camera.rgbd_frame", n=limit)
        frames = []
        for ev in events:
            payload = ev.payload or {}
            rgb_ref = payload.get("rgb_ref")
            if not rgb_ref:
                continue
            episode_id = ev.metadata.get("trace_id") or ev.metadata.get("episode_id") or "unknown"
            depth_ref = payload.get("depth_ref")
            frames.append(
                {
                    "practice_id": episode_id,
                    "event_id": ev.id,
                    "timestamp_utc": ev.timestamp.isoformat().replace("+00:00", "Z"),
                    "rgb_ref": rgb_ref,
                    "depth_ref": depth_ref,
                    "width": payload.get("width"),
                    "height": payload.get("height"),
                    "rgb_url": f"/api/artifacts/{episode_id}/{rgb_ref}",
                    "depth_url": f"/api/artifacts/{episode_id}/{depth_ref}" if depth_ref else None,
                }
            )
        if frames:
            return {"found": True, "count": len(frames), "frames": frames}

    layout = PracticeLayout(data_root or _practice_data_root())
    frames: list[dict[str, Any]] = []
    for ep in _list_episodes(layout.data_root):
        timeline = _read_jsonl(layout.timeline_jsonl_path(ep["practice_id"]))
        for ev in reversed(timeline):
            if ev.get("source") == "camera" and ev.get("event_type") == "rgbd_frame":
                payload = ev.get("payload", {})
                rgb_ref = payload.get("rgb_ref")
                if not rgb_ref:
                    continue
                depth_ref = payload.get("depth_ref")
                frames.append(
                    {
                        "practice_id": ep["practice_id"],
                        "event_id": ev.get("event_id"),
                        "timestamp_utc": ev.get("timestamp_utc"),
                        "rgb_ref": rgb_ref,
                        "depth_ref": depth_ref,
                        "width": payload.get("width"),
                        "height": payload.get("height"),
                        "rgb_url": f"/api/artifacts/{ep['practice_id']}/{rgb_ref}",
                        "depth_url": f"/api/artifacts/{ep['practice_id']}/{depth_ref}"
                        if depth_ref
                        else None,
                    }
                )
                if len(frames) >= limit:
                    break
        if len(frames) >= limit:
            break

    if frames:
        return {"found": True, "count": len(frames), "frames": frames}
    return {"found": False, "message": "No RealSense frames recorded yet.", "command": command}


def _list_realsense_streams(
    query_api: RuntimeQueryAPI | None = None,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Infer available RealSense streams from the latest recorded frame."""
    command = (
        "rosclaw practice run --robot d405_lab_01 "
        "--skill realsense_capture_rgbd --provider cosmos-reason2-lan "
        "--output-root ./episode"
    )

    if query_api is not None:
        ev = query_api.latest("camera.rgbd_frame")
        if ev is not None:
            payload = ev.payload or {}
            rgb_ref = payload.get("rgb_ref")
            if rgb_ref:
                episode_id = (
                    ev.metadata.get("trace_id") or ev.metadata.get("episode_id") or "unknown"
                )
                depth_ref = payload.get("depth_ref")
                streams = [
                    {
                        "name": "color",
                        "type": "rgb",
                        "encoding": payload.get("rgb_encoding", "png"),
                        "ref": rgb_ref,
                        "url": f"/api/artifacts/{episode_id}/{rgb_ref}",
                    }
                ]
                if depth_ref:
                    streams.append(
                        {
                            "name": "depth",
                            "type": "depth",
                            "encoding": payload.get("depth_encoding", "png16"),
                            "ref": depth_ref,
                            "url": f"/api/artifacts/{episode_id}/{depth_ref}",
                        }
                    )
                return {
                    "found": True,
                    "practice_id": episode_id,
                    "streams": streams,
                }

    layout = PracticeLayout(data_root or _practice_data_root())
    for ep in _list_episodes(layout.data_root):
        timeline = _read_jsonl(layout.timeline_jsonl_path(ep["practice_id"]))
        for ev in reversed(timeline):
            if ev.get("source") == "camera" and ev.get("event_type") == "rgbd_frame":
                payload = ev.get("payload", {})
                rgb_ref = payload.get("rgb_ref")
                depth_ref = payload.get("depth_ref")
                if not rgb_ref:
                    continue
                streams = [
                    {
                        "name": "color",
                        "type": "rgb",
                        "encoding": payload.get("rgb_encoding", "png"),
                        "ref": rgb_ref,
                        "url": f"/api/artifacts/{ep['practice_id']}/{rgb_ref}",
                    }
                ]
                if depth_ref:
                    streams.append(
                        {
                            "name": "depth",
                            "type": "depth",
                            "encoding": payload.get("depth_encoding", "png16"),
                            "ref": depth_ref,
                            "url": f"/api/artifacts/{ep['practice_id']}/{depth_ref}",
                        }
                    )
                return {
                    "found": True,
                    "practice_id": ep["practice_id"],
                    "streams": streams,
                }
    return {
        "found": False,
        "message": "No RealSense streams recorded yet.",
        "command": command,
    }


class WebSocketClient:
    """Adapter to make FastAPI WebSocket look like DashboardServer client."""

    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket

    async def send_text(self, message: str) -> None:
        await self._ws.send_text(message)

    async def send(self, message: str) -> None:
        await self._ws.send_text(message)


class DashboardWebServer:
    """FastAPI + WebSocket server wrapping DashboardServer."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        runtime_bus: RuntimeBus | None = None,
        trace_store: Any | None = None,
    ) -> None:
        self.metrics = DashboardMetrics()
        self.server = DashboardServer(self.metrics, host=host, port=port)
        self.app = FastAPI(title="ROSClaw Dashboard", version="1.0.0")
        self._runtime_bus = runtime_bus
        self._query_api = RuntimeQueryAPI(runtime_bus) if runtime_bus is not None else None
        if trace_store is None:
            from rosclaw.observability.store import TraceStore

            trace_store = TraceStore()
        self._trace_store = trace_store
        self._setup_routes()

    def attach_to_event_bus(self, event_bus: Any) -> None:
        """Subscribe to episode/praxis events for live metrics updates.

        Accepts either a legacy ``EventBus`` or a ``RuntimeBus``. When a
        ``RuntimeBus`` is supplied, the dashboard also exposes a query API over
        runtime history so RealSense endpoints can avoid scanning disk.
        """
        from rosclaw.runtime import RuntimeBus as _RuntimeBus

        if isinstance(event_bus, _RuntimeBus):
            self._runtime_bus = event_bus
            self._query_api = RuntimeQueryAPI(event_bus)
            self.server.attach_to_event_bus(event_bus.event_bus)
        else:
            self.server.attach_to_event_bus(event_bus)

        # Subscribe to episode completion events
        def on_episode_complete(event: Any) -> None:
            payload = getattr(event, "payload", {})
            episode_id = payload.get("episode_id", "unknown")
            robot_id = payload.get("robot_id", "unknown")
            success = payload.get("success", False)
            duration = payload.get("duration_sec")
            self.metrics.record_episode(
                episode_id=episode_id,
                robot_id=robot_id,
                status="success" if success else "failed",
                reward=1.0 if success else 0.0,
                duration_sec=duration,
            )

        event_bus.subscribe("praxis.completed", on_episode_complete)
        event_bus.subscribe("rosclaw.sandbox.episode.finished", on_episode_complete)

    def _setup_routes(self) -> None:
        @self.app.get("/health")
        async def health() -> dict[str, Any]:
            return self.server.get_health()

        @self.app.get("/snapshot")
        async def snapshot() -> dict[str, Any]:
            return self.server.get_snapshot()

        @self.app.get("/api/body")
        async def api_body() -> dict[str, Any]:
            return self.server.get_body_summary()

        @self.app.get("/api/body/effective")
        async def api_body_effective() -> dict[str, Any]:
            return self.server.get_body_effective()

        @self.app.get("/api/body/skills")
        async def api_body_skills() -> dict[str, Any]:
            return self.server.get_body_skills()

        @self.app.get("/api/body/history")
        async def api_body_history(limit: int = 20) -> dict[str, Any]:
            return self.server.get_body_history(limit=limit)

        @self.app.get("/api/body/provider-health")
        async def api_body_provider_health() -> dict[str, Any]:
            return self.server.get_body_provider_health()

        @self.app.get("/body")
        async def body_page() -> Any:
            from fastapi.responses import HTMLResponse

            return HTMLResponse(_BODY_PAGE_HTML)

        @self.app.get("/api/firstboot")
        async def api_firstboot() -> dict[str, Any]:
            return self.server.get_firstboot_state()

        # ── Structured Trace API ──────────────────────────────────────

        @self.app.get("/traces")
        async def traces_page() -> Any:
            from rosclaw.dashboard.trace_page import TRACE_PAGE_HTML

            return HTMLResponse(TRACE_PAGE_HTML)

        @self.app.get("/api/traces")
        async def api_traces(
            trace_id: str | None = None,
            kind: str | None = None,
            status: str | None = None,
            limit: int = 100,
        ) -> dict[str, Any]:
            safe_limit = max(1, min(limit, 5000))
            if trace_id or kind or status:
                spans = self._trace_store.read(
                    trace_id=trace_id,
                    kinds={part.strip().upper() for part in kind.split(",")} if kind else None,
                    statuses={part.strip().upper() for part in status.split(",")}
                    if status
                    else None,
                    limit=safe_limit,
                )
                return {"count": len(spans), "spans": spans}
            traces = self._trace_store.list_traces(limit=safe_limit)
            return {"count": len(traces), "traces": traces}

        @self.app.get("/api/traces/events/{event_id}")
        async def api_trace_event(event_id: str) -> dict[str, Any]:
            event = self._trace_store.find_event(event_id)
            if event is None:
                raise HTTPException(status_code=404, detail="Trace event not found")
            return event

        @self.app.get("/api/traces/{trace_id}")
        async def api_trace(trace_id: str) -> dict[str, Any]:
            trace = self._trace_store.get_trace(trace_id)
            if not trace["spans"]:
                raise HTTPException(status_code=404, detail="Trace not found")
            return trace

        @self.app.post("/api/firstboot/preview")
        async def firstboot_preview(request: Request) -> dict[str, Any]:
            try:
                choices = await request.json()
                if not isinstance(choices, dict):
                    raise HTTPException(status_code=400, detail="Expected JSON object")
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc
            return {
                "preview": preview_firstboot_config(choices),
                "command": build_firstboot_command(choices),
            }

        @self.app.get("/firstboot")
        async def firstboot_page() -> Any:
            from fastapi.responses import HTMLResponse

            return HTMLResponse(FIRSTBOOT_PAGE_HTML)

        @self.app.get("/events/counts")
        async def event_counts() -> dict[str, int]:
            return self.metrics.get_event_counts()

        @self.app.get("/metrics/provider")
        async def provider_metrics() -> dict[str, Any]:
            return self.metrics.get_provider_stats()

        @self.app.get("/metrics/sandbox")
        async def sandbox_metrics() -> dict[str, Any]:
            return self.metrics.get_sandbox_stats()

        @self.app.get("/metrics/episode")
        async def episode_metrics() -> dict[str, Any]:
            return self.metrics.get_episode_stats()

        @self.app.post("/metrics/provider")
        async def record_provider(
            provider: str, capability: str, latency_ms: float, status: str
        ) -> dict[str, str]:
            self.metrics.record_provider_call(provider, capability, latency_ms, status)
            return {"status": "recorded"}

        @self.app.post("/metrics/sandbox")
        async def record_sandbox(
            action_type: str, is_safe: bool, violations: list[str] | None = None
        ) -> dict[str, str]:
            self.metrics.record_sandbox_validation(action_type, is_safe, violations)
            return {"status": "recorded"}

        @self.app.post("/metrics/episode")
        async def record_episode(
            episode_id: str,
            robot_id: str,
            status: str,
            reward: float | None = None,
            duration_sec: float | None = None,
        ) -> dict[str, str]:
            self.metrics.record_episode(episode_id, robot_id, status, reward, duration_sec)
            return {"status": "recorded"}

        @self.app.post("/event/{topic}")
        async def record_event(topic: str) -> dict[str, str]:
            self.metrics.increment_event(topic)
            return {"status": "recorded"}

        @self.app.post("/health/{module}")
        async def set_module_health(module: str, status: str) -> dict[str, str]:
            self.metrics.set_module_health(module, status)
            return {"status": "updated"}

        # ── Practice episode API ────────────────────────────────────────

        @self.app.get("/api/practice/episodes")
        async def api_practice_episodes(data_root: str | None = None) -> dict[str, Any]:
            root = _practice_data_root(data_root)
            episodes = _list_episodes(root)
            return {
                "data_root": str(root),
                "count": len(episodes),
                "episodes": episodes,
                "command": (
                    "rosclaw practice run --robot d405_lab_01 "
                    "--skill realsense_capture_rgbd --provider cosmos-reason2-lan "
                    "--output-root ./episode"
                ),
            }

        @self.app.get("/api/practice/episodes/{episode_id}")
        async def api_practice_episode(
            episode_id: str, data_root: str | None = None
        ) -> dict[str, Any]:
            root = _practice_data_root(data_root)
            session_dir = _episode_dir(root, episode_id)
            episode_path = session_dir / "episode.json"
            if not episode_path.exists():
                raise HTTPException(
                    status_code=404, detail=f"episode.json not found for {episode_id}"
                )
            try:
                episode = json.loads(episode_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise HTTPException(
                    status_code=500, detail=f"Failed to read episode.json: {exc}"
                ) from exc
            return {
                "episode": episode,
                "session_dir": str(session_dir),
            }

        @self.app.get("/api/practice/episodes/{episode_id}/timeline")
        async def api_practice_timeline(
            episode_id: str, data_root: str | None = None
        ) -> dict[str, Any]:
            root = _practice_data_root(data_root)
            session_dir = _episode_dir(root, episode_id)
            layout = PracticeLayout(root)
            timeline = _read_jsonl(layout.timeline_jsonl_path(session_dir.name))
            return {"practice_id": session_dir.name, "count": len(timeline), "timeline": timeline}

        @self.app.get("/api/practice/episodes/{episode_id}/artifacts")
        async def api_practice_artifacts(
            episode_id: str, data_root: str | None = None
        ) -> dict[str, Any]:
            root = _practice_data_root(data_root)
            session_dir = _episode_dir(root, episode_id)
            files: list[str] = []
            for item in session_dir.rglob("*"):
                if item.is_file():
                    with contextlib.suppress(ValueError):
                        files.append(str(item.relative_to(session_dir)))
            files.sort()
            return {"practice_id": session_dir.name, "count": len(files), "artifacts": files}

        @self.app.get("/api/practice/episodes/{episode_id}/provider")
        async def api_practice_provider(
            episode_id: str, data_root: str | None = None
        ) -> dict[str, Any]:
            root = _practice_data_root(data_root)
            session_dir = _episode_dir(root, episode_id)
            provider_files = sorted(
                (session_dir / "provider").glob("provider_result_*.json"),
                key=lambda p: p.name,
            )
            if not provider_files:
                raise HTTPException(status_code=404, detail="No provider result for this episode")
            provider_path = provider_files[-1]
            try:
                provider_data = json.loads(provider_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise HTTPException(
                    status_code=500, detail=f"Failed to read provider result: {exc}"
                ) from exc
            return {"practice_id": session_dir.name, "provider": provider_data}

        @self.app.get("/api/practice/episodes/{episode_id}/sandbox")
        async def api_practice_sandbox(
            episode_id: str, data_root: str | None = None
        ) -> dict[str, Any]:
            root = _practice_data_root(data_root)
            session_dir = _episode_dir(root, episode_id)
            layout = PracticeLayout(root)
            events = _read_jsonl(layout.events_jsonl_path(session_dir.name))
            sandbox_events = [ev for ev in events if ev.get("source") == "sandbox"]
            return {
                "practice_id": session_dir.name,
                "count": len(sandbox_events),
                "decisions": sandbox_events,
            }

        # ── RealSense page and API ──────────────────────────────────────

        @self.app.get("/realsense")
        async def realsense_page() -> Any:
            return HTMLResponse(_REALSENSE_PAGE_HTML)

        @self.app.get("/api/realsense/status")
        async def api_realsense_status(data_root: str | None = None) -> dict[str, Any]:
            from rosclaw.body.resolver import BodyResolver
            from rosclaw.runtime import RobotRegistry

            profile_exists = False
            try:
                registry = RobotRegistry()
                profile_exists = registry.get("realsense_d405") is not None
            except Exception:
                pass

            body_linked = False
            body_id: str | None = None
            try:
                resolver = BodyResolver()
                body_linked = resolver.is_linked()
                body_id = resolver.get_current_body_id()
            except Exception:
                pass

            latest = await _latest_frame_info(
                query_api=self._query_api,
                data_root=_practice_data_root(data_root),
            )
            return {
                "profile_exists": profile_exists,
                "body_linked": body_linked,
                "body_id": body_id,
                "latest_frame": latest,
                "command": (
                    "rosclaw practice run --robot d405_lab_01 "
                    "--skill realsense_capture_rgbd --provider cosmos-reason2-lan "
                    "--output-root ./episode"
                ),
            }

        @self.app.get("/api/realsense/latest-frame")
        async def api_realsense_latest_frame(data_root: str | None = None) -> dict[str, Any]:
            return await _latest_frame_info(
                query_api=self._query_api,
                data_root=_practice_data_root(data_root),
            )

        @self.app.get("/api/realsense/streams")
        async def api_realsense_streams(data_root: str | None = None) -> dict[str, Any]:
            return _list_realsense_streams(
                query_api=self._query_api,
                data_root=_practice_data_root(data_root),
            )

        @self.app.get("/api/realsense/frames")
        async def api_realsense_frames(
            data_root: str | None = None, limit: int = 50
        ) -> dict[str, Any]:
            return _list_realsense_frames(
                query_api=self._query_api,
                data_root=_practice_data_root(data_root),
                limit=limit,
            )

        # ── Artifact serving ────────────────────────────────────────────
        # Registered AFTER the RealSense routes so the greedy {path:path}
        # pattern does not shadow /api/realsense/streams or /api/realsense/frames.

        @self.app.get("/api/artifacts/{path:path}")
        async def serve_artifact(path: str, data_root: str | None = None) -> Any:
            root = _practice_data_root(data_root)
            base = (root / "sessions").resolve()
            artifact_path = (base / path).resolve()
            try:
                artifact_path.relative_to(base)
            except ValueError as exc:
                raise HTTPException(status_code=403, detail="Access denied") from exc
            if not artifact_path.exists() or not artifact_path.is_file():
                raise HTTPException(status_code=404, detail="Artifact not found")
            return FileResponse(artifact_path)

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            await websocket.accept()
            client = WebSocketClient(websocket)
            self.server.register_client(client)
            try:
                # Send initial snapshot
                snapshot = self.server.get_snapshot()
                snapshot["body"] = self.server.get_body_summary()
                await websocket.send_text(json.dumps({"type": "snapshot", "data": snapshot}))
                # Keep connection alive and handle incoming messages
                while True:
                    msg = await websocket.receive_text()
                    data = json.loads(msg)
                    if data.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
            except WebSocketDisconnect:
                pass
            finally:
                self.server.unregister_client(client)

    async def start(self) -> None:
        await self.server.start()

    async def stop(self) -> None:
        await self.server.stop()


# FastAPI app instance for uvicorn
_metrics = DashboardMetrics()
_server = DashboardWebServer()
app = _server.app


async def main() -> None:
    ws = DashboardWebServer()
    await ws.start()
    print("DashboardWebServer started on http://0.0.0.0:8765")
    print("WebSocket: ws://localhost:8765/ws")
    print("Health: http://localhost:8765/health")
    print("Snapshot: http://localhost:8765/snapshot")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await ws.stop()
        print("DashboardWebServer stopped")


if __name__ == "__main__":
    asyncio.run(main())
