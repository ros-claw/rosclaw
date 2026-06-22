"""FastAPI wrapper for DashboardServer — provides HTTP API + WebSocket streaming."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect

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

    def __init__(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        self.metrics = DashboardMetrics()
        self.server = DashboardServer(self.metrics, host=host, port=port)
        self.app = FastAPI(title="ROSClaw Dashboard", version="1.0.0")
        self._setup_routes()

    def attach_to_event_bus(self, event_bus: Any) -> None:
        """Subscribe to episode/praxis events for live metrics updates."""
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

        @self.app.get("/body")
        async def body_page() -> Any:
            from fastapi.responses import HTMLResponse
            return HTMLResponse(_BODY_PAGE_HTML)

        @self.app.get("/api/firstboot")
        async def api_firstboot() -> dict[str, Any]:
            return self.server.get_firstboot_state()

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
