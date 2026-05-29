#!/usr/bin/env python3
"""Start Dashboard server for v1.0 acceptance.

Usage:
    python scripts/start_dashboard.py
    # Open http://localhost:8765 for HTTP API
    # Open ws://localhost:8765/ws for WebSocket stream
"""
import asyncio
import sys


def main():
    from rosclaw.dashboard.web_server import DashboardWebServer
    from rosclaw.core.event_bus import EventBus

    event_bus = EventBus()
    ws = DashboardWebServer(host="0.0.0.0", port=8765)

    # Subscribe dashboard metrics to EventBus for live updates
    ws.server.attach_to_event_bus(event_bus)

    # Set initial module health
    ws.metrics.set_module_health("dashboard", "HEALTHY")
    ws.metrics.set_module_health("event_bus", "HEALTHY")
    ws.metrics.set_module_health("runtime", "HEALTHY")

    print("=" * 60)
    print("ROSClaw Dashboard v1.0")
    print("=" * 60)
    print("HTTP API:   http://0.0.0.0:8765")
    print("Health:     http://0.0.0.0:8765/health")
    print("Snapshot:   http://0.0.0.0:8765/snapshot")
    print("WebSocket:  ws://0.0.0.0:8765/ws")
    print("=" * 60)
    print("Press Ctrl+C to stop")
    print("=" * 60)

    try:
        asyncio.run(ws.start())
    except KeyboardInterrupt:
        print("\n[Dashboard] Shutdown requested")
    finally:
        ws.server.detach_from_event_bus()
        print("[Dashboard] Stopped")


if __name__ == "__main__":
    main()
