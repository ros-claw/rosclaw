"""LIMO-like MCP server fixture for PR-05 tests (real MCP protocol over stdio).

Exposes:
- ``limo.localization.get_pose`` — readOnlyHint=True → must classify OBSERVE
- ``limo.speaker.play_tone`` — action verb, no readOnlyHint → PHYSICAL_ACTION
- ``limo.misc.ambiguous`` — no annotations, no verb → fail-closed PHYSICAL_ACTION

This is a REAL MCP server (mcp SDK, stdio transport), not a mock of the
adapter — discovery and calls exercise the full JSON-RPC path.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

server = FastMCP("limo-ros-mcp-fixture")


def _get_pose(frame: str = "map") -> str:
    return (
        f'{{"frame": "{frame}", "x": 1.25, "y": -0.5, "theta": 0.03, '
        '"timestamp": "2026-08-03T00:00:00Z", "fresh": true}'
    )


def _play_tone(frequency_hz: int = 660, duration_sec: float = 0.25) -> str:
    # A real action tool would side-effect the speaker; the fixture only
    # reports the request. It must NEVER reach this point through the catalog.
    return f'{{"played": true, "frequency_hz": {frequency_hz}, "duration_sec": {duration_sec}}}'


def _ambiguous() -> str:
    return '{"status": "ok"}'


server.add_tool(
    _get_pose,
    name="limo.localization.get_pose",
    description="Get the current LIMO localization pose (read-only observation).",
    annotations=ToolAnnotations(readOnlyHint=True),
)
server.add_tool(
    _play_tone,
    name="limo.speaker.play_tone",
    description="Play a tone on the LIMO speaker (physical action).",
)
server.add_tool(
    _ambiguous,
    name="limo.misc.ambiguous",
    description="A tool with no annotations at all.",
)

if __name__ == "__main__":
    server.run()
