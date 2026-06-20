"""Thin adapter around the ROSClaw episode recorder for MCP tools."""

from __future__ import annotations

from typing import Any


class PracticeClient:
    """Read-only client that lists practice episodes from EpisodeRecorder."""

    def __init__(self, recorder: Any) -> None:
        self._recorder = recorder

    def query(self, *, episode_id: str | None = None, limit: int = 10) -> dict[str, Any]:
        """Return one episode by ID or the most recent episodes up to ``limit``."""
        if episode_id:
            episode = self._recorder.get_episode(episode_id)
            episodes = [episode] if episode else []
        else:
            episodes = self._recorder.list_episodes()[:limit]
        return {
            "episodes": episodes,
            "count": len(episodes),
            "mode": "live",
        }
