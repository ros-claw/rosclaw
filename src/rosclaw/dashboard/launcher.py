"""Dashboard launcher with optional full rosclaw-dashboard package."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("rosclaw.dashboard.launcher")


def serve_dashboard(host: str = "0.0.0.0", port: int = 8765, workspace: str | Path | None = None) -> None:
    """Serve the ROSClaw dashboard.

    If ``rosclaw_dashboard`` (the full Next.js + FastAPI package) is installed,
    delegate to it. Otherwise fall back to the lightweight built-in dashboard.
    """
    try:
        from rosclaw_dashboard.serve import serve

        logger.info("Using full rosclaw-dashboard package")
        serve(host=host, port=port, workspace=workspace)
    except ImportError:
        logger.info("rosclaw-dashboard not installed; using built-in lightweight dashboard")
        import uvicorn
        from rosclaw.dashboard.web_server import app

        uvicorn.run(app, host=host, port=port, log_level="info")


def get_dashboard_app() -> Any:
    """Return the dashboard ASGI app.

    Tries the full package first, then the built-in lightweight app.
    """
    try:
        from rosclaw_dashboard.serve import get_app

        return get_app()
    except ImportError:
        from rosclaw.dashboard.web_server import app

        return app
