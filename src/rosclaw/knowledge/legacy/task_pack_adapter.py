"""TaskPackAdapter — runtime-facing call to rosclaw_know.task_pack_builder.

This is the v1.5 pre-flight knowledge surface for agents.  The runtime
calls :func:`task_pack_for` before selecting a provider so the agent
sees relevant FailureMode + FixPattern context.

Optional dependency:  ``rosclaw_know>=1.5.0a1``.  When the package is
absent, the adapter returns a safe empty pack and logs a one-time
warning.  Existing curated patterns in :class:`KnowledgeInterface`
keep working.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("rosclaw.know.task_pack_adapter")

try:
    from rosclaw_know.asset_loader import load_task_pack_assets
    from rosclaw_know.schemas import TaskPackQuery
    from rosclaw_know.task_pack_builder import TaskCardNotFoundError, build_task_pack

    _KNOW_AVAILABLE = True
except ImportError as exc:  # rosclaw_know not installed
    logger.info("rosclaw-know not available (%s); task packs disabled", exc)
    _KNOW_AVAILABLE = False


_assets_cache: dict[str, Any] | None = None
"""Module-level cache so we only YAML-parse once per asset bundle."""

_assets_cache_dir: Path | None = None


def _empty_pack() -> dict[str, Any]:
    return {
        "task_id": "",
        "summary": "",
        "failure_modes": [],
        "fix_patterns": [],
        "anti_patterns": [],
        "expected_signals": [],
        "warnings": ["rosclaw-know not installed"] if not _KNOW_AVAILABLE else [],
        "token_estimate": 0,
    }


def _load(assets_dir: Path) -> dict[str, Any] | None:
    """Cached asset loader.  Reloads only when ``assets_dir`` changes."""
    global _assets_cache, _assets_cache_dir
    if not _KNOW_AVAILABLE:
        return None
    if _assets_cache_dir != assets_dir or _assets_cache is None:
        assets = load_task_pack_assets(assets_dir)
        if assets is None:
            return None
        _assets_cache = assets
        _assets_cache_dir = assets_dir
    return _assets_cache


def reload_assets() -> None:
    """Force the next ``task_pack_for`` call to re-read YAMLs.

    Used by :mod:`assets_loader` when the bundle on disk changes.
    """
    global _assets_cache
    _assets_cache = None


def task_pack_for(
    task_id: str,
    *,
    embodiment_id: str | None = None,
    assets_dir: str | Path = "data/knowledge_assets",
    top_k_patterns: int = 5,
) -> dict[str, Any]:
    """Return the pre-flight knowledge pack for ``task_id``.

    ``task_id`` is the agent-facing identifier — it gets mapped to
    rosclaw-know's :class:`TaskPackQuery.task_name`.  ``embodiment_id``
    is accepted for forward compatibility (the v1.5 builder does not
    branch on it yet, but the integration design reserves the slot).

    Pure-ish: reads the cached YAMLs (parsed once at first call) and
    runs ``build_task_pack``.  No network.  Safe to call from any
    runtime hot path (~ ms range).
    """
    if not _KNOW_AVAILABLE:
        return _empty_pack()

    assets_dir = Path(assets_dir)
    assets = _load(assets_dir)
    if assets is None:
        pack = _empty_pack()
        pack["task_id"] = task_id
        pack["warnings"] = [f"assets not found in {assets_dir}"]
        return pack

    try:
        query = TaskPackQuery(
            task_name=task_id,
            top_k_patterns=top_k_patterns,
        )
        pack = build_task_pack(
            query,
            catalog=assets["tasks"],
            patterns=assets["patterns"],
            failures=assets["failures"],
        )
        result = pack.model_dump()
        # Surface the requesting task_id alongside the resolved name
        # so callers can correlate without re-parsing TaskPackQuery.
        result.setdefault("task_id", task_id)
        if embodiment_id is not None:
            result["embodiment_id"] = embodiment_id
        return result
    except TaskCardNotFoundError:
        pack = _empty_pack()
        pack["task_id"] = task_id
        pack["warnings"] = [f"no TaskCard for {task_id}"]
        return pack
    except Exception as exc:  # noqa: BLE001
        logger.warning("Task pack build failed for %s: %s", task_id, exc)
        pack = _empty_pack()
        pack["task_id"] = task_id
        pack["warnings"] = [f"build error: {exc}"]
        return pack
