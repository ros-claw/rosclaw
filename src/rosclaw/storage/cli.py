"""CLI commands for ROSClaw storage diagnostics.

Adds ``rosclaw db status`` and ``rosclaw db doctor``.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

from rosclaw.firstboot.config import load_rosclaw_yaml
from rosclaw.firstboot.workspace import resolve_home
from rosclaw.storage.factory import StorageFactory, _sanitize_url
from rosclaw.storage.migrations import MigrationRunner
from rosclaw.storage.outbox import OutboxStore


def _close_client(client: Any) -> None:
    """Best-effort close/disconnect for a knowledge-store client."""
    close = getattr(client, "disconnect", None) or getattr(client, "close", None)
    if close:
        with contextlib.suppress(Exception):
            close()


def _load_storage_config(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve storage configuration from CLI args, rosclaw.yaml, and env."""
    home = resolve_home()
    cfg = load_rosclaw_yaml(home) or {}
    runtime_cfg = cfg.get("runtime", {})
    storage_cfg = cfg.get("storage", {})
    practice_cfg = cfg.get("practice", {})

    backend = getattr(args, "backend", None) or runtime_cfg.get("seekdb_backend") or "sqlite"
    backend = backend.lower()

    url = getattr(args, "url", None)
    if url is None:
        url = runtime_cfg.get("seekdb_url") or os.environ.get("ROSCLAW_SEEKDB_URL")

    path = getattr(args, "path", None)
    if path is None:
        path = runtime_cfg.get("seekdb_path") or str(home / "data" / "memory" / "knowledge.sqlite")

    outbox_enabled = storage_cfg.get("outbox_enabled", False)
    outbox_path = storage_cfg.get("outbox_path") or str(home / "storage" / "outbox.sqlite")

    practice_data_root = practice_cfg.get("output_dir") or "/data/rosclaw/practice"

    return {
        "home": home,
        "config_found": bool(cfg),
        "config_overridden": any(
            getattr(args, name, None) is not None for name in ("backend", "url", "path")
        ),
        "backend": backend,
        "url": url,
        "path": path,
        "pool_size": storage_cfg.get("pool_size", 4),
        "vector_enabled": storage_cfg.get("vector_enabled", False),
        "outbox_enabled": outbox_enabled,
        "outbox_path": outbox_path,
        "practice_data_root": practice_data_root,
    }


def _create_client(cfg: dict[str, Any]) -> Any:
    """Create a knowledge-store client from resolved config."""
    return StorageFactory.create_knowledge_store(
        backend=cfg["backend"],
        url=cfg["url"],
        path=cfg["path"],
        pool_size=cfg["pool_size"],
        vector_enabled=cfg["vector_enabled"],
    )


def _sqlite_pragmas(connection: sqlite3.Connection) -> dict[str, Any]:
    """Read current SQLite PRAGMA settings."""
    pragmas = {}
    for name in ("journal_mode", "synchronous", "busy_timeout", "foreign_keys"):
        row = connection.execute(f"PRAGMA {name}").fetchone()
        pragmas[name] = row[0] if row else None
    return pragmas


def _sqlite_table_names(connection: sqlite3.Connection) -> list[str]:
    cursor = connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    return [row[0] for row in cursor.fetchall()]


def _sqlite_wal_size(path: str) -> int:
    db_path = Path(path).expanduser()
    wal_path = db_path.parent / f"{db_path.name}-wal"
    return wal_path.stat().st_size if wal_path.exists() else 0


def _mysql_table_names(connection: Any) -> list[str]:
    """List table names from a raw MySQL connection."""
    with connection.cursor() as cursor:
        cursor.execute("SHOW TABLES")
        rows = cursor.fetchall()
    names: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            names.append(next(iter(row.values())))
        elif isinstance(row, (list, tuple)) and row:
            names.append(row[0])
    return names


def _migration_status(client: Any, backend: str) -> dict[str, Any]:
    """Return pending migration versions for a connected client."""
    if backend == "memory":
        return {"pending": 0, "versions": []}
    try:
        runner = MigrationRunner()
        if backend == "sqlite":
            pending = runner.pending(client._connection, backend)
        else:
            with client._connection as connection:
                pending = runner.pending(connection, backend)
        return {"pending": len(pending), "versions": pending}
    except Exception as exc:  # noqa: BLE001
        return {"pending": None, "error": str(exc)}


def _outbox_status(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Return outbox stats, or None if the outbox is disabled."""
    if not cfg["outbox_enabled"]:
        return None
    try:
        outbox = OutboxStore(db_path=cfg["outbox_path"])
        stats = outbox.stats()
        outbox.close()
        return stats
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _vector_status(client: Any) -> dict[str, Any]:
    """Return vector-store status for SQLite-backed clients."""
    status: dict[str, Any] = {"enabled": False}
    if type(client).__name__ != "SQLiteKnowledgeStore":
        return status
    enabled = getattr(client, "_vector_enabled", False)
    status["enabled"] = enabled
    if not enabled:
        return status
    status["warmed"] = getattr(client, "_embedder_warmed", False)
    try:
        connection = client._connection
        vec_tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'vec_%'"
            ).fetchall()
        ]
        status["indexed_counts"] = {
            table[4:]: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in vec_tables
        }
    except Exception as exc:  # noqa: BLE001
        status["error"] = str(exc)
    return status


def _practice_status(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return practice catalog and latest-session summary."""
    data_root = cfg["practice_data_root"]
    catalog_path = _practice_catalog_path(data_root)
    status: dict[str, Any] = {"catalog_exists": catalog_path.exists()}
    if catalog_path.exists():
        try:
            catalog_conn = sqlite3.connect(str(catalog_path), check_same_thread=False)
            catalog_conn.row_factory = sqlite3.Row
            count = catalog_conn.execute("SELECT COUNT(*) FROM practices").fetchone()[0]
            catalog_conn.close()
            status["practices"] = count
        except Exception as exc:  # noqa: BLE001
            status["error"] = str(exc)
    latest_session_dir = _latest_session_dir(data_root)
    if latest_session_dir is not None:
        event_count, events_lines, consistent, timeline_exists = _session_event_consistency(
            latest_session_dir
        )
        status["latest_session"] = {
            "name": latest_session_dir.name,
            "event_count": event_count,
            "events_jsonl_lines": events_lines,
            "consistent": consistent,
            "timeline_exists": timeline_exists,
        }
    return status


def _practice_catalog_path(data_root: str) -> Path:
    """Return the catalog path used by PracticeLayout.

    The current layout stores the catalog under ``indexes/``; the legacy root
    path is used only as a fallback when it already exists.
    """
    indexes = Path(data_root) / "indexes" / "practice_catalog.sqlite"
    legacy = Path(data_root) / "practice_catalog.sqlite"
    return indexes if indexes.exists() or not legacy.exists() else legacy


def _latest_session_dir(data_root: str) -> Path | None:
    """Return the most recently modified session directory, if any."""
    sessions_root = Path(data_root) / "sessions"
    if not sessions_root.exists():
        return None
    dirs = [p for p in sessions_root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def _session_event_consistency(
    session_dir: Path,
) -> tuple[int | None, int, bool, bool]:
    """Return (episode event_count, events.jsonl lines, consistent?, timeline_exists?)."""
    episode_path = session_dir / "episode.json"
    events_path = session_dir / "raw" / "events.jsonl"
    timeline_path = session_dir / "timeline.jsonl"

    event_count: int | None = None
    if episode_path.exists():
        with contextlib.suppress(Exception):
            event_count = json.loads(episode_path.read_text(encoding="utf-8")).get("event_count")

    events_lines = 0
    if events_path.exists():
        with events_path.open("rb") as f:
            for _ in f:
                events_lines += 1

    timeline_exists = timeline_path.exists()
    consistent = event_count is None or event_count == events_lines
    return event_count, events_lines, consistent, timeline_exists


def cmd_db_status(args: argparse.Namespace) -> int:
    """Show storage backend status and capabilities."""
    cfg = _load_storage_config(args)
    try:
        client = _create_client(cfg)
    except Exception as exc:  # noqa: BLE001
        if args.json:
            print(
                json.dumps(
                    {"backend": cfg["backend"], "connected": False, "error": str(exc)},
                    indent=2,
                )
            )
        else:
            print(f"[rosclaw db status] Failed to create backend: {exc}", file=sys.stderr)
        return 1

    try:
        ping = StorageFactory.ping(client)
        caps = StorageFactory.capabilities(client)
        extras: dict[str, Any] = {}
        if cfg["backend"] in {"sqlite", "mysql"}:
            extras["migrations"] = _migration_status(client, cfg["backend"])
        outbox_stats = _outbox_status(cfg)
        if outbox_stats is not None:
            extras["outbox"] = outbox_stats
        extras["vector"] = _vector_status(client)
        extras["practice"] = _practice_status(cfg)
    finally:
        _close_client(client)

    display_url = _sanitize_url(str(cfg.get("url") or cfg.get("path") or ""))
    result: dict[str, Any] = {
        "backend": cfg["backend"],
        "url": display_url,
        "capabilities": caps,
        "ping": ping,
        **extras,
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return 0 if ping.get("connected") else 1

    print("=" * 60)
    print("ROSClaw Storage Status")
    print("=" * 60)
    print(f"  Backend:      {cfg['backend']}")
    print(f"  URL/Path:     {display_url}")
    print("  Capabilities:")
    for name, value in caps.items():
        print(f"    {name}: {value}")
    print("  Ping:")
    print(f"    connected:  {ping.get('connected')}")
    if ping.get("latency_ms") is not None:
        print(f"    latency_ms: {ping['latency_ms']}")
    if ping.get("wal_size_mb") is not None:
        print(f"    wal_size_mb: {ping['wal_size_mb']}")
    if ping.get("error"):
        print(f"    error:      {ping['error']}")
    if extras.get("migrations"):
        mig = extras["migrations"]
        print("  Migrations:")
        print(f"    pending:    {mig.get('pending')}")
    if extras.get("outbox"):
        ob = extras["outbox"]
        print("  Outbox:")
        print(f"    total:      {ob.get('total')}")
        print(f"    pending:    {ob.get('pending')}")
    if extras.get("vector"):
        vec = extras["vector"]
        print("  Vector:")
        print(f"    enabled:    {vec.get('enabled')}")
        print(f"    warmed:     {vec.get('warmed')}")
    if extras.get("practice"):
        prac = extras["practice"]
        print("  Practice:")
        print(f"    catalog:    {prac.get('catalog_exists')}")
        if "practices" in prac:
            print(f"    practices:  {prac['practices']}")
        if prac.get("latest_session"):
            ls = prac["latest_session"]
            print(f"    latest:     {ls.get('name')} events={ls.get('events_jsonl_lines')}")
    print("=" * 60)
    return 0 if ping.get("connected") else 1


def cmd_db_doctor(args: argparse.Namespace) -> int:
    """Run storage health checks and optionally apply safe fixes."""
    cfg = _load_storage_config(args)
    issues: list[str] = []
    checks: list[tuple[str, str, bool]] = []
    fixes: list[str] = []
    result: dict[str, Any] = {"checks": [], "issues": [], "fixes": []}

    # 1. Config presence
    config_ok = cfg["config_found"]
    config_overridden = cfg["config_overridden"]
    checks.append(
        (
            "rosclaw.yaml",
            "found" if config_ok else ("overridden by CLI" if config_overridden else "missing"),
            config_ok or config_overridden,
        )
    )
    if not config_ok and not config_overridden:
        issues.append("No rosclaw.yaml found; using defaults.")

    # 2. Backend resolution and URL sanity
    backend = cfg["backend"]
    url = cfg["url"]
    resolved_ok = backend in {"memory", "sqlite", "mysql"}
    checks.append(("backend", backend, resolved_ok))
    if not resolved_ok:
        issues.append(f"Unknown backend '{backend}'.")

    if (
        backend in {"sqlite", "mysql"}
        and url
        and str(url).lower().startswith(("http://", "https://"))
    ):
        issues.append(
            f"{backend} backend configured but seekdb_url looks like HTTP ({url}). "
            "Use ROSCLAW_PRACTICE_HTTP_ADAPTER_URL for the HTTP bridge."
        )

    # 3. Connection + ping
    client = None
    ping: dict[str, Any] = {"connected": False, "error": None}
    try:
        client = _create_client(cfg)
        ping = StorageFactory.ping(client)
    except Exception as exc:  # noqa: BLE001
        ping["error"] = str(exc)
    finally:
        if client is None:
            issues.append(f"Cannot create storage client: {ping['error']}")

    checks.append(
        (
            "connect",
            "ok" if ping.get("connected") else ping.get("error") or "failed",
            ping.get("connected", False),
        )
    )
    if not ping.get("connected"):
        issues.append(f"Storage ping failed: {ping.get('error')}")

    # 4. Backend-specific checks
    tables: list[str] = []
    pragmas: dict[str, Any] = {}
    if client is not None and backend == "sqlite":
        try:
            connection = client._connection
            pragmas = _sqlite_pragmas(connection)
            tables = _sqlite_table_names(connection)
            checks.append(
                (
                    "journal_mode",
                    str(pragmas.get("journal_mode")),
                    pragmas.get("journal_mode") == "wal",
                )
            )
            if pragmas.get("journal_mode") != "wal":
                issues.append("SQLite journal_mode is not WAL.")
                if args.fix:
                    connection.execute("PRAGMA journal_mode=WAL")
                    fixes.append("Set SQLite journal_mode=WAL.")
                    pragmas["journal_mode"] = "wal"
            checks.append(
                (
                    "busy_timeout",
                    str(pragmas.get("busy_timeout")),
                    bool(pragmas.get("busy_timeout")),
                )
            )
            if not pragmas.get("busy_timeout"):
                issues.append("SQLite busy_timeout is not set.")
                if args.fix:
                    connection.execute("PRAGMA busy_timeout=5000")
                    fixes.append("Set SQLite busy_timeout=5000.")
                    pragmas["busy_timeout"] = 5000
            wal_size = _sqlite_wal_size(cfg["path"])
            wal_ok = wal_size < 100 * 1024 * 1024
            checks.append(("wal_size", f"{wal_size / (1024 * 1024):.2f} MB", wal_ok))
            if not wal_ok:
                issues.append("SQLite WAL is larger than 100 MB.")
                if args.fix:
                    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    fixes.append("Ran PRAGMA wal_checkpoint(TRUNCATE).")
            db_size = Path(cfg["path"]).stat().st_size
            size_ok = db_size < 2 * 1024 * 1024 * 1024
            checks.append(("db_size", f"{db_size / (1024 * 1024):.2f} MB", size_ok))
            if not size_ok:
                issues.append("SQLite database is larger than 2 GB.")

            # Vector-store checks for SQLite.
            vec = _vector_status(client)
            checks.append(("vector", "enabled" if vec["enabled"] else "disabled", True))
            if vec.get("enabled"):
                warmed = vec.get("warmed", False)
                if not warmed and args.fix:
                    try:
                        client.warmup_embedder()
                        warmed = True
                        fixes.append("Warmed vector embedder.")
                    except Exception as warmup_exc:  # noqa: BLE001
                        issues.append(f"Vector warmup failed: {warmup_exc}")
                checks.append(("vector warmup", "warmed" if warmed else "not warmed", warmed))
                if not warmed:
                    issues.append("Vector store is enabled but embedder is not warmed.")
        except Exception as exc:  # noqa: BLE001
            issues.append(f"SQLite introspection failed: {exc}")

    if client is not None and backend == "mysql":
        try:
            with client._connection as connection:
                tables = _mysql_table_names(connection)
        except Exception as exc:  # noqa: BLE001
            issues.append(f"MySQL table listing failed: {exc}")

    # 5. Schema migrations
    if client is not None and backend in {"sqlite", "mysql"}:
        try:
            runner = MigrationRunner()
            if backend == "sqlite":
                pending = runner.pending(client._connection, backend)
            else:
                with client._connection as connection:
                    pending = runner.pending(connection, backend)

            if args.fix:
                if backend == "mysql":
                    with client._connection as connection:
                        applied = runner.apply(connection, backend)
                else:
                    applied = runner.apply(client._connection, backend)
                # Recompute pending after applying so the check reflects the
                # post-fix state rather than the pre-fix count.
                if backend == "mysql":
                    with client._connection as connection:
                        pending_after = runner.pending(connection, backend)
                else:
                    pending_after = runner.pending(client._connection, backend)
                checks.append(
                    (
                        "migrations",
                        f"{len(applied)} applied / {len(pending_after)} pending",
                        len(pending_after) == 0,
                    )
                )
                fixes.extend(f"Applied migration {v}." for v in applied)
                if pending_after:
                    issues.append(f"Pending migrations after fix: {pending_after}")
            else:
                checks.append(("migrations", f"{len(pending)} pending", len(pending) == 0))
                if pending:
                    issues.append(f"Pending migrations: {pending}")
        except Exception as exc:  # noqa: BLE001
            issues.append(f"Migration check failed: {exc}")
            checks.append(("migrations", "failed", False))

    checks.append(
        (
            "schema_migrations table",
            "present"
            if "schema_migrations" in tables
            else ("n/a" if backend == "memory" else "missing"),
            "schema_migrations" in tables or backend == "memory",
        )
    )
    if "schema_migrations" not in tables and backend != "memory":
        issues.append("schema_migrations table is missing.")

    # 6. Outbox check
    if cfg["outbox_enabled"]:
        try:
            outbox = OutboxStore(db_path=cfg["outbox_path"])
            stats = outbox.stats()
            outbox.close()
            oldest_sec = stats.get("oldest_pending_sec")
            oldest_str = f"oldest={oldest_sec}s" if oldest_sec is not None else "oldest=n/a"
            checks.append(
                (
                    "outbox",
                    f"{stats['total']} total / {stats['pending']} pending / {stats.get('dead_letters', 0)} dead ({oldest_str})",
                    True,
                )
            )
            if stats["failed"]:
                issues.append(f"Outbox has {stats['failed']} failed records.")
            if stats.get("dead_letters"):
                issues.append(f"Outbox has {stats['dead_letters']} dead-letter records.")
            result["outbox"] = stats
        except Exception as exc:  # noqa: BLE001
            issues.append(f"Outbox check failed: {exc}")
            checks.append(("outbox", "failed", False))

    # 7. Practice catalog check
    catalog_path = _practice_catalog_path(cfg["practice_data_root"])
    if catalog_path.exists():
        try:
            catalog_conn = sqlite3.connect(str(catalog_path), check_same_thread=False)
            catalog_conn.row_factory = sqlite3.Row
            count = catalog_conn.execute("SELECT COUNT(*) FROM practices").fetchone()[0]
            catalog_conn.close()
            checks.append(("practice catalog", f"{count} practices", True))
            result["practice_catalog"] = {"path": str(catalog_path), "practices": count}
        except Exception as exc:  # noqa: BLE001
            issues.append(f"Practice catalog check failed: {exc}")
            checks.append(("practice catalog", "failed", False))
    else:
        checks.append(("practice catalog", "not found (no sessions yet)", True))

    # 8. Latest practice session consistency
    latest_session_dir = _latest_session_dir(cfg["practice_data_root"])
    if latest_session_dir is not None:
        try:
            event_count, events_lines, consistent, timeline_exists = _session_event_consistency(
                latest_session_dir
            )
            checks.append(
                (
                    "latest session events",
                    f"{events_lines} lines / episode.event_count={event_count}",
                    consistent,
                )
            )
            if not consistent:
                issues.append(
                    f"Latest session {latest_session_dir.name} event_count "
                    f"({event_count}) != events.jsonl lines ({events_lines})."
                )
            timeline_ok = not timeline_exists
            checks.append(
                (
                    "latest session timeline",
                    "present" if timeline_exists else "absent",
                    timeline_ok,
                )
            )
            if timeline_exists:
                issues.append(f"Latest session {latest_session_dir.name} still has timeline.jsonl.")
            result["latest_session"] = {
                "path": str(latest_session_dir),
                "event_count": event_count,
                "events_jsonl_lines": events_lines,
                "timeline_exists": timeline_exists,
            }
        except Exception as exc:  # noqa: BLE001
            issues.append(f"Latest session consistency check failed: {exc}")

    if client is not None:
        _close_client(client)

    result["checks"] = [{"name": n, "value": v, "ok": ok} for n, v, ok in checks]
    result["issues"] = issues
    result["fixes"] = fixes
    exit_code = 0 if not issues else 1

    if args.json:
        print(json.dumps(result, indent=2))
        return exit_code

    print("=" * 60)
    print("ROSClaw Storage Doctor")
    print("=" * 60)
    for name, value, ok in checks:
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name:<30} {value}")
    print("=" * 60)
    if fixes:
        print(f"\n🔧 Fixes applied ({len(fixes)}):")
        for fx in fixes:
            print(f"  • {fx}")
    if issues:
        print(f"\n⚠️  Issues found ({len(issues)}):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        return 1
    print("\n✅ All storage checks passed.")
    return 0


def add_db_subparser(subparsers: Any) -> Any:
    """Add ``rosclaw db`` subcommands."""
    db_parser = subparsers.add_parser("db", help="Storage backend diagnostics")
    db_subparsers = db_parser.add_subparsers(dest="db_command")

    status_parser = db_subparsers.add_parser("status", help="Show storage backend status")
    status_parser.add_argument("--json", action="store_true", help="Output JSON")
    status_parser.add_argument("--backend", default=None, help="Override backend")
    status_parser.add_argument("--url", default=None, help="Override SQL URL")
    status_parser.add_argument("--path", default=None, help="Override SQLite path")

    doctor_parser = db_subparsers.add_parser("doctor", help="Run storage health checks")
    doctor_parser.add_argument("--json", action="store_true", help="Output JSON")
    doctor_parser.add_argument("--fix", action="store_true", help="Apply safe fixes")
    doctor_parser.add_argument("--backend", default=None, help="Override backend")
    doctor_parser.add_argument("--url", default=None, help="Override SQL URL")
    doctor_parser.add_argument("--path", default=None, help="Override SQLite path")
    return db_parser


# contextlib is imported lazily here to avoid a top-level dependency on
# an unused module for the happy path.
import contextlib  # noqa: E402
