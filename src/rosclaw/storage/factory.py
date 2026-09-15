"""Storage factory for ROSClaw.

Centralizes selection and health-checking of the knowledge-store backend
(Memory / SQLite / MySQL-compatible SeekDB/OceanBase).  Other modules should
use :class:`StoreFactory` instead of importing backend classes directly so
that backend detection, URL validation, and observability stay in one place.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rosclaw.memory.seekdb_client import (
    InMemoryStructuredStore,
    SeekDBSQLStore,
    SQLiteStructuredStore,
    StructuredStore,
)
from rosclaw.storage.vector import TfidfEmbedder

logger = logging.getLogger("rosclaw.storage.factory")

_SQL_SCHEMES = {"sqlite", "mysql", "mysql+pymysql", "seekdb"}


def _sanitize_url(url: str) -> str:
    """Return a display-safe version of a SQL DSN with password redacted."""
    try:
        parsed = urlparse(url)
        if parsed.password:
            return url.replace(f":{parsed.password}@", ":***@", 1)
    except Exception:  # noqa: BLE001
        pass
    return url


def _is_http_url(url: str) -> bool:
    return str(url).lower().startswith(("http://", "https://"))


def _detect_backend_from_url(url: str | None) -> str | None:
    """Return a backend short name if *url* is unambiguous, else None."""
    if not url:
        return None
    parsed = urlparse(str(url))
    scheme = parsed.scheme.lower()
    if scheme in _SQL_SCHEMES:
        if scheme == "sqlite":
            return "sqlite"
        return "mysql"
    if _is_http_url(str(url)):
        return "http"
    return None


class StoreFactory:
    """Create and inspect knowledge-store backends from runtime configuration."""

    @staticmethod
    def create_structured_store(
        *,
        backend: str | None = None,
        url: str | None = None,
        path: str | None = None,
        pool_size: int = 4,
        vector_enabled: bool = False,
        embedder: Any | None = None,
    ) -> StructuredStore:
        """Return a :class:`StructuredStore` for the chosen backend.

        Resolution order:
        1. If ``backend`` is explicitly provided, it wins (after validating the
           URL is not an HTTP URL for a SQL backend).
        2. If ``url`` is provided and has a recognizable scheme, derive the
           backend from it.
        3. Otherwise default to ``memory``.

        :param backend: ``memory``, ``sqlite``, or ``mysql``.
        :param url: SQL DSN or bare path. HTTP URLs are rejected for SQL backends.
        :param path: SQLite file path; used when ``url`` is absent or empty.
        :param pool_size: Reserved for future connection-pool sizing; currently
            passed through to MySQL-compatible backends.
        :raises ValueError: on ambiguous or unsupported backend configuration.
        """
        detected = _detect_backend_from_url(url)
        # "memory" is the neutral/ephemeral default. If a concrete URL scheme is
        # provided, let it select the real backend so callers that only set
        # ``ROSCLAW_SEEKDB_URL`` get the right implementation.
        if backend == "memory" and detected:
            chosen = detected
        else:
            chosen = (backend or detected or "memory").lower()

        # PR-SDB-140-4 (outline §五): ROSCLAW_SEEKDB_MODE makes the SeekDB
        # deployment mode EXPLICIT instead of guessed from path/host:
        #   legacy_embedded | local_runtime | server
        # An explicit mode wins over URL-derived detection and validates the
        # pairing (fail closed on contradiction — §十六).
        mode = os.environ.get("ROSCLAW_SEEKDB_MODE", "").strip().lower()
        if mode:
            mode_map = {
                "legacy_embedded": "seekdb_embedded",
                "local_runtime": "local_runtime",
                "server": "seekdb_server",
            }
            if mode not in mode_map:
                raise ValueError(
                    f"ROSCLAW_SEEKDB_MODE={mode!r} is not a deployment mode "
                    f"(supported: {', '.join(mode_map)})."
                )
            mapped = mode_map[mode]
            if backend and backend.lower() not in (mapped, "memory"):
                raise ValueError(
                    f"ROSCLAW_SEEKDB_MODE={mode} conflicts with seekdb_backend={backend!r}; "
                    "set one or the other, not both."
                )
            chosen = mapped

        if chosen == "local_runtime":
            # The 1.4 background-process embedded path.  Lifecycle is owned by
            # SeekDBLocalRuntime; storage semantics stay in the store adapter.
            # Fail closed when the bindings wheel is unavailable (aarch64
            # today) — never silently fall back to another backend.
            from rosclaw.storage.seekdb_runtime import SeekDBLocalRuntime

            if not SeekDBLocalRuntime.available():
                from rosclaw.storage.seekdb_runtime import LocalRuntimeUnavailableError

                raise LocalRuntimeUnavailableError(
                    "ROSCLAW_SEEKDB_MODE=local_runtime but the 'seekdb' bindings "
                    "package is unavailable on this platform (1.4.0.dev2 ships "
                    "x86_64 wheels only).  Choose server or legacy_embedded."
                )
            rt_dir = path or os.environ.get("ROSCLAW_SEEKDB_PATH") or ""
            if not rt_dir:
                raise ValueError(
                    "local_runtime requires seekdb_path (or ROSCLAW_SEEKDB_PATH) "
                    "for the runtime's db directory."
                )
            runtime = SeekDBLocalRuntime(rt_dir)
            runtime.recover_if_crashed()
            info = runtime.start()
            options = info.connection_options
            from rosclaw.storage.seekdb_native import SeekDBServerRetrievalStore

            logger.info(
                "Knowledge store backend: local_runtime (%s, pid=%s)",
                rt_dir,
                info.pid,
            )
            return SeekDBServerRetrievalStore(
                host=options.get("host", "127.0.0.1"),
                port=int(options.get("port", 2881)),
                database=(options.get("database") or "rosclaw"),
            )

        if chosen == "http":
            raise ValueError(
                "backend='http' is not a knowledge-store backend. "
                "For the rosclaw_practice HTTP bridge use seekdb_http_url / "
                "ROSCLAW_PRACTICE_HTTP_ADAPTER_URL; for SQL use sqlite:// or mysql://."
            )

        if chosen == "memory":
            if detected == "http":
                raise ValueError(
                    f"seekdb_backend='memory' but seekdb_url looks like an HTTP endpoint ({url}). "
                    f"Use seekdb_http_url / ROSCLAW_PRACTICE_HTTP_ADAPTER_URL for the HTTP bridge."
                )
            logger.info("Knowledge store backend: memory")
            return InMemoryStructuredStore()

        if chosen == "sqlite":
            db_path = None
            if url:
                db_path = str(url)
                if db_path.lower().startswith("sqlite://"):
                    db_path = db_path[len("sqlite://") :]
            if not db_path:
                db_path = path
            if not db_path:
                raise ValueError("seekdb_backend='sqlite' requires seekdb_path or a sqlite:// URL.")
            logger.info("Knowledge store backend: sqlite (%s)", db_path)
            return SQLiteStructuredStore(
                db_path,
                vector_enabled=vector_enabled,
                embedder=embedder or (TfidfEmbedder() if vector_enabled else None),
            )

        if chosen == "mysql":
            if not url:
                raise ValueError(
                    "seekdb_backend='mysql' requires seekdb_url (e.g. "
                    "mysql://root@127.0.0.1:2881/rosclaw). "
                    "Use ROSCLAW_SEEKDB_URL to set the SQL DSN."
                )
            if _is_http_url(url):
                raise ValueError(
                    f"seekdb_backend='mysql' but seekdb_url looks like an HTTP endpoint ({url}). "
                    f"For the rosclaw_practice HTTP bridge use seekdb_http_url / "
                    f"ROSCLAW_PRACTICE_HTTP_ADAPTER_URL; for SQL use mysql:// or seekdb://."
                )
            logger.info("Knowledge store backend: mysql (%s)", _sanitize_url(str(url)))
            return SeekDBSQLStore(
                str(url),
                pool_size=pool_size,
                connect_timeout=5.0,
                read_timeout=10.0,
                write_timeout=10.0,
            )

        if chosen == "seekdb_embedded":
            from rosclaw.storage.seekdb_native import SeekDBEmbeddedRetrievalStore

            db_path = path or (str(url) if url and not _is_http_url(url) else None)
            logger.info("Knowledge store backend: seekdb_embedded (%s)", db_path or "default")
            if db_path:
                return SeekDBEmbeddedRetrievalStore(path=db_path)
            return SeekDBEmbeddedRetrievalStore()

        if chosen == "seekdb_server":
            from urllib.parse import urlparse

            from rosclaw.storage.seekdb_native import SeekDBServerRetrievalStore

            if not url:
                raise ValueError(
                    "seekdb_backend='seekdb_server' requires seekdb_url "
                    "(e.g. mysql://root@127.0.0.1:2881/rosclaw or seekdb://root@host:2881/db)."
                )
            if _is_http_url(url):
                raise ValueError(
                    f"seekdb_backend='seekdb_server' but seekdb_url looks like HTTP ({url})."
                )
            parsed = urlparse(str(url))
            logger.info("Knowledge store backend: seekdb_server (%s)", _sanitize_url(str(url)))
            return SeekDBServerRetrievalStore(
                host=parsed.hostname or "127.0.0.1",
                port=parsed.port or 2881,
                user=parsed.username or "root",
                password=parsed.password or "",
                database=parsed.path.lstrip("/") or "rosclaw",
            )

        raise ValueError(
            f"Unknown knowledge-store backend '{chosen}'. "
            "Supported: memory, sqlite, mysql, seekdb_embedded, seekdb_server."
        )

    @staticmethod
    def resolve_backend(
        *,
        backend: str | None = None,
        url: str | None = None,
    ) -> str:
        """Return the backend that :meth:`create_structured_store` would select."""
        detected = _detect_backend_from_url(url)
        if backend == "memory" and detected:
            return detected
        return (backend or detected or "memory").lower()

    @staticmethod
    def ping(client: StructuredStore) -> dict[str, Any]:
        """Ping *client* and return latency/health metadata.

        The client is connected if necessary.  For SQLite, the WAL size is also
        reported so operators can spot checkpoint pressure.
        """
        result: dict[str, Any] = {
            "backend": type(client).__name__,
            "connected": False,
            "latency_ms": None,
            "error": None,
        }
        import time

        try:
            client.connect()
            t0 = time.perf_counter()
            # InMemoryStructuredStore does not support arbitrary SQL; count a known table.
            if isinstance(client, InMemoryStructuredStore):
                client.count("experience_graph", {})
            else:
                client.count("experience_graph", {})
            result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            result["connected"] = True
        except Exception as exc:  # noqa: BLE001
            result["error"] = str(exc)
            return result

        if isinstance(client, SQLiteStructuredStore):
            try:
                db_path = Path(client._db_path).expanduser()
                wal_path = db_path.parent / f"{db_path.name}-wal"
                result["wal_size_bytes"] = wal_path.stat().st_size if wal_path.exists() else 0
                result["wal_size_mb"] = round(result["wal_size_bytes"] / (1024 * 1024), 3)
            except Exception:  # noqa: BLE001
                pass

        return result

    @staticmethod
    def create_knowledge_store(**kwargs: Any) -> StructuredStore:
        """Deprecated alias for :meth:`create_structured_store` (ADR-0010)."""
        return StoreFactory.create_structured_store(**kwargs)

    @staticmethod
    def capabilities(client: StructuredStore) -> dict[str, bool]:
        """Return capability flags for *client*."""
        from rosclaw.storage.seekdb_native import SeekDBRetrievalStore

        has_vector = False
        if isinstance(client, SQLiteStructuredStore):
            has_vector = getattr(client, "_vector_enabled", False)
        elif isinstance(client, SeekDBRetrievalStore):
            has_vector = True
        return {
            "persistent": not isinstance(client, InMemoryStructuredStore),
            "sql": isinstance(client, (SQLiteStructuredStore, SeekDBSQLStore)),
            "mysql": isinstance(client, SeekDBSQLStore),
            "sqlite": isinstance(client, SQLiteStructuredStore),
            "vector": has_vector,
            "native_seekdb": isinstance(client, SeekDBRetrievalStore),
        }


# ADR-0010 compatibility aliases (PR-DF-01).
StorageFactory = StoreFactory


# ADR-0010 compatibility aliases (PR-DF-01).
StorageFactory = StoreFactory
