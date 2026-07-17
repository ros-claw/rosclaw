"""Storage factory for ROSClaw.

Centralizes selection and health-checking of the knowledge-store backend
(Memory / SQLite / MySQL-compatible SeekDB/OceanBase).  Other modules should
use :class:`StorageFactory` instead of importing backend classes directly so
that backend detection, URL validation, and observability stay in one place.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rosclaw.memory.seekdb_client import (
    InMemoryKnowledgeStore,
    SeekDBClient,
    SeekDBMySQLClient,
    SQLiteKnowledgeStore,
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


class StorageFactory:
    """Create and inspect knowledge-store backends from runtime configuration."""

    @staticmethod
    def create_knowledge_store(
        *,
        backend: str | None = None,
        url: str | None = None,
        path: str | None = None,
        pool_size: int = 4,
        vector_enabled: bool = False,
        embedder: Any | None = None,
    ) -> SeekDBClient:
        """Return a :class:`SeekDBClient` for the chosen backend.

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
            return InMemoryKnowledgeStore()

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
            return SQLiteKnowledgeStore(
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
            return SeekDBMySQLClient(
                str(url),
                pool_size=pool_size,
                connect_timeout=5.0,
                read_timeout=10.0,
                write_timeout=10.0,
            )

        if chosen == "seekdb_embedded":
            from rosclaw.storage.seekdb_native import SeekDBEmbeddedStore

            db_path = path or (str(url) if url and not _is_http_url(url) else None)
            logger.info("Knowledge store backend: seekdb_embedded (%s)", db_path or "default")
            if db_path:
                return SeekDBEmbeddedStore(path=db_path)
            return SeekDBEmbeddedStore()

        if chosen == "seekdb_server":
            from urllib.parse import urlparse

            from rosclaw.storage.seekdb_native import SeekDBServerStore

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
            return SeekDBServerStore(
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
        """Return the backend that :meth:`create_knowledge_store` would select."""
        detected = _detect_backend_from_url(url)
        if backend == "memory" and detected:
            return detected
        return (backend or detected or "memory").lower()

    @staticmethod
    def ping(client: SeekDBClient) -> dict[str, Any]:
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
            # InMemoryKnowledgeStore does not support arbitrary SQL; count a known table.
            if isinstance(client, InMemoryKnowledgeStore):
                client.count("experience_graph", {})
            else:
                client.count("experience_graph", {})
            result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            result["connected"] = True
        except Exception as exc:  # noqa: BLE001
            result["error"] = str(exc)
            return result

        if isinstance(client, SQLiteKnowledgeStore):
            try:
                db_path = Path(client._db_path).expanduser()
                wal_path = db_path.parent / f"{db_path.name}-wal"
                result["wal_size_bytes"] = wal_path.stat().st_size if wal_path.exists() else 0
                result["wal_size_mb"] = round(result["wal_size_bytes"] / (1024 * 1024), 3)
            except Exception:  # noqa: BLE001
                pass

        return result

    @staticmethod
    def capabilities(client: SeekDBClient) -> dict[str, bool]:
        """Return capability flags for *client*."""
        has_vector = False
        if isinstance(client, SQLiteKnowledgeStore):
            has_vector = getattr(client, "_vector_enabled", False)
        return {
            "persistent": not isinstance(client, InMemoryKnowledgeStore),
            "sql": isinstance(client, (SQLiteKnowledgeStore, SeekDBMySQLClient)),
            "mysql": isinstance(client, SeekDBMySQLClient),
            "sqlite": isinstance(client, SQLiteKnowledgeStore),
            "vector": has_vector,
        }
