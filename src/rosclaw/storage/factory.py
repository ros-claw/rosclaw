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
            db_path = path
            if url and not db_path:
                db_path = str(url)
                if db_path.lower().startswith("sqlite://"):
                    db_path = db_path[len("sqlite://") :]
            if not db_path:
                raise ValueError("seekdb_backend='sqlite' requires seekdb_path or a sqlite:// URL.")
            logger.info("Knowledge store backend: sqlite (%s)", db_path)
            return SQLiteKnowledgeStore(db_path)

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
            return SeekDBMySQLClient(str(url))

        raise ValueError(
            f"Unknown knowledge-store backend '{chosen}'. Supported: memory, sqlite, mysql."
        )

    @staticmethod
    def resolve_backend(
        *,
        backend: str | None = None,
        url: str | None = None,
    ) -> str:
        """Return the backend that :meth:`create_knowledge_store` would select."""
        detected = _detect_backend_from_url(url)
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
        return {
            "persistent": not isinstance(client, InMemoryKnowledgeStore),
            "sql": isinstance(client, (SQLiteKnowledgeStore, SeekDBMySQLClient)),
            "mysql": isinstance(client, SeekDBMySQLClient),
            "sqlite": isinstance(client, SQLiteKnowledgeStore),
            "vector": False,  # populated by VectorStore wrapper in Phase 1.3
        }
