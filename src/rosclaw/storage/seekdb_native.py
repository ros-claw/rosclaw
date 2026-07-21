"""Native SeekDB backend via pyseekdb (PR-SDB-1, §7).

``SeekDBNativeStore`` implements the :class:`SeekDBClient` knowledge-store
interface on top of native SeekDB collections — one collection per knowledge
table.  Records live as collection metadata; the built-in server-side
embedder vectorizes the document text, giving:

* native vector search (:meth:`similar`);
* native BM25 full-text (``hybrid_search`` ``where_document.$contains``);
* native metadata filters (``where``);
* native RRF hybrid fusion (:meth:`hybrid_search`).

Nothing here is a Python full-table scan masquerading as native SeekDB:
relational ``query()`` with filters uses SeekDB's own metadata filtering;
when an ``order_by`` is requested the result is sorted client-side *after*
the filtered fetch, which is documented in the docstring.

Two deployments share the class:

* embedded — ``SeekDBNativeStore(path="~/.rosclaw/data/seekdb")``;
* server — ``SeekDBNativeStore(host=..., port=2881, user=..., password=...)``
  (also works against OceanBase, which speaks the same protocol).
"""

from __future__ import annotations

import logging
from contextlib import ExitStack
from pathlib import Path
from threading import Lock
from typing import Any

from rosclaw.memory.seekdb_client import SeekDBClient

logger = logging.getLogger("rosclaw.storage.seekdb_native")

_EMBEDDED_PATH_LOCK = Lock()
_EMBEDDED_PROCESS_TARGET: tuple[str, str] | None = None

# Fields whose text is used as the embedding document, per table family.
# Task/outcome fields lead: without them the episode document degrades to an
# opaque ``artifact_uri=...`` key-value dump, which is useless for retrieval
# (found while decomposing a CJK-vs-English BM25 check on the RPS episodes).
_TEXT_FIELDS = (
    "title",
    "document",
    "instruction",
    "description",
    "summary",
    "task_name",
    "task_id",
    "task",
    "skill_id",
    "outcome",
    "result",
    "error_details",
    "subject",
    "predicate",
    "object",
    "condition",
    "action",
    "name",
    "root_cause",
    "recovery_hint",
    "hypothesis",
    "rejection_reason",
)


def _require_pyseekdb():
    try:
        import pyseekdb
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "pyseekdb is required for the native SeekDB backend. "
            "Install it with: pip install 'rosclaw[seekdb]' (or pip install pyseekdb)"
        ) from exc
    return pyseekdb


class SeekDBNativeStore(SeekDBClient):
    """Native SeekDB knowledge store (embedded or server)."""

    def __init__(
        self,
        *,
        path: str | None = None,
        host: str | None = None,
        port: int = 2881,
        user: str = "root",
        password: str = "",
        database: str = "rosclaw",
        protocol: str | None = None,
    ):
        if path is None and host is None:
            raise ValueError("SeekDBNativeStore requires either path (embedded) or host (server)")
        self._path = path
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = database
        self._client: Any | None = None
        self._client_stack: ExitStack | None = None
        self._collections: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if self._client is not None:
            return
        pyseekdb = _require_pyseekdb()
        if self._path is not None:
            self._claim_embedded_target()
            with pyseekdb.AdminClient(path=self._path) as admin:
                self._ensure_database(admin)
            client_context = pyseekdb.Client(path=self._path, database=self._database)
        else:
            with pyseekdb.AdminClient(
                host=self._host, port=self._port, user=self._user, password=self._password
            ) as admin:
                self._ensure_database(admin)
            client_context = pyseekdb.Client(
                host=self._host,
                port=self._port,
                user=self._user,
                password=self._password,
                database=self._database,
            )
        stack = ExitStack()
        try:
            self._client = stack.enter_context(client_context)
            self._client_stack = stack
            self._wait_ready()
        except BaseException:
            self._client = None
            self._client_stack = None
            stack.close()
            raise
        logger.info(
            "SeekDBNativeStore connected (%s, database=%s)",
            f"embedded:{self._path}" if self._path else f"server:{self._host}:{self._port}",
            self._database,
        )

    def _claim_embedded_target(self) -> None:
        """Prevent pylibseekdb from silently reusing another process-global target."""
        if self._path is None:
            return
        path = str(Path(self._path).resolve())
        target = (path, self._database)
        global _EMBEDDED_PROCESS_TARGET
        with _EMBEDDED_PATH_LOCK:
            if _EMBEDDED_PROCESS_TARGET is None:
                _EMBEDDED_PROCESS_TARGET = target
            elif target != _EMBEDDED_PROCESS_TARGET:
                claimed_path, claimed_database = _EMBEDDED_PROCESS_TARGET
                raise RuntimeError(
                    "pylibseekdb supports one embedded path/database target per process; "
                    f"this process already uses path={claimed_path!r}, "
                    f"database={claimed_database!r} and cannot open path={path!r}, "
                    f"database={self._database!r}. Reuse the existing target or start "
                    "a separate process."
                )
        self._path = path

    def _wait_ready(self, timeout_s: float = 30.0) -> None:
        """Block until the (embedded) engine answers catalog queries.

        The embedded SeekDB engine opens asynchronously; under load (full
        suite) early catalog calls can hit a not-yet-ready engine and lose
        writes or report missing collections.  Probe with retries so callers
        never see a half-open engine.
        """
        import time

        deadline = time.monotonic() + timeout_s
        last_exc: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self._client.list_collections()
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(0.5)
        raise RuntimeError(
            f"SeekDBNativeStore engine not ready within {timeout_s}s: {last_exc}"
        ) from last_exc

    def _ensure_database(self, admin: Any) -> None:
        try:
            admin.create_database(self._database)
            logger.info("Created SeekDB database %s", self._database)
        except Exception as exc:  # noqa: BLE001
            # Database already exists — expected on every start after the first.
            logger.debug("create_database(%s): %s", self._database, exc)

    def is_connected(self) -> bool:
        return self._client is not None

    def disconnect(self) -> None:
        stack = self._client_stack
        self._client = None
        self._client_stack = None
        self._collections = {}
        if stack is not None:
            stack.close()

    # ------------------------------------------------------------------
    # Collections
    # ------------------------------------------------------------------

    def _collection(self, table: str) -> Any:
        if table in self._collections:
            return self._collections[table]
        client = self._client
        if client is None:
            raise RuntimeError("SeekDBNativeStore is not connected")
        try:
            collection = client.get_collection(table)
        except Exception:  # noqa: BLE001
            collection = client.create_collection(name=table)
        self._collections[table] = collection
        return collection

    @staticmethod
    def _document_text(record: dict) -> str:
        # Text fields from the top level AND the nested metadata dict — the
        # episode schema nests task_name/session info under ``metadata``, and
        # without looking inside, the document degenerates to an opaque
        # ``artifact_uri=...`` dump (useless for both BM25 and embedding).
        metadata = record.get("metadata")
        sources = [record]
        if isinstance(metadata, dict):
            sources.append(metadata)
        seen: set[str] = set()
        parts: list[str] = []
        for source in sources:
            for key in _TEXT_FIELDS:
                value = source.get(key)
                if value and str(value) not in seen:
                    seen.add(str(value))
                    parts.append(str(value))
        if parts:
            return "\n".join(parts)
        # Fallback: embed a compact dump so every record is searchable.
        return " ".join(
            f"{key}={value}" for key, value in sorted(record.items()) if value is not None
        )

    @staticmethod
    def _metadata(record: dict) -> dict[str, Any]:
        """Collection metadata must be JSON-primitive."""
        meta: dict[str, Any] = {}
        for key, value in record.items():
            if value is None or isinstance(value, (str, int, float, bool)):
                meta[key] = value
            else:
                import json

                meta[key] = json.dumps(value, ensure_ascii=False)
        return meta

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def insert(self, table: str, record: dict) -> str:
        collection = self._collection(table)
        record_id = str(record.get("id") or record.get("memory_id") or "")
        if not record_id:
            import uuid

            record_id = str(uuid.uuid4())
            record = {**record, "id": record_id}
        collection.upsert(
            ids=[record_id],
            documents=[self._document_text(record)],
            metadatas=[self._metadata(record)],
        )
        return record_id

    def insert_many(self, table: str, records: list[dict]) -> int:
        """Batch upsert, then one index refresh (vector visibility is
        eventually consistent; refreshing per record is prohibitively slow)."""
        if not records:
            return 0
        collection = self._collection(table)
        ids = []
        documents = []
        metadatas = []
        import uuid

        for record in records:
            record_id = str(record.get("id") or record.get("memory_id") or uuid.uuid4())
            ids.append(record_id)
            documents.append(self._document_text(record))
            metadatas.append(self._metadata(record))
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        self.refresh_index(table)
        return len(ids)

    def refresh_index(self, table: str) -> None:
        """Force the vector index to pick up recent writes."""
        try:
            self._collection(table).refresh_index()
        except Exception as exc:  # noqa: BLE001
            logger.debug("refresh_index(%s): %s", table, exc)

    def query(
        self,
        table: str,
        filters: dict | None = None,
        order_by: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        collection = self._collection(table)
        result = collection.get(
            where=filters or None,
            limit=limit,
            include=["metadatas"],
        )
        records = self._records_from_result(result)
        if order_by:
            reverse = order_by.startswith("-")
            key = order_by.lstrip("+-")
            records.sort(key=lambda r: (r.get(key) is None, r.get(key)), reverse=reverse)
        return records[:limit]

    @staticmethod
    def _records_from_result(result: dict | None) -> list[dict]:
        if not result:
            return []
        ids = result.get("ids") or []
        metadatas = result.get("metadatas") or []
        # get() returns flat lists; query() returns nested lists.
        if ids and isinstance(ids[0], list):
            ids = ids[0]
            metadatas = metadatas[0] if metadatas else []
        records = []
        for record_id, metadata in zip(ids, metadatas, strict=False):
            record = dict(metadata or {})
            record.setdefault("id", record_id)
            records.append(record)
        return records

    def update(self, table: str, record_id: str, updates: dict) -> bool:
        collection = self._collection(table)
        existing = collection.get(ids=[record_id], include=["metadatas", "documents"])
        ids = (existing or {}).get("ids") or []
        if not ids:
            return False
        metadata = dict((existing.get("metadatas") or [{}])[0] or {})
        metadata.update(self._metadata(updates))
        document = updates and self._document_text({**metadata, **updates})
        collection.update(
            ids=[record_id],
            documents=[document] if document else None,
            metadatas=[metadata],
        )
        return True

    def count(self, table: str, filters: dict | None = None) -> int:
        collection = self._collection(table)
        if not filters:
            return int(collection.count())
        return len(self.query(table, filters=filters, limit=100_000))

    def delete(self, table: str, record_id: str) -> bool:
        collection = self._collection(table)
        existing = collection.get(ids=[record_id], include=[])
        if not (existing or {}).get("ids"):
            return False
        collection.delete(ids=[record_id])
        return True

    def delete_where(self, table: str, filters: dict) -> int:
        records = self.query(table, filters=filters, limit=100_000)
        ids = [record["id"] for record in records]
        if ids:
            self._collection(table).delete(ids=ids)
        return len(ids)

    # ------------------------------------------------------------------
    # Native retrieval
    # ------------------------------------------------------------------

    def similar(
        self,
        table: str,
        query_text: str,
        filters: dict | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """Native vector search with optional metadata pre-filter."""
        collection = self._collection(table)
        result = collection.query(
            query_texts=[query_text],
            where=filters or None,
            n_results=limit,
            include=["metadatas", "distances"],
        )
        records = self._records_from_result(result)
        distances = (result or {}).get("distances") or [[]]
        flat = distances[0] if distances and isinstance(distances[0], list) else distances
        for record, distance in zip(records, flat, strict=False):
            record["score"] = 1.0 - float(distance) if distance is not None else 0.0
        return records

    def hybrid_search(
        self,
        table: str,
        query_text: str,
        filters: dict | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """Native BM25 + vector + metadata filter with RRF fusion."""
        collection = self._collection(table)
        result = collection.hybrid_search(
            query={"where_document": {"$contains": query_text}, "n_results": limit},
            knn={
                "query_texts": [query_text],
                "n_results": limit,
                **({"where": filters} if filters else {}),
            },
            rank={"rrf": {}},
            n_results=limit,
            include=["metadatas"],
        )
        return self._records_from_result(result)

    def embedding_info(self, table: str) -> dict[str, Any]:
        """Model identity of the collection's built-in embedder (for §6.5 registry)."""
        collection = self._collection(table)
        info: dict[str, Any] = {}
        try:
            ef = collection.embedding_function
            info["embedder_type"] = type(ef).__name__ if ef is not None else None
            info["model_name"] = getattr(ef, "model_name", None) or getattr(ef, "_model_name", None)
        except Exception:  # noqa: BLE001
            pass
        try:
            info["dimension"] = collection.dimension
        except Exception:  # noqa: BLE001
            info["dimension"] = None
        return info

    # ------------------------------------------------------------------
    # Diagnostics (db doctor)
    # ------------------------------------------------------------------

    def list_collections(self) -> list[str]:
        """Names of all collections in the current database (engine catalog)."""
        client = self._client
        if client is None:
            raise RuntimeError("SeekDBNativeStore is not connected")
        return sorted(getattr(c, "name", str(c)) for c in client.list_collections())

    def deployment_info(self) -> dict[str, Any]:
        """Deployment descriptor for diagnostics (embedded path / server DSN).

        Never includes the password.
        """
        if self._path is not None:
            return {"mode": "embedded", "path": self._path, "database": self._database}
        return {
            "mode": "server",
            "host": self._host,
            "port": self._port,
            "user": self._user,
            "database": self._database,
        }


# Deployment-specific aliases matching the PR-SDB-1 naming (§7.3).
class SeekDBEmbeddedStore(SeekDBNativeStore):
    """Embedded SeekDB knowledge store.

    pylibseekdb owns process-global engine state, so one process must reuse the
    same path and database. Use another process for a different embedded target.
    """

    def __init__(self, path: str = "~/.rosclaw/data/seekdb", database: str = "rosclaw"):
        super().__init__(path=str(Path(path).expanduser()), database=database)


class SeekDBServerStore(SeekDBNativeStore):
    """Server-mode SeekDB / OceanBase knowledge store."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 2881,
        user: str = "root",
        password: str = "",
        database: str = "rosclaw",
    ):
        super().__init__(host=host, port=port, user=user, password=password, database=database)
