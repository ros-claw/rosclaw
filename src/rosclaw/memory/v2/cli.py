"""Memory 2.0 CLI handlers — registered into the existing ``rosclaw memory`` group.

New subcommands: ``verify``, ``consolidate``, ``forget``, ``index``,
``benchmark``, ``distill``.  The legacy ``status``/``query``/``explain``
subcommands gain a ``--v2`` flag that routes to the Memory 2.0 handlers
without changing legacy behavior.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

from rosclaw.firstboot.config import load_rosclaw_yaml
from rosclaw.firstboot.workspace import resolve_home
from rosclaw.memory.seekdb_client import SQLiteKnowledgeStore
from rosclaw.memory.v2.consolidate import MemoryConsolidator
from rosclaw.memory.v2.distill import distill_session_dir
from rosclaw.memory.v2.gate import MemoryWriteGate
from rosclaw.memory.v2.index import (
    SQLITE_HARD_MAX_RECORDS,
    EmbeddingIndexManager,
    IndexModelMismatchError,
    memory_embedding_text,
)
from rosclaw.memory.v2.repository import MemoryRepository
from rosclaw.memory.v2.retrieval import MemoryQuery, MemoryRetriever, SafetyRetrievalPolicy
from rosclaw.storage.factory import StorageFactory
from rosclaw.storage.seekdb_native import SeekDBNativeStore
from rosclaw.storage.vector import SQLiteVectorStore, TfidfEmbedder

# Score-semantics disclosure: a vector score only has meaning relative to the
# backend that produced it.  Never present a local TF-IDF cosine as a native
# SeekDB vector score (or vice versa).
_SCORE_SEMANTICS = {
    "tfidf_cosine_local": (
        "vector_score is cosine similarity of a locally-fitted TF-IDF embedding "
        "(SQLite store); corpus-dependent, NOT comparable to native SeekDB scores"
    ),
    "seekdb_native_1_minus_distance": (
        "vector_score is 1 - distance from the SeekDB server-side embedder, "
        "computed natively by the SeekDB engine"
    ),
}


class _QueryTextPassthrough:
    """Embedder stand-in that hands the raw query text to a native backend."""

    def encode(self, text: str) -> str:
        return text


class _NativeSeekDBVectorAdapter:
    """Adapt :meth:`SeekDBNativeStore.similar` to the retriever's
    ``vector_store.search(table, embedding, limit)`` protocol.

    The server embeds the query text itself, so ``embedding`` here is the raw
    query text supplied by :class:`_QueryTextPassthrough`.
    """

    def __init__(self, client: SeekDBNativeStore):
        self._client = client

    def search(self, table: str, embedding: Any, *, limit: int) -> list[dict[str, Any]]:
        hits = self._client.similar(table, str(embedding), limit=limit)
        return [{"record_id": hit.get("id"), "score": float(hit.get("score", 0.0))} for hit in hits]


def _resolve_store_path(args: argparse.Namespace) -> str:
    if getattr(args, "v2_path", None):
        return args.v2_path
    home = resolve_home()
    cfg = load_rosclaw_yaml(home) or {}
    runtime_cfg = cfg.get("runtime", {})
    return runtime_cfg.get("seekdb_path") or str(home / "data" / "memory" / "knowledge.sqlite")


def _open_stack(args: argparse.Namespace, *, with_vector: bool = False):
    """Open the knowledge store via :class:`StorageFactory` and build the
    retrieval stack on top.

    Backend resolution order: ``--backend`` → ``--seekdb-url`` /
    ``ROSCLAW_SEEKDB_URL`` → legacy SQLite path (``--v2-path`` /
    rosclaw.yaml / default).  Returns ``(client, repo, vector, embedder,
    meta)`` where ``meta`` discloses the actual backend and score semantics so
    callers never present TF-IDF scores as native SeekDB vector scores.
    """
    import os

    backend = getattr(args, "backend", None)
    url = getattr(args, "seekdb_url", None) or os.environ.get("ROSCLAW_SEEKDB_URL")
    path = _resolve_store_path(args)
    client = StorageFactory.create_knowledge_store(
        backend=backend or ("sqlite" if not url else None),
        url=url,
        path=path,
    )
    client.connect()
    repo = MemoryRepository(client)

    meta: dict[str, Any] = {
        "backend": type(client).__name__,
        "store": url or path,
    }
    vector = None
    embedder = None
    if with_vector:
        if isinstance(client, SeekDBNativeStore):
            vector = _NativeSeekDBVectorAdapter(client)
            embedder = _QueryTextPassthrough()
            meta["vector_source"] = "seekdb_native"
            meta["score_semantics"] = _SCORE_SEMANTICS["seekdb_native_1_minus_distance"]
            with contextlib.suppress(Exception):
                meta["embedder"] = client.embedding_info("memory_items")
        elif isinstance(client, SQLiteKnowledgeStore):
            vector = SQLiteVectorStore(client)
            embedder = TfidfEmbedder()
            corpus = [
                memory_embedding_text(item) for item in repo.query(limit=SQLITE_HARD_MAX_RECORDS)
            ]
            embedder.fit(corpus)
            meta["vector_source"] = "sqlite_tfidf"
            meta["score_semantics"] = _SCORE_SEMANTICS["tfidf_cosine_local"]
        else:
            meta["vector_source"] = "unavailable"
            meta["score_semantics"] = (
                f"backend {type(client).__name__} has no vector path in the v2 CLI; "
                "retrieval is lexical+metadata only"
            )
    return client, repo, vector, embedder, meta


def _close(client: Any) -> None:
    with contextlib.suppress(Exception):
        client.disconnect()


def _emit(payload: Any, **json_kwargs: Any) -> None:
    """Print a JSON payload and flush stdout immediately.

    The embedded SeekDB engine's teardown path can bypass Python's stdio
    flush at process exit; without an explicit flush, block-buffered output
    (pipe/file redirect) is silently lost — the command exits 0 having
    printed nothing.
    """
    print(json.dumps(payload, **json_kwargs))
    sys.stdout.flush()


def cmd_memory_v2_status(args: argparse.Namespace) -> int:
    client, repo, _, _, meta = _open_stack(args)
    try:
        by_type: dict[str, int] = {}
        for row in client.query("memory_items", limit=10000):
            if row.get("status", "active") == "active":
                memory_type = row.get("memory_type", "?")
                by_type[memory_type] = by_type.get(memory_type, 0) + 1
        if isinstance(client, SeekDBNativeStore):
            index_info: Any = {
                "mode": "seekdb_native_server_side",
                "memory_items": client.embedding_info("memory_items"),
            }
        elif isinstance(client, SQLiteKnowledgeStore):
            index_info = EmbeddingIndexManager(client, SQLiteVectorStore(client)).status()["active"]
        else:
            index_info = {"mode": "unsupported", "backend": type(client).__name__}
        result = {
            "backend": meta["backend"],
            "store": meta["store"],
            "store_path": _resolve_store_path(args),
            "total_active": sum(by_type.values()),
            "by_type": by_type,
            "legacy_experience_graph": client.count("experience_graph"),
            "index": index_info,
        }
    finally:
        _close(client)
    _emit(result, indent=2, ensure_ascii=False)
    return 0


def cmd_memory_v2_query(args: argparse.Namespace) -> int:
    client, repo, vector, embedder, meta = _open_stack(args, with_vector=not args.no_vector)
    try:
        if vector is not None and not isinstance(client, SeekDBNativeStore):
            manager = EmbeddingIndexManager(client, vector)
            if manager.active_index() is not None:
                manager.check_query_embedder(embedder)
        retriever = MemoryRetriever(repo, vector_store=vector, embedder=embedder)
        query = MemoryQuery(
            text=args.query,
            memory_types=getattr(args, "type", None) or [],
            tenant_id=getattr(args, "tenant_id", None),
            project_id=getattr(args, "project_id", None),
            site_id=getattr(args, "site_id", None),
            robot_id=getattr(args, "robot_id", None),
            body_id=getattr(args, "body_id", None),
            task_id=getattr(args, "task_id", None),
            limit=args.limit,
        )
        results = retriever.retrieve(query)
        if getattr(args, "safety_filter", False):
            results = SafetyRetrievalPolicy().filter(results, query)
        output = {
            "retrieval": meta,
            "results": [r.to_dict() for r in results],
        }
    except IndexModelMismatchError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    finally:
        _close(client)
    _emit(output, indent=2, ensure_ascii=False)
    return 0


def cmd_memory_v2_explain(args: argparse.Namespace) -> int:
    if not getattr(args, "memory_id", None):
        print("memory explain --v2 requires --memory-id", file=sys.stderr)
        return 2
    client, repo, vector, embedder, meta = _open_stack(args, with_vector=not args.no_vector)
    try:
        if vector is not None and not isinstance(client, SeekDBNativeStore):
            manager = EmbeddingIndexManager(client, vector)
            if manager.active_index() is not None:
                manager.check_query_embedder(embedder)
        retriever = MemoryRetriever(repo, vector_store=vector, embedder=embedder)
        query = MemoryQuery(
            text=getattr(args, "text", "") or "",
            tenant_id=getattr(args, "tenant_id", None),
            project_id=getattr(args, "project_id", None),
            site_id=getattr(args, "site_id", None),
            robot_id=getattr(args, "robot_id", None),
        )
        output = retriever.explain(args.memory_id, query)
        output["trace"] = repo.trace(args.memory_id)
        # Retrieval-backend disclosure (P0): backend + score semantics so a
        # TF-IDF cosine is never mistaken for a native SeekDB vector score.
        output["retrieval"] = meta
    except IndexModelMismatchError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    finally:
        _close(client)
    _emit(output, indent=2, ensure_ascii=False)
    return 0 if output.get("found", True) else 1


def cmd_memory_v2_verify(args: argparse.Namespace) -> int:
    """Verify every active memory is traceable to evidence."""
    client, repo, _, _, _meta = _open_stack(args)
    try:
        items = repo.query(limit=10000)
        untraceable = [i.memory_id for i in items if not repo.trace(i.memory_id)["traceable"]]
        result = {
            "checked": len(items),
            "untraceable": len(untraceable),
            "untraceable_ids": untraceable[:20],
            "passed": not untraceable,
        }
    finally:
        _close(client)
    _emit(result, indent=2, ensure_ascii=False)
    return 0 if result["passed"] else 1


def cmd_memory_v2_consolidate(args: argparse.Namespace) -> int:
    client, repo, _, _, _meta = _open_stack(args)
    try:
        result = MemoryConsolidator(repo).consolidate(robot_id=getattr(args, "robot_id", None))
        output = vars(result)
    finally:
        _close(client)
    _emit(output, indent=2, ensure_ascii=False)
    return 0


def cmd_memory_v2_forget(args: argparse.Namespace) -> int:
    """Delete a memory and sync the vector index."""
    client, repo, vector, _, _meta = _open_stack(args, with_vector=True)
    try:
        item = repo.get(args.memory_id)
        if item is None:
            _emit({"deleted": False, "reason": "not found"})
            return 1
        client.delete("memory_items", args.memory_id)
        for evidence in repo.evidence_for(args.memory_id):
            client.delete("memory_evidence", evidence.evidence_id)
        if isinstance(client, SeekDBNativeStore):
            # The server-side embedder re-indexes on delete; nudge visibility.
            client.refresh_index("memory_items")
            index_deleted = True
        else:
            index_deleted = EmbeddingIndexManager(client, vector).on_memory_deleted(args.memory_id)
        output = {"deleted": True, "memory_id": args.memory_id, "index_synced": index_deleted}
    finally:
        _close(client)
    _emit(output, indent=2)
    return 0


def cmd_memory_v2_distill(args: argparse.Namespace) -> int:
    """Distill a practice session into Memory 2.0."""
    client, repo, _, _, _meta = _open_stack(args)
    try:
        gate = MemoryWriteGate(repo)
        result = distill_session_dir(args.session_dir, gate=gate, repository=repo)
        output = {
            "practice_id": result.practice_id,
            "candidates": result.candidates,
            "stored": result.stored,
            "merged": result.merged,
            "updated": result.updated,
            "ignored": result.ignored,
            "quarantined": result.quarantined,
        }
    finally:
        _close(client)
    _emit(output, indent=2, ensure_ascii=False)
    return 0


def cmd_memory_v2_index_status(args: argparse.Namespace) -> int:
    client, repo, vector, _, _meta = _open_stack(args, with_vector=True)
    try:
        if isinstance(client, SeekDBNativeStore):
            output = {
                "mode": "seekdb_native_server_side",
                "detail": (
                    "embedding is computed by the SeekDB server-side embedder; "
                    "there is no local TF-IDF index registry"
                ),
                "memory_items": client.embedding_info("memory_items"),
                "count": client.count("memory_items"),
            }
        elif isinstance(client, SQLiteKnowledgeStore):
            output = EmbeddingIndexManager(client, vector).status()
        else:
            output = {"mode": "unsupported", "backend": type(client).__name__}
    finally:
        _close(client)
    _emit(output, indent=2, ensure_ascii=False, default=str)
    return 0


def cmd_memory_v2_index_rebuild(args: argparse.Namespace) -> int:
    client, repo, vector, _, _meta = _open_stack(args, with_vector=True)
    try:
        if isinstance(client, SeekDBNativeStore):
            client.refresh_index("memory_items")
            output = {
                "rebuilt": False,
                "mode": "seekdb_native_server_side",
                "detail": (
                    "server-side embedding is automatic on write; "
                    "issued refresh_index instead of a local TF-IDF rebuild"
                ),
            }
        elif isinstance(client, SQLiteKnowledgeStore):
            embedder = TfidfEmbedder()
            items = repo.query(limit=200000)
            manager = EmbeddingIndexManager(client, vector)
            record = manager.build(items, embedder)
            output = {"rebuilt": True, "index": record}
        else:
            output = {"rebuilt": False, "mode": "unsupported", "backend": type(client).__name__}
    finally:
        _close(client)
    _emit(output, indent=2, ensure_ascii=False, default=str)
    return 0


def cmd_memory_v2_benchmark(args: argparse.Namespace) -> int:
    import subprocess

    module_path = Path(__file__).resolve()
    candidates = [
        module_path.parents[2] / "benchmarks" / "memory" / "run_benchmark.py",
        module_path.parents[4] / "benchmarks" / "memory" / "run_benchmark.py",
    ]
    script = next((candidate for candidate in candidates if candidate.is_file()), None)
    if script is None:
        searched = ", ".join(str(candidate) for candidate in candidates)
        print(f"benchmark harness not found (searched: {searched})", file=sys.stderr)
        return 1
    return subprocess.call([sys.executable, str(script)])


def _add_backend_arguments(p: Any) -> None:
    """Add StorageFactory backend selection args to a v2 parser."""
    p.add_argument(
        "--backend",
        choices=["sqlite", "seekdb_embedded", "seekdb_server", "mysql", "memory"],
        default=None,
        help="Knowledge-store backend (default: sqlite at --v2-path, or derived from --seekdb-url)",
    )
    p.add_argument(
        "--seekdb-url",
        default=None,
        help="Backend DSN, e.g. seekdb://root@127.0.0.1:2881/rosclaw (or ROSCLAW_SEEKDB_URL)",
    )


def register_memory_v2_commands(memory_subparsers: Any) -> None:
    """Register Memory 2.0 subcommands into the existing memory group."""
    p = memory_subparsers.add_parser("verify", help="Verify all memories are traceable (v2)")
    p.add_argument("--v2-path", default=None, help="SQLite knowledge store path")
    _add_backend_arguments(p)
    p.set_defaults(v2_handler=cmd_memory_v2_verify)

    p = memory_subparsers.add_parser("consolidate", help="Run memory consolidation (v2)")
    p.add_argument("--robot-id", default=None)
    p.add_argument("--v2-path", default=None)
    _add_backend_arguments(p)
    p.set_defaults(v2_handler=cmd_memory_v2_consolidate)

    p = memory_subparsers.add_parser("forget", help="Delete a memory and sync the index (v2)")
    p.add_argument("memory_id")
    p.add_argument("--v2-path", default=None)
    _add_backend_arguments(p)
    p.set_defaults(v2_handler=cmd_memory_v2_forget)

    p = memory_subparsers.add_parser("distill", help="Distill a practice session into memory (v2)")
    p.add_argument("session_dir", help="Practice session directory")
    p.add_argument("--v2-path", default=None)
    _add_backend_arguments(p)
    p.set_defaults(v2_handler=cmd_memory_v2_distill)

    p = memory_subparsers.add_parser("index", help="Embedding index operations (v2)")
    index_sub = p.add_subparsers(dest="index_command")
    pi = index_sub.add_parser("status", help="Index registry status")
    pi.add_argument("--v2-path", default=None)
    _add_backend_arguments(pi)
    pi.set_defaults(v2_handler=cmd_memory_v2_index_status)
    pi = index_sub.add_parser("rebuild", help="Rebuild the embedding index")
    pi.add_argument("--v2-path", default=None)
    _add_backend_arguments(pi)
    pi.set_defaults(v2_handler=cmd_memory_v2_index_rebuild)

    p = memory_subparsers.add_parser("benchmark", help="Run the retrieval benchmark (v2)")
    p.set_defaults(v2_handler=cmd_memory_v2_benchmark)


def extend_legacy_memory_parsers(
    status_parser: Any, query_parser: Any, explain_parser: Any
) -> None:
    """Add ``--v2`` routing flags to the legacy memory subcommands."""
    status_parser.add_argument("--v2", action="store_true", help="Use Memory 2.0 status")
    status_parser.add_argument("--v2-path", default=None)
    _add_backend_arguments(status_parser)
    status_parser.set_defaults(v2_handler=cmd_memory_v2_status)

    query_parser.add_argument("--v2", action="store_true", help="Use Memory 2.0 hybrid retrieval")
    query_parser.add_argument("--v2-path", default=None)
    query_parser.add_argument("--type", action="append", help="Memory type filter (v2)")
    query_parser.add_argument("--tenant-id", default=None)
    query_parser.add_argument("--project-id", default=None)
    query_parser.add_argument("--site-id", default=None)
    query_parser.add_argument("--robot-id", default=None)
    query_parser.add_argument("--body-id", default=None)
    query_parser.add_argument("--task-id", default=None)
    query_parser.add_argument("--no-vector", action="store_true")
    query_parser.add_argument("--safety-filter", action="store_true")
    _add_backend_arguments(query_parser)
    query_parser.set_defaults(v2_handler=cmd_memory_v2_query)

    explain_parser.add_argument("--v2", action="store_true", help="Use Memory 2.0 explain")
    explain_parser.add_argument("--v2-path", default=None)
    explain_parser.add_argument("--memory-id", default=None, help="Memory id (v2)")
    explain_parser.add_argument("--text", default="", help="Query text (v2)")
    explain_parser.add_argument("--tenant-id", default=None)
    explain_parser.add_argument("--project-id", default=None)
    explain_parser.add_argument("--site-id", default=None)
    explain_parser.add_argument("--robot-id", default=None)
    explain_parser.add_argument("--no-vector", action="store_true")
    _add_backend_arguments(explain_parser)
    explain_parser.set_defaults(v2_handler=cmd_memory_v2_explain)
