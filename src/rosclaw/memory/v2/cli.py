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
from rosclaw.memory.v2.index import EmbeddingIndexManager
from rosclaw.memory.v2.repository import MemoryRepository
from rosclaw.memory.v2.retrieval import MemoryQuery, MemoryRetriever, SafetyRetrievalPolicy
from rosclaw.storage.vector import SQLiteVectorStore, TfidfEmbedder


def _resolve_store_path(args: argparse.Namespace) -> str:
    if getattr(args, "v2_path", None):
        return args.v2_path
    home = resolve_home()
    cfg = load_rosclaw_yaml(home) or {}
    runtime_cfg = cfg.get("runtime", {})
    return runtime_cfg.get("seekdb_path") or str(home / "data" / "memory" / "knowledge.sqlite")


def _open_stack(args: argparse.Namespace, *, with_vector: bool = False):
    path = _resolve_store_path(args)
    client = SQLiteKnowledgeStore(path)
    client.connect()
    repo = MemoryRepository(client)
    vector = SQLiteVectorStore(client) if with_vector else None
    embedder = None
    if with_vector:
        embedder = TfidfEmbedder()
        corpus = [f"{i.title}\n{i.document}" for i in repo.query(limit=1000)]
        if corpus:
            embedder.fit(corpus)
    return client, repo, vector, embedder


def _close(client: Any) -> None:
    with contextlib.suppress(Exception):
        client.disconnect()


def cmd_memory_v2_status(args: argparse.Namespace) -> int:
    client, repo, _, _ = _open_stack(args)
    try:
        by_type: dict[str, int] = {}
        for row in client.query("memory_items", limit=10000):
            if row.get("status", "active") == "active":
                memory_type = row.get("memory_type", "?")
                by_type[memory_type] = by_type.get(memory_type, 0) + 1
        result = {
            "store_path": _resolve_store_path(args),
            "total_active": sum(by_type.values()),
            "by_type": by_type,
            "legacy_experience_graph": client.count("experience_graph"),
            "index": EmbeddingIndexManager(client, SQLiteVectorStore(client)).status()["active"],
        }
    finally:
        _close(client)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_memory_v2_query(args: argparse.Namespace) -> int:
    client, repo, vector, embedder = _open_stack(args, with_vector=not args.no_vector)
    try:
        retriever = MemoryRetriever(repo, vector_store=vector, embedder=embedder)
        query = MemoryQuery(
            text=args.query,
            memory_types=getattr(args, "type", None) or [],
            robot_id=getattr(args, "robot_id", None),
            body_id=getattr(args, "body_id", None),
            task_id=getattr(args, "task_id", None),
            limit=args.limit,
        )
        results = retriever.retrieve(query)
        if getattr(args, "safety_filter", False):
            results = SafetyRetrievalPolicy().filter(results, query)
        output = [r.to_dict() for r in results]
    finally:
        _close(client)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


def cmd_memory_v2_explain(args: argparse.Namespace) -> int:
    client, repo, vector, embedder = _open_stack(args, with_vector=not args.no_vector)
    try:
        retriever = MemoryRetriever(repo, vector_store=vector, embedder=embedder)
        query = MemoryQuery(
            text=getattr(args, "text", "") or "",
            robot_id=getattr(args, "robot_id", None),
        )
        output = retriever.explain(args.memory_id, query)
        output["trace"] = repo.trace(args.memory_id)
    finally:
        _close(client)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0 if output.get("found", True) else 1


def cmd_memory_v2_verify(args: argparse.Namespace) -> int:
    """Verify every active memory is traceable to evidence."""
    client, repo, _, _ = _open_stack(args)
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
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 1


def cmd_memory_v2_consolidate(args: argparse.Namespace) -> int:
    client, repo, _, _ = _open_stack(args)
    try:
        result = MemoryConsolidator(repo).consolidate(robot_id=getattr(args, "robot_id", None))
        output = vars(result)
    finally:
        _close(client)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


def cmd_memory_v2_forget(args: argparse.Namespace) -> int:
    """Delete a memory and sync the vector index."""
    client, repo, vector, _ = _open_stack(args, with_vector=True)
    try:
        item = repo.get(args.memory_id)
        if item is None:
            print(json.dumps({"deleted": False, "reason": "not found"}))
            return 1
        client.delete("memory_items", args.memory_id)
        for evidence in repo.evidence_for(args.memory_id):
            client.delete("memory_evidence", evidence.evidence_id)
        index_deleted = EmbeddingIndexManager(client, vector).on_memory_deleted(args.memory_id)
        output = {"deleted": True, "memory_id": args.memory_id, "index_synced": index_deleted}
    finally:
        _close(client)
    print(json.dumps(output, indent=2))
    return 0


def cmd_memory_v2_distill(args: argparse.Namespace) -> int:
    """Distill a practice session into Memory 2.0."""
    client, repo, _, _ = _open_stack(args)
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
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


def cmd_memory_v2_index_status(args: argparse.Namespace) -> int:
    client, repo, vector, _ = _open_stack(args, with_vector=True)
    try:
        output = EmbeddingIndexManager(client, vector).status()
    finally:
        _close(client)
    print(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    return 0


def cmd_memory_v2_index_rebuild(args: argparse.Namespace) -> int:
    client, repo, vector, _ = _open_stack(args, with_vector=True)
    try:
        embedder = TfidfEmbedder()
        items = repo.query(limit=200000)
        embedder.fit([f"{i.title}\n{i.document}" for i in items])
        manager = EmbeddingIndexManager(client, vector)
        record = manager.build(items, embedder)
        output = {"rebuilt": True, "index": record}
    finally:
        _close(client)
    print(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    return 0


def cmd_memory_v2_benchmark(args: argparse.Namespace) -> int:
    import subprocess

    script = Path(__file__).resolve().parents[3] / "benchmarks" / "memory" / "run_benchmark.py"
    if not script.exists():
        print(f"benchmark harness not found at {script}", file=sys.stderr)
        return 1
    return subprocess.call([sys.executable, str(script)])


def register_memory_v2_commands(memory_subparsers: Any) -> None:
    """Register Memory 2.0 subcommands into the existing memory group."""
    p = memory_subparsers.add_parser("verify", help="Verify all memories are traceable (v2)")
    p.add_argument("--v2-path", default=None, help="SQLite knowledge store path")
    p.set_defaults(v2_handler=cmd_memory_v2_verify)

    p = memory_subparsers.add_parser("consolidate", help="Run memory consolidation (v2)")
    p.add_argument("--robot-id", default=None)
    p.add_argument("--v2-path", default=None)
    p.set_defaults(v2_handler=cmd_memory_v2_consolidate)

    p = memory_subparsers.add_parser("forget", help="Delete a memory and sync the index (v2)")
    p.add_argument("memory_id")
    p.add_argument("--v2-path", default=None)
    p.set_defaults(v2_handler=cmd_memory_v2_forget)

    p = memory_subparsers.add_parser("distill", help="Distill a practice session into memory (v2)")
    p.add_argument("session_dir", help="Practice session directory")
    p.add_argument("--v2-path", default=None)
    p.set_defaults(v2_handler=cmd_memory_v2_distill)

    p = memory_subparsers.add_parser("index", help="Embedding index operations (v2)")
    index_sub = p.add_subparsers(dest="index_command")
    pi = index_sub.add_parser("status", help="Index registry status")
    pi.add_argument("--v2-path", default=None)
    pi.set_defaults(v2_handler=cmd_memory_v2_index_status)
    pi = index_sub.add_parser("rebuild", help="Rebuild the embedding index")
    pi.add_argument("--v2-path", default=None)
    pi.set_defaults(v2_handler=cmd_memory_v2_index_rebuild)

    p = memory_subparsers.add_parser("benchmark", help="Run the retrieval benchmark (v2)")
    p.set_defaults(v2_handler=cmd_memory_v2_benchmark)


def extend_legacy_memory_parsers(
    status_parser: Any, query_parser: Any, explain_parser: Any
) -> None:
    """Add ``--v2`` routing flags to the legacy memory subcommands."""
    status_parser.add_argument("--v2", action="store_true", help="Use Memory 2.0 status")
    status_parser.add_argument("--v2-path", default=None)
    status_parser.set_defaults(v2_handler=cmd_memory_v2_status)

    query_parser.add_argument("--v2", action="store_true", help="Use Memory 2.0 hybrid retrieval")
    query_parser.add_argument("--v2-path", default=None)
    query_parser.add_argument("--type", action="append", help="Memory type filter (v2)")
    query_parser.add_argument("--robot-id", default=None)
    query_parser.add_argument("--body-id", default=None)
    query_parser.add_argument("--task-id", default=None)
    query_parser.add_argument("--no-vector", action="store_true")
    query_parser.add_argument("--safety-filter", action="store_true")
    query_parser.set_defaults(v2_handler=cmd_memory_v2_query)

    explain_parser.add_argument("--v2", action="store_true", help="Use Memory 2.0 explain")
    explain_parser.add_argument("--v2-path", default=None)
    explain_parser.add_argument("--memory-id", default=None, help="Memory id (v2)")
    explain_parser.add_argument("--text", default="", help="Query text (v2)")
    explain_parser.add_argument("--robot-id", default=None)
    explain_parser.add_argument("--no-vector", action="store_true")
    explain_parser.set_defaults(v2_handler=cmd_memory_v2_explain)
