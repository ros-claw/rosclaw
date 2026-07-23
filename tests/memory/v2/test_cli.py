"""Memory 2.0 CLI regression tests."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest

from rosclaw.memory.seekdb_client import SQLiteKnowledgeStore
from rosclaw.memory.v2.cli import (
    _close,
    _open_stack,
    cmd_memory_v2_benchmark,
    cmd_memory_v2_explain,
    cmd_memory_v2_query,
)
from rosclaw.memory.v2.index import EmbeddingIndexManager
from rosclaw.memory.v2.models import MemoryItem
from rosclaw.memory.v2.repository import MemoryRepository
from rosclaw.memory.v2.retrieval import MemoryQuery, MemoryRetriever
from rosclaw.storage.vector import SQLiteVectorStore, TfidfEmbedder


def test_benchmark_command_resolves_source_harness(monkeypatch) -> None:
    called: list[list[str]] = []

    def fake_call(command: list[str]) -> int:
        called.append(command)
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)
    assert cmd_memory_v2_benchmark(argparse.Namespace()) == 0
    assert called
    assert Path(called[0][1]).is_file()
    assert called[0][1].endswith("benchmarks/memory/run_benchmark.py")


def test_explain_v2_requires_memory_id(capsys) -> None:
    args = argparse.Namespace(memory_id=None)
    assert cmd_memory_v2_explain(args) == 2
    assert "--memory-id" in capsys.readouterr().err


def test_cli_reopens_index_with_identical_tfidf_revision(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite"
    client = SQLiteKnowledgeStore(str(path))
    client.connect()
    repo = MemoryRepository(client)
    item = MemoryItem(
        memory_type="episodic",
        robot_id="r1",
        title="calibration note",
        document="completed successfully",
        tags=["rare_tag_token"],
        evidence_refs=["evt_1"],
    )
    repo.store(item)
    manager = EmbeddingIndexManager(client, SQLiteVectorStore(client))
    manager.build(repo.query(limit=100), TfidfEmbedder())
    client.disconnect()

    reopened, reopened_repo, vector, embedder, meta = _open_stack(
        argparse.Namespace(v2_path=str(path)),
        with_vector=True,
    )
    try:
        assert vector is not None
        assert meta["vector_source"] == "sqlite_tfidf"
        assert "score_semantics" in meta
        EmbeddingIndexManager(reopened, vector).check_query_embedder(embedder)
        results = MemoryRetriever(
            reopened_repo,
            vector_store=vector,
            embedder=embedder,
        ).retrieve(MemoryQuery(text="rare_tag_token"))
        assert results and results[0].memory_id == item.memory_id
        assert results[0].vector_score is not None
    finally:
        _close(reopened)


pyseekdb = pytest.importorskip("pyseekdb", reason="native SeekDB engine not installed")


def _native_stack_args(path: str) -> argparse.Namespace:
    return argparse.Namespace(
        v2_path=path,
        backend="seekdb_embedded",
        seekdb_url=None,
        no_vector=False,
    )


def test_open_stack_native_seekdb_backend(shared_embedded_seekdb_target) -> None:
    """P0-3: --backend seekdb_embedded routes via StorageFactory, not SQLite."""
    client, repo, vector, embedder, meta = _open_stack(
        _native_stack_args(shared_embedded_seekdb_target["path"]), with_vector=True
    )
    try:
        from rosclaw.storage.seekdb_native import SeekDBNativeStore

        assert isinstance(client, SeekDBNativeStore)
        assert meta["backend"] == "SeekDBEmbeddedStore"
        assert meta["vector_source"] == "seekdb_native"
        assert "SeekDB" in meta["score_semantics"]
        # The native adapter must NOT be a SQLite TF-IDF stack.
        assert not isinstance(vector, SQLiteVectorStore)
        assert not isinstance(embedder, TfidfEmbedder)
    finally:
        _close(client)


def _query_args(path: str, query: str) -> argparse.Namespace:
    return argparse.Namespace(
        **{
            **vars(_native_stack_args(path)),
            "query": query,
            "limit": 5,
            "type": None,
            "tenant_id": None,
            "project_id": None,
            "site_id": None,
            "robot_id": None,
            "body_id": None,
            "task_id": None,
            "safety_filter": False,
            "purpose": "human_search",
            "reranker": False,
        }
    )


def _build_active_collection(path: str, records: list[dict]) -> str:
    """Build + activate a versioned ACTIVE collection with the fake profile."""
    from rosclaw.storage.versioned_collections import VersionedCollectionManager
    from tests.embedding.test_embedding_providers import FAKE_PROFILE, FakeProvider

    client, _, _, _, _ = _open_stack(_native_stack_args(path), with_vector=False)
    try:
        mgr = VersionedCollectionManager(client, FakeProvider(FAKE_PROFILE))
        mgr.build("memory_items", records, analyzer="ik")
        activated = mgr.activate("memory_items", analyzer="ik")
        return str(activated["physical_collection"])
    finally:
        _close(client)


def test_query_native_without_active_declares_fallback(
    shared_embedded_seekdb_target, capsys
) -> None:
    """PR-MEM-5 (v4 §3.3): a missing ACTIVE pointer must NOT silently fall
    back to the logical collection's built-in embedder — the query declares
    the fallback chain and its reason."""
    path = shared_embedded_seekdb_target["path"]
    client, repo, _, _, _ = _open_stack(_native_stack_args(path), with_vector=False)
    client.delete_where("memory_items", {})
    repo.store(
        MemoryItem(
            memory_type="episodic",
            robot_id="r1",
            title="native calibration note",
            document="native seekdb retrieval path completed successfully",
            evidence_refs=["evt_native_1"],
        )
    )
    client.refresh_index("memory_items")
    _close(client)

    try:
        assert cmd_memory_v2_query(_query_args(path, "native calibration")) == 0
    finally:
        cleanup, _, _, _, _ = _open_stack(_native_stack_args(path), with_vector=False)
        cleanup.delete_where("memory_items", {})
        _close(cleanup)
    output = json.loads(capsys.readouterr().out)
    assert output["retrieval"]["backend"] == "SeekDBEmbeddedStore"
    # No ACTIVE pointer and (in this environment) no sqlite fallback file:
    # the response abstains with the declared reason.
    assert output["retrieval"]["fallback"] is True
    assert "no_active_pointer" in (output["retrieval"]["fallback_reason"] or "") or (
        output["retrieval"]["retrieval_mode"] == "sqlite_memory_v2_lexical"
    )
    # The built-in-embedder logical collection never served this query.
    assert output["retrieval"]["retrieval_mode"] != "seekdb_native"


def test_query_native_active_serves_physical_collection(
    shared_embedded_seekdb_target, capsys
) -> None:
    """PR-MEM-5: with an ACTIVE pointer, query --v2 serves the ACTIVE
    physical collection (BM25-only here: the fake profile has no production
    provider, and the fallback is declared, never another model)."""
    path = shared_embedded_seekdb_target["path"]
    record = {
        "id": "mem_active_1",
        "memory_type": "episodic",
        "robot_id": "r1",
        "title": "active native calibration note",
        "document": "active physical collection retrieval path",
        "status": "active",
        "evidence_refs": ["evt_native_1"],
    }
    physical = _build_active_collection(path, [record])

    try:
        assert cmd_memory_v2_query(_query_args(path, "active native calibration")) == 0
    finally:
        cleanup, _, _, _, _ = _open_stack(_native_stack_args(path), with_vector=False)
        cleanup.delete_where("memory_items", {})
        cleanup.delete_where("projection_registry", {})
        for name in cleanup.list_collections():
            if name.startswith("memory_items__"):
                cleanup._client.delete_collection(name)
        _close(cleanup)
    output = json.loads(capsys.readouterr().out)
    assert output["retrieval"]["backend"] == "SeekDBEmbeddedStore"
    assert output["retrieval"]["retrieval_mode"] == "active_bm25_metadata"
    assert output["retrieval"]["physical_collection"] == physical
    assert output["retrieval"]["fallback"] is True
    assert output["retrieval"]["fallback_reason"].startswith("embedding_provider_unavailable")
    assert output["results"], "expected the ACTIVE collection to serve results"
    assert output["results"][0]["memory_id"] == "mem_active_1"
    assert output["results"][0]["physical_collection"] == physical


def test_explain_native_includes_retrieval_block(shared_embedded_seekdb_target, capsys) -> None:
    """PR-MEM-5: explain --v2 reports the ACTIVE truth (collection, mode,
    score semantics) instead of the built-in embedder's."""
    path = shared_embedded_seekdb_target["path"]
    record = {
        "id": "mem_explain_1",
        "memory_type": "episodic",
        "robot_id": "r1",
        "title": "explain target",
        "document": "explain retrieval disclosure check",
        "status": "active",
        "evidence_refs": ["evt_explain_1"],
    }
    physical = _build_active_collection(path, [record])

    args = argparse.Namespace(
        **{
            **vars(_native_stack_args(path)),
            "memory_id": "mem_explain_1",
            "text": "explain target",
            "tenant_id": None,
            "project_id": None,
            "site_id": None,
            "robot_id": None,
            "purpose": "human_search",
            "reranker": False,
        }
    )
    try:
        assert cmd_memory_v2_explain(args) == 0
    finally:
        cleanup, _, _, _, _ = _open_stack(_native_stack_args(path), with_vector=False)
        cleanup.delete_where("memory_items", {})
        cleanup.delete_where("projection_registry", {})
        for name in cleanup.list_collections():
            if name.startswith("memory_items__"):
                cleanup._client.delete_collection(name)
        _close(cleanup)
    output = json.loads(capsys.readouterr().out)
    assert output["retrieval"]["physical_collection"] == physical
    assert output["retrieval"]["retrieval_mode"] == "active_bm25_metadata"
    assert "score_semantics" in output["retrieval"]
    assert output["memory_id"] == "mem_explain_1"
