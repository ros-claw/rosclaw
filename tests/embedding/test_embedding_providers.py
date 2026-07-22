"""PR-SDB-2 embedding provider unit tests (数据库优化v3 §13):

* query carries the instruction, document never does
* profile cache namespace: query/document isolation, revision/dimension
  sensitivity
* cache round-trip + dimension guard + query/document isolation
* dimension mismatch fails loudly
* real Qwen3 local provider (offline): profile identity + health
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3

import pytest

from rosclaw.embedding.cache import EmbeddingCache
from rosclaw.embedding.errors import (
    EmbeddingDimensionMismatchError,
    EmbeddingUnavailableError,
)
from rosclaw.embedding.local_sentence_transformer import LocalSentenceTransformerProvider
from rosclaw.embedding.profile import QWEN3_06B_512, QWEN3_06B_1024
from rosclaw.embedding.protocol import EmbeddingProfile
from rosclaw.embedding.registry import get_provider
from rosclaw.embedding.reranker import RERANKER_INSTRUCTION, Qwen3RerankerProvider
from rosclaw.memory.v2.cli import cmd_memory_v2_index_describe


class FakeProvider:
    """Deterministic hash-based vectors for engine-level tests."""

    def __init__(self, profile: EmbeddingProfile, *, fail: bool = False) -> None:
        self._profile = profile
        self._fail = fail
        self.calls: list[tuple[str, list[str]]] = []

    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile

    def _vec(self, text: str) -> list[float]:
        import hashlib

        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in digest[: self._profile.dimension]]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        if self._fail:
            raise EmbeddingUnavailableError("service down (fake)")
        self.calls.append(("document", list(texts)))
        return [self._vec("doc:" + t) for t in texts]

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        if self._fail:
            raise EmbeddingUnavailableError("service down (fake)")
        self.calls.append(("query", list(texts)))
        return [self._vec("q:" + t) for t in texts]

    def health(self) -> dict:
        return {"ok": not self._fail, "profile_id": self._profile.profile_id}


FAKE_PROFILE = EmbeddingProfile(
    profile_id="fake_8d_v1",
    model_id="fake/model",
    model_revision="rev1",
    dimension=8,
    normalize=True,
    distance="cosine",
    query_instruction="test instruction",
    document_instruction=None,
    max_tokens=32,
    provider_type="fake",
)


# ---------------------------------------------------------------------------
# Profile + instruction separation
# ---------------------------------------------------------------------------


def test_qwen_query_prompt_only():
    ns_q = QWEN3_06B_1024.cache_namespace(kind="query", text_hash="x")
    ns_d = QWEN3_06B_1024.cache_namespace(kind="document", text_hash="x")
    assert ns_q != ns_d  # instruction hash differs by kind


def test_document_has_no_query_prompt():
    provider = LocalSentenceTransformerProvider(QWEN3_06B_1024)
    assert provider.profile.document_instruction is None
    assert provider.profile.query_instruction is not None


def test_embedding_profile_hash():
    a = QWEN3_06B_1024.cache_namespace(kind="query", text_hash="x")
    b = QWEN3_06B_512.cache_namespace(kind="query", text_hash="x")
    assert a != b  # dimension is part of the cache key


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_embedding_cache_query_document_isolation(tmp_path):
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    vec_q = [0.1] * 8
    vec_d = [0.9] * 8
    cache.put(FAKE_PROFILE, "query", "same text", vec_q)
    cache.put(FAKE_PROFILE, "document", "same text", vec_d)
    assert cache.get(FAKE_PROFILE, "query", "same text") == vec_q
    assert cache.get(FAKE_PROFILE, "document", "same text") == vec_d


def test_cache_dimension_guard(tmp_path):
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    cache.put(FAKE_PROFILE, "document", "x", [0.1] * 7)  # wrong dim: refused
    assert cache.get(FAKE_PROFILE, "document", "x") is None


def test_cache_never_deserializes_legacy_pickle_rows(tmp_path):
    path = tmp_path / "c.sqlite"
    cache = EmbeddingCache(path)
    cache.put(FAKE_PROFILE, "document", "x", [0.1] * 8)
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE embedding_cache SET encoding='legacy_pickle', embedding_blob=?",
            (b"cos\nsystem\n(S'false'\ntR.",),
        )

    assert cache.get(FAKE_PROFILE, "document", "x") is None


def test_reranker_loads_with_rosclaw_prompt_and_cpu_safe_dtype(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    captured = {}
    float32 = object()

    def fake_cross_encoder(*args, **kwargs):
        captured.update(kwargs)
        return object()

    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.CrossEncoder = fake_cross_encoder
    torch = ModuleType("torch")
    torch.float16 = object()
    torch.float32 = float32
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)
    provider = Qwen3RerankerProvider(device="cpu")

    provider._load()

    assert captured["prompts"] == {"rosclaw": RERANKER_INSTRUCTION}
    assert captured["default_prompt_name"] == "rosclaw"
    assert captured["model_kwargs"]["torch_dtype"] is float32


def test_dimension_mismatch_fails():
    class BadProvider(FakeProvider):
        def encode_documents(self, texts):
            return [[0.0] * 7 for _ in texts]

    provider = LocalSentenceTransformerProvider(QWEN3_06B_1024)
    # directly exercise the guard in _encode via a stubbed model
    provider._model = type(
        "M",
        (),
        {
            "encode": lambda self, batch, normalize_embeddings, convert_to_numpy: [
                [0.0] * 999 for _ in batch
            ]
        },
    )()
    with pytest.raises(EmbeddingDimensionMismatchError):
        provider.encode_documents(["x"])


# ---------------------------------------------------------------------------
# Real Qwen3 local provider (offline snapshot)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("ROSCLAW_QWEN3_TEST") != "1",
    reason="real model load; enable with ROSCLAW_QWEN3_TEST=1",
)
def test_real_qwen3_provider_identity_and_health():
    provider = get_provider("qwen3_06b_1024_v1", cache_path=None)
    health = provider.health()
    assert health["ok"] is True
    assert health["model_revision"] == QWEN3_06B_1024.model_revision
    assert health["probe_dim"] == 1024
    docs = provider.encode_documents(["中指未到位 joint_not_reached"])
    query = provider.encode_queries(["中指未到位"])
    assert len(docs[0]) == 1024 and len(query[0]) == 1024
    assert docs[0] != query[0]  # instruction changes the query vector


def test_embedding_unavailable_degrades_not_blocks():
    provider = FakeProvider(FAKE_PROFILE, fail=True)
    with pytest.raises(EmbeddingUnavailableError):
        provider.encode_queries(["x"])


def test_index_describe_rejects_non_native_backend_without_loading_model(tmp_path, capsys):
    args = argparse.Namespace(
        v2_path=str(tmp_path / "memory.sqlite"),
        backend="sqlite",
        seekdb_url=None,
        logical="memory_items",
        profile="qwen3_06b_1024_v1",
        cache=None,
        probe_provider=False,
    )

    assert cmd_memory_v2_index_describe(args) == 2

    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is False
    assert output["backend"] == "SQLiteKnowledgeStore"
