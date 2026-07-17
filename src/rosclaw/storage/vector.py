"""Optional vector and hybrid retrieval extensions for ROSClaw storage.

This module provides lightweight, dependency-minimal semantic search:

- :class:`TfidfEmbedder` implements a pure-Python/Numpy TF-IDF embedder so the
  vector path works offline without ``scikit-learn``.
- :class:`SentenceTransformerEmbedder` is an optional backend for higher-quality
  embeddings when ``sentence-transformers`` is installed.
- :class:`SQLiteVectorStore` stores embeddings in a SQLite table and performs
  cosine search in memory.  If ``sqlite-vec`` is installed it can be enabled for
  native vector indices, but the fallback requires no C extensions.

The intent is *optional augmentation* of :class:`SQLiteKnowledgeStore`.  If
vector support is disabled or dependencies are missing, callers fall back to the
existing keyword/BM25 paths.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rosclaw.memory.seekdb_client import SQLiteKnowledgeStore

logger = logging.getLogger("rosclaw.storage.vector")

# SQLite identifiers we interpolate into SQL must be simple names.
_VALID_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(name: str, kind: str = "identifier") -> None:
    if not isinstance(name, str) or not _VALID_IDENTIFIER.fullmatch(name):
        raise ValueError(f"Invalid SQL {kind}: {name!r}")


def _tokenize(text: str) -> list[str]:
    """Simple lower-case word tokenization."""
    return re.findall(r"[a-z0-9]+", text.lower())


class Embedder(ABC):
    """Abstract text embedder."""

    @property
    @abstractmethod
    def dim(self) -> int | None: ...

    @abstractmethod
    def encode(self, text: str) -> list[float]: ...

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.encode(t) for t in texts]


class TfidfEmbedder(Embedder):
    """Lightweight TF-IDF embedder backed by NumPy.

    The vocabulary is built from the first ``fit()`` call.  If ``encode()`` is
    called before fitting, it auto-fits on the provided text so it always
    returns a vector.
    """

    def __init__(self, max_features: int = 10_000) -> None:
        self._max_features = max_features
        self._vocab: dict[str, int] | None = None
        self._idf: np.ndarray | None = None

    @property
    def dim(self) -> int | None:
        return len(self._vocab) if self._vocab else None

    def fit(self, corpus: list[str]) -> TfidfEmbedder:
        """Build vocabulary and IDF weights from *corpus*."""
        if not corpus:
            self._vocab = {}
            self._idf = np.zeros(0)
            return self

        doc_freq: dict[str, int] = {}
        for text in corpus:
            seen = set(_tokenize(text))
            for tok in seen:
                doc_freq[tok] = doc_freq.get(tok, 0) + 1

        # Keep most frequent terms up to max_features for stable ordering.
        sorted_terms = sorted(doc_freq.items(), key=lambda kv: (-kv[1], kv[0]))[
            : self._max_features
        ]
        self._vocab = {term: idx for idx, (term, _) in enumerate(sorted_terms)}
        n_docs = len(corpus)
        self._idf = np.array(
            [math.log((1 + n_docs) / (1 + doc_freq[term])) + 1.0 for term, _ in sorted_terms],
            dtype=np.float32,
        )
        return self

    def encode(self, text: str) -> list[float]:
        if self._vocab is None:
            self.fit([text])
        assert self._vocab is not None and self._idf is not None
        tokens = _tokenize(text)
        if not tokens:
            return np.zeros(len(self._vocab), dtype=np.float32).tolist()
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1
        vec = np.zeros(len(self._vocab), dtype=np.float32)
        for tok, count in tf.items():
            idx = self._vocab.get(tok)
            if idx is not None:
                vec[idx] = float(count) * self._idf[idx]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


class SentenceTransformerEmbedder(Embedder):
    """Optional sentence-transformers embedder."""

    def __init__(self, model_name: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedder. "
                "Install it with: pip install rosclaw[vector]"
            ) from exc

        model_name = model_name or __import__("os").environ.get(
            "ROSCLAW_EMBEDDER_MODEL", "all-MiniLM-L6-v2"
        )
        self._model = SentenceTransformer(model_name)

    @property
    def dim(self) -> int | None:
        return self._model.get_sentence_embedding_dimension()

    def encode(self, text: str) -> list[float]:
        return self._model.encode(text, convert_to_numpy=True).tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, convert_to_numpy=True).tolist()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two dense vectors."""
    av = np.array(a, dtype=np.float32)
    bv = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    if denom == 0:
        return 0.0
    return float(np.dot(av, bv) / denom)


def _rrf_fusion(*rankings: list[tuple[str, float]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion across multiple result lists.

    Each ranking is a list of ``(record_id, score)`` ordered by relevance.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (record_id, _score) in enumerate(ranking, start=1):
            scores[record_id] = scores.get(record_id, 0.0) + 1.0 / (k + rank)
    # Sort by descending RRF score.
    return sorted(scores.items(), key=lambda kv: -kv[1])


class VectorStore(ABC):
    """Abstract vector index."""

    @abstractmethod
    def upsert(
        self,
        table: str,
        record_id: str,
        text: str,
        embedding: list[float] | None = None,
    ) -> None: ...

    @abstractmethod
    def search(
        self,
        table: str,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    def hybrid_search(
        self,
        table: str,
        query_text: str,
        embedder: Embedder,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]: ...


class SQLiteVectorStore(VectorStore):
    """Vector store backed by a SQLite knowledge store connection.

    Embeddings are stored in a side table ``vec_<table>`` so existing tables are
    not modified.  Search scans the side table and ranks by cosine similarity.
    """

    def __init__(self, store: SQLiteKnowledgeStore) -> None:
        self._store = store

    def _table_name(self, table: str) -> str:
        _validate_identifier(table, "table name")
        return f"vec_{table}"

    def _filter_record_ids(self, table: str, filters: dict[str, Any]) -> set[str] | None:
        """Return main-table record ids matching *filters*, or None if empty."""
        if not filters:
            return None
        _validate_identifier(table, "table name")
        cursor = self._store._connection.execute(f"PRAGMA table_info({table})")
        columns = {row["name"] for row in cursor.fetchall()}
        if "id" not in columns:
            return None

        conditions: list[str] = []
        params: list[Any] = []
        for key, value in filters.items():
            _validate_identifier(key, "column name")
            if key not in columns:
                raise ValueError(f"Filter column {key!r} not found in table {table!r}")
            if isinstance(value, (list, tuple, set)):
                value = list(value)
                if not value:
                    continue
                placeholders = ", ".join("?" for _ in value)
                conditions.append(f"{key} IN ({placeholders})")
                params.extend(value)
            else:
                conditions.append(f"{key} = ?")
                params.append(value)
        if not conditions:
            return None

        sql = f"SELECT id FROM {table} WHERE " + " AND ".join(conditions)
        with self._store._lock:
            rows = self._store._connection.execute(sql, params).fetchall()
        return {row[0] for row in rows}

    def _ensure_table(self, table: str) -> None:
        sql = f"""
        CREATE TABLE IF NOT EXISTS {self._table_name(table)} (
            record_id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            updated_at REAL NOT NULL
        )
        """
        with self._store._lock:
            self._store._connection.execute(sql)
            self._store._connection.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self._table_name(table)}_updated "
                f"ON {self._table_name(table)}(updated_at)"
            )
            self._store._connection.commit()

    def upsert(
        self,
        table: str,
        record_id: str,
        text: str,
        embedding: list[float] | None = None,
    ) -> None:
        self._ensure_table(table)
        with self._store._lock:
            self._store._connection.execute(
                f"""
                INSERT INTO {self._table_name(table)} (record_id, text, embedding_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(record_id) DO UPDATE SET
                    text=excluded.text,
                    embedding_json=excluded.embedding_json,
                    updated_at=excluded.updated_at
                """,
                (record_id, text, json.dumps(embedding or []), time.time()),
            )
            self._store._connection.commit()

    def upsert_many(
        self,
        table: str,
        records: list[tuple[str, str, list[float] | None]],
    ) -> None:
        """Batch upsert embeddings for a single table in one transaction."""
        self._ensure_table(table)
        if not records:
            return
        now = time.time()
        params = [
            (record_id, text, json.dumps(embedding or []), now)
            for record_id, text, embedding in records
        ]
        with self._store._lock:
            self._store._connection.executemany(
                f"""
                INSERT INTO {self._table_name(table)} (record_id, text, embedding_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(record_id) DO UPDATE SET
                    text=excluded.text,
                    embedding_json=excluded.embedding_json,
                    updated_at=excluded.updated_at
                """,
                params,
            )
            self._store._connection.commit()

    def delete(self, table: str, record_id: str) -> bool:
        """Remove *record_id* from the vector index (delete sync, §6.6)."""
        self._ensure_table(table)
        with self._store._lock:
            cursor = self._store._connection.execute(
                f"DELETE FROM {self._table_name(table)} WHERE record_id = ?",
                (record_id,),
            )
            self._store._connection.commit()
        return cursor.rowcount > 0

    def count(self, table: str) -> int:
        """Number of indexed records for *table*."""
        self._ensure_table(table)
        with self._store._lock:
            row = self._store._connection.execute(
                f"SELECT COUNT(*) FROM {self._table_name(table)}"
            ).fetchone()
        return int(row[0]) if row else 0

    def _all_rows(self, table: str) -> list[sqlite3.Row]:
        with self._store._lock:
            return self._store._connection.execute(
                f"SELECT record_id, text, embedding_json FROM {self._table_name(table)}"
            ).fetchall()

    def search(
        self,
        table: str,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        limit: int = 10,
        _allowed_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_table(table)
        allowed_ids = _allowed_ids
        if allowed_ids is None and filters:
            allowed_ids = self._filter_record_ids(table, filters)
        rows = self._all_rows(table)
        scored = []
        for row in rows:
            if allowed_ids is not None and row["record_id"] not in allowed_ids:
                continue
            emb = json.loads(row["embedding_json"])
            if not emb:
                continue
            score = _cosine_similarity(query_embedding, emb)
            scored.append((row["record_id"], row["text"], score))
        scored.sort(key=lambda x: -x[2])
        return [
            {"id": record_id, "text": text, "score": score}
            for record_id, text, score in scored[:limit]
        ]

    def hybrid_search(
        self,
        table: str,
        query_text: str,
        embedder: Embedder,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        self._ensure_table(table)
        query_embedding = embedder.encode(query_text)
        allowed_ids = self._filter_record_ids(table, filters) if filters else None
        vector_results = self.search(
            table, query_embedding, limit=limit * 4, _allowed_ids=allowed_ids
        )
        keyword_results = self._keyword_search(
            table, query_text, limit=limit * 4, _allowed_ids=allowed_ids
        )

        vector_ranking = [(r["id"], r["score"]) for r in vector_results]
        keyword_ranking = [(r["id"], r["score"]) for r in keyword_results]
        fused = _rrf_fusion(vector_ranking, keyword_ranking)

        # Build lookup of texts from both result sets.
        texts = {r["id"]: r["text"] for r in vector_results + keyword_results}
        return [
            {"id": record_id, "text": texts.get(record_id, ""), "score": round(score, 6)}
            for record_id, score in fused[:limit]
        ]

    def _keyword_search(
        self,
        table: str,
        query_text: str,
        limit: int = 10,
        _allowed_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        query_tokens = set(_tokenize(query_text))
        if not query_tokens:
            return []
        rows = self._all_rows(table)
        scored = []
        for row in rows:
            if _allowed_ids is not None and row["record_id"] not in _allowed_ids:
                continue
            tokens = set(_tokenize(row["text"]))
            overlap = len(query_tokens & tokens)
            if overlap:
                # Jaccard-ish score normalized by query length.
                score = overlap / len(query_tokens)
                scored.append((row["record_id"], row["text"], score))
        scored.sort(key=lambda x: -x[2])
        return [
            {"id": record_id, "text": text, "score": score}
            for record_id, text, score in scored[:limit]
        ]
