"""SQLite embedding cache (数据库优化v3 §7.4).

Cache key material: model_id + revision + dimension + kind
(query|document) + instruction hash + text hash — query and document
caches can NEVER mix, and a model/dimension change naturally invalidates
old entries (a revision change is a different key, §17.4).
"""

from __future__ import annotations

import os
import sqlite3
import struct
import threading
import time
from pathlib import Path

from .protocol import EmbeddingProfile

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embedding_cache (
    cache_key TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    text_sha256 TEXT NOT NULL,
    text_type TEXT NOT NULL,
    embedding_blob BLOB NOT NULL,
    encoding TEXT NOT NULL DEFAULT 'f64le',
    dimension INTEGER NOT NULL,
    created_at REAL NOT NULL,
    last_accessed_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embedding_cache_profile ON embedding_cache(profile_id);
"""


class EmbeddingCache:
    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._lock = threading.RLock()
        parent = Path(self._path).parent
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(embedding_cache)")}
            if "encoding" not in columns:
                conn.execute(
                    "ALTER TABLE embedding_cache ADD COLUMN encoding TEXT NOT NULL "
                    "DEFAULT 'legacy_pickle'"
                )
        if self._path != ":memory:":
            os.chmod(self._path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    @staticmethod
    def _text_hash(text: str) -> str:
        import hashlib

        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, profile: EmbeddingProfile, kind: str, text: str) -> list[float] | None:
        key = profile.cache_namespace(kind=kind, text_hash=self._text_hash(text))
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT embedding_blob, dimension, encoding FROM embedding_cache WHERE cache_key=?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            blob, dim, encoding = row
            if dim != profile.dimension or encoding != "f64le":
                return None  # defensive: key already encodes dimension
            if len(blob) != dim * 8:
                return None
            conn.execute(
                "UPDATE embedding_cache SET last_accessed_at=? WHERE cache_key=?",
                (time.time(), key),
            )
        return list(struct.unpack(f"<{dim}d", blob))

    def put(self, profile: EmbeddingProfile, kind: str, text: str, vector: list[float]) -> None:
        if len(vector) != profile.dimension:
            return  # never cache a mismatched vector (dimension guard)
        key = profile.cache_namespace(kind=kind, text_hash=self._text_hash(text))
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO embedding_cache"
                " (cache_key, profile_id, text_sha256, text_type, embedding_blob, encoding,"
                "  dimension, created_at, last_accessed_at)"
                " VALUES (?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(cache_key) DO UPDATE SET"
                " embedding_blob=excluded.embedding_blob,"
                " encoding=excluded.encoding,"
                " last_accessed_at=excluded.last_accessed_at",
                (
                    key,
                    profile.profile_id,
                    self._text_hash(text),
                    kind,
                    struct.pack(f"<{len(vector)}d", *vector),
                    "f64le",
                    profile.dimension,
                    now,
                    now,
                ),
            )

    def count(self, profile_id: str | None = None) -> int:
        with self._lock, self._connect() as conn:
            if profile_id:
                (n,) = conn.execute(
                    "SELECT COUNT(*) FROM embedding_cache WHERE profile_id=?",
                    (profile_id,),
                ).fetchone()
            else:
                (n,) = conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()
        return int(n)


class CachedEmbeddingProvider:
    """Read-through cache wrapper around another provider."""

    def __init__(self, inner, cache: EmbeddingCache) -> None:
        self._inner = inner
        self._cache = cache

    @property
    def profile(self):
        return self._inner.profile

    def _encode(self, texts: list[str], *, kind: str, encode_fn) -> list[list[float]]:
        out: list[list[float] | None] = [None] * len(texts)
        missing: list[tuple[int, str]] = []
        for i, text in enumerate(texts):
            hit = self._cache.get(self.profile, kind, text)
            if hit is None:
                missing.append((i, text))
            else:
                out[i] = hit
        if missing:
            vectors = encode_fn([t for _, t in missing])
            for (i, text), vector in zip(missing, vectors, strict=True):
                self._cache.put(self.profile, kind, text, vector)
                out[i] = vector
        return [v for v in out if v is not None]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, kind="document", encode_fn=self._inner.encode_documents)

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, kind="query", encode_fn=self._inner.encode_queries)

    def health(self) -> dict:
        status = self._inner.health()
        status["cache_entries"] = self._cache.count(self.profile.profile_id)
        return status
