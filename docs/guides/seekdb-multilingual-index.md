# Guide: Versioned Multilingual SeekDB Index（数据库优化v3 §8）

## Why versioned

The 384-dim MiniLM collection is never modified in place. Every multilingual index generation is a NEW physical collection with a registry row, so a bad index is a registry flip away from the previous good one (无损回滚, §18).

## Registry

Table `projection_registry` on the native store:

```text
logical_name           memory_items
physical_collection    memory_items__qwen3_06b_1024_v1__ik
embedding_profile_id   qwen3_06b_1024_v1
model_id / revision    Qwen/Qwen3-Embedding-0.6B @ 97b0c614...
dimension              1024
analyzer               ik | ngram
corpus_hash            sha256 over sorted (id, document)
record_count           N
status                 BUILDING | READY | ACTIVE | OLD | FAILED
created_at / activated_at
```

## Build flow

```python
from rosclaw.embedding.registry import get_provider
from rosclaw.storage.versioned_collections import VersionedCollectionManager

provider = get_provider("qwen3_06b_1024_v1")
mgr = VersionedCollectionManager(store, provider)
mgr.build("memory_items", records, analyzer="ik")      # manual embeddings, batched
mgr.verify("memory_items", analyzer="ik")              # count + dimension
rows = mgr.shadow_query("memory_items", "左手 剪刀 失败", analyzer="ik")
mgr.activate("memory_items", analyzer="ik")            # atomic flip
mgr.rollback("memory_items")                           # back to previous ACTIVE
```

Notes:

- Collections are created with `embedding_function=None` and the profile's dimension; documents are embedded by the PROVIDER, not the collection (query needs an instruction, documents must not get one — §8.1).
- pyseekdb specifics on this rig: the legacy `Configuration(hnsw=..., fulltext_config=...)` form is the working no-embedder shape; the newer `Schema` path still forces the 384-dim default embedder.
- `has_collection`/`delete_collection` are used for idempotent rebuilds.

## Analyzer: IK vs ngram

Both were built on the real corpus (441 memories, server): CJK probes tie 9/9; IK kept ACTIVE. Error-code-heavy corpora should prefer ngram (§8.2); re-decide with the growing corpus benchmark.

## Retrieval path

```text
metadata hard filter (exact body when unambiguous)
→ hybrid BM25 + KNN, BOTH legs filtered, candidate_window=20
→ exact-entity boost/demote (middle≠thumb_rot, left≠right)
→ (optional) Qwen3-Reranker on high-risk queries
→ top-5
```

The hard body filter exists because of a measured failure mode: without it, a 左手 query returned 20/20 right-body rows on the qwen3 index — exact semantics beat more vectors.

## Truth fields (§13)

`rosclaw memory index describe --logical memory_items --profile qwen3_06b_1024_v1` prints: backend, active collection, model + revision + dimension + instruction, analyzer, vector source, score semantics (cosine/BM25/RRF — never a single "similarity"), reranker, fallback state, and the full registry.
