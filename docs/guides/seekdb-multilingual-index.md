# Guide: Versioned Multilingual SeekDB Index（数据库优化v3 §8）

## Why versioned

The 384-dim MiniLM collection is never modified in place. Every multilingual index generation is a NEW physical collection with a registry row, so the versioned-manager pointer can be rolled back without rebuilding (§18).

## Registry

Table `projection_registry` on the native store:

```text
logical_name           memory_items
physical_collection    memory_items__qwen3_06b_1024_v1__ik__g4f6cbe8a21
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
mgr.activate("memory_items", analyzer="ik")            # atomic pointer switch
mgr.rollback("memory_items")                           # back to previous ACTIVE
```

Notes:

- Collections are created with `embedding_function=None` and the profile's dimension; documents are embedded by the PROVIDER, not the collection (query needs an instruction, documents must not get one — §8.1).
- pyseekdb specifics on this rig: the legacy `Configuration(hnsw=..., fulltext_config=...)` form is the working no-embedder shape; the newer `Schema` path still forces the 384-dim default embedder.
- Every build uses a fresh generation suffix. Rebuilding a profile never deletes or mutates the current ACTIVE collection.
- Activation is one upsert of the canonical active pointer after count and dimension verification; rollback swaps it with the previous pointer.

## Analyzer: IK vs ngram

Both were built in a historical real-server run. Simple CJK hit-count probes tied,
which is diagnostic only and cannot choose an analyzer. Select neither for
production until the corrected labeled benchmark is rerun.

## Retrieval path

```text
metadata hard filter (exact body when unambiguous)
→ hybrid BM25 + KNN, BOTH legs filtered, candidate_window=20
→ exact-entity boost/demote (middle≠thumb_rot, left≠right)
→ (optional) Qwen3-Reranker on high-risk queries
→ top-5
```

The hard body filter exists because of a measured failure mode: without it, a 左手 query returned 20/20 right-body rows on the qwen3 index — exact semantics beat more vectors.

This is currently an explicit versioned-manager shadow path. `activate()` changes
the manager's canonical registry pointer, but the general
`rosclaw memory query --v2` command does not yet consume that pointer. Do not
interpret activation as a fleet-wide runtime switch.

## Truth fields (§13)

`rosclaw memory index describe --backend seekdb_server --seekdb-url seekdb://root@127.0.0.1:2881/rosclaw --logical memory_items --profile qwen3_06b_1024_v1` prints: backend, active collection, the ACTIVE model + revision + dimension, requested-provider match state, analyzer, vector source, score semantics (cosine/BM25/RRF — never a single "similarity"), reranker, fallback state, and the full registry. Add `--probe-provider` only when loading the local model for an actual health check is intended.

The bake-off defaults to the dedicated `rosclaw_benchmark` database and uses a
fresh baseline collection name on every run; it never deletes a fixed collection.
