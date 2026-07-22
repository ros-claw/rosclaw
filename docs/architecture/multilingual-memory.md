# Multilingual Embodied Memory Architecture（数据库优化v3）

Status: experimental implementation (PR-MEM-3 / PR-SDB-2 / PR-MEM-4).
RH56 distillation and SeekDB execution have real-system coverage; embedding-model
promotion remains blocked until the corrected benchmark is rerun.

## 0. Design principles (discipline §17)

1. Correctness before model upgrades — retrieval constraints and data semantics first, embeddings second.
2. SQLite is the source of truth; SeekDB is a projection. Rebuilds are always possible.
3. Collections are versioned and never modified in place; rollbacks are registry flips.
4. Models are pinned (revision), profiles are explicit (dimension, instruction, normalization).
5. Query and document encoding paths are separate (Qwen instruction on queries only).
6. Exact entities (joints, hands, gestures, error codes, devices) outrank semantic similarity.
7. Embedding never blocks the robot; every remote path has a stated degradation chain.
8. Memory must carry evidence; correlation is never written as causation.

## 1. Pipeline

```text
Practice raw events (JSONL/MCAP, untouched)
        ↓
Memory V2 Distillation (generic extractors + TaskDistillationAdapter)
        ↓
MemoryWriteGate (evidence required; empty body cognition IGNORE; secrets/CoT stripped)
        ↓
SQLite Memory Repository (source of truth)
        ↓
Outbox / Projection (idempotent, rebuildable)
        ↓
Versioned SeekDB collections (manual embeddings, analyzer per build; shadow path)
        ↓
Retrieval: metadata hard filter → hybrid BM25+vector (both legs filtered)
        → exact-entity constraints → (Qwen3-Reranker, high-risk only)
        → safety policy → HOW / KNOW / Auto
```

## 2. Task distillation adapters

`memory/v2/adapters/` — protocol + registry + `rh56_rps.py`.

The RH56 RPS adapter understands the stress protocol:

- `rps.stress.round.resolved result=invalid` is an IMPLICIT failure even with no explicit failure event;
- failures are emitted per failing gesture (invalid rounds, valid rounds, and between-round gestures);
- linkage priority: `round_id` → time window (gesture events carry no round_id);
- episode quality is a verified-rate distribution with task-declared thresholds (`success ≥0.98`, `partial ≥0.80`), not a blanket SUCCESS;
- body temperature is an observation (`observed_*`, `causal_status: observed_correlation`), never a `thermal_limits` claim;
- joint attribution is filled only when recorded — never invented (`joint_name=None` when absent).

Validated: Golden Path session → 52/52 unverified gestures as failure memories (right rock 22, left left_scissors 27, right scissors 2, right ready 1); 2h session → 379 failure memories; idempotent re-distill delta 0.

## 3. Multilingual documents

`memory/v2/document.py` + `resources/robotics_lexicon.yaml`.

Every failure memory carries parallel sections:

```text
[ZH]   中文自然语言（领域词典翻译，非模型自动翻译）
[EN]   English narrative
[CANONICAL]  robot=RH56 hand=right joint=middle gesture=scissors failure=joint_not_reached round=37 temperature_c=46
[ALIASES]    关节未到位 joint_not_reached position tracking failure ...
```

Exact-entity extraction (query side): joints (中指→middle, 拇指根关节→thumb_rot), hands (左手/左手 hand only — bare "left"/"right" excluded), gestures, failure types, error codes (EIO/-110), devices. Latin matching is `\b`-bounded (`\bthumb\b` never matches inside `thumb_rot`); CJK aliases use substring with longest-entity-wins (拇指根关节 → thumb_rot, not thumb).

## 4. Retrieval constraints

Storage layer (`seekdb_native.py`):

- BOTH hybrid legs (BM25 + KNN) receive the SAME engine-side hard metadata filter;
- server mode uses native RRF; embedded pyseekdb 1.3.0 runs the two filtered native legs separately and applies deterministic RRF client-side because its combined-query SQL generator is malformed;
- `candidate_window` widens the shortlist before RRF fusion (rank_constant=60);
- `query_embedding` switches the KNN leg to caller-supplied vectors (manual multilingual embeddings);
- `order_by` raises `UnsupportedOperationError` (no fake global ordering); `count`/`delete_where` paginate by cursor;
- exception narrowing: only a genuine collection-not-found creates a collection; 1049/auth/permission/network errors surface.

Retrieval layer (`memory/v2/retrieval.py` + `storage/versioned_collections.py`):

- metadata hard filter first (when the query names an unambiguous body — measured: a 左手 query returned 20/20 right-body rows without it);
- exact-entity multipliers post-fusion: matching joint/hand/failure_type ×1.3-1.5, wrong joint ×0.25, wrong hand ×0.3, unattributed (None) neutral;
- score semantics are explicit (cosine / BM25 / RRF are never mixed up as "similarity").

Current integration boundary: `VersionedCollectionManager` provides build, verify,
shadow-query, pointer activation, and rollback. The general
`rosclaw memory query --v2` path does **not** yet resolve that pointer; it continues
to use the logical `memory_items` collection and its built-in embedding path.
Accordingly, `ACTIVE` below means the versioned-manager registry selection, not a
claim that every ROSClaw memory consumer has switched to that model.

## 5. Embedding providers

`src/rosclaw/embedding/` — protocol, pinned profiles, local sentence-transformer provider, SQLite cache (query/document isolation, dimension guard), health, registry (offline-first, `HF_HUB_OFFLINE=1` with pinned snapshots).

Available candidate profiles (none is auto-promoted):

| profile_id | model | dim | note |
|---|---|---|---|
| qwen3_06b_1024_v1 | Qwen/Qwen3-Embedding-0.6B @97b0c614 | 1024 | baseline build |
| qwen3_06b_768_v1 | same | 768 | Matryoshka candidate; promotion pending corrected bake-off |
| qwen3_06b_512_v1 | same | 512 | fallback |

Qwen query instruction is applied on queries only; Matryoshka truncation +
renormalization is used for 768/512. Deployment-specific latency and memory
measurements are evidence, not portable guarantees.

## 6. Versioned collections

`storage/versioned_collections.py` — `projection_registry` (BUILDING/READY/ACTIVE/OLD/FAILED).

```text
create physical collection (embedding_function=None, profile dimension, analyzer)
→ backfill with manual document embeddings
→ verify (record count, dimension)
→ shadow query (exact-entity boosted)
→ benchmark
→ activate (atomic canonical-pointer switch after verification)
→ keep OLD for rollback (registry flip, no rebuild)
```

Physical names embed profile and generation, for example
`memory_items__qwen3_06b_1024_v1__ik__g4f6cbe8a21`. Every build gets a new
physical collection. A single canonical registry pointer selects ACTIVE and
stores the immediately previous generation for rollback; audit status rows are
secondary to that pointer. Collections are never shared across models or
dimensions (§17.5).

Historical lab runs built both `ik` and `ngram`; simple CJK probes did not
distinguish them. That observation is not analyzer-selection evidence. No analyzer
or embedding profile is production-promoted by this PR.

## 7. Reranker

`embedding/reranker.py` — Qwen3-Reranker-0.6B @e61197ed cross-encoder.

High-risk paths only: failure recovery, HOW rule selection, safety-adjacent
history, and hard negatives (middle vs thumb_rot). Candidate window 20 → final
5. The provider uses the pinned model with the ROSClaw task instruction. The
previous reranker numbers were invalidated with the benchmark labels and must
be rerun before this path is enabled by default.

## 8. Degradation chain (§11)

```text
Qwen provider available   → versioned-manager shadow hybrid (manual embeddings)
provider unavailable      → versioned-manager SeekDB BM25 + metadata filter
SeekDB unavailable        → SQLite lexical memory
all remote unavailable    → curated KNOW / verified HOW rules
```

A different model's vector is NEVER used against another model's collection.

## 9. Benchmark evidence status

The original v3 run is not promotion evidence. Review found that its
hard-negative generator selected same-body rows as "opposite-body" negatives,
error-code queries were bound to unrelated source memories, and an empty
English-memory lane silently fell back to CJK rows. Generated real-session
datasets and per-query output are therefore excluded from Git and belong in an
access-controlled evidence store. The corrected generator now fails closed
when a lane lacks valid sources and validates source, relevance, and forbidden
bindings before inference.
