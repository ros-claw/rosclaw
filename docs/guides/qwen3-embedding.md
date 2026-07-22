# Guide: Qwen3 Embedding on ROSClaw（数据库优化v3 §7）

## Model and profile

Production profile: **`qwen3_06b_768_v1`**

```yaml
model: Qwen/Qwen3-Embedding-0.6B
revision: 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3   # pinned, §17.4
dimension: 768            # Matryoshka truncation + renormalization
normalize: true
distance: cosine
query_instruction: English task instruction (query side ONLY)
provider: local_sentence_transformer (Jetson GPU, FP16)
```

Selection evidence (`reports/embedding_bakeoff/v3_run1/`): R@1 0.970 vs 1024-dim 0.940 and MiniLM 0.910; error-code queries 0.8 vs 0.4; nDCG 0.655 vs 0.448; joint confusion 0; query p50 14 ms. The 768 profile satisfies the §1.2 downsize rule (no Recall@1 loss vs 1024, no confusion increase).

## Query instruction (§7.2)

Qwen's model card recommends an English instruction on the **query** side only. ROSClaw:

```text
Given a robot embodied-memory query, retrieve the most relevant
evidence-backed past experience, failure, intervention, body-state
pattern, or skill. Preserve robot, body, joint, gesture, task,
error-code, and temporal specificity.
```

Queries are encoded as `Instruct: <instruction>\nQuery: <text>`; documents are encoded bare. The provider enforces this split (§17.6).

## Offline / network

HuggingFace direct is blocked on this rig; use the mirror and offline mode:

```bash
export HF_ENDPOINT=https://hf-mirror.com   # snapshot download
export HF_HUB_OFFLINE=1                    # runtime, pinned snapshot only
```

## Usage

```python
from rosclaw.embedding.registry import get_provider

provider = get_provider("qwen3_06b_768_v1")            # cached, offline-first
docs = provider.encode_documents(["RH56 右手中指剪刀未到位"])
vec = provider.encode_queries(["中指未到位"])[0]        # instruction applied
health = provider.health()                              # profile + probe + cache stats
```

## Cache

`~/.rosclaw/embedding_cache.sqlite` — key = sha256(model_id + revision + dimension + kind(query|document) + instruction_hash + text_hash). Query/document caches never mix; a revision/dimension change is a different key by construction (no stale reuse).

## Degradation

`EmbeddingUnavailableError` → caller degrades to SeekDB BM25 + metadata filter. Never substitute another model's vectors into a Qwen collection (§11).

## Reranker (high-risk paths only)

```python
from rosclaw.embedding.reranker import Qwen3RerankerProvider
reranker = Qwen3RerankerProvider()          # pinned e61197ed
top5 = reranker.rerank(query, top20_rows, top_k=5)
```

Use for failure recovery / HOW / hard negatives only; ~266 ms per call is acceptable there, not on the dashboard or hot path.
