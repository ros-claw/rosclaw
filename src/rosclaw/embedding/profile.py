"""Built-in embedding profiles (数据库优化v3 §1.2/§7).

Revisions are PINNED (§17.4).  The Qwen query instruction follows the
official model card: English instruction on the query side only,
documents never carry one.
"""

from __future__ import annotations

from .protocol import EmbeddingProfile

QWEN3_QUERY_INSTRUCTION = (
    "Given a robot embodied-memory query, retrieve the most relevant "
    "evidence-backed past experience, failure, intervention, body-state "
    "pattern, or skill. Preserve robot, body, joint, gesture, task, "
    "error-code, and temporal specificity."
)

QWEN3_EMBEDDING_06B_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
QWEN3_RERANKER_06B_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"

QWEN3_06B_1024 = EmbeddingProfile(
    profile_id="qwen3_06b_1024_v1",
    model_id="Qwen/Qwen3-Embedding-0.6B",
    model_revision=QWEN3_EMBEDDING_06B_REVISION,
    dimension=1024,
    normalize=True,
    distance="cosine",
    query_instruction=QWEN3_QUERY_INSTRUCTION,
    document_instruction=None,
    max_tokens=32768,
    provider_type="local_sentence_transformer",
)

QWEN3_06B_768 = EmbeddingProfile(
    profile_id="qwen3_06b_768_v1",
    model_id="Qwen/Qwen3-Embedding-0.6B",
    model_revision=QWEN3_EMBEDDING_06B_REVISION,
    dimension=768,
    normalize=True,
    distance="cosine",
    query_instruction=QWEN3_QUERY_INSTRUCTION,
    document_instruction=None,
    max_tokens=32768,
    provider_type="local_sentence_transformer",
)

QWEN3_06B_512 = EmbeddingProfile(
    profile_id="qwen3_06b_512_v1",
    model_id="Qwen/Qwen3-Embedding-0.6B",
    model_revision=QWEN3_EMBEDDING_06B_REVISION,
    dimension=512,
    normalize=True,
    distance="cosine",
    query_instruction=QWEN3_QUERY_INSTRUCTION,
    document_instruction=None,
    max_tokens=32768,
    provider_type="local_sentence_transformer",
)

PROFILES: dict[str, EmbeddingProfile] = {
    p.profile_id: p for p in (QWEN3_06B_1024, QWEN3_06B_768, QWEN3_06B_512)
}
