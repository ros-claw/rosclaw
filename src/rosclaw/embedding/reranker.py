"""Qwen3-Reranker-0.6B provider (数据库优化v3 §9).

Cross-encoder re-ranking for HIGH-RISK paths only (failure recovery,
HOW rule selection, safety-adjacent history, hard-negative pairs like
``middle`` vs ``thumb_rot``).  Candidate window 20 -> final 5.  Never on
the dashboard path; never inside the robot motion hot path.
"""

from __future__ import annotations

import time
from typing import Any

from .errors import EmbeddingUnavailableError
from .profile import QWEN3_RERANKER_06B_REVISION

RERANKER_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"

RERANKER_INSTRUCTION = (
    "Given a robot embodied-memory query, retrieve the most relevant "
    "evidence-backed past experience, failure, intervention, body-state "
    "pattern, or skill. Preserve robot, body, joint, gesture, task, "
    "error-code, and temporal specificity."
)


class Qwen3RerankerProvider:
    """Local cross-encoder reranker (Qwen3-Reranker-0.6B, pinned)."""

    def __init__(
        self,
        *,
        revision: str = QWEN3_RERANKER_06B_REVISION,
        device: str | None = None,
        max_length: int = 8192,
    ) -> None:
        self._revision = revision
        self._device = device
        self._max_length = max_length
        self._model: Any | None = None

    @property
    def model_id(self) -> str:
        return RERANKER_MODEL_ID

    @property
    def revision(self) -> str:
        return self._revision

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingUnavailableError(
                f"sentence-transformers/torch unavailable for reranker: {exc}"
            ) from exc
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        try:
            self._model = CrossEncoder(
                RERANKER_MODEL_ID,
                revision=self._revision,
                max_length=self._max_length,
                device=device,
                prompts={"rosclaw": RERANKER_INSTRUCTION},
                default_prompt_name="rosclaw",
                model_kwargs={"torch_dtype": dtype},
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingUnavailableError(
                f"failed to load {RERANKER_MODEL_ID}@{self._revision}: {exc}"
            ) from exc
        return self._model

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Relevance score per (query, document) pair; higher = more relevant."""
        if not documents:
            return []
        model = self._load()
        pairs = [[query, doc] for doc in documents]
        try:
            scores = model.predict(pairs)
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingUnavailableError(f"reranker predict failed: {exc}") from exc
        return [float(s) for s in scores]

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        text_key: str = "document",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return candidates re-ordered by cross-encoder score (top_k)."""
        if not candidates:
            return []
        docs = [str(c.get(text_key) or c.get("title") or "") for c in candidates]
        scores = self.score(query, docs)
        order = sorted(
            zip(candidates, scores, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        out = []
        for candidate, score in order[:top_k]:
            enriched = dict(candidate)
            enriched["rerank_score"] = score
            out.append(enriched)
        return out

    def health(self) -> dict:
        t0 = time.monotonic()
        status: dict[str, Any] = {
            "model_id": RERANKER_MODEL_ID,
            "model_revision": self._revision,
            "provider_type": "local_cross_encoder",
        }
        try:
            self.score("health probe", ["probe document"])
            status["ok"] = True
            status["probe_ms"] = round((time.monotonic() - t0) * 1000.0, 1)
        except Exception as exc:  # noqa: BLE001
            status["ok"] = False
            status["error"] = f"{type(exc).__name__}: {exc}"
        return status
