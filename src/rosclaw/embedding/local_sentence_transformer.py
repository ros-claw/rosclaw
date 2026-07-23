"""Local sentence-transformers embedding provider (数据库优化v3 §7.1/§7.3).

Runs the pinned model on this machine (Jetson GPU measured: load ~0.7 s
warm, ~103 ms/doc in small batches, ~1.3 GB VRAM for Qwen3-0.6B FP16).
Offline-first: a pinned local snapshot is used as-is; downloads only
happen through an explicit endpoint, never implicitly inside the robot
hot path (§17.7 — embedding never blocks motion).
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any

from .errors import EmbeddingDimensionMismatchError, EmbeddingUnavailableError
from .protocol import EmbeddingProfile


class LocalSentenceTransformerProvider:
    """Qwen3-Embedding (or any ST model) on local torch."""

    def __init__(
        self,
        profile: EmbeddingProfile,
        *,
        device: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        self._profile = profile
        self._device = device
        self._trust_remote_code = trust_remote_code
        self._model: Any | None = None

    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - env dependent
            raise EmbeddingUnavailableError(
                f"sentence-transformers/torch unavailable: {exc}"
            ) from exc
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        try:
            self._model = SentenceTransformer(
                self._profile.model_id,
                revision=self._profile.model_revision,
                model_kwargs=kwargs,
                tokenizer_kwargs={"padding_side": "left"},
                device=device,
                trust_remote_code=self._trust_remote_code,
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingUnavailableError(
                f"failed to load {self._profile.model_id}@{self._profile.model_revision}: {exc}"
            ) from exc
        return self._model

    def _encode(self, texts: list[str], *, kind: str) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        batch = list(texts)
        if kind == "query" and self._profile.query_instruction:
            # Qwen model card: instruction on the QUERY side only (§7.2).
            batch = [
                f"Instruct: {self._profile.query_instruction}\nQuery: {text}" for text in texts
            ]
        elif kind == "document" and self._profile.document_instruction:
            batch = [
                f"Instruct: {self._profile.document_instruction}\nDocument: {text}"
                for text in texts
            ]
        try:
            vectors = model.encode(
                batch,
                normalize_embeddings=self._profile.normalize,
                convert_to_numpy=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingUnavailableError(f"encode failed ({kind}): {exc}") from exc
        out = [list(map(float, row)) for row in vectors]
        fixed: list[list[float]] = []
        for row in out:
            if len(row) != self._profile.dimension:
                if len(row) > self._profile.dimension:
                    # Matryoshka truncation (Qwen3 supports 32-1024 via
                    # prefix truncation + renormalization).
                    row = row[: self._profile.dimension]
                    if self._profile.normalize:
                        norm = sum(v * v for v in row) ** 0.5 or 1.0
                        row = [v / norm for v in row]
                else:
                    raise EmbeddingDimensionMismatchError(
                        f"{self._profile.profile_id} produced dim {len(row)}, "
                        f"expected {self._profile.dimension}"
                    )
            fixed.append(row)
        return fixed

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, kind="document")

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, kind="query")

    def health(self) -> dict:
        t0 = time.monotonic()
        status: dict[str, Any] = {
            "profile_id": self._profile.profile_id,
            "model_id": self._profile.model_id,
            "model_revision": self._profile.model_revision,
            "dimension": self._profile.dimension,
            "provider_type": self._profile.provider_type,
            "offline": bool(os.environ.get("HF_HUB_OFFLINE")),
        }
        try:
            vec = self.encode_queries(["health probe"])
            status["ok"] = True
            status["probe_ms"] = round((time.monotonic() - t0) * 1000.0, 1)
            status["probe_dim"] = len(vec[0])
            status["text_sha256"] = hashlib.sha256(b"health probe").hexdigest()[:12]
        except Exception as exc:  # noqa: BLE001
            status["ok"] = False
            status["error"] = f"{type(exc).__name__}: {exc}"
        return status
