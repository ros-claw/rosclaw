"""Retrieval health probe (数据库优化v4 §3.2 health.py / §14).

Reports what the serving path would do WITHOUT loading model weights:
ACTIVE descriptor, provider identity check, fallback availability.  A full
model-load probe is opt-in (``probe_provider=True``) because it is expensive
on robot hardware.
"""

from __future__ import annotations

import logging
from typing import Any

from rosclaw.embedding.profile import PROFILES

from .active_resolver import ActiveCollectionResolver, ActiveIndexUnavailableError
from .fallback import MODE_ABSTAIN, MODE_ACTIVE_BM25, MODE_SQLITE_LEXICAL, hybrid_mode_for
from .provider_resolver import EmbeddingProviderResolver, ProviderUnavailableError

logger = logging.getLogger("rosclaw.memory.v2.runtime_retrieval.health")


class RetrievalHealthProbe:
    """Introspects the facade's serving path for CLI/status output."""

    def __init__(
        self,
        *,
        native_store: Any | None,
        sqlite_store: Any | None,
        provider_resolver: EmbeddingProviderResolver | None = None,
        logical_name: str = "memory_items",
    ) -> None:
        self._native_store = native_store
        self._sqlite_store = sqlite_store
        self._provider_resolver = provider_resolver or EmbeddingProviderResolver()
        self._logical_name = logical_name

    def probe(self, *, probe_provider: bool = False) -> dict[str, Any]:
        report: dict[str, Any] = {
            "logical_collection": self._logical_name,
            "native_store": type(self._native_store).__name__ if self._native_store else None,
            "sqlite_fallback": (type(self._sqlite_store).__name__ if self._sqlite_store else None),
            "active": None,
            "serving_mode": None,
            "fallback_chain": [
                "<profile>_hybrid",
                MODE_ACTIVE_BM25,
                MODE_SQLITE_LEXICAL,
                MODE_ABSTAIN,
            ],
        }
        if self._native_store is None:
            report["serving_mode"] = (
                MODE_SQLITE_LEXICAL if self._sqlite_store is not None else MODE_ABSTAIN
            )
            report["active"] = {"ok": False, "reason": "seekdb_native_store_unavailable"}
            return report

        try:
            descriptor = ActiveCollectionResolver(self._native_store).resolve(self._logical_name)
        except ActiveIndexUnavailableError as exc:
            report["active"] = {"ok": False, "reason": exc.reason, "detail": exc.detail}
            report["serving_mode"] = (
                MODE_SQLITE_LEXICAL if self._sqlite_store is not None else MODE_ABSTAIN
            )
            return report
        except Exception as exc:  # noqa: BLE001
            report["active"] = {"ok": False, "reason": f"native_store_error:{exc}"}
            report["serving_mode"] = (
                MODE_SQLITE_LEXICAL if self._sqlite_store is not None else MODE_ABSTAIN
            )
            return report

        provider_status: dict[str, Any] = {"identity_known": False}
        profile = PROFILES.get(descriptor.embedding_profile_id)
        if profile is not None:
            provider_status = {
                "identity_known": True,
                "profile_id": profile.profile_id,
                "identity_matches_active": (
                    profile.model_id == descriptor.model_id
                    and profile.model_revision == descriptor.model_revision
                    and profile.dimension == descriptor.dimension
                ),
            }
        if probe_provider:
            try:
                provider = self._provider_resolver.resolve(descriptor)
                provider_status["load_probe"] = provider.health()
            except ProviderUnavailableError as exc:
                provider_status["load_probe"] = {"ok": False, "reason": exc.reason}
            except Exception as exc:  # noqa: BLE001
                provider_status["load_probe"] = {"ok": False, "reason": str(exc)}

        report["active"] = {
            "ok": True,
            "physical_collection": descriptor.physical_collection,
            "embedding_profile_id": descriptor.embedding_profile_id,
            "model_id": descriptor.model_id,
            "model_revision": descriptor.model_revision,
            "dimension": descriptor.dimension,
            "analyzer": descriptor.analyzer,
            "activated_at": descriptor.activated_at,
        }
        report["provider"] = provider_status
        report["serving_mode"] = hybrid_mode_for(descriptor.embedding_profile_id)
        return report
