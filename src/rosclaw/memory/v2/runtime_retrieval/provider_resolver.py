"""Embedding provider resolution pinned to the ACTIVE descriptor (v4 §3.4).

The ACTIVE registry row records the exact model identity that built the
collection.  The resolver refuses any substitution: a Qwen 1024 provider can
never query a Qwen 768 collection, and a MiniLM/GTE provider can never query
a Qwen collection.  When the pinned provider cannot be constructed the caller
enters BM25-only fallback — it never falls back to another model's vectors.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rosclaw.embedding.profile import PROFILES
from rosclaw.embedding.protocol import EmbeddingProvider

from .active_resolver import ActiveIndexDescriptor

# Machine-readable reason codes.
PROVIDER_PROFILE_UNKNOWN = "provider_profile_unknown"
PROVIDER_IDENTITY_MISMATCH = "provider_identity_mismatch"
PROVIDER_CONSTRUCTION_FAILED = "provider_construction_failed"


class ProviderUnavailableError(RuntimeError):
    """The pinned provider cannot serve this descriptor; carries ``reason``."""

    def __init__(self, reason: str, detail: str | None = None) -> None:
        self.reason = reason
        self.detail = detail or reason
        super().__init__(self.detail)


class EmbeddingProviderResolver:
    """Resolve the provider that exactly matches an ACTIVE descriptor.

    ``provider_factory`` defaults to :func:`rosclaw.embedding.registry.get_provider`
    and is injectable for tests.  Providers are constructed lazily (the
    sentence-transformer backend loads weights on first encode) and cached per
    profile id for the lifetime of the process.
    """

    def __init__(
        self,
        provider_factory: Callable[..., EmbeddingProvider] | None = None,
        *,
        cache_path: str | None = None,
        device: str | None = None,
        profiles: dict[str, Any] | None = None,
    ) -> None:
        if provider_factory is None:
            from rosclaw.embedding.registry import get_provider

            provider_factory = get_provider
        self._factory = provider_factory
        self._cache_path = cache_path
        self._device = device
        # Injectable profile registry (tests use a fake profile; production
        # uses the pinned built-ins).
        self._profiles: dict[str, Any] = profiles if profiles is not None else PROFILES
        self._providers: dict[str, EmbeddingProvider] = {}
        import threading

        self._lock = threading.Lock()

    def resolve(self, descriptor: ActiveIndexDescriptor) -> EmbeddingProvider:
        profile_id = descriptor.embedding_profile_id
        profile = self._profiles.get(profile_id)
        if profile is None:
            raise ProviderUnavailableError(
                PROVIDER_PROFILE_UNKNOWN,
                f"ACTIVE profile {profile_id!r} is not a known embedding profile",
            )
        mismatches = []
        if profile.model_id != descriptor.model_id:
            mismatches.append(f"model_id {profile.model_id!r} != {descriptor.model_id!r}")
        if profile.model_revision != descriptor.model_revision:
            mismatches.append("model_revision differs")
        if profile.dimension != descriptor.dimension:
            mismatches.append(f"dimension {profile.dimension} != {descriptor.dimension}")
        if mismatches:
            raise ProviderUnavailableError(
                PROVIDER_IDENTITY_MISMATCH,
                f"provider profile {profile_id!r} does not match ACTIVE descriptor: "
                + "; ".join(mismatches),
            )
        if profile_id not in self._providers:
            with self._lock:
                if profile_id not in self._providers:
                    kwargs: dict[str, Any] = {}
                    if self._cache_path is not None:
                        kwargs["cache_path"] = self._cache_path
                    if self._device is not None:
                        kwargs["device"] = self._device
                    try:
                        self._providers[profile_id] = self._factory(profile_id, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        raise ProviderUnavailableError(
                            PROVIDER_CONSTRUCTION_FAILED,
                            f"could not construct provider for {profile_id!r}: {exc}",
                        ) from exc
        return self._providers[profile_id]
