"""ROSClaw Provider - ProviderRegistry.

Central registry for provider discovery, health tracking, and lifecycle management.
"""

import asyncio
import time
from typing import Any, Callable

from rosclaw.provider.core.errors import ProviderNotFoundError
from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.provider import Provider


class ProviderRegistry:
    """Central registry for all ROSClaw capability providers.

    Responsibilities:
    1. Register / unregister providers by manifest + factory
    2. Look up providers by name or capability
    3. Track health status (async health checks)
    4. Manage provider lifecycle (load / unload)

    Thread-safety: Not thread-safe by default. Use within a single event loop
    or protect with locks if accessed from multiple threads.
    """

    def __init__(self, health_check_interval_sec: float = 30.0):
        self._providers: dict[str, Provider] = {}
        self._factories: dict[str, Callable[[ProviderManifest], Provider]] = {}
        self._manifests: dict[str, ProviderManifest] = {}
        self._health: dict[str, dict[str, Any]] = {}
        self._health_interval = health_check_interval_sec
        self._health_task: asyncio.Task | None = None
        self._shutdown: bool = False

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(
        self,
        manifest: ProviderManifest,
        factory: Callable[[ProviderManifest], Provider],
        auto_load: bool = True,
    ) -> Provider:
        """Register a provider from manifest + factory.

        Args:
            manifest: Parsed provider.yaml manifest.
            factory: Callable that creates a Provider from the manifest.
            auto_load: If True, immediately call provider.load().

        Returns:
            The created Provider instance.
        """
        name = manifest.name
        if name in self._providers:
            raise ProviderNotFoundError(f"Provider '{name}' is already registered")

        self._manifests[name] = manifest
        self._factories[name] = factory

        provider = factory(manifest)
        self._providers[name] = provider

        if auto_load:
            try:
                asyncio.get_event_loop().run_until_complete(provider.load())
                provider._healthy = True
            except Exception as e:
                provider._healthy = False
                provider._load_error = str(e)

        self._health[name] = {"last_check": 0.0, "ok": provider._healthy}
        return provider

    def unregister(self, name: str) -> None:
        """Unregister and unload a provider."""
        provider = self._providers.pop(name, None)
        if provider is not None:
            try:
                asyncio.get_event_loop().run_until_complete(provider.unload())
            except Exception:
                pass
        self._factories.pop(name, None)
        self._manifests.pop(name, None)
        self._health.pop(name, None)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    def get(self, name: str) -> Provider:
        """Get provider by exact name."""
        if name not in self._providers:
            raise ProviderNotFoundError(f"Provider '{name}' not found")
        return self._providers[name]

    def get_manifest(self, name: str) -> ProviderManifest:
        """Get manifest by provider name."""
        if name not in self._manifests:
            raise ProviderNotFoundError(f"Manifest for '{name}' not found")
        return self._manifests[name]

    def list_providers(self) -> list[str]:
        """List all registered provider names."""
        return list(self._providers.keys())

    def find_by_capability(
        self,
        capability: str,
        healthy_only: bool = True,
    ) -> list[Provider]:
        """Find all providers that support a given capability."""
        results: list[Provider] = []
        for name, provider in self._providers.items():
            if capability not in provider.capabilities:
                continue
            if healthy_only and not self._is_healthy(name):
                continue
            results.append(provider)
        return results

    def find_by_type(self, provider_type: str) -> list[Provider]:
        """Find all providers of a given type (llm, vlm, skill, etc.)."""
        return [
            p for p in self._providers.values()
            if p.manifest.type == provider_type
        ]

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    async def check_health(self, name: str) -> dict[str, Any]:
        """Run health check for a single provider."""
        provider = self._providers.get(name)
        if provider is None:
            return {"ok": False, "error": "not_registered"}
        try:
            result = await provider.health()
            result["timestamp"] = time.time()
            self._health[name] = result
            provider._healthy = result.get("ok", False)
            return result
        except Exception as e:
            provider._healthy = False
            self._health[name] = {"ok": False, "error": str(e), "timestamp": time.time()}
            return self._health[name]

    async def check_all_health(self) -> dict[str, dict[str, Any]]:
        """Run health checks for all registered providers concurrently."""
        names = list(self._providers.keys())
        results = await asyncio.gather(
            *(self.check_health(n) for n in names),
            return_exceptions=True,
        )
        return {n: (r if not isinstance(r, Exception) else {"ok": False, "error": str(r)})
                for n, r in zip(names, results)}

    def _is_healthy(self, name: str) -> bool:
        health = self._health.get(name, {})
        return health.get("ok", False)

    def is_healthy(self, name: str) -> bool:
        """Return cached health status (does not run a new check)."""
        return self._is_healthy(name)

    # ------------------------------------------------------------------
    # Background health monitor
    # ------------------------------------------------------------------
    async def _health_monitor_loop(self) -> None:
        while not self._shutdown:
            try:
                await self.check_all_health()
            except Exception:
                pass
            await asyncio.sleep(self._health_interval)

    def start_health_monitor(self) -> None:
        """Start background health checks. Idempotent."""
        if self._health_task is None or self._health_task.done():
            self._shutdown = False
            self._health_task = asyncio.create_task(self._health_monitor_loop())

    def stop_health_monitor(self) -> None:
        """Stop background health checks."""
        self._shutdown = True
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------
    async def load_all(self) -> None:
        """Load all registered providers."""
        await asyncio.gather(
            *(p.load() for p in self._providers.values()),
            return_exceptions=True,
        )

    async def unload_all(self) -> None:
        """Unload all registered providers."""
        await asyncio.gather(
            *(p.unload() for p in self._providers.values()),
            return_exceptions=True,
        )

    def get_statistics(self) -> dict[str, Any]:
        """Registry-wide statistics."""
        total = len(self._providers)
        healthy = sum(1 for n in self._providers if self._is_healthy(n))
        by_type: dict[str, int] = {}
        for m in self._manifests.values():
            by_type[m.type] = by_type.get(m.type, 0) + 1
        return {
            "total_providers": total,
            "healthy_providers": healthy,
            "unhealthy_providers": total - healthy,
            "by_type": by_type,
        }
