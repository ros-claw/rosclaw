"""ROSClaw Provider - ProviderRegistry.

Central registry for provider discovery, health tracking, and lifecycle management.
"""

import asyncio
import contextlib
import time
from collections.abc import Callable
from typing import Any

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

    def __init__(
        self,
        event_bus: Any | None = None,
        health_check_interval_sec: float = 30.0,
    ):
        self._providers: dict[str, Provider] = {}
        self._factories: dict[str, Callable[[ProviderManifest], Provider]] = {}
        self._manifests: dict[str, ProviderManifest] = {}
        self._health: dict[str, dict[str, Any]] = {}
        self._health_interval = health_check_interval_sec
        self._health_task: asyncio.Task | None = None
        self._shutdown: bool = False
        self._event_bus = event_bus

    def _publish_event(self, topic: str, payload: dict[str, Any]) -> None:
        """Publish a lifecycle event if EventBus is available."""
        if self._event_bus is None:
            return
        try:
            from rosclaw.core.event_bus import Event

            self._event_bus.publish(
                Event(topic=topic, payload=payload, source="provider_registry")
            )
        except Exception:
            pass

    def _publish_health_changed(
        self, name: str, ok: bool, reason: str = ""
    ) -> None:
        """Publish provider_health_changed event."""
        self._publish_event(
            "provider_health_changed",
            {
                "provider": name,
                "ok": ok,
                "reason": reason,
                "timestamp": time.time(),
            },
        )

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

        self._publish_event(
            "provider_registered",
            {
                "provider": name,
                "type": manifest.type,
                "capabilities": manifest.capabilities,
                "auto_load": auto_load,
            },
        )

        if auto_load:
            self._load_provider(provider)

        self._health[name] = {"last_check": 0.0, "ok": provider._healthy}
        return provider

    def _load_provider(self, provider: Provider) -> None:
        """Load a provider, handling both sync and async calling contexts."""
        from rosclaw.core.async_utils import run_sync
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop — load synchronously.
            try:
                run_sync(provider.load())
                provider._healthy = True
                self._publish_health_changed(provider.name, True, "load_success")
            except Exception as e:
                provider._healthy = False
                provider._load_error = str(e)
                self._publish_health_changed(provider.name, False, f"load_failed: {e}")
        else:
            # Called from within an async context — schedule deferred load.
            # The caller must await asyncio.sleep() or similar to let the
            # task complete before relying on provider._healthy.
            loop.create_task(self._deferred_load(provider))

    async def _deferred_load(self, provider: Provider) -> None:
        """Async-load a provider from within an existing event loop."""
        try:
            await provider.load()
            provider._healthy = True
            self._health[provider.name] = {"last_check": time.time(), "ok": True}
            self._publish_health_changed(provider.name, True, "load_success")
        except Exception as e:
            provider._healthy = False
            provider._load_error = str(e)
            self._health[provider.name] = {"last_check": time.time(), "ok": False, "error": str(e)}
            self._publish_health_changed(provider.name, False, f"load_failed: {e}")

    def unregister(self, name: str) -> None:
        """Unregister and unload a provider."""
        from rosclaw.core.async_utils import run_sync
        provider = self._providers.pop(name, None)
        if provider is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                with contextlib.suppress(Exception):
                    run_sync(provider.unload())
            else:
                loop.create_task(provider.unload())
        self._factories.pop(name, None)
        self._manifests.pop(name, None)
        self._health.pop(name, None)
        self._publish_event(
            "provider_unregistered",
            {"provider": name},
        )

    def set_provider_health(
        self,
        name: str,
        ok: bool,
        error: str = "",
    ) -> None:
        """Set the cached health status for a provider.

        This is the public API for marking pre-initialized providers
        (e.g. mock providers registered with auto_load=False) as
        healthy without bypassing the registry boundary.
        """
        provider = self._providers.get(name)
        if provider is not None:
            provider._healthy = ok
            if error:
                provider._load_error = error
        self._health[name] = {
            "last_check": time.time(),
            "ok": ok,
            "error": error,
        }
        self._publish_health_changed(name, ok, error or "manual_set")

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
            new_ok = result.get("ok", False)
            if provider._healthy != new_ok:
                self._publish_health_changed(name, new_ok, "health_check")
            provider._healthy = new_ok
            return result
        except Exception as e:
            provider._healthy = False
            self._health[name] = {"ok": False, "error": str(e), "timestamp": time.time()}
            self._publish_health_changed(name, False, f"health_check_error: {e}")
            return self._health[name]

    async def check_all_health(self) -> dict[str, dict[str, Any]]:
        """Run health checks for all registered providers concurrently."""
        names = list(self._providers.keys())
        results = await asyncio.gather(
            *(self.check_health(n) for n in names),
            return_exceptions=True,
        )
        return {n: (r if not isinstance(r, Exception) else {"ok": False, "error": str(r)})
                for n, r in zip(names, results, strict=False)}

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
            with contextlib.suppress(Exception):
                await self.check_all_health()
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

    def get_reasoner(self, provider_id: str | None = None) -> Any:
        """Return a ``PhysicalReasoner`` facade for the requested provider.

        Falls back to the provider factory abstraction (``rosclaw.provider.reasoner``)
        when no registered provider matches ``provider_id``.
        """
        from rosclaw.provider.reasoner import get_reasoner

        if provider_id:
            try:
                manifest = self.get_manifest(provider_id)
                return get_reasoner(manifest.type or provider_id)
            except ProviderNotFoundError:
                pass
        return get_reasoner(provider_id)
