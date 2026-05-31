"""ProviderLoader - Discover and load providers from YAML manifests.

Scans directories for provider.yaml files, creates Provider instances,
and registers them with a ProviderRegistry.

Usage:
    loader = ProviderLoader(registry)
    loader.scan_directory("./providers")
    loader.scan_directory("~/.rosclaw/providers")
"""

import logging
import os
from pathlib import Path
from typing import Any

from rosclaw.provider.adapters.generic import GenericProvider
from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.provider import Provider
from rosclaw.provider.core.registry import ProviderRegistry

logger = logging.getLogger("rosclaw.provider.loader")


class ProviderLoader:
    """Discover and load providers from the filesystem."""

    def __init__(self, registry: ProviderRegistry):
        self.registry = registry
        self._loaded_paths: dict[str, Path] = {}

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def scan_directory(self, directory: str | Path) -> list[str]:
        """Scan a directory tree for provider.yaml files and load them.

        Returns list of successfully loaded provider names.
        """
        dir_path = Path(directory).expanduser().resolve()
        if not dir_path.exists():
            return []

        loaded: list[str] = []
        for yaml_path in dir_path.rglob("provider.yaml"):
            name = self._load_from_yaml(yaml_path)
            if name:
                loaded.append(name)
        return loaded

    def load_file(self, path: str | Path) -> str | None:
        """Load a single provider.yaml file.

        Returns provider name on success, None on failure.
        """
        return self._load_from_yaml(Path(path))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _load_from_yaml(self, path: Path) -> str | None:
        """Load provider from a single YAML file."""
        try:
            manifest = ProviderManifest.from_yaml(path)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", path, e)
            return None

        name = manifest.name
        if name in self._loaded_paths:
            logger.info("Provider '%s' already loaded from %s", name, self._loaded_paths[name])
            return None

        # Allow custom subclass via env hint: provider_class = "my_module.MyProvider"
        provider_class = self._resolve_provider_class(manifest)

        def factory(m: ProviderManifest) -> Provider:
            return provider_class(m)

        try:
            self.registry.register(manifest, factory, auto_load=False)
            self._loaded_paths[name] = path
            logger.info("Registered '%s' from %s", name, path)
        except Exception as e:
            logger.warning("Failed to register '%s': %s", name, e)
            return None

        return name

    # Allowed top-level module prefixes for dynamic provider class loading.
    # tests.* is allowed for test-time custom provider classes.
    _ALLOWED_MODULE_PREFIXES: tuple[str, ...] = ("rosclaw.", "tests.")

    @classmethod
    def _resolve_provider_class(cls, manifest: ProviderManifest) -> type[Provider]:
        """Resolve the Provider class to use for this manifest.

        Priority:
        1. Custom class specified in manifest.extra["provider_class"]
        2. GenericProvider (default)

        Security: module_name must start with an allowed prefix and must not
        contain path traversal characters.
        """
        class_path = manifest.extra.get("provider_class", "")
        if not class_path:
            return GenericProvider

        try:
            module_name, class_name = class_path.rsplit(".", 1)
        except ValueError:
            print(
                f"[ProviderLoader] Invalid provider_class '{class_path}'; "
                f"falling back to GenericProvider"
            )
            return GenericProvider

        # Security: block path traversal and disallowed modules
        if ".." in module_name or "/" in module_name or "\\" in module_name:
            print(
                f"[ProviderLoader] Blocked unsafe module_name '{module_name}'; "
                f"falling back to GenericProvider"
            )
            return GenericProvider

        if not any(module_name.startswith(prefix) for prefix in cls._ALLOWED_MODULE_PREFIXES):
            print(
                f"[ProviderLoader] Module '{module_name}' not in allowed prefixes "
                f"{cls._ALLOWED_MODULE_PREFIXES}; falling back to GenericProvider"
            )
            return GenericProvider

        try:
            module = __import__(module_name, fromlist=[class_name])
            cls_obj = getattr(module, class_name)
            if not isinstance(cls_obj, type) or not issubclass(cls_obj, Provider):
                print(
                    f"[ProviderLoader] {class_path} is not a Provider subclass; "
                    f"falling back to GenericProvider"
                )
                return GenericProvider
            return cls_obj
        except Exception as e:
            print(
                f"[ProviderLoader] Could not resolve {class_path}: {e}; "
                f"falling back to GenericProvider"
            )
            return GenericProvider

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def list_loaded(self) -> dict[str, str]:
        """Return map of provider name -> yaml path."""
        return {k: str(v) for k, v in self._loaded_paths.items()}

    def unload(self, name: str) -> None:
        """Unload a provider by name."""
        self.registry.unregister(name)
        self._loaded_paths.pop(name, None)
