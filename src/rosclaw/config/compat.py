"""Legacy → typed config normalization (PR-DF-18 / phase-II §15.4).

The ONLY place pre-DF-18 shapes are interpreted: flat RuntimeConfig
fields (``seekdb_backend``/``seekdb_path``/``seekdb_url``,
``know_store_mode``/``know_store_path``, ``enable_auto``,
``enable_darwin``, ``enable_knowledge``) and the dict-shaped
``storage``/``darwin`` sections.  RuntimeConfig.__post_init__ calls these
exactly once; the rest of the runtime reads typed models only.

Precedence: an explicitly-passed typed model (or a dict key) always wins;
legacy flat fields only fill slots the caller never set.
"""

from __future__ import annotations

import logging
from typing import Any

from .models import (
    DarwinConfig,
    EvolutionConfig,
    KnowledgeConfig,
    StorageConfig,
)

logger = logging.getLogger("rosclaw.config.compat")

_LOGGED: set[str] = set()


def _deprecated_once(legacy: str, canonical: str) -> None:
    key = f"{legacy}->{canonical}"
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    logger.warning(
        "DEPRECATED CONFIG: %s -> %s (accepted once; migrate to the typed field)",
        legacy,
        canonical,
    )


def normalize_storage_config(value: Any, *, legacy: Any) -> StorageConfig:
    """Fold ``storage`` (dict | StorageConfig | None) + flat seekdb_* fields."""
    if isinstance(value, StorageConfig):
        return value
    data = value if isinstance(value, dict) else {}
    cfg = StorageConfig.from_dict(data)

    # Flat dict keys from the pre-v2 storage section.
    if "pool_size" in data:
        cfg.structured.pool_size = int(data["pool_size"])
    if "vector_enabled" in data:
        cfg.retrieval.enabled = bool(data["vector_enabled"])
        _deprecated_once("storage.vector_enabled", "storage.retrieval.enabled")
    if "outbox_enabled" in data:
        cfg.outbox.enabled = bool(data["outbox_enabled"])
        _deprecated_once("storage.outbox_enabled", "storage.outbox.enabled")
    if "outbox_path" in data:
        cfg.outbox.path = data["outbox_path"]
    if "outbox_max_records" in data:
        cfg.outbox.max_records = int(data["outbox_max_records"])
    if "outbox_flush_interval_sec" in data:
        cfg.outbox.flush_interval_sec = float(data["outbox_flush_interval_sec"])

    # Flat RuntimeConfig seekdb_* fields only fill unset slots.
    structured = data.get("structured", {}) if isinstance(data.get("structured"), dict) else {}
    if "backend" not in structured:
        backend = getattr(legacy, "seekdb_backend", None)
        if backend:
            cfg.structured.backend = backend
    if "path" not in structured:
        path = getattr(legacy, "seekdb_path", None)
        if path:
            cfg.structured.path = path
    if "dsn" not in structured:
        dsn = getattr(legacy, "seekdb_url", None)
        if dsn:
            cfg.structured.dsn = dsn
    return cfg


def normalize_knowledge_config(value: Any, *, legacy: Any) -> KnowledgeConfig:
    if isinstance(value, KnowledgeConfig):
        return value
    data = value if isinstance(value, dict) else {}
    cfg = KnowledgeConfig.from_dict(data)
    if "enabled" not in data:
        cfg.enabled = bool(getattr(legacy, "enable_knowledge", True))
    if "mode" not in data:
        cfg.mode = getattr(legacy, "knowledge_v2_mode", "disabled") or "disabled"
    if "store_mode" not in data:
        store_mode = getattr(legacy, "know_store_mode", None)
        if store_mode:
            cfg.store_mode = store_mode
    if "store_path" not in data:
        store_path = getattr(legacy, "know_store_path", None)
        if store_path:
            cfg.store_path = store_path
    if "url" not in data:
        cfg.url = getattr(legacy, "know_url", None)
    if "api_key" not in data:
        cfg.api_key = getattr(legacy, "know_api_key", None)
    if "timeout" not in data:
        cfg.timeout = float(getattr(legacy, "knowledge_timeout", 15.0))
    if "curated_registry_enabled" not in data:
        cfg.curated_registry_enabled = bool(
            getattr(legacy, "know_curated_registry_enabled", False)
        )
    return cfg


def normalize_evolution_config(value: Any, *, legacy: Any) -> EvolutionConfig:
    if isinstance(value, EvolutionConfig):
        return value
    data = value if isinstance(value, dict) else {}
    cfg = EvolutionConfig.from_dict(data)
    if "enabled" not in data:
        cfg.enabled = bool(getattr(legacy, "enable_auto", True))
    return cfg


def normalize_darwin_config(value: Any, *, legacy: Any) -> DarwinConfig:
    if isinstance(value, DarwinConfig):
        return value
    data = value if isinstance(value, dict) else {}
    cfg = DarwinConfig.from_dict(data)
    if "enabled" not in data:
        cfg.enabled = bool(getattr(legacy, "enable_darwin", False))
    return cfg
