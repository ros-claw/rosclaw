"""Typed Runtime configuration models (PR-DF-18 / phase-II §15.1-15.3).

These are the canonical in-process config objects.  The flat legacy
RuntimeConfig fields (seekdb_backend, know_store_mode, enable_auto, ...)
and the dict-shaped ``storage``/``darwin`` sections exist only as
constructor inputs; ``rosclaw.config.compat`` folds them into these
models exactly once.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def _default_structured_path() -> str:
    # Deliberately the pre-v2 data location (phase-II §16: no path move
    # in DF-18 — a move needs its own migration PR with a mover).
    from rosclaw.firstboot.workspace import get_rosclaw_home

    return str(get_rosclaw_home() / "data" / "memory" / "knowledge.sqlite")


@dataclass
class StructuredStoreConfig:
    backend: str = "sqlite"
    path: str | None = None
    dsn: str | None = None
    pool_size: int = 4

    def __post_init__(self) -> None:
        if self.path is None:
            self.path = _default_structured_path()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StructuredStoreConfig:
        return cls(
            backend=data.get("backend", "sqlite"),
            path=data.get("path"),
            dsn=data.get("dsn"),
            pool_size=int(data.get("pool_size", 4)),
        )


@dataclass
class RetrievalStoreConfig:
    enabled: bool = False
    backend: str = "seekdb_native"
    mode: str = "embedded"
    path: str | None = None
    host: str | None = None
    port: int = 2881
    database: str = "rosclaw"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievalStoreConfig:
        return cls(
            enabled=bool(data.get("enabled", False)),
            backend=data.get("backend", "seekdb_native"),
            mode=data.get("mode", "embedded"),
            path=data.get("path"),
            host=data.get("host"),
            port=int(data.get("port", 2881)),
            database=data.get("database", "rosclaw"),
        )


@dataclass
class OutboxConfig:
    enabled: bool = False
    path: str | None = None
    batch_size: int = 100
    flush_interval_sec: float = 5.0
    max_records: int = 100_000

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutboxConfig:
        return cls(
            enabled=bool(data.get("enabled", False)),
            path=data.get("path"),
            batch_size=int(data.get("batch_size", 100)),
            flush_interval_sec=float(data.get("flush_interval_sec", 5.0)),
            max_records=int(data.get("max_records", 100_000)),
        )


@dataclass
class ArtifactStoreConfig:
    backend: str = "filesystem"
    root: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactStoreConfig:
        return cls(backend=data.get("backend", "filesystem"), root=data.get("root"))


@dataclass
class StorageConfig:
    structured: StructuredStoreConfig = field(default_factory=StructuredStoreConfig)
    retrieval: RetrievalStoreConfig = field(default_factory=RetrievalStoreConfig)
    outbox: OutboxConfig = field(default_factory=OutboxConfig)
    artifacts: ArtifactStoreConfig = field(default_factory=ArtifactStoreConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StorageConfig:
        return cls(
            structured=StructuredStoreConfig.from_dict(data.get("structured", {})),
            retrieval=RetrievalStoreConfig.from_dict(data.get("retrieval", {})),
            outbox=OutboxConfig.from_dict(data.get("outbox", {})),
            artifacts=ArtifactStoreConfig.from_dict(data.get("artifacts", {})),
        )


@dataclass
class KnowledgeConfig:
    enabled: bool = True
    mode: str = "disabled"  # versioned Know/How integration mode
    store_mode: str = field(
        default_factory=lambda: os.environ.get("ROSCLAW_KNOW_STORE_MODE", "embedded")
    )
    store_path: str | None = field(
        default_factory=lambda: os.environ.get("ROSCLAW_KNOW_SEEKDB_PATH")
    )
    url: str | None = field(default_factory=lambda: os.environ.get("ROSCLAW_KNOW_URL"))
    api_key: str | None = field(
        default_factory=lambda: os.environ.get("ROSCLAW_KNOW_API_KEY")
    )
    timeout: float = 15.0
    curated_registry_enabled: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeConfig:
        return cls(
            enabled=bool(data.get("enabled", True)),
            mode=data.get("mode", "disabled"),
            store_mode=data.get("store_mode", "embedded"),
            store_path=data.get("store_path"),
            url=data.get("url"),
            api_key=data.get("api_key"),
            timeout=float(data.get("timeout", 15.0)),
            curated_registry_enabled=bool(data.get("curated_registry_enabled", False)),
        )


@dataclass
class EvolutionConfig:
    enabled: bool = True
    require_human_approval: bool = True
    allow_code_patch: bool = False
    trigger_failure_threshold: int = 3

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvolutionConfig:
        return cls(
            enabled=bool(data.get("enabled", True)),
            require_human_approval=bool(data.get("require_human_approval", True)),
            allow_code_patch=bool(data.get("allow_code_patch", False)),
            trigger_failure_threshold=int(data.get("trigger_failure_threshold", 3)),
        )


@dataclass
class DarwinConfig:
    enabled: bool = False
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    episodes: int = 50

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DarwinConfig:
        return cls(
            enabled=bool(data.get("enabled", False)),
            seeds=list(data.get("seeds", [0, 1, 2])),
            episodes=int(data.get("episodes", 50)),
        )
