"""Typed runtime config loader (PR-DF-18 / phase-II §15.1).

Builds the canonical typed config models from a firstboot rosclaw.yaml
(which ``FirstbootConfig`` has already normalized to the Config v2
sections).  Intended for CLI/entrypoint code that wants typed configs
without constructing a Runtime first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import (
    DarwinConfig,
    EvolutionConfig,
    KnowledgeConfig,
    StorageConfig,
)


def load_typed_configs(
    home: Path,
) -> tuple[StorageConfig, KnowledgeConfig, EvolutionConfig, DarwinConfig]:
    """Load (storage, knowledge, evolution, darwin) from rosclaw.yaml.

    Missing/partial files yield defaults; the file-format legacy sections
    are already folded by ``FirstbootConfig._normalize_legacy_config``.
    """
    from rosclaw.firstboot.config import FirstbootConfig, load_rosclaw_yaml

    raw: dict[str, Any] = load_rosclaw_yaml(home)
    fb = FirstbootConfig(
        workspace=raw.get("workspace", {}),
        runtime=raw.get("runtime", {}),
        memory=raw.get("memory", {}),
        knowledge=raw.get("knowledge", {}),
        evolution=raw.get("evolution", {}),
        darwin=raw.get("darwin", {}),
        storage=raw.get("storage", {}),
        know=raw.get("know", {}),
        auto=raw.get("auto", {}),
    )
    return (
        StorageConfig.from_dict(fb.storage),
        KnowledgeConfig.from_dict(fb.knowledge),
        EvolutionConfig.from_dict(fb.evolution),
        DarwinConfig.from_dict(fb.darwin),
    )
