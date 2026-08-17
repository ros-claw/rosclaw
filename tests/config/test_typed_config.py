"""PR-DF-18 (phase-II §15): typed Config v2 as the Runtime's native config.

Proves the legacy fold happens exactly once at the constructor boundary,
typed models win over legacy fields, the legacy mirror stays consistent
for pre-DF-18 readers, and runtime.py carries no legacy reads / bare
``seekdb`` locals (§15.5/§15.6).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from rosclaw.config import (
    DarwinConfig,
    EvolutionConfig,
    KnowledgeConfig,
    StorageConfig,
    load_typed_configs,
)
from rosclaw.core.runtime import RuntimeConfig

RUNTIME_PY = Path(__file__).resolve().parents[2] / "src" / "rosclaw" / "core" / "runtime.py"


# -- models ------------------------------------------------------------


def test_storage_model_defaults_keep_pre_v2_path():
    cfg = StorageConfig()
    assert cfg.structured.backend == "sqlite"
    assert cfg.structured.path.endswith("data/memory/knowledge.sqlite")  # §16: no path move
    assert cfg.retrieval.enabled is False
    assert cfg.outbox.batch_size == 100
    assert cfg.artifacts.backend == "filesystem"


def test_models_from_dict():
    storage = StorageConfig.from_dict(
        {
            "structured": {"backend": "mysql", "dsn": "mysql://root@h:2881/db"},
            "retrieval": {"enabled": True, "port": 2882},
            "outbox": {"enabled": True, "batch_size": 50},
        }
    )
    assert storage.structured.backend == "mysql"
    assert storage.structured.dsn == "mysql://root@h:2881/db"
    assert storage.retrieval.enabled is True and storage.retrieval.port == 2882
    assert storage.outbox.enabled is True and storage.outbox.batch_size == 50
    assert KnowledgeConfig.from_dict({"enabled": False, "mode": "service"}).mode == "service"
    assert EvolutionConfig.from_dict({"trigger_failure_threshold": 5}).trigger_failure_threshold == 5
    assert DarwinConfig.from_dict({"enabled": True, "seeds": [9], "episodes": 3}).seeds == [9]


# -- §15.4 legacy fold (once, at the boundary) -------------------------


def test_legacy_flat_seekdb_fields_fold_into_typed():
    cfg = RuntimeConfig(
        seekdb_backend="mysql",
        seekdb_url="mysql://root@127.0.0.1:2881/rosclaw",
        seekdb_path="/tmp/k.sqlite",
    )
    assert cfg.storage.structured.backend == "mysql"
    assert cfg.storage.structured.dsn == "mysql://root@127.0.0.1:2881/rosclaw"
    assert cfg.storage.structured.path == "/tmp/k.sqlite"


def test_legacy_dict_storage_folds():
    cfg = RuntimeConfig(
        storage={
            "pool_size": 8,
            "vector_enabled": True,
            "outbox_enabled": True,
            "outbox_max_records": 5000,
            "outbox_flush_interval_sec": 1.5,
        }
    )
    assert cfg.storage.structured.pool_size == 8
    assert cfg.storage.retrieval.enabled is True
    assert cfg.storage.outbox.enabled is True
    assert cfg.storage.outbox.max_records == 5000
    assert cfg.storage.outbox.flush_interval_sec == 1.5


def test_legacy_enable_flags_fold():
    cfg = RuntimeConfig(enable_auto=False, enable_darwin=True, enable_knowledge=False)
    assert cfg.evolution.enabled is False
    assert cfg.darwin.enabled is True
    assert cfg.knowledge.enabled is False


def test_legacy_darwin_dict_folds():
    cfg = RuntimeConfig(darwin={"seeds": [7, 8], "episodes": 9})
    assert cfg.darwin.seeds == [7, 8] and cfg.darwin.episodes == 9


def test_typed_config_wins_over_legacy():
    typed = StorageConfig.from_dict({"structured": {"backend": "memory"}})
    cfg = RuntimeConfig(storage=typed, seekdb_backend="mysql")
    assert cfg.storage.structured.backend == "memory"


# -- mirror-back (pre-DF-18 readers stay consistent) --------------------


def test_legacy_fields_mirror_typed():
    cfg = RuntimeConfig(
        storage=StorageConfig.from_dict({"structured": {"backend": "mysql"}}),
        darwin=DarwinConfig(enabled=True),
        evolution=EvolutionConfig(enabled=False),
    )
    assert cfg.seekdb_backend == "mysql"
    assert cfg.enable_darwin is True
    assert cfg.enable_auto is False


def test_defaults_unchanged_for_plain_construction():
    cfg = RuntimeConfig()
    assert cfg.storage.structured.backend == "sqlite"
    assert cfg.evolution.enabled is True
    assert cfg.darwin.enabled is False
    assert cfg.knowledge.enabled is True
    assert cfg.knowledge.mode == "disabled"


# -- §15.5/§15.6 runtime source discipline -------------------------------


def test_runtime_has_no_legacy_config_reads_or_bare_seekdb():
    src = RUNTIME_PY.read_text(encoding="utf-8")
    assert not re.search(r"^\s*seekdb\s*=", src, re.MULTILINE), "bare `seekdb =` local (§15.6)"
    body = src.split("def __post_init__", 1)[-1]  # field defaults keep legacy names as INPUTS
    for forbidden in (
        "self.config.seekdb_backend",
        "self.config.seekdb_url",
        "self.config.enable_auto",
        "self.config.enable_darwin",
        "self.config.know_store_mode",
        "self.config.storage.get(",
        "self.config.darwin.get(",
    ):
        assert forbidden not in body, f"legacy read {forbidden} survived (§15.5)"


# -- loader ---------------------------------------------------------------


def test_load_typed_configs_from_legacy_yaml(tmp_path):
    home = tmp_path / "home"
    (home / "config").mkdir(parents=True)
    (home / "config" / "rosclaw.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "workspace": {"home": str(home)},
                "runtime": {"seekdb_backend": "mysql", "seekdb_url": "mysql://root@h:2881/db"},
                "storage": {"vector_enabled": True},
                "auto": {"enabled": True, "trigger_failure_threshold": 5},
            }
        ),
        encoding="utf-8",
    )
    storage, knowledge, evolution, darwin = load_typed_configs(home)
    assert storage.structured.backend == "mysql"
    assert storage.structured.dsn == "mysql://root@h:2881/db"
    assert storage.retrieval.enabled is True
    assert evolution.trigger_failure_threshold == 5
    assert knowledge.enabled is True
    assert darwin.episodes == 50


def test_load_typed_configs_missing_file_is_defaults(tmp_path):
    storage, knowledge, evolution, darwin = load_typed_configs(tmp_path / "nope")
    assert storage.structured.backend == "sqlite"
    assert evolution.trigger_failure_threshold == 3
    assert darwin.enabled is False
