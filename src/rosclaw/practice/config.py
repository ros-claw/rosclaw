"""Configuration and session objects for rosclaw-practice."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rosclaw.firstboot.workspace import get_rosclaw_home

DEFAULT_DATA_ROOT = "/data/rosclaw/practice"
DEFAULT_FALLBACK_DIR = "/data/rosclaw/practice/fallback"
DEFAULT_INDEX_DIR = "/data/rosclaw/practice/indexes"
DEFAULT_CONFIG_ROOT = get_rosclaw_home() / "practice"


@dataclass
class RecorderConfig:
    """Writer settings for a practice session."""

    jsonl_enabled: bool = True
    jsonl_rotate_mb: float = 512.0

    mcap_enabled: bool = False
    mcap_compression: str = "zstd"
    mcap_chunk_size_bytes: int = 4 * 1024 * 1024

    frames_enabled: bool = False
    rgb_format: str = "jpg"
    depth_format: str = "png16"


@dataclass
class SeekDBConfig:
    """SeekDB integration settings."""

    enabled: bool = False
    url: str | None = field(
        default_factory=lambda: os.environ.get("ROSCLAW_SEEKDB_URL")
    )
    fallback_dir: str = field(
        default_factory=lambda: os.environ.get(
            "ROSCLAW_SEEKDB_FALLBACK_DIR", DEFAULT_FALLBACK_DIR
        )
    )
    table: str = "praxis_events"
    timeout_sec: float = 2.0


@dataclass
class SourceConfig:
    """Which data sources to record."""

    dds: bool = False
    ros2: bool = False
    camera: bool = False
    agent: bool = True
    provider: bool = False
    sandbox: bool = False
    runtime: bool = True
    human: bool = False


@dataclass
class PracticeConfig:
    """Top-level configuration for a PracticeCoordinator."""

    robot_id: str = "default_robot"
    robot_type: str | None = None
    task_id: str | None = None
    task_name: str | None = None
    skill_id: str | None = None
    session_name: str | None = None

    data_root: str = DEFAULT_DATA_ROOT
    config_root: Path = field(default_factory=lambda: get_rosclaw_home() / "practice")
    sources: SourceConfig = field(default_factory=SourceConfig)
    recorder: RecorderConfig = field(default_factory=RecorderConfig)
    seekdb: SeekDBConfig = field(default_factory=SeekDBConfig)

    # Runtime knobs
    mock: bool = False
    duration_sec: float | None = None
    publish_to_event_bus: bool = True

    # Optional pre-built objects (used by Runtime and tests)
    event_bus: Any | None = None
    seekdb_bridge: Any | None = None

    @property
    def data_root_path(self) -> Path:
        return Path(self.data_root)

    @property
    def sessions_dir(self) -> Path:
        return self.data_root_path / "sessions"

    @property
    def indexes_dir(self) -> Path:
        return self.data_root_path / "indexes"

    @property
    def fallback_dir(self) -> Path:
        return Path(self.seekdb.fallback_dir)


@dataclass
class PracticeSession:
    """Live practice session handle."""

    practice_id: str
    robot_id: str
    task_id: str | None
    task_name: str | None
    skill_id: str | None
    session_dir: Path
    start_time_ns: int
    start_time_utc: str
    robot_type: str | None = None
    session_id: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PracticeSummary:
    """Result returned when a practice session stops."""

    practice_id: str
    robot_id: str
    outcome: str = "UNKNOWN"
    reward: float | None = None
    duration_ms: float | None = None
    event_count: int = 0
    artifact_dir: Path | None = None
    seekdb_committed: bool | None = None
    failure_labels: list[str] = field(default_factory=list)
