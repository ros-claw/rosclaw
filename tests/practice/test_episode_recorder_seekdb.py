"""Integration tests for EpisodeRecorder -> SeekDBBridge forwarding."""

import sys
import types
from unittest.mock import MagicMock

import pytest

from rosclaw.core.event_bus import Event, EventBus
from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.practice.episode_recorder import EpisodeRecorder


def _publish_praxis_completed(bus: EventBus, episode_id: str, instruction: str = "pick the cup") -> None:
    bus.publish(
        Event(
            topic="agent.command",
            payload={"episode_id": episode_id, "instruction": instruction},
            source="test",
        )
    )
    bus.publish(
        Event(
            topic="praxis.completed",
            payload={"episode_id": episode_id, "outcome": {"reward": 1.0}},
            source="test",
        )
    )


class TestEpisodeRecorderSeekDBForwarding:
    def test_commit_called_when_bridge_configured(self, tmp_path):
        bus = EventBus()
        bridge = MagicMock()
        recorder = EpisodeRecorder(
            robot_id="r1",
            event_bus=bus,
            artifact_base_dir=str(tmp_path),
            seekdb_bridge=bridge,
        )
        recorder.initialize()

        episode_id = "ep_test_001"
        _publish_praxis_completed(bus, episode_id)

        bridge.commit.assert_called_once()
        event = bridge.commit.call_args.args[0]
        assert event.event_id == episode_id
        assert event.event_type == "success"
        assert event.metadata["reward"] == 1.0
        assert event.robot_id == "r1"

        recorder.stop()

    def test_commit_failure_is_non_fatal(self, tmp_path):
        bus = EventBus()
        recorded = []
        bus.subscribe("praxis.recorded", recorded.append)

        bridge = MagicMock()
        bridge.commit.side_effect = RuntimeError("SeekDB unreachable")
        recorder = EpisodeRecorder(
            robot_id="r1",
            event_bus=bus,
            artifact_base_dir=str(tmp_path),
            seekdb_bridge=bridge,
        )
        recorder.initialize()

        episode_id = "ep_err_001"
        _publish_praxis_completed(bus, episode_id)

        bridge.commit.assert_called_once()
        assert len(recorded) == 1
        assert (tmp_path / "episodes" / episode_id / "metadata.json").exists()

        recorder.stop()

    def test_no_commit_when_bridge_absent(self, tmp_path):
        bus = EventBus()
        recorded = []
        bus.subscribe("praxis.recorded", recorded.append)

        recorder = EpisodeRecorder(
            robot_id="r1",
            event_bus=bus,
            artifact_base_dir=str(tmp_path),
            seekdb_bridge=None,
        )
        recorder.initialize()

        episode_id = "ep_no_bridge_001"
        _publish_praxis_completed(bus, episode_id)

        assert len(recorded) == 1
        assert (tmp_path / "episodes" / episode_id / "metadata.json").exists()

        recorder.stop()


class TestRuntimeConfigSeekDB:
    def test_runtime_config_reads_seekdb_env_vars(self, monkeypatch):
        monkeypatch.setenv("ROSCLAW_SEEKDB_URL", "http://seekdb.example:2881")
        monkeypatch.setenv("ROSCLAW_SEEKDB_FALLBACK_DIR", "/tmp/seekdb_fallback")

        config = RuntimeConfig()
        assert config.seekdb_url == "http://seekdb.example:2881"
        assert config.seekdb_fallback_dir == "/tmp/seekdb_fallback"

    def test_runtime_config_seekdb_defaults(self, monkeypatch):
        monkeypatch.delenv("ROSCLAW_SEEKDB_URL", raising=False)
        monkeypatch.delenv("ROSCLAW_SEEKDB_FALLBACK_DIR", raising=False)

        config = RuntimeConfig()
        assert config.seekdb_url is None
        assert config.seekdb_fallback_dir == "/data/rosclaw/fallback"


class TestRuntimeSeekDBAssembly:
    def test_runtime_assembles_seekdb_bridge_when_url_configured(self, tmp_path, monkeypatch):
        pytest.importorskip("rosclaw_practice")
        monkeypatch.setenv("ROSCLAW_SEEKDB_URL", "http://localhost:2881")

        config = RuntimeConfig(
            robot_id="test_bot",
            enable_firewall=False,
            enable_memory=False,
            enable_practice=True,
            enable_skill_manager=False,
            enable_knowledge=False,
            enable_how=False,
            enable_auto=False,
            enable_provider=False,
            enable_sense=False,
            seekdb_fallback_dir=str(tmp_path / "fallback"),
        )
        runtime = Runtime(config)
        runtime.initialize()

        assert runtime._episode_recorder is not None
        assert runtime._episode_recorder._seekdb_bridge is not None

        runtime.stop()

    def test_runtime_initializes_gracefully_when_rosclaw_practice_missing(self, monkeypatch):
        # Simulate an environment where rosclaw_practice is not installed by
        # injecting a fake seekdb_bridge module whose SeekDBBridge attribute
        # raises ImportError when accessed.
        fake_module = types.ModuleType("rosclaw.practice.seekdb_bridge")

        def _raise_import_error(*_args, **_kwargs):
            raise ImportError("rosclaw-practice is required")

        fake_module.__getattr__ = lambda name: _raise_import_error() if name == "SeekDBBridge" else None
        monkeypatch.delitem(sys.modules, "rosclaw.practice.seekdb_bridge", raising=False)
        monkeypatch.setitem(sys.modules, "rosclaw.practice.seekdb_bridge", fake_module)
        monkeypatch.setenv("ROSCLAW_SEEKDB_URL", "http://localhost:2881")

        config = RuntimeConfig(
            robot_id="test_bot",
            enable_firewall=False,
            enable_memory=False,
            enable_practice=True,
            enable_skill_manager=False,
            enable_knowledge=False,
            enable_how=False,
            enable_auto=False,
            enable_provider=False,
            enable_sense=False,
        )
        runtime = Runtime(config)
        runtime.initialize()

        assert runtime._episode_recorder is not None
        assert runtime._episode_recorder._seekdb_bridge is None

        runtime.stop()
