"""Tests for Runtime wiring of private know/how adapters."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from rosclaw.core.event_bus import Event, EventPriority
from rosclaw.core.runtime import Runtime, RuntimeConfig


class TestRuntimeHowWiring:
    """Runtime picks HowClient when configured and reachable, otherwise local engine."""

    def test_creates_how_client_when_url_healthy(self, monkeypatch):
        config = RuntimeConfig(
            robot_id="test",
            enable_firewall=False,
            enable_memory=False,
            enable_practice=False,
            enable_skill_manager=False,
            enable_provider=False,
            enable_auto=False,
            how_url="http://how:8088",
        )
        fake_client = MagicMock()
        fake_client.initialize = AsyncMock(return_value=None)
        fake_engine = MagicMock()
        fake_engine.initialize = AsyncMock(return_value=None)
        fake_engine.seed_defaults = AsyncMock(return_value=0)
        # Runtime imports these names inside _create_how_engine, so patch the
        # source modules rather than rosclaw.core.runtime.
        monkeypatch.setattr("rosclaw.how.client.HowClient", lambda *a, **kw: fake_client)
        monkeypatch.setattr("rosclaw.how.engine.HeuristicEngine", lambda *a, **kw: fake_engine)

        rt = Runtime(config)
        engine = rt._create_how_engine(None)
        assert engine is fake_client

    def test_falls_back_to_heuristic_engine_when_how_unhealthy(self, monkeypatch):
        config = RuntimeConfig(
            robot_id="test",
            enable_firewall=False,
            enable_memory=False,
            enable_practice=False,
            enable_skill_manager=False,
            enable_provider=False,
            enable_auto=False,
            how_url="http://how:8088",
        )
        fake_client = MagicMock()
        fake_client.initialize = AsyncMock(side_effect=RuntimeError("down"))
        fake_engine = MagicMock()
        fake_engine.initialize = AsyncMock(return_value=None)
        fake_engine.seed_defaults = AsyncMock(return_value=0)
        monkeypatch.setattr("rosclaw.how.client.HowClient", lambda *a, **kw: fake_client)
        monkeypatch.setattr("rosclaw.how.engine.HeuristicEngine", lambda *a, **kw: fake_engine)

        rt = Runtime(config)
        engine = rt._create_how_engine(MagicMock())
        assert engine is fake_engine

    def test_uses_heuristic_engine_when_no_how_url(self, monkeypatch):
        config = RuntimeConfig(
            robot_id="test",
            enable_firewall=False,
            enable_memory=False,
            enable_practice=False,
            enable_skill_manager=False,
            enable_provider=False,
            enable_auto=False,
        )
        fake_engine = MagicMock()
        fake_engine.initialize = AsyncMock(return_value=None)
        fake_engine.seed_defaults = AsyncMock(return_value=0)
        monkeypatch.setattr("rosclaw.how.engine.HeuristicEngine", lambda *a, **kw: fake_engine)

        rt = Runtime(config)
        engine = rt._create_how_engine(MagicMock())
        assert engine is fake_engine


class TestRuntimeKnowledgeWiring:
    """Runtime passes the registry flag into KnowledgeInterface."""

    def test_knowledge_interface_receives_registry_flag(self, monkeypatch):
        config = RuntimeConfig(
            robot_id="test",
            enable_firewall=False,
            enable_memory=False,
            enable_practice=False,
            enable_skill_manager=False,
            enable_provider=False,
            enable_auto=False,
            know_curated_registry_enabled=True,
        )
        captured: dict = {}

        def fake_ki(*args, **kwargs):
            captured["kwargs"] = kwargs
            return MagicMock()

        monkeypatch.setattr("rosclaw.knowledge.legacy.interface.KnowledgeInterface", fake_ki)
        rt = Runtime(config)
        rt.initialize()
        assert captured["kwargs"].get("use_rosclaw_know_registry") is True


class TestRuntimeRecoveryForwardsPlateauTelemetry:
    """Runtime failure handlers must pass previous_scores/current_iteration to HOW."""

    @pytest.fixture
    def capture_recovery_call(self, monkeypatch):
        """Patch RecoveryEngine.generate_recovery_hint to record kwargs."""
        captured: dict = {}

        class FakeRecoveryEngine:
            def __init__(self, *args, **kwargs):
                pass

            async def generate_recovery_hint(self, *args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs
                return None

            def format_for_eventbus(self, *args, **kwargs):
                return {}

        monkeypatch.setattr("rosclaw.how.recovery.RecoveryEngine", FakeRecoveryEngine)
        return captured

    def _runtime_with_how(self, monkeypatch, fake_how):
        config = RuntimeConfig(
            robot_id="test",
            enable_firewall=False,
            enable_memory=True,
            enable_practice=False,
            enable_skill_manager=False,
            enable_provider=False,
            enable_auto=False,
            seekdb_backend="memory",
        )
        monkeypatch.setattr("rosclaw.how.engine.HeuristicEngine", lambda *a, **kw: fake_how)
        rt = Runtime(config)
        rt.initialize()
        return rt

    def test_sandbox_episode_failure_forwards_scores(self, monkeypatch, capture_recovery_call):
        fake_how = MagicMock()
        fake_how.suggest_recovery = AsyncMock(return_value=None)
        fake_how.initialize = AsyncMock(return_value=None)
        fake_how.seed_defaults = AsyncMock(return_value=0)
        rt = self._runtime_with_how(monkeypatch, fake_how)

        rt.event_bus.publish(
            Event(
                topic="rosclaw.sandbox.episode.failed",
                payload={
                    "failure_type": "velocity divergence",
                    "request_id": "r1",
                    "previous_scores": [0.5, 0.5, 0.5, 0.5],
                    "current_iteration": 10,
                },
                source="test",
                priority=EventPriority.HIGH,
            )
        )

        kwargs = capture_recovery_call["kwargs"]
        assert kwargs.get("previous_scores") == [0.5, 0.5, 0.5, 0.5]
        assert kwargs.get("current_iteration") == 10
        assert kwargs.get("request_id") == "r1"

    def test_sandbox_action_blocked_forwards_scores(self, monkeypatch, capture_recovery_call):
        fake_how = MagicMock()
        fake_how.suggest_recovery = AsyncMock(return_value=None)
        fake_how.initialize = AsyncMock(return_value=None)
        fake_how.seed_defaults = AsyncMock(return_value=0)
        rt = self._runtime_with_how(monkeypatch, fake_how)

        rt.event_bus.publish(
            Event(
                topic="rosclaw.sandbox.action.blocked",
                payload={
                    "reason": "joint limit exceeded",
                    "request_id": "r2",
                    "previous_scores": [0.6, 0.6, 0.55],
                    "current_iteration": 7,
                },
                source="test",
                priority=EventPriority.HIGH,
            )
        )

        kwargs = capture_recovery_call["kwargs"]
        assert kwargs.get("previous_scores") == [0.6, 0.6, 0.55]
        assert kwargs.get("current_iteration") == 7

    def test_runtime_execution_failed_forwards_scores(self, monkeypatch, capture_recovery_call):
        fake_how = MagicMock()
        fake_how.suggest_recovery = AsyncMock(return_value=None)
        fake_how.initialize = AsyncMock(return_value=None)
        fake_how.seed_defaults = AsyncMock(return_value=0)
        rt = self._runtime_with_how(monkeypatch, fake_how)

        rt.event_bus.publish(
            Event(
                topic="rosclaw.runtime.execution.failed",
                payload={
                    "error_type": "torque overflow",
                    "request_id": "r3",
                    "previous_scores": [0.4, 0.4, 0.4, 0.4],
                    "current_iteration": 12,
                },
                source="test",
                priority=EventPriority.HIGH,
            )
        )

        kwargs = capture_recovery_call["kwargs"]
        assert kwargs.get("previous_scores") == [0.4, 0.4, 0.4, 0.4]
        assert kwargs.get("current_iteration") == 12

    def test_missing_scores_defaults_to_none(self, monkeypatch, capture_recovery_call):
        fake_how = MagicMock()
        fake_how.suggest_recovery = AsyncMock(return_value=None)
        fake_how.initialize = AsyncMock(return_value=None)
        fake_how.seed_defaults = AsyncMock(return_value=0)
        rt = self._runtime_with_how(monkeypatch, fake_how)

        rt.event_bus.publish(
            Event(
                topic="rosclaw.sandbox.episode.failed",
                payload={"failure_type": "collision detected", "request_id": "r4"},
                source="test",
                priority=EventPriority.HIGH,
            )
        )

        kwargs = capture_recovery_call["kwargs"]
        assert kwargs.get("previous_scores") is None
        assert kwargs.get("current_iteration") is None
