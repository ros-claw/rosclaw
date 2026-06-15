"""Tests for Runtime wiring of private know/how adapters."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

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

        monkeypatch.setattr("rosclaw.know.interface.KnowledgeInterface", fake_ki)
        rt = Runtime(config)
        rt.initialize()
        assert captured["kwargs"].get("use_rosclaw_know_registry") is True
