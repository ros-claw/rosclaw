from __future__ import annotations

from rosclaw.knowledge.how_client import DisabledHowClient, HttpHowClient
from rosclaw.knowledge.know_client import DisabledKnowClient, HttpKnowClient
from rosclaw.knowledge.service_manager import KnowledgeServiceConfig, KnowledgeServiceManager


def test_disabled_mode_has_truthful_health():
    manager = KnowledgeServiceManager(KnowledgeServiceConfig(mode="disabled"))
    health = manager.health()
    assert health["status"] == "disabled"
    assert health["memory_boundary"] == "isolated"
    assert isinstance(manager.know, DisabledKnowClient)
    assert isinstance(manager.how, DisabledHowClient)


def test_service_mode_builds_clients_without_network_call():
    manager = KnowledgeServiceManager(
        KnowledgeServiceConfig(
            mode="service",
            know_url="http://know.test:8087",
            how_url="http://how.test:8088",
        )
    )
    assert isinstance(manager.know, HttpKnowClient)
    assert isinstance(manager.how, HttpHowClient)


def test_missing_service_url_degrades_without_legacy_fallback():
    manager = KnowledgeServiceManager(KnowledgeServiceConfig(mode="service"))
    health = manager.health()
    assert health["status"] == "degraded"
    assert isinstance(manager.know, DisabledKnowClient)
    assert isinstance(manager.how, DisabledHowClient)
    assert "requires" in health["startup_error"]


def test_inprocess_server_uses_server_coordinates_not_embedded_path(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_create_know_store(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr("rosclaw_know.store.create_know_store", fake_create_know_store)
    manager = KnowledgeServiceManager(
        KnowledgeServiceConfig(
            mode="inprocess",
            know_store_mode="server",
            know_store_path="/must/not/reach/server",
            seekdb_host="seekdb.internal",
            seekdb_port=2881,
            seekdb_tenant="tenant-a",
            seekdb_user="know-reader",
            seekdb_password="fixture-secret",
            know_database="rosclaw_know",
            memory_database="rosclaw_memory",
            practice_database="rosclaw_practice",
        ),
        how_client=DisabledHowClient(),
    )
    assert manager.startup_error is None
    assert captured["mode"] == "server"
    assert captured["host"] == "seekdb.internal"
    assert captured["database"] == "rosclaw_know"
    assert captured["memory_database"] == "rosclaw_memory"
    assert "path" not in captured
