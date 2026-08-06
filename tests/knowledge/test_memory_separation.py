from __future__ import annotations

from rosclaw.knowledge.service_manager import KnowledgeServiceConfig, KnowledgeServiceManager

from .conftest import FakeHow, FakeKnow


def test_manager_never_accepts_or_exposes_memory_store(reference_pack):
    memory_store = object()
    know = FakeKnow(reference_pack)
    manager = KnowledgeServiceManager(
        KnowledgeServiceConfig(mode="disabled"),
        know_client=know,
        how_client=FakeHow(reference_pack, know),
    )
    assert manager.know is know
    assert manager.know is not memory_store
    assert manager.health()["memory_boundary"] == "isolated"
    assert "memory" not in vars(manager)
