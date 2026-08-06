from __future__ import annotations

from rosclaw.knowledge.agent_tools import KnowledgeAgentTools
from rosclaw.knowledge.facade import KnowledgeFacade
from rosclaw.knowledge.service_manager import KnowledgeServiceConfig, KnowledgeServiceManager

from .conftest import FakeHow, FakeKnow


def test_agent_tools_are_read_only_or_advisory(reference_pack):
    know = FakeKnow(reference_pack)
    manager = KnowledgeServiceManager(
        KnowledgeServiceConfig(mode="disabled"),
        know_client=know,
        how_client=FakeHow(reference_pack, know),
    )
    tools = KnowledgeAgentTools(KnowledgeFacade(manager))
    definitions = tools.definitions()
    assert definitions
    assert all(item["read_only"] or item["advisory"] for item in definitions)
    text = repr(definitions).casefold()
    assert "permit" not in text
    assert "shell" not in text
    assert "execute robot" not in text


def test_reference_pack_progressive_open(reference_pack):
    know = FakeKnow(reference_pack)
    manager = KnowledgeServiceManager(
        KnowledgeServiceConfig(mode="disabled"),
        know_client=know,
        how_client=FakeHow(reference_pack, know),
    )
    tools = KnowledgeAgentTools(KnowledgeFacade(manager))
    result = tools.call(
        "rosclaw_know_build_reference_pack",
        {"query": "camera error", "context": {"task": "diagnose"}, "top_k": 3},
    )
    assert result["reference_pack_id"] == "pack_1"
    opened = tools.call("rosclaw_know_open_reference_pack", {"reference_pack_id": "pack_1"})
    assert opened["items"][0]["evidence_refs"][0]["snapshot_id"] == "snap_commit_abc"
