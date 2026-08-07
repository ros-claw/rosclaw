from __future__ import annotations

from rosclaw.knowledge.agent_tools import KnowledgeAgentTools
from rosclaw.knowledge.facade import KnowledgeFacade
from rosclaw.knowledge.mcp_tools import KnowledgeMCPTools
from rosclaw.knowledge.service_manager import KnowledgeServiceConfig, KnowledgeServiceManager

from .conftest import FakeHow, FakeKnow


async def test_mcp_tools_delegate_without_action_channel(reference_pack):
    know = FakeKnow(reference_pack)
    manager = KnowledgeServiceManager(
        KnowledgeServiceConfig(mode="disabled"),
        know_client=know,
        how_client=FakeHow(reference_pack, know),
    )
    tools = KnowledgeMCPTools(KnowledgeAgentTools(KnowledgeFacade(manager)))
    result = await tools.call("rosclaw_know_open_reference_pack", {"reference_pack_id": "pack_1"})
    assert result["reference_pack_id"] == "pack_1"
    assert all(spec["read_only"] or spec["advisory"] for spec in tools.specs())
