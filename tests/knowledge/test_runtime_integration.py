from __future__ import annotations

from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.knowledge.how_client import HttpHowClient
from rosclaw.knowledge.know_client import HttpKnowClient


def test_runtime_v2_service_does_not_reuse_memory_store():
    runtime = Runtime(
        RuntimeConfig(
            robot_id="fixture",
            enable_firewall=False,
            enable_memory=True,
            enable_practice=False,
            enable_swarm=False,
            enable_skill_manager=False,
            enable_knowledge=True,
            enable_how=True,
            enable_auto=False,
            enable_provider=False,
            enable_sense=False,
            enable_event_persistence=False,
            enable_tracing=False,
            seekdb_backend="memory",
            knowledge_v2_mode="service",
            know_url="http://know.test:8087",
            how_url="http://how.test:8088",
        )
    )
    runtime.initialize()
    assert runtime.memory is not None
    assert runtime.knowledge_v2 is not None
    assert runtime._knowledge is None
    assert runtime._how is None
    assert isinstance(runtime._knowledge_v2_manager.know, HttpKnowClient)
    assert isinstance(runtime._knowledge_v2_manager.how, HttpHowClient)
    assert runtime._knowledge_v2_manager.know is not runtime._memory.seekdb_client
    runtime.stop()
