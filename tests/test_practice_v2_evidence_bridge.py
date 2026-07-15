"""Cross-module evidence bridge tests for Practice v2 records."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from rosclaw.cli import cmd_practice_record
from rosclaw.how.engine import HeuristicEngine
from rosclaw.know.interface import KnowledgeInterface
from rosclaw.memory.interface import MemoryInterface
from rosclaw.memory.seekdb_client import InMemoryKnowledgeStore

FIXTURE = Path(__file__).parent / "fixtures" / "practice" / "rh56_minimal_loop.json"
EPISODE_ID = "episode_rh56_minimal_loop"


def _record_practice_v2_fixture(data_root: Path) -> None:
    args = SimpleNamespace(
        fixture=str(FIXTURE),
        out=str(data_root),
        data_root=None,
        json=True,
    )
    assert cmd_practice_record(args) == 0


def test_memory_ingest_consumes_practice_v2_episode(tmp_path, capsys):
    data_root = tmp_path / "practice"
    _record_practice_v2_fixture(data_root)
    capsys.readouterr()

    client = InMemoryKnowledgeStore()
    memory = MemoryInterface(robot_id="rh56", seekdb_client=client)
    memory._do_initialize()

    result = memory.ingest_episode(EPISODE_ID, data_root=str(data_root))

    assert result["status"] == "success"
    assert result["event_count"] == 9
    assert result["outcome"] == "success"

    experiences = client.query("experience_graph")
    assert len(experiences) == 1
    assert experiences[0]["event_type"] == "practice_episode"
    assert "skill_ok_contact" in experiences[0]["instruction"]

    metadata = experiences[0]["metadata"]
    assert metadata["episode"]["practice_id"] == "practice_rh56_minimal_loop"
    assert metadata["episode"]["robot_id"] == "rh56"
    assert len(metadata["events"]) == 9


def test_know_compile_consumes_practice_v2_episode(tmp_path, capsys):
    data_root = tmp_path / "practice"
    _record_practice_v2_fixture(data_root)
    capsys.readouterr()

    know = KnowledgeInterface(robot_id="rh56")
    know._do_initialize()

    card = know.compile_task_card(
        "recover from over contact",
        episode_id=EPISODE_ID,
        data_root=str(data_root),
    )

    assert card["episode_id"] == EPISODE_ID
    assert card["robot_id"] == "rh56"
    assert str(card["outcome"]).lower() == "success"
    assert card["evidence"]["event_count"] == 9
    assert {"runtime", "sandbox"}.issubset(set(card["evidence"]["sources"]))


def test_how_advise_consumes_practice_v2_how_intervention(tmp_path, capsys):
    data_root = tmp_path / "practice"
    _record_practice_v2_fixture(data_root)
    capsys.readouterr()

    engine = HeuristicEngine(seekdb_client=InMemoryKnowledgeStore())
    result = asyncio.run(
        engine.advise(
            body_id="body_rh56_left",
            failure="over_contact",
            episode_id=EPISODE_ID,
            data_root=str(data_root),
        )
    )

    assert result["evidence"]["event_count"] == 9
    assert {"runtime", "sandbox"}.issubset(set(result["evidence"]["sources"]))
    assert result["intervention"]["rule_id"] == "how_rh56_001"
    assert result["intervention"]["source"] == "practice_how_intervention"
    assert "back off thumb target" in result["intervention"]["action"]
    assert result["intervention"]["action_taken"]["thumb_target_delta"] == -60.0
