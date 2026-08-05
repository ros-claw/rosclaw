"""Basic chat operator-surface lifecycle tests."""

from argparse import Namespace
from types import SimpleNamespace

import pytest

from rosclaw.agentd.cli import _chat_repl


@pytest.mark.asyncio
async def test_basic_chat_starts_operator_projection_before_input(monkeypatch) -> None:
    calls: list[str] = []

    class Service:
        def create_mission(self, goal: str, *, mode: str):
            assert goal == "operator socket smoke test"
            assert mode == "REAL"
            return SimpleNamespace(mission_id="mis_chat_socket", mode=SimpleNamespace(value="REAL"))

        async def start_operator_socket(self) -> None:
            calls.append("start")

        async def close(self) -> None:
            calls.append("close")

    monkeypatch.setattr("builtins.input", lambda _prompt: "/quit")
    args = Namespace(
        mission=None,
        goal="operator socket smoke test",
        mode="REAL",
    )

    result = await _chat_repl(Service(), args)

    assert result == 0
    assert calls == ["start", "close"]
