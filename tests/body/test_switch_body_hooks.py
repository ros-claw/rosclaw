"""Tests for body switch runtime hooks."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.body.hooks import BodyHookEvent, BodySwitchHooks, get_default_hooks
from rosclaw.body.service import BodyInstanceService


@pytest.fixture
def multi_body_workspace(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    BodyInstanceService(workspace=tmp_path).create_or_init(
        robot="unitree-g1", name="g1-a", mode="registry", update_registry=True, switch_active=True
    )
    BodyInstanceService(workspace=tmp_path).create_or_init(
        robot="unitree-g1", name="g1-b", mode="registry", update_registry=True
    )
    return tmp_path


def test_default_hooks_dispatch_without_failure(multi_body_workspace: Path):
    hooks = BodySwitchHooks()
    result = hooks.on_active_body_switched(
        workspace=multi_body_workspace,
        body_instance_id="g1-b",
        reason="test switch",
    )
    assert result["success"] is True
    assert result["failures"] == []


def test_subscriber_receives_event(multi_body_workspace: Path):
    hooks = BodySwitchHooks()
    received: list[tuple[str, str]] = []

    def callback(event_type: str, context: object) -> None:
        received.append((event_type, context.body_instance_id))

    hooks.subscribe(BodyHookEvent.BODY_ACTIVE_SWITCHED, callback)
    hooks.on_active_body_switched(
        workspace=multi_body_workspace,
        body_instance_id="g1-b",
        reason="test switch",
    )

    assert len(received) == 1
    assert received[0][0] == BodyHookEvent.BODY_ACTIVE_SWITCHED
    assert received[0][1] == "g1-b"


def test_failing_subscriber_does_not_block_in_non_strict_mode(multi_body_workspace: Path):
    hooks = BodySwitchHooks()

    def failing_callback(event_type: str, context: object) -> None:
        raise RuntimeError("hook failure")

    hooks.subscribe(BodyHookEvent.BODY_ACTIVE_SWITCHED, failing_callback)
    result = hooks.on_active_body_switched(
        workspace=multi_body_workspace,
        body_instance_id="g1-b",
        strict=False,
    )

    assert result["success"] is False
    assert len(result["failures"]) == 1
    assert "hook failure" in result["failures"][0]


def test_failing_subscriber_blocks_in_strict_mode(multi_body_workspace: Path):
    hooks = BodySwitchHooks()

    def failing_callback(event_type: str, context: object) -> None:
        raise RuntimeError("strict hook failure")

    hooks.subscribe(BodyHookEvent.BODY_ACTIVE_SWITCHED, failing_callback)
    with pytest.raises(RuntimeError, match="strict hook failure"):
        hooks.on_active_body_switched(
            workspace=multi_body_workspace,
            body_instance_id="g1-b",
            strict=True,
        )


def test_global_hook_instance_is_singleton():
    h1 = get_default_hooks()
    h2 = get_default_hooks()
    assert h1 is h2
