"""Protocol tests for the ROSClaw interaction coordinator."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from rosclaw.interaction import InteractionClient, InteractionCoordinator
from rosclaw.interaction.capability_detection import detect_interaction_capabilities


class FakeBackend:
    def __init__(self) -> None:
        self.confirmed = 0
        self.cancelled: list[str] = []

    async def confirm_operator_action(self, prepared: Any, **kwargs: Any) -> dict[str, Any]:
        self.confirmed += 1
        assert kwargs["principal_id"].startswith("mcp:")
        return {
            "state": "QUEUED",
            "action_id": "action-1",
            "permit_id": "must-not-leak",
            "operator_confirmation": {"permit_injected": True},
        }

    async def cancel_action(self, action_id: str) -> None:
        self.cancelled.append(action_id)


class PendingBackend(FakeBackend):
    async def defer_operator_action(self, prepared: Any, **kwargs: Any) -> dict[str, Any]:
        assert prepared is not None
        assert kwargs["ttl_sec"] == 60.0
        return {
            "proposal": {
                "request_id": "proposal-1",
                "state": "CREATED",
                "action_intent_hash": "sha256:must-not-leak",
            },
            "permit_exposed": False,
        }


class FakeContext:
    request_id = "request-1"

    def __init__(self, *, action: str = "accept", form: bool = True, url: bool = False) -> None:
        self.action = action
        self.messages: list[str] = []
        self.progress: list[float] = []
        capabilities = SimpleNamespace(
            elicitation=SimpleNamespace(
                form=SimpleNamespace() if form else None,
                url=SimpleNamespace() if url else None,
            ),
            tasks=None,
        )
        self.client_params = SimpleNamespace(capabilities=capabilities)
        self.request_context = SimpleNamespace(session=self)

    async def elicit_form(self, **kwargs: Any) -> Any:
        self.messages.append(kwargs["message"])
        return SimpleNamespace(action=self.action)

    async def report_progress(self, *, progress: float, **kwargs: Any) -> None:
        self.progress.append(progress)


def _approval() -> dict[str, Any]:
    return {
        "action_id": "action-1",
        "action_intent_hash": "sha256:private",
        "risk_class": "HIGH",
        "display": {
            "title": "Move robot",
            "summary": "Move to one waypoint.",
            "target_pose": {"x": 1.0, "y": 0.0},
            "physical_effect": "The base moves.",
        },
    }


def test_capability_detection_uses_negotiated_capabilities() -> None:
    caps = detect_interaction_capabilities(FakeContext(form=True, url=True))
    assert caps.form_elicitation is True
    assert caps.url_elicitation is True
    assert caps.progress is True


@pytest.mark.asyncio
async def test_accept_dispatches_and_redacts_internal_control_plane_fields() -> None:
    backend = FakeBackend()
    context = FakeContext()
    result = await InteractionCoordinator(InteractionClient(backend)).complete_prepared(
        context,
        prepared_action=object(),
        approval_request=_approval(),
    )
    assert result["state"] == "QUEUED"
    assert result["interaction"]["decision"] == "CONFIRMED"
    assert "permit" not in str(result).lower()
    assert "sha256:private" not in str(result)
    assert backend.confirmed == 1
    assert context.progress == [0.1, 0.5, 1.0]
    assert "Move robot" in context.messages[0]


@pytest.mark.asyncio
async def test_decline_and_missing_channel_never_dispatch() -> None:
    backend = FakeBackend()
    coordinator = InteractionCoordinator(InteractionClient(backend))
    declined = await coordinator.complete_prepared(
        FakeContext(action="decline"),
        prepared_action=object(),
        approval_request=_approval(),
    )
    unavailable = await coordinator.complete_prepared(
        FakeContext(form=False),
        prepared_action=object(),
        approval_request=_approval(),
    )
    assert declined["decision"] == "CANCELLED"
    assert unavailable["decision"] == "APPROVAL_CHANNEL_UNAVAILABLE"
    assert backend.confirmed == 0


@pytest.mark.asyncio
async def test_url_capability_returns_pending_without_exposing_approval_request() -> None:
    result = await InteractionCoordinator(InteractionClient(PendingBackend())).complete_prepared(
        FakeContext(form=False, url=True),
        prepared_action=object(),
        approval_request=_approval(),
    )
    assert result["decision"] == "APPROVAL_PENDING"
    assert "action_intent_hash" not in str(result)
