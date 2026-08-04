"""CLI tests for the trusted local Operator Broker reference client."""

from __future__ import annotations

import json
from typing import Any

from rosclaw.operator import cli


class _FakeDaemonClient:
    decisions: list[dict[str, Any]] = []

    def __init__(self, **_kwargs: Any) -> None:
        pass

    def list_pending_operator_proposals(self) -> dict[str, Any]:
        return {
            "proposals": [
                {
                    "request_id": "proposal-1",
                    "action_id": "action-1",
                    "action_intent_hash": "sha256:exact",
                    "challenge_nonce": "operator-only-challenge",
                    "display": {"title": "Move one finger"},
                }
            ],
            "count": 1,
        }

    def decide_operator_proposal(self, request_id: str, **kwargs: Any) -> dict[str, Any]:
        self.decisions.append({"request_id": request_id, **kwargs})
        return {
            "proposal": {"request_id": request_id, "state": "SUBMITTED"},
            "permit_exposed": False,
            "action": {"action_id": "action-1", "state": "FINISHED"},
        }


def test_pending_output_redacts_decision_challenge(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "DaemonClient", _FakeDaemonClient)

    assert cli.dispatch_operator_argv(["operator", "pending", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["proposals"][0]["request_id"] == "proposal-1"
    assert "challenge_nonce" not in str(payload)
    assert "permit" not in str(payload).lower()


def test_decide_redirects_to_operatord(monkeypatch, capsys) -> None:
    """R1：admin CLI 不再直决——决定只属于 rosclaw-operatord。"""
    _FakeDaemonClient.decisions.clear()
    monkeypatch.setattr(cli, "DaemonClient", _FakeDaemonClient)

    from rosclaw.daemon.client import DaemonClientError

    try:
        cli.dispatch_operator_argv(
            [
                "operator",
                "decide",
                "proposal-1",
                "--accept",
                "--principal-id",
                "operator-shift-a",
                "--reason",
                "Reviewed exact action",
                "--json",
            ]
        )
    except DaemonClientError as exc:
        assert exc.code == "DECISION_CHANNEL_MOVED"
    else:  # 若 CLI 内部捕获，则必须非零退出且不产生任何决定
        assert not _FakeDaemonClient.decisions
    assert not _FakeDaemonClient.decisions, "admin CLI 绝不再直决"
