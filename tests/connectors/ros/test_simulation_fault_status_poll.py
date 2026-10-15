"""Synthetic fault binding contracts; no action or physical receipt fabricated."""

import importlib
from pathlib import Path

import pytest

from rosclaw.mcp.schemas.common import MCPError


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("simulation_fault_status_poll")


def data():
    policy = {
        "case": "rosbridge_disconnect",
        "phase": "during_motion",
        "run_id": "run",
        "body_snapshot_hash": "body",
    }
    ready = {"daemon_instance_id": "generation", "pid": 42}
    effect = {
        "schema_version": "rosclaw.owned_rosbridge_disconnect.v1",
        "case": policy["case"],
        "policy_sha256": "hash",
        "run_id": "run",
        "body_snapshot_hash": "body",
        "daemon_instance_id": ready["daemon_instance_id"],
        "daemon_pid": ready["pid"],
        "before_monotonic_ns": 1,
        "after_monotonic_ns": 2,
        "new_connections_opened": 0,
        "physics_server_signals_sent": 0,
        "original_navigation_sockets_before": [{"index": i, "connected": True} for i in range(4)],
        "original_navigation_sockets_after": [
            {"index": i, "closed": True, "websocket_released": True} for i in range(4)
        ],
    }
    return effect, policy, ready


def test_bound_original_fault_allows_only_read_retry(module):
    e, p, r = data()
    assert module.permits_unavailable_status_retry(
        e,
        p,
        r,
        policy_sha256="hash",
        error=MCPError(
            "DAEMON_UNAVAILABLE",
            "rosclawd is unavailable: timed out",
            details={"operation": "get_action_status"},
        ),
        now_ns=3,
    )


@pytest.mark.parametrize(
    "fault",
    [
        "run",
        "body",
        "generation",
        "pid",
        "phase",
        "policy",
        "reconnected",
        "future",
        "not_closed",
        "other_error",
        "wrong_code",
        "wrong_operation",
    ],
)
def test_unbound_or_unconfirmed_fault_cannot_hide_status_errors(module, fault):
    e, p, r = data()
    error = MCPError(
        "DAEMON_UNAVAILABLE",
        "rosclawd is unavailable: timed out",
        details={"operation": "get_action_status"},
    )
    if fault == "run":
        e["run_id"] = "other"
    elif fault == "body":
        e["body_snapshot_hash"] = "other"
    elif fault == "generation":
        e["daemon_instance_id"] = "other"
    elif fault == "pid":
        e["daemon_pid"] = 43
    elif fault == "phase":
        p["phase"] = "before_dispatch"
    elif fault == "policy":
        e["policy_sha256"] = "other"
    elif fault == "reconnected":
        e["new_connections_opened"] = 1
    elif fault == "future":
        e["after_monotonic_ns"] = 4
    elif fault == "not_closed":
        e["original_navigation_sockets_after"][0]["closed"] = False
    elif fault == "wrong_code":
        error.code = "AUTHORIZATION_REQUIRED"
    elif fault == "wrong_operation":
        error.details["operation"] = "request_action"
    else:
        error = RuntimeError("invalid original receipt")
    assert not module.permits_unavailable_status_retry(
        e, p, r, policy_sha256="hash", error=error, now_ns=3
    )


def test_actual_adapter_preserves_daemon_unavailable_code(module):
    from rosclaw.daemon.client import DaemonUnavailableError
    from rosclaw.mcp.adapters.runtime_client import RuntimeClient

    with pytest.raises(MCPError) as captured:
        RuntimeClient._raise_daemon_error(
            None,
            "get_action_status",
            DaemonUnavailableError("DAEMON_UNAVAILABLE", "rosclawd is unavailable: timed out"),
        )
    effect, policy, ready = data()
    assert module.permits_unavailable_status_retry(
        effect, policy, ready, policy_sha256="hash", error=captured.value, now_ns=3
    )
