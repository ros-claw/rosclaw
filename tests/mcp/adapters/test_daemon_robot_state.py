"""Configured Sense crosses authenticated UDS without an Agent-side Runtime.

All telemetry below is explicitly a software fixture. No ROS/DDS/model/device.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rosclaw.daemon.client import DaemonRequestError
from rosclaw.mcp.adapters.runtime_client import RuntimeClient
from rosclaw.mcp.schemas.common import MCPError
from rosclaw.sense.schemas import BodyState

pytest_plugins = ("tests.daemon.test_server",)


def sense(source="hardware:fixture", stale=False, collector="ros2"):
    state = BodyState("rh56-test", 123.0, source=source)
    state.energy.battery_percent = 42.0
    return SimpleNamespace(
        config=SimpleNamespace(collector=collector),
        get_latest_state=lambda: state,
        get_body_sense=lambda: {},
        get_readiness=lambda: {},
        is_stale=stale,
        state_age_ms=42,
        stop=lambda: None,
    )


def mcp(daemon_client, tmp_path, monkeypatch, observed):
    rpc, runtime, _, _ = daemon_client
    runtime._sense = observed
    client = RuntimeClient(
        project_root=tmp_path, robot_id="rh56-test", runtime_profile={}, daemon_client=rpc
    )
    monkeypatch.setattr(
        client, "_ensure_runtime", Mock(side_effect=AssertionError("Agent Runtime forbidden"))
    )
    monkeypatch.setattr(
        "rosclaw.core.runtime.Runtime.__init__",
        Mock(side_effect=AssertionError("No Runtime construction during read")),
    )
    return client


@pytest.mark.parametrize(
    "source,stale,mode,trust",
    [
        ("hardware:fixture", False, "live", "OBSERVED"),
        ("hardware:fixture", True, "stale", "STALE"),
        ("file_replay:hardware:fixture", False, "replay", "RECORDED"),
    ],
)
async def test_original_state_and_provenance_over_authenticated_socket(
    daemon_client, tmp_path, monkeypatch, source, stale, mode, trust
):
    observed = sense(source, stale)
    client = mcp(daemon_client, tmp_path, monkeypatch, observed)
    result = await client.get_robot_state()
    assert result["robot_id"] == "rh56-test"
    assert result["mode"] == mode and result["trust_level"] == trust
    assert result["body_state"]["timestamp"] == 123.0
    assert result["body_state"]["source"] == source
    assert result["body_state"]["energy"]["battery_percent"] == 42
    assert result["usable_for_real_execution"] is False
    assert result["execution_mode"] == ("REPLAY" if mode == "replay" else "UNKNOWN")
    assert result["observation_owner"] == "rosclawd" and result["daemon_instance_id"]
    assert result["daemon_peer"]["uid"] >= 0
    assert client._runtime is None
    observed.get_latest_state().energy.battery_percent = None
    assert (await client.get_robot_state())["body_state"]["energy"]["battery_percent"] is None
    assert daemon_client[0].get_runtime_status()["queue"]["RUNNING"] == 0


@pytest.mark.parametrize(
    "source,collector,code",
    [
        ("mock:normal", "mock", "SYNTHETIC_SOURCE_NOT_ALLOWED"),
        ("mock:normal", "ros2", "SYNTHETIC_SOURCE_NOT_ALLOWED"),
        ("unconfigured", "unconfigured", "STATE_PROVENANCE_UNAVAILABLE"),
        ("file_replay:empty", "file_replay", "STATE_PROVENANCE_UNAVAILABLE"),
        ("ros2:stub", "ros2", "STATE_PROVENANCE_UNAVAILABLE"),
        ("unknown", "ros2", "STATE_PROVENANCE_UNAVAILABLE"),
        (None, "ros2", "STATE_PROVENANCE_UNAVAILABLE"),
        ([], "ros2", "STATE_PROVENANCE_UNAVAILABLE"),
    ],
)
async def test_unavailable_sources_stay_unavailable_without_local_fallback(
    daemon_client, tmp_path, monkeypatch, source, collector, code
):
    client = mcp(daemon_client, tmp_path, monkeypatch, sense(source, False, collector))
    with pytest.raises(MCPError) as error:
        await client.get_robot_state()
    assert error.value.code == code
    assert error.value.details["trust_level"] == "UNAVAILABLE"
    assert error.value.details["usable_for_real_execution"] is False
    assert client._runtime is None


async def test_disabled_daemon_sense_is_unknown_and_does_not_enable_mock(
    daemon_client, tmp_path, monkeypatch
):
    client = mcp(daemon_client, tmp_path, monkeypatch, None)
    with pytest.raises(MCPError) as error:
        await client.get_robot_state()
    assert error.value.code == "SENSE_NOT_CONFIGURED"
    assert daemon_client[1].sense is None


async def test_older_daemon_does_not_fall_back_to_local_runtime(tmp_path, monkeypatch):
    rpc = SimpleNamespace(
        get_robot_state=Mock(side_effect=DaemonRequestError("METHOD_NOT_ALLOWED", "old daemon"))
    )
    client = RuntimeClient(
        project_root=tmp_path, robot_id="fixture", runtime_profile={}, daemon_client=rpc
    )
    monkeypatch.setattr(
        client, "_ensure_runtime", Mock(side_effect=AssertionError("No local fallback"))
    )
    with pytest.raises(MCPError) as error:
        await client.get_robot_state()
    assert error.value.code == "METHOD_NOT_ALLOWED" and client._runtime is None


def test_read_endpoint_rejects_caller_collector_or_body_configuration(daemon_client):
    rpc, runtime, _, _ = daemon_client
    with pytest.raises(DaemonRequestError) as error:
        rpc.call("observation.robot_state", {"collector": "mock"})
    assert error.value.code == "INVALID_PARAMS" and runtime.sense is None


@pytest.mark.parametrize(
    "change",
    [
        {"usable_for_real_execution": True},
        {"trust_level": "SIMULATED"},
        {"observation_owner": "local"},
        {"daemon_instance_id": ""},
        {"mode": {}},
        {"execution_mode": []},
    ],
)
async def test_malformed_or_promoted_daemon_projection_is_rejected(tmp_path, change):
    result = {
        "mode": "live",
        "execution_mode": "UNKNOWN",
        "trust_level": "OBSERVED",
        "usable_for_real_execution": False,
        "observation_owner": "rosclawd",
        "daemon_instance_id": "fixture",
        "body_state": {},
        "readiness": {},
    }
    result.update(change)
    client = RuntimeClient(
        project_root=tmp_path,
        robot_id="fixture",
        runtime_profile={},
        daemon_client=SimpleNamespace(get_robot_state=lambda: result),
    )
    with pytest.raises(MCPError) as error:
        await client.get_robot_state()
    assert error.value.code == "STATE_PROVENANCE_UNAVAILABLE"
