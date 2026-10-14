"""Owned socket fault contracts with synthetic sockets, never host ROS connections."""

import importlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rosclaw.connectors.ros.mission.navigation_binding import SimulationNavigationBinding
from rosclaw.connectors.ros.transport.base import RosbridgeEndpoint
from rosclaw.connectors.ros.transport.rosbridge import RosbridgeTransport
from tests.connectors.ros.test_navigation_only_executor import compiled_body
from tests.connectors.ros.test_simulation_navigation_supervision import healthy_status


@pytest.fixture
def fixture(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("simulation_rosbridge_disconnect_fixture")
    binding = SimulationNavigationBinding.from_body(compiled_body())
    closed = []

    class Socket:
        connected = True
        sock = None

        def close(self):
            self.connected = False
            closed.append(self)

    transports = [
        RosbridgeTransport(RosbridgeEndpoint.from_url("ws://127.0.0.1:9090")) for _ in range(4)
    ]
    for transport in transports:
        transport._ws = Socket()
    policy = {
        "case": "rosbridge_disconnect",
        "body_snapshot_hash": binding.body_snapshot_hash,
        "run_id": binding.run_id,
        "action_id": "first",
    }
    sample = {
        "body_id": binding.body_id,
        "body_snapshot_hash": binding.body_snapshot_hash,
        "run_id": binding.run_id,
        "frame_id": binding.frame_id,
        "evidence_domain": "GAZEBO_PHYSICS",
        "collision_source": "gazebo_contacts",
        "observation_complete": True,
        "collision_count": 0,
        "lease_enabled": True,
        "linear_speed_mps": binding.stop_speed_mps * 2,
        "angular_speed_rps": 0.0,
        "x": 0.0,
        "y": 0.0,
        "yaw": 0.0,
        "time_sec": 1.0,
        "captured_at": datetime.now(UTC).isoformat(),
    }
    trigger = {
        "policy_sha256": "a" * 64,
        "injection": {
            "actual_prior_physics": sample,
            "actual_prior_sink": {"daemon_instance_id": "daemon_synthetic", "action_id": "first"},
        },
    }

    def publish():
        (tmp_path / "governed-rosbridge-disconnect-trigger-original.json").write_text(
            json.dumps(trigger)
        )

    hook = module.OwnedRosbridgeDisconnectFixture(
        root=tmp_path,
        policy=policy,
        policy_sha256="a" * 64,
        binding=binding,
        transports=transports,
        daemon_status=lambda: healthy_status(binding.body_id),
        instance_id="daemon_synthetic",
    )
    return hook, transports, closed, trigger, publish, tmp_path


def test_only_four_existing_sockets_close_once_and_no_reconnect(fixture):
    hook, transports, closed, _, publish, root = fixture
    hook.poll()
    assert not closed
    publish()
    hook.poll()
    assert len(closed) == 4
    assert all(t._closed and t._ws is None for t in transports)
    hook.poll()
    assert len(closed) == 4
    result = json.loads((root / "governed-rosbridge-disconnect-effect-original.json").read_text())
    assert result["new_connections_opened"] == result["physics_server_signals_sent"] == 0


@pytest.mark.parametrize(
    "fault", ["stale", "stationary", "wrong_action", "wrong_sha", "already_disconnected"]
)
def test_invalid_or_nonmotion_trigger_never_closes_other_sockets(fixture, fault):
    hook, transports, closed, trigger, publish, _ = fixture
    if fault == "stale":
        trigger["injection"]["actual_prior_physics"]["captured_at"] = (
            datetime.now(UTC) - timedelta(seconds=1)
        ).isoformat()
    elif fault == "stationary":
        trigger["injection"]["actual_prior_physics"]["linear_speed_mps"] = 0.0
    elif fault == "wrong_action":
        trigger["injection"]["actual_prior_sink"]["action_id"] = "other"
    elif fault == "wrong_sha":
        trigger["policy_sha256"] = "b" * 64
    else:
        transports[0]._ws.connected = False
    publish()
    with pytest.raises(ValueError):
        hook.poll()
    assert closed == []
