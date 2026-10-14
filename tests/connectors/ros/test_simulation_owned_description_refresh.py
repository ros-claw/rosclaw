"""Synthetic identical-description refresh contracts; no ROS/network access."""

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("simulation_owned_description_refresh")
    assets = tmp_path / "assets"
    assets.mkdir()
    urdf = b'<robot name="original"/>'
    (assets / "robot.urdf").write_bytes(urdf)
    node = "/unusual/description_owner"
    topic = "/unusual/description"
    native = {
        "observations": {
            "urdf_descriptions": {
                node: {
                    "sha256": hashlib.sha256(urdf).hexdigest(),
                    "complete": True,
                    "source": node + "/get_parameters",
                }
            }
        },
        "graph": {
            "topics": [{"name": topic, "msg_type": "std_msgs/msg/String", "publishers": [node]}],
            "services": [
                {"name": node + "/" + suffix, "srv_type": "rcl_interfaces/srv/" + kind}
                for suffix, kind in [
                    ("get_parameters", "GetParameters"),
                    ("set_parameters", "SetParameters"),
                ]
            ],
        },
    }
    raw = json.dumps(native).encode()
    (tmp_path / "native.json").write_bytes(raw)
    monkeypatch.setattr(
        module,
        "read_current_native_pointer",
        lambda root: ({"file": "native.json", "sha256": hashlib.sha256(raw).hexdigest()}, 0),
    )

    class Transport:
        value = urdf.decode()
        ok = True
        ack = True

        def __init__(self):
            self.calls = []

        def call_service(self, name, request, **kwargs):
            self.calls.append((name, request, kwargs))
            if name.endswith("get_parameters"):
                values = {"values": [{"type": 4, "string_value": self.value}]}
            else:
                values = {"results": [{"successful": self.ack}]}
            return SimpleNamespace(ok=self.ok, data={"values": values}, error=None)

    transport = Transport()
    refresh = module.OwnedDescriptionRefresh(tmp_path, assets, transport)
    (tmp_path / "gazebo.log").write_bytes(
        ("Subscribing to '" + topic + "' topic for robot description.\n").encode()
    )
    return refresh, transport, tmp_path, urdf


def test_refresh_sets_only_byte_identical_original_description(fixture):
    refresh, transport, _, urdf = fixture
    result = refresh.after_create(0)
    writes = [r for name, r, _ in transport.calls if name.endswith("set_parameters")]
    assert writes == [
        {
            "parameters": [
                {"name": "robot_description", "value": {"type": 4, "string_value": urdf.decode()}}
            ]
        }
    ]
    assert (
        result["changed_URDF_bytes"] is False
        and result["new_publishers_created"] == result["new_TF_publishers_created"] == 0
    )
    assert result["urdf_sha256"] == hashlib.sha256(urdf).hexdigest()


@pytest.mark.parametrize("fault", ["changed_description", "transport", "acknowledgement"])
def test_failed_refresh_never_reports_original_description_ready(fixture, fault):
    refresh, transport, _, _ = fixture
    if fault == "changed_description":
        transport.value = '<robot name="different"/>'
    if fault == "transport":
        transport.ok = False
    if fault == "acknowledgement":
        transport.ack = False
    with pytest.raises(ValueError):
        refresh.after_create(0)
    if fault == "changed_description":
        assert not any(name.endswith("set_parameters") for name, _, _ in transport.calls)


def test_subscriber_marker_cannot_be_replayed_from_before_creation(fixture):
    refresh, _, root, _ = fixture
    with pytest.raises(ValueError, match="bounded original Gazebo log"):
        refresh.after_create((root / "gazebo.log").stat().st_size + 1)
