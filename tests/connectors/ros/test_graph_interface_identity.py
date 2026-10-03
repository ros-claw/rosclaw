"""Graph compilation must retain actual interface types or refuse the input."""

import json

import pytest

from rosclaw.connectors.ros.cli.ros_cli import cmd_ros_compile
from rosclaw.connectors.ros.compiler import CapabilityManifestCompiler
from rosclaw.connectors.ros.discovery.graph import RosGraphSnapshot
from tests.connectors.ros.test_ros_cli import FakeArgs

CASES = [
    ("topics", "msg_type", "std_msgs/msg/String"),
    ("services", "srv_type", "std_srvs/srv/Trigger"),
    ("actions", "action_type", "nav2_msgs/action/NavigateToPose"),
]


def graph(collection, interface):
    return {
        "ros_version": "ros2",
        "endpoint": "dds://fixture",
        "topics": [],
        "services": [],
        "actions": [],
        collection: [interface],
    }


@pytest.mark.parametrize("collection,field,type_name", CASES)
def test_public_type_alias_reaches_actual_compiled_interface(collection, field, type_name):
    snapshot = RosGraphSnapshot.from_dict(
        graph(collection, {"name": "/fixture", "type": type_name})
    )
    manifest = CapabilityManifestCompiler().compile(snapshot)
    assert manifest.capabilities[0].interface.msg_type == type_name
    assert getattr(getattr(snapshot, collection)[0], field) == type_name
    assert manifest.endpoint["execution_eligible"] is False


@pytest.mark.parametrize("collection,field,type_name", CASES)
@pytest.mark.parametrize("value", [None, "", "   ", 123, ["std_msgs/msg/String"]])
def test_invalid_canonical_type_never_compiles_as_success(collection, field, type_name, value):
    snapshot = RosGraphSnapshot.from_dict(graph(collection, {"name": "/fixture", field: value}))
    with pytest.raises(ValueError, match="ROS_INTERFACE_TYPE_INVALID"):
        CapabilityManifestCompiler().compile(snapshot)


@pytest.mark.parametrize("collection,field,type_name", CASES)
def test_conflicting_alias_cannot_override_declared_interface(collection, field, type_name):
    with pytest.raises(ValueError, match="ROS_INTERFACE_TYPE_CONFLICT"):
        RosGraphSnapshot.from_dict(
            graph(
                collection, {"name": "/fixture", field: type_name, "type": "other_msgs/msg/Other"}
            )
        )


def test_real_cli_rejects_missing_type_without_writing_manifest(tmp_path, capsys):
    source = tmp_path / "graph.json"
    output = tmp_path / "manifest.json"
    source.write_text(json.dumps(graph("topics", {"name": "/fixture"})))
    rc = cmd_ros_compile(
        FakeArgs(graph=str(source), output=str(output), robot_id="fixture", json=True)
    )
    assert rc == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
    assert not output.exists()
