"""Reject forged or incomplete originals without starting an SDK or ROS."""

import copy
import json
import os

import pytest

from benchmarks.ros_expert_ablation.rsk_nav2_original_review import (
    _check_container_resources,
    _check_pointer,
    _check_sink,
    _mounts,
    _original,
)


def sink():
    return {
        "run_id": "a" * 32,
        "body_snapshot_hash": None,
        "terminal": False,
        "action_id": None,
        "daemon_instance_id": None,
        "lease_expires_ns": 0,
        "lease_updates": 0,
        "received_commands": 0,
        "published_velocity_messages": 0,
    }


@pytest.mark.parametrize("missing", list(sink()))
def test_each_sink_field_is_required(missing):
    value = sink()
    del value[missing]
    with pytest.raises(ValueError, match="never-admitted"):
        _check_sink(value, "a" * 32)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", "b" * 32),
        ("body_snapshot_hash", "invented"),
        ("terminal", 0),
        ("action_id", "action"),
        ("daemon_instance_id", "daemon"),
    ]
    + [
        (key, value)
        for key in [
            "lease_expires_ns",
            "lease_updates",
            "received_commands",
            "published_velocity_messages",
        ]
        for value in [True, 1, 0.0, None]
    ],
)
def test_admission_command_and_coerced_counters_rejected(field, value):
    original = sink()
    original[field] = value
    with pytest.raises(ValueError, match="zero-command"):
        _check_sink(original, "a" * 32)


def pointer_context():
    pointer = {
        "file": "original-native-001.json",
        "sha256": "a" * 64,
        "captured_at": "2026-10-18T20:00:00+00:00",
        "received_monotonic_ns": 1,
    }
    native = {
        "captured_at": pointer["captured_at"],
        "environment": {"rmw": "rmw_cyclonedds_cpp", "ros_generation": "ros2"},
        "completeness": {"graph": True},
    }
    window = {
        "native_before": {"sha256": pointer["sha256"]},
        "native_original_file_before": pointer["file"],
        "host_observed_monotonic_ns_before": 500_000_001,
    }
    return pointer, native, window


def test_complete_source_only_originals_accepted():
    _check_sink(sink(), "a" * 32)
    _check_pointer(*pointer_context(), "before")


@pytest.mark.parametrize(
    "name",
    [
        "../original-native-001.json",
        "/tmp/frame.json",
        "original-native-1.json",
        "original-native-0000001.json",
        "live-native-pointer.json",
        "original-native-001.json/extra",
    ],
)
def test_native_filename_cannot_escape_closed_producer_namespace(name):
    pointer, native, window = pointer_context()
    pointer["file"] = window["native_original_file_before"] = name
    with pytest.raises(ValueError, match="pointer identity"):
        _check_pointer(pointer, native, window, "before")


@pytest.mark.parametrize(
    "change",
    [
        "sha",
        "time",
        "rmw",
        "generation",
        "complete",
        "bool-clock",
        "negative-clock",
        "extra",
        "missing",
    ],
)
def test_original_pointer_identity_rejects_tampering(change):
    pointer, native, window = copy.deepcopy(pointer_context())
    if change == "sha":
        pointer["sha256"] = "b" * 64
    elif change == "time":
        native["captured_at"] = "2026-10-18T20:00:01+00:00"
    elif change in {"rmw", "generation"}:
        native["environment"]["rmw" if change == "rmw" else "ros_generation"] = "other"
    elif change == "complete":
        native["completeness"]["graph"] = 1
    elif change in {"bool-clock", "negative-clock"}:
        pointer["received_monotonic_ns"] = True if change == "bool-clock" else -1
    elif change == "extra":
        pointer["extra"] = True
    else:
        del pointer["file"]
    with pytest.raises(ValueError, match="pointer identity"):
        _check_pointer(pointer, native, window, "before")


def test_original_reader_rejects_parent_and_leaf_symlinks(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    (original / "data.json").write_text('{"original":true}')
    for name, target in [("parent", original), ("leaf", original / "data.json")]:
        (tmp_path / name).symlink_to(target)
    for path in [tmp_path / "parent/data.json", tmp_path / "leaf"]:
        with pytest.raises(ValueError, match="symlink"):
            _original(tmp_path, path)
    data, reference = _original(tmp_path, original / "data.json")
    assert data == {"original": True}
    assert reference["path"] == "original/data.json"


def test_original_reader_rejects_outside_root_and_oversized_file(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    path = tmp_path / "outside.json"
    path.write_text(json.dumps({"data": "a" * 100}))
    with pytest.raises(ValueError):
        _original(root, path)
    with pytest.raises((ValueError, OSError)):
        _original(tmp_path, path, limit=10)


def test_duplicate_mount_destinations_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        _mounts({"Mounts": [{"Destination": "/repo"}, {"Destination": "/repo"}]})


@pytest.mark.parametrize("observed", [None, True, 0, 500_000_002])
def test_read_time_must_prove_original_pointer_within_500ms(observed):
    pointer, native, window = pointer_context()
    window["host_observed_monotonic_ns_before"] = observed
    with pytest.raises(ValueError, match="pointer identity"):
        _check_pointer(pointer, native, window, "before")


def container():
    return {
        "Image": "sha256:51ffb0e0aeb391f0efbef3013c7fc83586794c222b233e3dea688f3015ef1c6b",
        "HostConfig": {
            "NetworkMode": "none",
            "ReadonlyRootfs": True,
            "Privileged": False,
            "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges"],
            "NanoCpus": 6_000_000_000,
            "Memory": 10 * 1024**3,
            "PidsLimit": 512,
            "ShmSize": 512 * 1024**2,
        },
        "Config": {
            "User": f"{os.getuid()}:{os.getgid()}",
            "Entrypoint": ["bash"],
            "Cmd": [
                "-c",
                "source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash && "
                "export PYTHONPATH=/repo/src:/repo/integrations/ros_probe/acceptance:"
                "$PYTHONPATH && python3 /repo/integrations/ros_probe/acceptance/"
                "source_binding_fixture.py",
            ],
        },
    }


def test_fixed_sdk_resources_accepted():
    _check_container_resources(container())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("NetworkMode", "host"),
        ("ReadonlyRootfs", False),
        ("Privileged", True),
        ("Devices", [{"PathOnHost": "/dev/fake"}]),
        ("DeviceRequests", [{"Count": -1}]),
        ("DeviceCgroupRules", ["c *:* rwm"]),
        ("CapAdd", ["SYS_ADMIN"]),
        ("PidMode", "host"),
        ("IpcMode", "host"),
        ("CapDrop", []),
        ("SecurityOpt", []),
        ("NanoCpus", 0),
        ("NanoCpus", 6_000_000_001),
        ("Memory", 0),
        ("Memory", 10 * 1024**3 + 1),
        ("PidsLimit", -1),
        ("PidsLimit", 513),
        ("ShmSize", 0),
        ("ShmSize", 512 * 1024**2 + 1),
    ],
)
def test_added_privilege_devices_and_unbounded_resources_rejected(field, value):
    original = container()
    original["HostConfig"][field] = value
    with pytest.raises(ValueError, match="non-root SDK resources"):
        _check_container_resources(original)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("User", "0:1000"),
        ("User", "root"),
        ("User", ""),
        ("Entrypoint", ["sh"]),
        ("Cmd", ["-c", "arbitrary-command"]),
    ],
)
def test_owner_uid_and_closed_entrypoint_required(field, value):
    original = container()
    original["Config"][field] = value
    with pytest.raises(ValueError, match="non-root SDK resources"):
        _check_container_resources(original)
