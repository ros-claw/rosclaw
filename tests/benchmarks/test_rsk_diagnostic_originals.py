"""All Docker/Native records in this file are synthetic negative contracts."""

import copy
import hashlib
import json
import os
from datetime import timedelta

import pytest

from benchmarks.ros_expert_ablation.rsk_diagnostic_provider import (
    SDK_IMAGE,
    DiagnosticProbeOriginalsV1,
)
from benchmarks.ros_expert_ablation.rsk_episode_runner import OriginalRef
from rosclaw.connectors.ros.engineering.diagnostic_experiment import RosDiagnosticExperimentV1
from tests.benchmarks.test_rsk_operational_aci import facade, plan, provider_fixture
from tests.connectors.ros.test_operational_graph_contracts import NOW


def originals(tmp_path, monkeypatch, fault):
    aci = facade()
    provider, _ = provider_fixture(tmp_path, aci)
    experiment = RosDiagnosticExperimentV1.from_dict(plan(aci)["experiment"])
    selected = next(p for p in experiment.probes if p.request.operation == "ENDPOINT_QOS")
    execution = tmp_path / "execution"
    execution.mkdir()
    config = json.loads((tmp_path / "private-sensor-config.json").read_text())

    def put(path, value):
        path.write_text(json.dumps(value, indent=2))
        return OriginalRef(
            path=str(path.relative_to(tmp_path)),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    endpoints = [
        {
            "topic": config["topic"],
            "node": config["namespace"] + "/emitter",
            "kind": "publisher",
            "gid": "fixture-publisher",
            "reliability": "BEST_EFFORT",
            "durability": "VOLATILE",
        },
        {
            "topic": config["topic"],
            "node": config["namespace"] + "/receiver",
            "kind": "subscriber",
            "gid": "fixture-subscriber",
            "reliability": "RELIABLE",
            "durability": "VOLATILE",
        },
    ]
    native = {
        "schema_version": "rosclaw.ros_probe.v1",
        "captured_at": NOW.isoformat(),
        "environment": {"ros_generation": "ros2", "distro": "jazzy", "rmw": "rmw_cyclonedds_cpp"},
        "graph": {"nodes": [], "topics": [], "services": [], "actions": []},
        "qos": {"endpoints": endpoints},
        "completeness": {"graph": True},
    }
    later = copy.deepcopy(native)
    later["captured_at"] = (NOW + timedelta(seconds=5.2)).isoformat()
    if fault == "changed_endpoint":
        later["qos"]["endpoints"][0]["reliability"] = "RELIABLE"
    if fault == "future_native":
        later["captured_at"] = (NOW + timedelta(days=1)).isoformat()
    b = put(execution / "before.json", native)
    a = put(execution / "after.json", later)
    for name in [
        "integrations/ros_probe/ros2/probe.py",
        "benchmarks/ros_expert_ablation/rsk_native_diagnostic_fixture.py",
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("SYNTHETIC_UNIT_SOURCE_PIN_" + name)
    owner = {
        "schema_version": "rosclaw.rsk_sensor_observation_owner.v1",
        "run_id": "c" * 32,
        "source": "b" * 40,
        "execution_config_sha256": provider.policy.execution_config_sha256,
        "native_before_sha256": b.sha256,
        "native_after_sha256": a.sha256,
        "publisher_policy_changed_during_window": False,
        "observation_steady_sec": 5.2,
        "native_snapshot_call_durations_sec": [0.1, 0.1],
        "robot_driver_processes": [],
        "navigation_goals": 0,
        "control_commands": 0,
        "body_admission": "NOT_RUN",
        "physics": "NOT_RUN",
        "consumer_before": {"count": 3, "sent": 4},
        "consumer_after": {"count": 3, "sent": 5},
        "source_file_sha256": {
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            for name in [
                "integrations/ros_probe/ros2/probe.py",
                "benchmarks/ros_expert_ablation/rsk_native_diagnostic_fixture.py",
            ]
        },
    }
    if fault == "owner_policy_changed":
        owner["publisher_policy_changed_during_window"] = True
    if fault == "command":
        owner["control_commands"] = 1
    if fault == "driver":
        owner["robot_driver_processes"] = ["forbidden-driver"]
    if fault == "short_window":
        owner["observation_steady_sec"] = 2.0
    if fault == "slow_snapshot":
        owner["native_snapshot_call_durations_sec"][1] = 6.0
    if fault == "counter_backwards":
        owner["consumer_after"]["count"] = 2
    if fault == "extra_source_path":
        owner["source_file_sha256"][".env"] = "0" * 64
    created = {
        "Id": "d" * 64,
        "Image": SDK_IMAGE,
        "State": {"Running": False, "Pid": 0, "ExitCode": 0, "Status": "created"},
        "Config": {
            "Env": ["RSK_SOURCE_COMMIT=" + "b" * 40, "RSK_RUN_ID=" + "c" * 32],
            "User": f"{os.getuid()}:{os.getgid()}",
        },
        "HostConfig": {
            "NetworkMode": "none",
            "ReadonlyRootfs": True,
            "Privileged": False,
            "Devices": [],
            "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges"],
        },
        "Mounts": [
            {"Destination": "/repo", "Source": str(tmp_path), "RW": False},
            {"Destination": "/evidence", "Source": str(execution), "RW": True},
        ],
    }
    stopped = copy.deepcopy(created)
    stopped["State"]["Status"] = "exited"
    if fault == "network":
        created["HostConfig"]["NetworkMode"] = stopped["HostConfig"]["NetworkMode"] = "bridge"
    if fault == "devices":
        created["HostConfig"]["Devices"] = stopped["HostConfig"]["Devices"] = [
            {"PathOnHost": "/dev/fake"}
        ]
    if fault == "bad_exit":
        stopped["State"]["ExitCode"] = 1
    current = copy.deepcopy(stopped)
    if fault == "live_state_changed":
        current["State"]["Running"] = True
    import benchmarks.ros_expert_ablation.rsk_diagnostic_provider as module

    monkeypatch.setattr(
        module.subprocess, "check_output", lambda *args, **kw: json.dumps([current])
    )
    values = {
        "source_revision": "b" * 40,
        "original_run_id": "a" * 32,
        "probe_run_id": "c" * 32,
        "requested_operation": "ENDPOINT_QOS",
        "started_at": (NOW - timedelta(seconds=1)).isoformat(),
        "finished_at": (NOW + timedelta(seconds=7)).isoformat(),
        "execution_config_sha256": provider.policy.execution_config_sha256,
        "native_before": b,
        "native_after": a,
        "owner_observations": put(execution / "owner.json", owner),
        "container_before": put(tmp_path / "created.json", created),
        "container_after": put(tmp_path / "stopped.json", stopped),
    }
    if fault == "same_run":
        values["probe_run_id"] = values["original_run_id"]
    if fault == "wrong_source":
        values["source_revision"] = "e" * 40
    if fault == "reused_capture":
        values["native_after"] = b
    manifest = DiagnosticProbeOriginalsV1(**values)
    reference = put(tmp_path / "manifest.json", manifest.model_dump(mode="json"))
    return provider, reference, experiment, selected, aci


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("wrong_source", "frozen source"),
        ("same_run", "frozen source"),
        ("reused_capture", "distinct source"),
        ("live_state_changed", "current stopped"),
        ("bad_exit", "stopped successfully"),
        ("network", "pinned isolated"),
        ("devices", "pinned isolated"),
        ("owner_policy_changed", "unchanged policy"),
        ("command", "zero-command"),
        ("driver", "zero-command"),
        ("short_window", "steady observation"),
        ("counter_backwards", "backwards"),
        ("extra_source_path", "exact original"),
        ("changed_endpoint", "endpoint policies"),
        ("future_native", "within the same original"),
        ("slow_snapshot", "request duration"),
    ],
)
def test_rehashed_false_or_incomplete_originals_cannot_pass_observational_review(
    tmp_path, monkeypatch, fault, expected
):
    provider, reference, experiment, selected, aci = originals(tmp_path, monkeypatch, fault)
    with pytest.raises(ValueError, match=expected):
        provider.review_originals(
            reference, experiment, selected, aci.base, original_graph=aci._graph()
        )
    assert aci.base.probe_count == 0 and not aci.base.results
