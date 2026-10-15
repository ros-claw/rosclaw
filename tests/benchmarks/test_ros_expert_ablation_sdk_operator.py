"""No Docker or ROS calls: cleanup requires the exact owned SIM container."""

import json
from types import SimpleNamespace

import pytest

from benchmarks.ros_expert_ablation import sdk_operator as module


def operator(tmp_path):
    value = module.SdkExperiment.__new__(module.SdkExperiment)
    value.evidence_root = tmp_path
    path = tmp_path / "owned-world"
    path.mkdir()
    (path / "container-id.txt").write_text("original-cid")
    return value, path


def test_unowned_evidence_mount_cannot_be_stopped(tmp_path, monkeypatch):
    value, path = operator(tmp_path)
    calls = []
    response = {
        "Id": "original-cid",
        "Config": {"Image": module.SDK_IMAGE},
        "HostConfig": {"NetworkMode": "none"},
        "Mounts": [{"Destination": "/evidence", "Source": "/unrelated"}],
        "State": {"Running": True},
    }

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=json.dumps([response]).encode())

    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(AssertionError):
        value.cleanup_owned_world(path)
    assert len(calls) == 1 and calls[0][1] == "inspect"


def test_exact_owned_sdk_world_is_stopped_without_claiming_physical_evidence(tmp_path, monkeypatch):
    value, path = operator(tmp_path)
    calls = []
    response = {
        "Id": "original-cid",
        "Config": {"Image": module.SDK_IMAGE},
        "HostConfig": {"NetworkMode": "none"},
        "Mounts": [{"Destination": "/evidence", "Source": str(path / "execution")}],
        "State": {"Running": True},
    }

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(
            returncode=0, stdout=json.dumps([response]).encode() if argv[1] == "inspect" else b""
        )

    monkeypatch.setattr(module.subprocess, "run", run)
    value.cleanup_owned_world(path)
    assert calls[-1] == ["docker", "stop", "-t", "3", "original-cid"]
    proof = json.loads((path / "operator-timeout-cleanup-original.json").read_text())
    assert proof["World_terminated"] and proof["independent_physical_stop_claimed"] is False


@pytest.mark.parametrize(
    "case,seed,ordinal", [("unknown", 101, 1), ("scan_stale", 0, 1), ("scan_stale", 101, True)]
)
def test_operator_case_scope_rejected_before_process_creation(tmp_path, case, seed, ordinal):
    with pytest.raises(ValueError, match="closed experiment"):
        module.SdkExperiment(
            tmp_path,
            case,
            seed,
            ordinal,
            recovery_root=tmp_path,
            healthy_stage=tmp_path,
            fault_staging_root=tmp_path,
        )
