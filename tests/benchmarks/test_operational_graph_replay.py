"""Offline replay process integrity; no ROS transport or physical acceptance."""

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.ros_expert_ablation.replay_operational_graph import replay
from tests.connectors.ros.test_operational_graph_contracts import NOW, native_fixture


def inputs(tmp_path):
    m, u, r = native_fixture()
    entries = {}
    for name, raw in [
        ("snapshot", json.dumps(m.to_dict()).encode()),
        ("urdf", u),
        ("requirements", json.dumps(r).encode()),
    ]:
        p = tmp_path / name
        p.write_bytes(raw)
        entries[name] = {"path": str(p), "sha256": hashlib.sha256(raw).hexdigest()}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "rosclaw.operational_graph_replay_inputs.v1",
                "entries": [
                    {
                        "format": "sealed_system_model",
                        "observation": entries["snapshot"],
                        "urdf": entries["urdf"],
                        "requirements": entries["requirements"],
                        "reviewed_at": NOW.isoformat(),
                    }
                ],
            }
        )
    )
    return manifest


def test_actual_replay_outputs_are_byte_exact_without_mutating_inputs(tmp_path):
    manifest = inputs(tmp_path)
    originals = {p: p.read_bytes() for p in tmp_path.iterdir()}
    a, b = tmp_path / "out-a", tmp_path / "out-b"
    result = replay(manifest, a)
    assert result == replay(manifest, b)
    assert {p.name: p.read_bytes() for p in a.iterdir()} == {
        p.name: p.read_bytes() for p in b.iterdir()
    }
    assert result["new_Worlds"] == 0 and result["probes_executed"] == 0
    assert all(p.read_bytes() == raw for p, raw in originals.items())
    with pytest.raises(FileExistsError):
        replay(manifest, a)


@pytest.mark.parametrize(
    "fault", ["tamper", "symlink", "missing_file", "wrong_version", "wrong_expected_verdict"]
)
def test_replay_input_integrity_and_frozen_verdicts_fail_closed(tmp_path, fault):
    manifest = inputs(tmp_path)
    data = json.loads(manifest.read_text())
    snapshot = Path(data["entries"][0]["observation"]["path"])
    if fault == "tamper":
        snapshot.write_text("{}")
    if fault == "symlink":
        new = tmp_path / "link"
        new.symlink_to(snapshot)
        data["entries"][0]["observation"]["path"] = str(new)
    if fault == "missing_file":
        snapshot.unlink()
    if fault == "wrong_version":
        data["schema_version"] = "future.v999"
    if fault == "wrong_expected_verdict":
        data["entries"][0]["expected_constraints"] = {"F-TF-001": "VIOLATED"}
    manifest.write_text(json.dumps(data))
    with pytest.raises((ValueError, AssertionError)):
        replay(manifest, tmp_path / "output")


def test_native_normalization_has_its_own_document_and_explicit_original_binding(tmp_path):
    manifest = inputs(tmp_path)
    data = json.loads(manifest.read_text())
    row = data["entries"][0]
    m = json.loads(Path(row["observation"]["path"]).read_text())
    native = {
        k: v
        for k, v in m.items()
        if k not in {"snapshot_id", "snapshot_hash", "robot_id", "inference", "body"}
    }
    native["schema_version"] = "rosclaw.ros_probe.v1"
    original = tmp_path / "original-native.json"
    raw = json.dumps(native).encode()
    original.write_bytes(raw)
    row.update(
        format="native_probe",
        observation={"path": str(original), "sha256": hashlib.sha256(raw).hexdigest()},
    )
    manifest.write_text(json.dumps(data))
    output = tmp_path / "output"
    result = replay(manifest, output)
    reviewed = result["rows"][0]
    assert reviewed["source_ref"].startswith("nativeprojection_")
    assert (
        reviewed["original_observation_sha256"] == hashlib.sha256(original.read_bytes()).hexdigest()
    )
    assert (output / reviewed["normalized_model_artifact"]).is_file()
    assert (output / "001-case.json").is_file()
    assert original.read_bytes() == raw
