"""Completion markers never replace original fault audit bytes or accept partial data."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def fixture(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("simulation_completed_artifact")
    outcome = {
        "status": "synthetic-stop",
        "case": "lease_loss",
        "policy_sha256": "a" * 64,
        "injection": {"run_id": "original", "body_snapshot_hash": "b" * 64},
        "after_monotonic_ns": 1,
        "original_post_fault_physics": [{"synthetic": True}],
    }
    return module, tmp_path, outcome


def test_completed_marker_hashes_full_original_without_parsing_large_audit(fixture):
    module, root, outcome = fixture
    outcome["synthetic_padding"] = "x" * 300_000
    marker = module.publish_completed_fault_outcome(root, outcome)
    assert marker["artifact_bytes"] > 256_000
    assert module.read_completed_fault_outcome(root, expected_policy_sha256="a" * 64) == marker
    assert json.loads((root / module.ORIGINAL_NAME).read_text()) == outcome


@pytest.mark.parametrize("fault", ["contents", "policy", "size", "path", "symlink", "missing"])
def test_incomplete_or_altered_audit_is_never_a_completion(fixture, fault):
    module, root, outcome = fixture
    marker = module.publish_completed_fault_outcome(root, outcome)
    original = root / module.ORIGINAL_NAME
    if fault == "contents":
        original.write_bytes(original.read_bytes().replace(b"synthetic-stop", b"synthetic-lost"))
    elif fault == "policy":
        marker["policy_sha256"] = "c" * 64
    elif fault == "size":
        marker["artifact_bytes"] = 0
    elif fault == "path":
        marker["artifact_file"] = "../fixture.key"
    elif fault == "symlink":
        moved = root / "moved.json"
        original.rename(moved)
        original.symlink_to(moved)
    else:
        original.unlink()
    (root / module.MARKER_NAME).write_text(json.dumps(marker))
    with pytest.raises((ValueError, OSError)):
        module.read_completed_fault_outcome(root, expected_policy_sha256="a" * 64)


def test_existing_original_is_never_overwritten(fixture):
    module, root, outcome = fixture
    module.publish_completed_fault_outcome(root, outcome)
    before = (root / module.ORIGINAL_NAME).read_bytes()
    with pytest.raises(FileExistsError):
        module.publish_completed_fault_outcome(root, outcome)
    assert (root / module.ORIGINAL_NAME).read_bytes() == before
