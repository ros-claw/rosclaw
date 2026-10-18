"""Static original mesh proof does not survive replacement, tamper or symlinks."""

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from rosclaw.connectors.ros.context.collision_assets import (
    collision_assets_from_observations,
    staged_collision_resource_pins,
)
from tests.connectors.ros.test_collision_asset_originals import asset, binary


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("simulation_collision_resources")


def materialize(root):
    root.mkdir()
    (root / "collision-resources").mkdir()
    row = asset()
    row["uri"] = "/assets/collision-resources/" + row["sha256"] + ".stl"
    raw = (json.dumps([row], indent=2) + "\n").encode()
    (root / "collision-asset-originals.json").write_bytes(raw)
    path = root / "collision-resources" / (row["sha256"] + ".stl")
    path.write_bytes(binary())
    observation = {
        "prepared_collision_assets": {
            "schema_version": "rosclaw.prepared_collision_asset_observation.v1",
            "source": "/assets/collision-asset-originals.json",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "document": raw.decode(),
        }
    }
    return observation, path


def test_exact_manifest_and_original_bytes_are_rechecked_without_authority(module, tmp_path):
    observations, _ = materialize(tmp_path / "assets")
    parsed = collision_assets_from_observations(observations)
    assert parsed is not None and len(parsed[1]) == 1
    guard = module.CollisionResourceContinuity(tmp_path / "assets", observations)
    assert guard.verify(observations) == staged_collision_resource_pins(observations)
    assert len(guard.verify()) == 2
    assert module.CollisionResourceContinuity(tmp_path / "no-assets", {}).verify({}) == {}
    assert collision_assets_from_observations({}) is None


@pytest.mark.parametrize(
    "fault",
    [
        "bytes",
        "manifest",
        "file_replace_same_bytes",
        "file_symlink",
        "resource_dir_symlink",
        "root_symlink",
        "resource_dir_replace",
        "root_replace",
        "native_original_changed",
        "native_original_removed",
    ],
)
def test_any_original_resource_change_latches_even_after_rollback(module, tmp_path, fault):
    root = tmp_path / "assets"
    observations, path = materialize(root)
    guard = module.CollisionResourceContinuity(root, observations)
    current = observations
    if fault == "bytes":
        path.write_bytes(binary() + b"x")
    elif fault == "manifest":
        (root / "collision-asset-originals.json").write_text("[]")
    elif fault == "file_replace_same_bytes":
        path.unlink()
        path.write_bytes(binary())
    elif fault == "file_symlink":
        other = tmp_path / "other.stl"
        other.write_bytes(binary())
        path.unlink()
        path.symlink_to(other)
    elif fault == "resource_dir_symlink":
        (root / "collision-resources").rename(tmp_path / "elsewhere")
        (root / "collision-resources").symlink_to(tmp_path / "elsewhere")
    elif fault == "root_symlink":
        root.rename(tmp_path / "elsewhere")
        root.symlink_to(tmp_path / "elsewhere")
    elif fault == "resource_dir_replace":
        (root / "collision-resources").rename(tmp_path / "old")
        (root / "collision-resources").mkdir()
        (root / "collision-resources" / path.name).write_bytes(binary())
    elif fault == "root_replace":
        root.rename(tmp_path / "old")
        materialize(root)
    elif fault == "native_original_changed":
        current = {
            "prepared_collision_assets": {
                **observations["prepared_collision_assets"],
                "sha256": "0" * 64,
            }
        }
    else:
        current = {}
    with pytest.raises(ValueError, match="continuity rejected"):
        guard.verify(current)
    with pytest.raises(ValueError, match="permanently latched"):
        guard.verify(observations)


@pytest.mark.parametrize(
    "fault", ["document_sha", "manifest_source", "duplicate_json", "unbound_uri"]
)
def test_provenance_document_must_match_exact_originals_and_staged_names(tmp_path, fault):
    observations, _ = materialize(tmp_path / "assets")
    envelope = observations["prepared_collision_assets"]
    if fault == "document_sha":
        envelope["sha256"] = "0" * 64
    elif fault == "manifest_source":
        envelope["source"] = "/etc/arbitrary"
    elif fault == "duplicate_json":
        envelope["document"] = '[{"uri":"first","uri":"second","sha256":"x","content_base64":""}]'
        envelope["sha256"] = hashlib.sha256(envelope["document"].encode()).hexdigest()
    else:
        envelope["document"] = json.dumps([asset()])
        envelope["sha256"] = hashlib.sha256(envelope["document"].encode()).hexdigest()
    with pytest.raises(ValueError):
        staged_collision_resource_pins(observations)
