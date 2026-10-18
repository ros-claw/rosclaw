"""Small references require exact separate originals; static caches confer no L1."""

import hashlib
import importlib
import json
from collections import OrderedDict
from pathlib import Path

import pytest

from rosclaw.connectors.ros.context import collision_assets as assets
from tests.connectors.ros.test_collision_asset_originals import asset


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    monkeypatch.setattr(assets, "_REGISTERED_ORIGINALS", OrderedDict())


def document(indent=2):
    row = asset()
    row["uri"] = "/assets/collision-resources/" + row["sha256"] + ".stl"
    raw = json.dumps([row], indent=indent).encode()
    return raw, hashlib.sha256(raw).hexdigest(), row


def test_reference_is_small_and_original_facts_exactly_equal_legacy():
    raw, digest, _ = document()
    legacy = {
        "schema_version": "rosclaw.prepared_collision_asset_observation.v1",
        "source": "/assets/collision-asset-originals.json",
        "sha256": digest,
        "document": raw.decode(),
    }
    reference = assets.register_prepared_collision_originals(raw, digest)
    assert "document" not in reference and len(json.dumps(reference)) < 256
    assert assets.collision_assets_from_observations(
        {"prepared_collision_assets": reference}
    ) == assets.collision_assets_from_observations({"prepared_collision_assets": legacy})
    assert assets.staged_collision_resource_pins(
        {"prepared_collision_assets": reference}
    ) == assets.staged_collision_resource_pins({"prepared_collision_assets": legacy})


def test_unregistered_reference_never_reads_or_fetches_any_path():
    raw, digest, _ = document()
    ref = assets.register_prepared_collision_originals(raw, digest)
    assets._REGISTERED_ORIGINALS.clear()
    with pytest.raises(assets.PreparedCollisionOriginalsRequiredError, match=digest):
        assets.collision_assets_from_observations({"prepared_collision_assets": ref})


@pytest.mark.parametrize(
    "changed",
    [
        {"size_bytes": True},
        {"size_bytes": 0},
        {"size_bytes": 5_000_001},
        {"size_bytes": 1},
        {"sha256": "0" * 64},
        {"source": "/etc/passwd"},
        {"document": "invented"},
        {"schema_version": "unknown"},
    ],
)
def test_reference_type_size_digest_source_and_extra_fields_reject(changed):
    raw, digest, _ = document()
    ref = assets.register_prepared_collision_originals(raw, digest)
    with pytest.raises(ValueError):
        assets.collision_assets_from_observations({"prepared_collision_assets": {**ref, **changed}})


def test_wrong_original_hash_cannot_register_or_resolve_and_registry_is_bounded():
    raw, digest, _ = document()
    with pytest.raises(ValueError):
        assets.register_prepared_collision_originals(raw, "0" * 64)
    refs = []
    for indent in (1, 2, 3):
        raw, digest, _ = document(indent)
        refs.append(assets.register_prepared_collision_originals(raw, digest))
    assert len(assets._REGISTERED_ORIGINALS) == 2
    with pytest.raises(assets.PreparedCollisionOriginalsRequiredError):
        assets.collision_assets_from_observations({"prepared_collision_assets": refs[0]})
    assert assets.collision_assets_from_observations({"prepared_collision_assets": refs[-1]})


def test_sdk_continuity_registers_actual_document_and_latches_file_replacement(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("simulation_collision_resources")
    raw, digest, row = document()
    ref = assets.register_prepared_collision_originals(raw, digest)
    assets._REGISTERED_ORIGINALS.clear()
    (tmp_path / "collision-resources").mkdir()
    (tmp_path / "collision-asset-originals.json").write_bytes(raw)
    import base64

    (tmp_path / "collision-resources" / (row["sha256"] + ".stl")).write_bytes(
        base64.b64decode(row["content_base64"])
    )
    guard = module.CollisionResourceContinuity(tmp_path, {"prepared_collision_assets": ref})
    assets._REGISTERED_ORIGINALS.clear()
    assert guard.verify()
    original = tmp_path / "collision-asset-originals.json"
    original.unlink()
    original.write_bytes(raw)
    with pytest.raises(ValueError, match="file identity changed"):
        guard.verify()
    with pytest.raises(ValueError, match="permanently latched"):
        guard.verify()


def test_explicit_attachment_preserves_snapshot_hash_and_missing_originals_stay_unknown():
    from rosclaw.connectors.ros.context.discovery import discover_body_candidate
    from tests.connectors.ros.test_navigation_body_workspace import NOW, source

    raw_doc, digest, row = document()
    reference = assets.register_prepared_collision_originals(raw_doc, digest)
    model, urdf, _, _ = source()
    urdf = urdf.replace(
        b'<box size="0.4 0.2 0.1"/>',
        f'<mesh filename="{row["uri"]}" scale="0.01 0.01 0.01"/>'.encode(),
    )
    for value in model.observations["urdf_descriptions"].values():
        value.update(sha256=hashlib.sha256(urdf).hexdigest(), size_bytes=len(urdf))
    model.observations["prepared_collision_assets"] = reference
    model.seal()
    original_hash = model.snapshot_hash
    assets._REGISTERED_ORIGINALS.clear()
    unknown = discover_body_candidate(model, urdf, now=NOW)
    assert unknown["geometry"]["complete"] is False
    assert any(
        "explicit original collision bytes" in reason for reason in unknown["unknown_fields"]
    )
    assert assets.bind_prepared_collision_originals(model.observations, raw_doc) == digest
    complete = discover_body_candidate(model, urdf, now=NOW)
    assert complete["geometry"]["complete"] is True
    assert model.compute_snapshot_hash() == original_hash
    assert "document" not in model.to_dict()["observations"]["prepared_collision_assets"]


def test_cli_originals_are_explicit_bounded_and_must_match_reference(tmp_path):
    from argparse import Namespace
    from types import SimpleNamespace

    from rosclaw.connectors.ros.cli.ros_cli import _bind_model_originals

    raw, digest, _ = document()
    ref = assets.register_prepared_collision_originals(raw, digest)
    assets._REGISTERED_ORIGINALS.clear()
    model = SimpleNamespace(observations={"prepared_collision_assets": ref})
    assert _bind_model_originals(model, Namespace()) is None
    path = tmp_path / "originals.json"
    path.write_bytes(raw)
    assert _bind_model_originals(model, Namespace(asset_originals=str(path))) == digest
    path.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="length differs|SHA-pinned"):
        _bind_model_originals(model, Namespace(asset_originals=str(path)))
