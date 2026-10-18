"""Synthetic mesh source predicates; no Gazebo or motion acceptance."""

import hashlib
import json
from copy import deepcopy

import pytest

from rosclaw.connectors.ros.context.collision_assets import staged_collision_resource_pins
from rosclaw.connectors.ros.context.sim_source_binding import review_sim_source_binding
from tests.connectors.ros import test_sim_source_binding as baseline
from tests.connectors.ros.test_collision_asset_originals import asset


def fixture(tmp_path, monkeypatch):
    original = baseline.source
    row = asset()
    row["uri"] = "/assets/collision-resources/" + row["sha256"] + ".stl"
    doc = json.dumps([row], indent=2) + "\n"
    envelope = {
        "schema_version": "rosclaw.prepared_collision_asset_observation.v1",
        "source": "/assets/collision-asset-originals.json",
        "sha256": hashlib.sha256(doc.encode()).hexdigest(),
        "document": doc,
    }

    def with_originals():
        model, raw, requirements, policy = original()
        raw = raw.replace(b'<box size="0.4 0.2 0.1"/>', f'<mesh filename="{row["uri"]}"/>'.encode())
        assert b"<mesh filename=" in raw
        digest = hashlib.sha256(raw).hexdigest()
        for value in model.observations["urdf_descriptions"].values():
            value.update(sha256=digest, size_bytes=len(raw))
        model.observations["prepared_collision_assets"] = deepcopy(envelope)
        model.seal()
        policy.update(source_urdf_sha256=digest, source_snapshot_hash=model.snapshot_hash)
        return model, raw, requirements, policy

    monkeypatch.setattr(baseline, "source", with_originals)
    parts = list(baseline.fixture(tmp_path))
    mesh = f"<mesh><uri>{row['uri']}</uri><scale>1 1 1</scale></mesh>"
    for i in (5, 6):
        parts[i] = parts[i].replace(b"<box><size>0.4 0.3 0.2</size></box>", mesh.encode())
    for sample in parts[4]:
        sample["collisions"][0]["geometry_sdf"] = "<geometry>" + mesh + "</geometry>"
    assets = parts[2]["asset_sha256"]
    assets.update(
        staged_collision_resource_pins(parts[-1].observations),
        **{"converted.sdf": baseline.sha(parts[5]), "world.sdf": baseline.sha(parts[6])},
    )
    parts[2]["conversion"]["output_sha256"] = assets["converted.sdf"]
    parts[3].update(assets_before=deepcopy(assets), assets_after=deepcopy(assets))
    return parts


def test_original_manifest_and_resource_pins_match_all_converted_and_loaded_meshes(
    tmp_path, monkeypatch
):
    root, request, kwargs = baseline.pack(fixture(tmp_path, monkeypatch))
    result = review_sim_source_binding(root, request, **kwargs)
    assert result["result"] == "MATCHED", result
    assert len(result["collision_resource_source_pins"]) == 2
    assert result["execution_authority"] == "NONE" and result["L1"] == result["L2"] == "NOT_RUN"
    assert result["collision_resource_authority"] == "NONE_REQUIRES_CURRENT_RESOURCE_CONTINUITY"


@pytest.mark.parametrize(
    "fault",
    [
        "asset_missing",
        "asset_sha",
        "document_missing",
        "document_sha",
        "loaded_uri",
        "loaded_scale",
        "selected_submesh",
        "negative_scale",
    ],
)
def test_mesh_source_mismatch_never_binds_body_or_grants_authority(tmp_path, monkeypatch, fault):
    parts = fixture(tmp_path, monkeypatch)
    resources = staged_collision_resource_pins(parts[-1].observations)
    mesh_path = next(k for k in resources if k.endswith(".stl"))
    if fault == "asset_missing":
        parts[2]["asset_sha256"].pop(mesh_path)
    elif fault == "asset_sha":
        parts[2]["asset_sha256"][mesh_path] = "0" * 64
    elif fault == "document_missing":
        parts[2]["asset_sha256"].pop("collision-asset-originals.json")
    elif fault == "document_sha":
        parts[2]["asset_sha256"]["collision-asset-originals.json"] = "0" * 64
    else:
        for sample in parts[4]:
            old = sample["collisions"][0]["geometry_sdf"]
            if fault == "loaded_uri":
                old = old.replace("/assets/collision-resources/", "/unbound/")
            elif fault == "loaded_scale":
                old = old.replace("1 1 1", "2 1 1")
            elif fault == "selected_submesh":
                old = old.replace(
                    "</mesh>", "<submesh><name>part</name><center>false</center></submesh></mesh>"
                )
            else:
                old = old.replace("1 1 1", "-1 1 1")
            sample["collisions"][0]["geometry_sdf"] = old
    parts[3].update(
        assets_before=deepcopy(parts[2]["asset_sha256"]),
        assets_after=deepcopy(parts[2]["asset_sha256"]),
    )
    root, request, kwargs = baseline.pack(parts)
    result = review_sim_source_binding(root, request, **kwargs)
    assert result["result"] != "MATCHED", result
    assert result["execution_authority"] == "NONE" and result["real_authority"] is False
