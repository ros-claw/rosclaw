"""Synthetic geometric selection checks; never actual contact acceptance."""

import importlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("simulation_contact_sensitivity_fixture")


def sample():
    return {
        "inventory_complete": True,
        "collisions": [
            {
                "ancestry": [{"name": "other_body", "model": True}],
                "geometry_sdf": "<geometry><box><size>9 9 9</size></box></geometry>",
                "world_pose": [20, 20, 20, 0, 0, 0],
            },
            {
                "ancestry": [{"name": "declared_body", "model": True}],
                "geometry_sdf": "<geometry><box><size>.6 .3 .2</size></box></geometry>",
                "world_pose": [2, 3, 0.2, 0, 0, math.pi / 2],
            },
        ],
    }


def test_contact_surface_uses_declared_body_actual_world_pose_without_editing_body(module):
    original = sample()
    raw, collision = module.collider_bytes(original, "declared_body", "trial_collider")
    model = ET.fromstring(raw).find("model")
    assert model.get("name") == "trial_collider" and model.findtext("static") == "true"
    pose = [float(v) for v in model.findtext("pose").split()]
    assert pose[:3] == pytest.approx([2, 3.3, 0.2])
    assert collision == original["collisions"][1]
    assert original == sample()


@pytest.mark.parametrize("problem", ["incomplete", "missing", "tilted", "nonfinite"])
def test_ambiguous_or_invalid_surface_never_produces_test_collider(module, problem):
    original = sample()
    if problem == "incomplete":
        original["inventory_complete"] = False
    elif problem == "missing":
        original["collisions"] = original["collisions"][:1]
    elif problem == "tilted":
        original["collisions"][1]["world_pose"][3] = 0.2
    else:
        original["collisions"][1]["world_pose"][0] = math.nan
    with pytest.raises(ValueError):
        module.collider_bytes(original, "declared_body", "trial_collider")


@pytest.mark.parametrize("condition", ["clear", "active", "stale", "missing", "wrong_run"])
def test_raw_contacts_require_fresh_complete_streams_and_no_active_nonfloor_pair(
    module, tmp_path, monkeypatch, condition
):
    monkeypatch.setattr(module.time, "monotonic_ns", lambda: 1_000_000_000)
    sources = SimpleNamespace(
        run_id="run",
        contact_topics=("left", "right", "base"),
        ground_witness_topics=("left", "right"),
        ground_collision_names=("floor",),
    )
    rows = [
        {
            "run_id": "run",
            "topic": topic,
            "received_monotonic_ns": 950_000_000,
            "message": {"contacts": []},
        }
        for topic in ["left", "right"]
    ]
    if condition == "active":
        rows[0]["message"]["contacts"] = [
            {"collision1": {"name": "body"}, "collision2": {"name": "trial"}}
        ]
    elif condition == "stale":
        rows[0]["received_monotonic_ns"] = 1
    elif condition == "missing":
        rows.pop()
    elif condition == "wrong_run":
        rows[0]["run_id"] = "other"
    (tmp_path / "contact-messages-1.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    if condition == "wrong_run":
        with pytest.raises(ValueError, match="run/topic"):
            module.recent_original_contacts_clear(tmp_path, sources)
    else:
        assert module.recent_original_contacts_clear(tmp_path, sources) is (condition == "clear")
