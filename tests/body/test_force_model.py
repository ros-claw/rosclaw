"""Tests for generic ForceModel."""
from __future__ import annotations

from rosclaw.body.force_model import DofForceWindow, ForceBaseline, ForceModel


def test_net_force_subtracts_baseline():
    model = ForceModel(
        baseline={
            "thumb": ForceBaseline(mean=-50.0),
            "index": ForceBaseline(mean=-35.0),
        }
    )
    raw = {"thumb": 100.0, "index": 50.0, "middle": 0.0}
    net = model.net_force(raw)
    assert net["thumb"] == 150.0
    assert net["index"] == 85.0
    assert net["middle"] == 0.0


def test_contact_levels_with_window():
    model = ForceModel(
        contact_windows={
            "thumb": DofForceWindow(desired_min=80.0, desired_max=180.0, hard=250.0, emergency=350.0)
        }
    )
    assert model.contact_level(10.0, "thumb") == "none"
    assert model.contact_level(100.0, "thumb") == "desired"
    assert model.contact_level(200.0, "thumb") == "strong"
    assert model.contact_level(260.0, "thumb") == "hard"
    assert model.contact_level(400.0, "thumb") == "emergency"


def test_is_desired_contact_per_dof():
    model = ForceModel(
        contact_windows={
            "thumb": DofForceWindow(desired_min=80.0, desired_max=180.0),
            "index": DofForceWindow(desired_min=80.0, desired_max=200.0),
        }
    )
    assert model.is_desired_contact(190.0, "index")
    assert not model.is_desired_contact(190.0, "thumb")


def test_missing_baselines():
    model = ForceModel(
        baseline={
            "thumb": ForceBaseline(samples=10),
            "index": ForceBaseline(samples=0),
        }
    )
    assert model.list_missing_baselines(["thumb", "index"]) == ["index"]


def test_current_not_used_for_static_contact():
    model = ForceModel()
    assert model.policy["use_current_for_static_contact"] is False
    assert model.policy["use_force_for_static_contact"] is True
