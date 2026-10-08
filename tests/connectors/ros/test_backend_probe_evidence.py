"""Synthetic state-machine contracts; no backend or physical acceptance claim."""

import copy
import importlib
import json
from pathlib import Path

import pytest

from tests.connectors.ros.test_native_contact_components import packet_named, policy


@pytest.fixture
def probe_module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("backend_probe_evidence")


def specification():
    packet = packet_named("native_actual_contact_entity_names")
    native = policy(packet)
    native.update(
        schema_version="rosclaw.native_contact_policy.v2",
        component_topic="/instrument/probe_components",
        component_gz_topic="/rosclaw_sim/backend_probe_components",
    )
    return {
        "schema_version": "rosclaw.backend_cache_probe_policy.v1",
        "source": "simulator_operator_fixture_policy",
        "approved": True,
        "evidence_domain": "SIMULATION",
        "native_policy": native,
        "probe_xy": [6, 0],
        "sphere_radius_m": 0.05,
        "ground_z_m": 0,
        "lift_z_m": 10,
        "stable_samples": 10,
        "stable_sim_span_sec": 0.4,
        "refresh_sim_sec": 10,
        "refresh_wall_sec": 20,
    }


class Fixture:
    def __init__(self, module):
        self.probe = module.BackendCacheProbe(specification())
        self.packet = packet_named("native_actual_contact_entity_names")
        self.i = 0
        self.sim = 0.0
        self.wall = 100.0
        self.unix = 1_791_504_000_000_000_000

    def frame(self, z=0.05, touching=True, pose_time_offset=0, sim_step=0.05, wall_step=0.05):
        self.i += 1
        self.sim += sim_step
        self.wall += wall_step
        sim, wall = self.sim, self.wall
        packet = copy.deepcopy(self.packet)
        packet.update(
            sequence=self.i,
            iterations=self.i * 50,
            sim_time_sec=sim,
            captured_at_unix_ns=self.unix + round((wall - 100) * 1e9),
        )
        packet["body_world_pose"] = [6, 0, z, 1, 0, 0, 0]
        if not touching:
            packet["collision_contacts"][0]["contacts"] = []
        self.probe.pose(sim + pose_time_offset, wall, [6, 0, z, 1, 0, 0, 0])
        metadata = self.probe.observe(
            json.dumps(packet).encode(),
            received_monotonic_sec=wall + 0.001,
            received_unix_ns=packet["captured_at_unix_ns"] + 1_000_000,
        )
        return metadata

    def ground(self):
        for _ in range(12):
            self.frame()
        assert self.probe.phase == "READY_FOR_LIFT"

    def ack(self, **changes):
        ack = {
            "source": "owned_gazebo_world_set_pose_ack_not_pose_proof",
            "run_id": self.packet["run_id"],
            "model_name": self.packet["body_model_name"],
            "target_xyz": [6, 0, 10],
            "acknowledged_at_unix_ns": self.probe.previous[3],
            "request_sha256": "a" * 64,
            "service_success": True,
        }
        ack.update(changes)
        self.probe.acknowledge_lift(ack)

    def cycle(self):
        self.ground()
        self.ack()
        for i in range(12):
            self.frame(z=10 - 0.05 * i, touching=False)
        assert self.probe.phase == "WAIT_RECONTACT"
        self.ground()
        return self.probe.snapshot(100 + self.i * 0.05 + 0.002)


def test_full_actual_byte_pattern_needed_and_no_runtime_admission_granted(probe_module):
    fixture = Fixture(probe_module)
    initial = fixture.probe.snapshot(100)
    assert not initial["cache_update_pattern_observed"]
    fixture.ground()
    assert not fixture.probe.snapshot(100 + fixture.i * 0.05 + 0.002)[
        "cache_update_pattern_observed"
    ]
    fixture.ack()
    assert not fixture.probe.snapshot(100 + fixture.i * 0.05 + 0.002)[
        "cache_update_pattern_observed"
    ]
    for i in range(12):
        fixture.frame(z=10 - 0.05 * i, touching=False)
    assert fixture.probe.phase == "WAIT_RECONTACT"
    assert not fixture.probe.snapshot(100 + fixture.i * 0.05 + 0.002)[
        "cache_update_pattern_observed"
    ]
    fixture.ground()
    result = fixture.probe.snapshot(100 + fixture.i * 0.05 + 0.002)
    assert result["cache_update_pattern_observed"] and result["completed_cache_cycles"] == 1
    assert result["physical_acceptance"] == "NOT_VERIFIED" and result["authorization"] is False
    assert result["runtime_backend_admission"].startswith("NOT_VERIFIED")


def test_service_ack_without_measured_lift_or_clear_cache_is_not_evidence(probe_module):
    f = Fixture(probe_module)
    f.ground()
    f.ack()
    for _ in range(20):
        f.frame()
    assert f.probe.phase == "WAIT_CLEAR_AFTER_LIFT"
    assert not f.probe.snapshot(100 + f.i * 0.05 + 0.002)["cache_update_pattern_observed"]


def test_empty_cache_at_wrong_height_does_not_confirm_requested_lift(probe_module):
    f = Fixture(probe_module)
    f.ground()
    f.ack()
    for _ in range(20):
        f.frame(z=3, touching=False)
    assert not f.probe.lift_pose_confirmed
    assert f.probe.phase == "WAIT_CLEAR_AFTER_LIFT"


def test_lost_ground_readiness_cannot_arm_a_scene_lift(probe_module):
    f = Fixture(probe_module)
    f.ground()
    f.frame(z=0.2, touching=False)
    assert f.probe.phase == "WAIT_GROUND"
    with pytest.raises(ValueError, match="stable ground"):
        f.ack()


def test_lift_ack_cannot_use_an_expired_ground_contact(probe_module):
    f = Fixture(probe_module)
    f.ground()
    with pytest.raises(ValueError, match="acknowledgement"):
        f.ack(acknowledged_at_unix_ns=f.probe.previous[3] + 300_000_000)


@pytest.mark.parametrize(
    "fault", ["wrong_scope", "wrong_target", "no_success", "old_ack", "bad_hash", "wrong_source"]
)
def test_ack_identity_and_phase_are_closed_and_rejection_latches(probe_module, fault):
    f = Fixture(probe_module)
    f.ground()
    changes = {
        "wrong_scope": {"model_name": "robot"},
        "wrong_target": {"target_xyz": [0, 0, 0]},
        "no_success": {"service_success": False},
        "old_ack": {"acknowledged_at_unix_ns": 1},
        "bad_hash": {"request_sha256": "bad"},
        "wrong_source": {"source": "assistant_claim"},
    }[fault]
    with pytest.raises(ValueError):
        f.ack(**changes)
    assert f.probe.fault
    with pytest.raises(ValueError, match="latched"):
        f.frame()


@pytest.mark.parametrize(
    "fault",
    ["source_pause", "refresh_wall", "refresh_sim", "exact_time_mismatch", "stale_contact_cache"],
)
def test_admitted_pattern_loses_qualification_on_freshness_or_source_fault(probe_module, fault):
    f = Fixture(probe_module)
    assert f.cycle()["cache_update_pattern_observed"]
    if fault == "source_pause":
        result = f.probe.snapshot(100 + f.i * 0.05 + 0.31)
    elif fault == "refresh_wall":
        with pytest.raises(ValueError, match="refresh deadline"):
            for _ in range(202):
                f.frame(sim_step=0.025, wall_step=0.1)
        assert f.probe.previous[2] - f.probe.cycle_sim < 10
        result = f.probe.snapshot(f.wall + 0.002)
    elif fault == "refresh_sim":
        f.probe.cycle_sim -= 11
        result = f.probe.snapshot(100 + f.i * 0.05 + 0.002)
    elif fault == "exact_time_mismatch":
        with pytest.raises(ValueError, match="exact-time"):
            f.frame(pose_time_offset=0.001)
        result = f.probe.snapshot(100 + f.i * 0.05 + 0.002)
    else:
        f.ack()
        with pytest.raises(ValueError, match="inconsistent"):
            f.frame(z=10, touching=True)
        result = f.probe.snapshot(100 + f.i * 0.05 + 0.002)
    assert not result["cache_update_pattern_observed"] and result["source_fault"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("approved", False),
        ("evidence_domain", "REAL"),
        ("stable_samples", True),
        ("sphere_radius_m", 0),
        ("lift_z_m", 1),
        ("refresh_sim_sec", 100),
        ("probe_xy", [float("nan"), 0]),
    ],
)
def test_invalid_or_unbounded_probe_policy_is_refused(probe_module, field, value):
    s = specification()
    s[field] = value
    with pytest.raises(ValueError):
        probe_module.BackendCacheProbe(s)
