from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from rosclaw.simforge.g1_recovery_dream import (
    build_g1_recovery_context_curriculum,
    route_g1_recovery_context,
    run_g1_recovery_dream_cycle,
)


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _context(index: int) -> dict[str, float | bool | str]:
    offset = 0.1 * (index // 2)
    return {
        "schema_version": "rosclaw.simforge.g1_recovery_context.v1",
        "policy_phase": 0.45 + offset,
        "pelvis_height_m": 0.80 - 0.02 * offset,
        "projected_gravity_x": offset,
        "projected_gravity_y": -offset,
        "projected_gravity_z": -0.98,
        "base_linear_velocity_x_mps": offset,
        "base_linear_velocity_y_mps": 0.02 * index,
        "base_linear_velocity_z_mps": 0.0,
        "base_angular_velocity_x_rps": 0.1 * index,
        "base_angular_velocity_y_rps": -0.05 * index,
        "base_angular_velocity_z_rps": 0.01 * index,
        "ball_speed_mps": 8.0 + index,
        "ball_direction_x": 1.0,
        "ball_direction_y": 0.0,
        "ball_direction_z": 0.0,
        "left_contact": True,
        "right_contact": index % 2 == 0,
    }


def _report() -> dict[str, object]:
    collection = []
    for index in range(6):
        collection.append(
            {
                "scenario_id": f"scenario-{index // 2}",
                "contact_context": _context(index),
                "naturalness_reduction": 0.10 if index % 2 == 0 else -0.02,
                "elite": index % 2 == 0,
                "task_preserved": True,
            }
        )
    return {
        "schema_version": "rosclaw.simforge.g1_recovery_awr_validation.v1",
        "decision": "REJECTED",
        "hardware_authorized": False,
        "body_hash": _digest("body"),
        "parent_policy_hash": _digest("controller-parent"),
        "stable_artifact_hash": _digest("stable"),
        "retained_recovery_artifact_hash": _digest("retained"),
        "candidate_artifact_hash": _digest("retained"),
        "motion_prior_artifact_hash": _digest("prior"),
        "online_replay_hash": _digest("replay"),
        "collection": collection,
        "blockers": ["development_candidate_found"],
        "checks": {
            "in_sample_value_converged": True,
            "actor_updated_only_from_elites": True,
            "development_candidate_found": False,
            "retained_parent_exact_before_contact": True,
            "sim_only_boundary_preserved": True,
            "independent_strict_validation_passed": False,
        },
        "validation_gate": {
            "passed": False,
            "mean_naturalness_reduction": 0.0,
            "reasons": ["no_development_candidate"],
        },
        "parent_development": [],
        "parent_validation": [],
        "candidate_validation": [],
        "trust_runs": [],
    }


def test_context_curriculum_is_deterministic_and_keeps_mixed_domains_separate() -> None:
    report = _report()
    first = build_g1_recovery_context_curriculum(
        report,
        source_report_hash=_digest("report"),
    )
    second = build_g1_recovery_context_curriculum(
        report,
        source_report_hash=_digest("report"),
    )

    assert first.curriculum_hash == second.curriculum_hash
    assert first.routing_ready
    assert first.valid_context_count == 6
    assert first.scenario_count == 3
    assert len(first.clusters) == 3
    assert sum(item.sample_count for item in first.clusters) == 6
    assert all(item.elite_count == 1 for item in first.clusters)
    routed = route_g1_recovery_context(first, _context(0))
    assert routed.eligible
    assert routed.expert_cluster_id is not None
    no_elite = replace(
        first,
        clusters=tuple(
            replace(item, elite_count=0)
            if item.cluster_id == routed.expert_cluster_id
            else item
            for item in first.clusters
        ),
    )
    unavailable = route_g1_recovery_context(no_elite, _context(0))
    assert not unavailable.eligible
    assert unavailable.fallback_reason == "cluster_has_no_qualified_expert_data"
    outlier = _context(0)
    outlier["ball_speed_mps"] = 1_000.0
    fallback = route_g1_recovery_context(first, outlier)
    assert not fallback.eligible
    assert fallback.fallback_reason == "outside_sealed_context_envelope"


def test_dream_cycle_journals_rejected_learning_without_activating(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    import rosclaw.simforge.g1_recovery_dream as module

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    output = tmp_path / "evidence" / "cycle"
    pilot = tmp_path / "pilot.json"
    pilot.write_text("{}", encoding="utf-8")
    stable = SimpleNamespace(
        body_hash=_digest("body"),
        parent_policy_hash=_digest("controller-parent"),
        artifact_hash=_digest("stable"),
    )
    retained = SimpleNamespace(
        body_hash=_digest("body"),
        parent_policy_hash=_digest("controller-parent"),
        artifact_hash=_digest("retained"),
    )
    prior = SimpleNamespace(body_hash=_digest("body"), artifact_hash=_digest("prior"))

    def load_torque(path: Path, **_: object):  # type: ignore[no-untyped-def]
        return stable if path.name.startswith("stable") else retained

    def runner(**kwargs: object) -> dict[str, object]:
        root = Path(str(kwargs["output_dir"]))
        root.mkdir(parents=True)
        report = _report()
        (root / "g1-recovery-awr-report.json").write_text(
            json.dumps(report, sort_keys=True),
            encoding="utf-8",
        )
        return report

    monkeypatch.setattr(module, "load_g1_neural_torque_artifact", load_torque)
    monkeypatch.setattr(module, "load_g1_motion_prior_artifact", lambda _: prior)
    result = run_g1_recovery_dream_cycle(
        asset_root=tmp_path / "assets",
        motion_prior_path=tmp_path / "prior.bin",
        motiondecode_pilot_report_path=pilot,
        stable_artifact_path=tmp_path / "stable.bin",
        recovery_artifact_path=tmp_path / "recovery.bin",
        output_dir=output,
        source_checkout=checkout,
        device="cpu",
        validation_runner=runner,
    )

    assert result["decision"] == "reject"
    assert result["scheduler_status"]["state"] == "completed"
    assert result["scheduler_status"]["candidate_artifact_hashes"] == []
    assert result["activation_authorized"] is False
    assert result["hardware_authorized"] is False
    disposition = json.loads((output / "growth-disposition.json").read_text())
    assert disposition["active_policy_unchanged"] is True
    assert disposition["evolution"]["state"] == "REJECTED"
