"""A calibration task must provide observations independent of its base model."""

import json

import pytest

from benchmarks.harnessbench import oracle
from benchmarks.harnessbench.runner import stage_workspace
from rosclaw.sim.runtime import SimulationRuntime


def test_s01_staged_observations_recover_true_damping_without_truth_model(tmp_path):
    work = stage_workspace(tmp_path, "S01")
    observations = json.loads((work / "observations/sysid.json").read_text())
    runtime = SimulationRuntime(work)
    base = runtime.load_model("model/sysid_bot.xml")
    assert len(runtime.backend.store.list_children("models")) == 1
    receipt = runtime.sysid(
        {
            "schema_version": "rosclaw.sim.sysid_spec.v1",
            "base_model_ref": base["model_ref"],
            "dataset_ref": observations["dataset_ref"],
            "parameters": [{"type": "joint_damping", "joint": "hinge", "min": 0.01, "max": 2.0}],
            "train_sequences": [0, 1],
            "holdout_sequences": [2],
        }
    )
    assert receipt["parameters_after"]["hinge_damping"] == pytest.approx(0.3, abs=0.05)
    assert receipt["verdict"] == "IMPROVED"
    assert receipt["holdout_improvement"] > 0.5
    assert not list(work.rglob("truth.xml"))
    (work / "answer.json").write_text(
        json.dumps(
            {
                "identified_damping": receipt["parameters_after"]["hinge_damping"],
                "sysid_receipt_ref": receipt["receipt_ref"],
            }
        )
    )
    # A later unrelated receipt cannot replace the explicitly reported result.
    runtime.backend.store.put(
        "experiments",
        {
            **receipt,
            "parameters_after": {"hinge_damping": 0.01},
            "receipt_ref": "",
        },
    )
    assert oracle.judge("S01", work)["verified_success"] is True


def test_both_legs_receive_identical_raw_observations(tmp_path):
    a = stage_workspace(tmp_path / "a", "S01", leg="A")
    b = stage_workspace(tmp_path / "b", "S01", leg="B")
    assert (a / "observations/sysid.json").read_bytes() == (
        b / "observations/sysid.json"
    ).read_bytes()
    assert not (a / "sim").exists()
    assert (b / "sim/traces").is_dir()


def test_static_observation_task_has_actual_zero_motion(tmp_path):
    work = stage_workspace(tmp_path, "S02")
    observations = json.loads((work / "observations/sysid.json").read_text())
    assert observations["sequences"]
    assert all(
        state["qpos"] == [0.0] and state["qvel"] == [0.0]
        for seq in observations["sequences"]
        for state in seq["states"]
    )


def test_s01_rejects_fitting_data_generated_from_base_instead_of_observations(tmp_path):
    work = stage_workspace(tmp_path, "S01")
    runtime = SimulationRuntime(work)
    base = runtime.load_model("model/sysid_bot.xml")
    own_data = runtime.backend.record_dataset(
        base["model_ref"],
        sequences=[
            {"controller": {"hold": True}, "duration_s": 1.0, "qpos0": [q]}
            for q in [0.6, -0.4, 0.9]
        ],
    )
    receipt = runtime.sysid(
        {
            "schema_version": "rosclaw.sim.sysid_spec.v1",
            "base_model_ref": base["model_ref"],
            "dataset_ref": own_data,
            "parameters": [{"type": "joint_damping", "joint": "hinge", "min": 0.01, "max": 2.0}],
            "train_sequences": [0, 1],
            "holdout_sequences": [2],
        }
    )
    (work / "answer.json").write_text(
        json.dumps(
            {
                "identified_damping": 0.3,
                "sysid_receipt_ref": receipt["receipt_ref"],
            }
        )
    )
    verdict = oracle.judge("S01", work)
    assert verdict["verified_success"] is False
    assert verdict["reason"] == "sysid_observation_dataset_mismatch"
