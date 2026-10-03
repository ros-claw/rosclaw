"""Actual generic servo records: an incomplete earlier iteration cannot mask a repair."""

import json

import pytest

from benchmarks.harnessbench import dynamic_oracle, oracle
from benchmarks.harnessbench.tasks_v2 import V2_TASKS
from rosclaw.sim.backends.mujoco.backend import MujocoBackend


@pytest.fixture
def iterations(tmp_path, monkeypatch):
    task = V2_TASKS["R04"]
    backend = MujocoBackend(tmp_path)
    original = backend.load_model_xml(
        task.staged_files["model/unstable_servo.xml"], source={"kind": "fixture"}
    ).model_ref
    with pytest.raises(ValueError, match="SIM_DIVERGED"):
        backend.run_experiment(original, controller={"position_targets": [1]}, steps=500)
    candidates = []
    for kp in (100, 120):
        patched = backend.patch_model(
            original,
            [
                {
                    "op": "set",
                    "target": {"type": "actuator", "name": "srv"},
                    "field": "kp",
                    "value": kp,
                },
                {
                    "op": "set",
                    "target": {"type": "joint", "name": "j1"},
                    "field": "damping",
                    "value": 4,
                },
            ],
        ).new_model_ref
        receipt = backend.run_experiment(
            patched,
            controller={"position_targets": [1]},
            steps=500,
            budgets={"max_record_points": 501},
        )
        candidates.append((patched, receipt))
    monkeypatch.setattr(
        dynamic_oracle,
        "_runtime_parts",
        lambda *_: (oracle, backend, original, False, {p for p, _ in candidates}),
    )
    records = dynamic_oracle._records
    monkeypatch.setattr(
        dynamic_oracle,
        "_records",
        lambda b, refs, **kw: (
            records(b, refs, **kw)
            if original in refs
            else sorted(
                records(b, refs, **kw),
                key=lambda r: next(
                    i for i, (p, _) in enumerate(candidates) if p == r[1]["model_ref"]
                ),
            )
        ),
    )
    # This is a fixture tool transcript, not a claim of actual Kimi execution.
    report = backend.strict_replay(candidates[1][1].receipt_ref)
    directory = tmp_path / "rh/agent/sessions"
    directory.mkdir(parents=True)
    messages = [
        {
            "message": {
                "content": [
                    {
                        "type": "toolCall",
                        "id": "fixture_replay",
                        "name": "sim_strict_replay",
                        "arguments": {"receipt_ref": candidates[1][1].receipt_ref},
                    }
                ]
            }
        },
        {
            "message": {
                "role": "toolResult",
                "toolCallId": "fixture_replay",
                "content": [{"type": "text", "text": json.dumps(report)}],
            }
        },
    ]
    (directory / "fixture.jsonl").write_text("\n".join(json.dumps(m) for m in messages))
    return tmp_path, task, candidates


def test_later_replayed_candidate_survives_earlier_unreplayed_iteration(iterations):
    root, task, candidates = iterations
    (root / "answer.json").write_text("{}")
    verdict = dynamic_oracle.judge_dynamic_repair(root, task)
    assert verdict["verified_success"], verdict
    assert verdict["candidate_ref"] == candidates[1][0]
    assert verdict["physics_steps_by_oracle"] == 0


def test_claimed_complete_iteration_preferred_and_unreplayed_claim_rejected(iterations):
    root, task, candidates = iterations
    (root / "answer.json").write_text(json.dumps({"fixed_model_ref": candidates[1][0]}))
    assert dynamic_oracle.judge_dynamic_repair(root, task)["verified_success"]
    (root / "answer.json").write_text(json.dumps({"fixed_model_ref": candidates[0][0]}))
    verdict = dynamic_oracle.judge_dynamic_repair(root, task)
    assert not verdict["verified_success"]
    assert verdict["false_success"]
