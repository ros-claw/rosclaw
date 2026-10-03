"""A default-state replay cannot serve as evidence for a named keyframe repair."""

import json

import numpy as np
import pytest

from benchmarks.harnessbench import oracle
from benchmarks.harnessbench.keyframe_oracle import keyframe_only_patch, keyframe_rollout_evidence
from benchmarks.harnessbench.tasks_v2 import V2_TASKS
from rosclaw.sim.runtime import SimulationRuntime


def _workspace(tmp_path):
    task = V2_TASKS["R03"]
    path = tmp_path / "model/bad_reset.xml"
    path.parent.mkdir()
    path.write_text(task.staged_files["model/bad_reset.xml"])
    runtime = SimulationRuntime(tmp_path)
    loaded = runtime.load_model("model/bad_reset.xml")
    fixed = runtime.patch_model(
        loaded["model_ref"],
        [
            {
                "op": "set",
                "target": {"type": "keyframe", "name": "home"},
                "field": "qpos",
                "value": [0, 0, 0.05, 1, 0, 0, 0],
            }
        ],
    )["new_model_ref"]
    (tmp_path / "answer.json").write_text(json.dumps({"fixed_model_ref": fixed}))
    return runtime, fixed


def _replay_fixture(root, runtime, receipt):
    # Clearly synthetic transcript, not actual native-agent execution evidence.
    report = runtime.strict_replay(receipt["receipt_ref"])
    directory = root / "rh/agent/sessions"
    directory.mkdir(parents=True)
    messages = [
        {
            "message": {
                "content": [
                    {
                        "type": "toolCall",
                        "id": "fixture",
                        "name": "sim_strict_replay",
                        "arguments": {"receipt_ref": receipt["receipt_ref"]},
                    }
                ]
            }
        },
        {
            "message": {
                "role": "toolResult",
                "toolCallId": "fixture",
                "content": [{"type": "text", "text": json.dumps(report)}],
            }
        },
    ]
    (directory / "fixture.jsonl").write_text("\n".join(json.dumps(m) for m in messages))


def test_default_state_rollout_does_not_verify_keyframe_repair(tmp_path):
    runtime, fixed = _workspace(tmp_path)
    receipt = runtime.rollout(fixed, controller={"hold": True}, duration_s=0.5)
    _replay_fixture(tmp_path, runtime, receipt)
    verdict = oracle.judge("R03", tmp_path)
    assert not verdict["verified_success"], verdict


def test_actual_named_keyframe_rollout_and_native_replay_binding_pass(tmp_path):
    runtime, fixed = _workspace(tmp_path)
    receipt = runtime.rollout(fixed, keyframe="home", controller={"hold": True}, duration_s=0.5)
    _replay_fixture(tmp_path, runtime, receipt)
    verdict = oracle.judge("R03", tmp_path)
    assert verdict["verified_success"], verdict
    assert verdict["physics_steps_by_oracle"] == 0


@pytest.mark.parametrize("mutation", ["vector", "keyref", "trace_init", "validation", "replay"])
def test_keyframe_evidence_mutations_rejected(tmp_path, monkeypatch, mutation):
    runtime, fixed = _workspace(tmp_path)
    receipt = runtime.rollout(fixed, keyframe="home", controller={"hold": True}, duration_s=0.5)
    if mutation != "replay":
        _replay_fixture(tmp_path, runtime, receipt)
    backend = runtime.backend
    initial = backend.store.get(receipt["initial_state_ref"])
    keyref = initial["initialization"]["keyframe_ref"]
    vectorref = initial["state_vector_ref"]
    get = backend.store.get

    def altered(ref):
        value = get(ref)
        if mutation == "vector" and ref == vectorref:
            vector = np.frombuffer(value, dtype=np.float64).copy()
            vector[3] += 0.1
            return vector.tobytes()
        if mutation == "keyref" and ref == keyref:
            return {**value, "model_ref": "simmdl_" + "0" * 16}
        if ref == receipt["trace_ref"]:
            if mutation == "trace_init":
                return {**value, "initialization": {}}
            if mutation == "validation":
                return {**value, "runtime_validation": None}
        return value

    monkeypatch.setattr(backend.store, "get", altered)
    assert keyframe_rollout_evidence(backend, tmp_path, fixed, "home") is None


@pytest.mark.parametrize(
    "mutation",
    [
        ('mass="0.5"', 'mass="0.6"'),
        ("<worldbody>", '<option gravity="0 0 0"/><worldbody>'),
        ("<worldbody>", '<option timestep=".004"/><worldbody>'),
        ("<worldbody>", '<option integrator="RK4"/><worldbody>'),
        ('name="home" qpos=', 'name="home" qvel="1 0 0 0 0 0" qpos='),
    ],
)
def test_non_keyframe_physics_or_other_initializer_edits_rejected(mutation):
    source = V2_TASKS["R03"].staged_files["model/bad_reset.xml"]
    changed = source.replace('qpos="0 0 0.02', 'qpos="0 0 0.05')
    assert keyframe_only_patch(source, changed, "home")
    mutated = changed.replace(*mutation)
    assert mutated != changed
    assert not keyframe_only_patch(source, mutated, "home")
