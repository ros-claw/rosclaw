"""Recorded validation must describe actual execution, including rejected resets."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from rosclaw.sim.backends.mujoco.backend import MujocoBackend

XML = """<mujoco><option timestep=".002"/><worldbody>
<body pos="0 0 .4"><joint name="j" axis="0 1 0" damping=".01"/>
<geom type="capsule" size=".04 .2" pos="0 0 -.2" mass="1.2"/>
</body></worldbody><actuator><position joint="j" kp="2000000"/>
</actuator></mujoco>"""


def test_success_records_actual_serial_validation(tmp_path):
    backend = MujocoBackend(tmp_path)
    model = backend.load_model_xml(
        XML.replace('kp="2000000"', 'kp="10"'), source={"kind": "fixture"}
    )
    receipt = backend.run_experiment(model.model_ref, controller={"hold": True}, steps=10)
    trace = backend.store.get(receipt.trace_ref)
    validation = trace["runtime_validation"]
    assert receipt.runtime_validation == validation
    assert validation["status"] == "PASS"
    assert validation["method"] == "serial_each_step"
    assert validation["steps_checked"] == 10
    assert validation["warning_counts"] == [0] * 7
    assert validation["per_step_warning_check"] is True
    assert validation["actual_elapsed_s"] == pytest.approx(0.02)
    assert validation["expected_elapsed_s"] == pytest.approx(0.02)
    assert "qacc" in validation["checked_finite_fields"]


@pytest.mark.parametrize("route", ["rollout", "run_experiment"])
def test_failed_solver_reset_keeps_inspectable_partial_evidence(tmp_path, route):
    backend = MujocoBackend(tmp_path)
    model = backend.load_model_xml(XML, source={"kind": "fixture"})
    with pytest.raises(ValueError, match="SIM_DIVERGED") as caught:
        getattr(backend, route)(model.model_ref, controller={"position_targets": [1]}, steps=500)
    failure_ref = re.search(r"failure_ref=(simexp_[a-f0-9]{16})", str(caught.value))
    assert failure_ref, str(caught.value)
    failure = backend.store.get(failure_ref[1])
    assert failure["kind"] == "simulation_failure"
    assert failure["simulation_valid"] is False
    assert failure["task_success"] is None
    assert failure["physical_audit_pass"] is None
    trace = backend.store.get(failure["trace_ref"])
    assert trace["kind"] == "failed_simulation_trace"
    assert trace["model_ref"] == model.model_ref
    assert trace["controller"] == {"position_targets": [1]}
    assert trace["failed_step"] == 3
    assert trace["failure_point"]["warning_counts"][int(mujoco.mjtWarning.mjWARN_BADQACC)] > 0
    assert trace["last_valid_state"]["t"] == pytest.approx(0.004)
    assert abs(trace["last_valid_state"]["qvel"][0]) > 1e6
    assert trace["failure_point"]["finite_fields"]["qvel"] is True
    assert failure["usable_for_real_execution"] is False
    assert backend.store.verify_digest(failure_ref[1])
    assert backend.store.verify_digest(failure["trace_ref"])
    with pytest.raises(ValueError, match="not a simulation trace"):
        backend.audit(model.model_ref, trace_ref=failure["trace_ref"])
    with pytest.raises(ValueError, match="not a simulation receipt"):
        backend.strict_replay(failure_ref[1])
    with pytest.raises(ValueError, match="failure_ref=" + failure_ref[1]):
        getattr(backend, route)(model.model_ref, controller={"position_targets": [1]}, steps=500)


def test_nonfinite_failure_is_json_safe_and_preserves_actual_field(monkeypatch, tmp_path):
    backend = MujocoBackend(tmp_path)
    model = backend.load_model_xml(
        XML.replace('kp="2000000"', 'kp="10"'), source={"kind": "fixture"}
    )
    actual_step = mujoco.mj_step

    def inject(model, data):
        actual_step(model, data)
        data.qacc[0] = np.nan

    monkeypatch.setattr(mujoco, "mj_step", inject)
    with pytest.raises(ValueError, match="SIM_DIVERGED") as caught:
        backend.rollout(model.model_ref, controller={"hold": True}, steps=10)
    trace_ref = re.search(r"trace_ref=(simtrc_[a-f0-9]{16})", str(caught.value))[1]
    trace = backend.store.get(trace_ref)
    assert trace["failure_point"]["qacc"] == ["NaN"]
    assert trace["failure_point"]["finite_fields"]["qacc"] is False
    assert trace["failure_point"]["warning_counts"] == [0] * 7
    assert trace["last_valid_state"]["t"] == 0
    assert trace["failed_step"] == 1


def test_native_cli_exposes_failed_refs_without_success_receipt(tmp_path):
    backend = MujocoBackend(tmp_path)
    ref = backend.load_model_xml(XML, source={"kind": "fixture"}).model_ref
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rosclaw.entrypoint",
            "sim",
            "--root",
            str(tmp_path),
            "rollout",
            ref,
            "--controller",
            '{"position_targets":[1]}',
            "--steps",
            "500",
        ],
        env={**os.environ, "PYTHONPATH": str(repo / "src")},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode != 0
    error = json.loads(result.stderr.splitlines()[-1])
    assert error["ok"] is False
    assert "failure_ref=simexp_" in error["error"]
    assert "trace_ref=simtrc_" in error["error"]
    assert not result.stdout.strip()
