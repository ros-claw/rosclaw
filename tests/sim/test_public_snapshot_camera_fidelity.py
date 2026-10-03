"""Public default snapshots must bind camera evidence to an actual full initial state."""

import asyncio
import json
import os
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from benchmarks.harnessbench.tasks_v2 import V02_MODEL
from benchmarks.harnessbench.vision_oracle import verify_camera
from rosclaw.mcp.adapters.runtime_client import RuntimeClient
from rosclaw.sim.backends.mujoco.backend import MujocoBackend
from rosclaw.sim.runtime import SimulationRuntime


def _fixture(root):
    (root / "model").mkdir(exist_ok=True)
    (root / "model/seg_world.xml").write_text(V02_MODEL)


def _full_initial_state(backend, model_ref, state_ref):
    meta = backend.store.get(state_ref)
    assert meta["kind"] == "state_snapshot_v2"
    assert meta["fidelity"] == "FULL_INTEGRATION"
    assert meta["model_ref"] == model_ref
    model = mujoco.MjModel.from_xml_string(V02_MODEL)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    expected = np.zeros(mujoco.mj_stateSize(model, spec), dtype=np.float64)
    mujoco.mj_getState(model, data, expected, spec)
    assert backend.store.get(meta["state_vector_ref"]) == expected.tobytes()
    assert meta["time"] == 0


def test_public_runtime_default_snapshot_is_full_and_explicit_legacy_remains_partial(
    tmp_path, monkeypatch
):
    _fixture(tmp_path)
    runtime = SimulationRuntime(tmp_path)
    loaded = runtime.load_model("model/seg_world.xml")
    monkeypatch.setattr(
        mujoco,
        "mj_step",
        lambda *_: (_ for _ in ()).throw(AssertionError("snapshot must not step")),
    )
    snap = runtime.snapshot(loaded["model_ref"])
    _full_initial_state(runtime.backend, loaded["model_ref"], snap["state_ref"])
    legacy = runtime.backend.initial_state(loaded["model_ref"])
    resumed = runtime.snapshot(loaded["model_ref"], state_ref=legacy)
    assert runtime.backend.state_fidelity(resumed["state_ref"]) == "LEGACY_PARTIAL"


def test_actual_cli_default_snapshot_observe_passes_strong_camera_binding(tmp_path):
    _fixture(tmp_path)
    repo = Path(__file__).resolve().parents[2]
    prefix = [
        str(repo / ".venv/bin/python"),
        "-m",
        "rosclaw.entrypoint",
        "sim",
        "--root",
        str(tmp_path),
    ]

    def cli(*args):
        run = subprocess.run(
            [*prefix, *args],
            cwd=repo,
            env={**os.environ, "PYTHONPATH": str(repo / "src")},
            text=True,
            capture_output=True,
            timeout=60,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        return json.loads(run.stdout)

    model = cli("load", "model/seg_world.xml")["model_ref"]
    snapshot = cli("snapshot", model)
    backend = MujocoBackend(tmp_path)
    _full_initial_state(backend, model, snapshot["state_ref"])
    observed = cli("observe", model, snapshot["state_ref"], "--channels", "camera_segmentation:cam")
    camera = observed["values"]["camera_segmentation:cam"]
    truth = verify_camera(
        backend,
        camera["observation_manifest_ref"],
        source_xml=V02_MODEL,
        camera="cam",
        body="blue_box",
        kind="segmentation",
    )
    assert truth["pixel_count"] > 4 and truth["physics_steps_by_oracle"] == 0


def test_native_mcp_default_snapshot_has_full_integration_binding(tmp_path, monkeypatch):
    _fixture(tmp_path)
    monkeypatch.setenv("ROSCLAW_SIM_TASK_ROOT", str(tmp_path))
    monkeypatch.setattr(
        mujoco,
        "mj_step",
        lambda *_: (_ for _ in ()).throw(AssertionError("snapshot must not step")),
    )
    client = RuntimeClient(project_root=tmp_path, robot_id=None, runtime_profile={})
    loaded = asyncio.run(client.sim_load_model("model/seg_world.xml"))
    snapshot = asyncio.run(client.sim_snapshot(loaded["model_ref"]))
    _full_initial_state(MujocoBackend(tmp_path), loaded["model_ref"], snapshot["state_ref"])
