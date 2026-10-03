"""Named keyframe initialization uses the actual model-bound reset operator."""

import json
import os
import subprocess
from pathlib import Path

import mujoco
import pytest

from rosclaw.sim.backends.mujoco.backend import MujocoBackend
from rosclaw.sim.runtime import SimulationRuntime

XML = """<mujoco><option gravity="0 0 0" timestep=".002"/><worldbody>
<body name="box"><freejoint/><geom name="g" type="box" size=".1 .1 .1" mass="1"/></body>
</worldbody><keyframe><key name="home" time="2" qpos="0 0 .8 1 0 0 0" qvel=".1 0 0 0 0 0"/></keyframe></mujoco>"""


@pytest.fixture
def loaded(tmp_path):
    backend = MujocoBackend(tmp_path)
    ref = backend.load_model_xml(XML, source={"kind": "fixture"})
    return backend, ref.model_ref


def test_named_and_reference_initialization_match_actual_reset_operator(loaded):
    backend, ref = loaded
    selected = backend.initial_state_v2(ref, keyframe="home")
    meta = backend.store.get(selected)
    assert meta["initialization"]["kind"] == "keyframe"
    keyref = meta["initialization"]["keyframe_ref"]
    assert keyref.startswith("simkey_")
    initializer = backend.store.get(keyref)
    assert initializer["model_ref"] == ref
    repeated = backend.initial_state_v2(ref, keyframe_ref=keyref)
    assert repeated == selected
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    assert meta["time"] == data.time
    assert meta["qpos"] == list(data.qpos)
    assert meta["qvel"] == list(data.qvel)
    assert backend.store.get(backend.initial_state_v2(ref))["qpos"][2] == 0


def test_direct_rollout_and_snapshot_resume_keep_keyframe_lineage(loaded):
    backend, ref = loaded
    snap = backend.initial_state_v2(ref, keyframe="home")
    for args in [{"keyframe": "home"}, {"state_ref": snap}]:
        receipt = backend.run_experiment(
            ref, controller={"hold": True}, steps=5, audit=False, **args
        )
        initial = backend.store.get(receipt.initial_state_ref)
        assert initial["initialization"]["keyframe_name"] == "home"
        trace = backend.store.get(receipt.trace_ref)
        assert trace["initial_state_ref"] == receipt.initial_state_ref
        assert trace["initialization"] == initial["initialization"]
        assert trace["states"][0]["qpos"][2] == 0.8
        assert trace["states"][0]["qvel"][0] == 0.1
        assert trace["states"][0]["t"] == 2
        assert receipt.simulation_time_s == pytest.approx(0.01)
        assert receipt.metrics["duration_s"] == pytest.approx(0.01)
        assert backend.strict_replay(receipt.receipt_ref)["mode"] == "RAW_EXACT"


@pytest.mark.parametrize(
    "args",
    [
        {"keyframe": "missing"},
        {"keyframe": ""},
        {"keyframe": "home", "keyframe_ref": "simkey_" + "0" * 16},
    ],
)
def test_unknown_or_ambiguous_initializer_never_falls_back(loaded, args):
    backend, ref = loaded
    with pytest.raises(ValueError, match="KEYFRAME|INITIAL_STATE"):
        backend.initial_state_v2(ref, **args)


def test_cross_model_keyframe_reference_and_state_plus_keyframe_rejected(loaded):
    backend, ref = loaded
    snap = backend.initial_state_v2(ref, keyframe="home")
    keyref = backend.store.get(snap)["initialization"]["keyframe_ref"]
    child = backend.patch_model(
        ref, [{"op": "set", "target": {"type": "geom", "name": "g"}, "field": "mass", "value": 2}]
    ).new_model_ref
    with pytest.raises(ValueError, match="KEYFRAME_REF_MISMATCH"):
        backend.initial_state_v2(child, keyframe_ref=keyref)
    with pytest.raises(ValueError, match="INITIAL_STATE"):
        backend.run_experiment(
            ref, state_ref=snap, keyframe="home", controller={"hold": True}, steps=1
        )


def test_cli_selects_actual_named_keyframe_with_auditable_reference(tmp_path):
    backend = MujocoBackend(tmp_path)
    ref = backend.load_model_xml(XML, source={"kind": "fixture"}).model_ref
    repo = Path(__file__).resolve().parents[2]
    command = [
        str(repo / ".venv/bin/python"),
        "-m",
        "rosclaw.entrypoint",
        "sim",
        "--root",
        str(tmp_path),
        "snapshot",
        ref,
        "--keyframe",
        "home",
    ]
    process = subprocess.run(
        command,
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo / "src")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    payload = json.loads(process.stdout)
    assert payload["keyframe_ref"].startswith("simkey_")
    assert backend.store.get(payload["state_ref"])["qpos"][2] == 0.8
    runtime = SimulationRuntime(tmp_path)
    assert (
        runtime.snapshot(ref, keyframe_ref=payload["keyframe_ref"])["state_ref"]
        == payload["state_ref"]
    )
    assert (
        runtime.snapshot(ref, state_ref=payload["state_ref"])["state_ref"] == payload["state_ref"]
    )
    failed = subprocess.run(
        [*command[:-1], "missing"],
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo / "src")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert failed.returncode != 0
    assert "KEYFRAME_NOT_FOUND" in failed.stdout + failed.stderr


def test_native_mcp_client_passes_explicit_initializer_without_model_change(tmp_path, monkeypatch):
    import asyncio

    from rosclaw.mcp.adapters.runtime_client import RuntimeClient

    source = tmp_path / "fixture.xml"
    source.write_text(XML)
    monkeypatch.setenv("ROSCLAW_SIM_TASK_ROOT", str(tmp_path))
    client = RuntimeClient(project_root=tmp_path, robot_id=None, runtime_profile={})
    loaded = asyncio.run(client.sim_load_model("fixture.xml"))
    snapshot = asyncio.run(client.sim_snapshot(loaded["model_ref"], keyframe="home"))
    assert snapshot["keyframe_ref"].startswith("simkey_")
    receipt = asyncio.run(
        client.sim_rollout(
            loaded["model_ref"], {"hold": True}, steps=5, keyframe_ref=snapshot["keyframe_ref"]
        )
    )
    backend = MujocoBackend(tmp_path)
    initial = backend.store.get(receipt["initial_state_ref"])
    assert initial["qpos"][2] == 0.8
    assert initial["initialization"]["keyframe_ref"] == snapshot["keyframe_ref"]
    assert receipt["model_ref"] == loaded["model_ref"]
