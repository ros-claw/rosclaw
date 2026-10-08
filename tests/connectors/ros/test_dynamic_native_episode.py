"""Host launcher refusal/orchestration contracts; no Docker/model dispatch."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance"


@pytest.fixture
def launcher(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "dynamic_episode_test", ROOT / "dynamic_native_episode.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def specification():
    return {
        "schema_version": "rosclaw.dynamic_native_episode.v1",
        "case": "D2",
        "source_commit": "a" * 40,
        "p0_merge_commit": "b" * 40,
        "image": "example/immutable:fixture",
        "image_id": "sha256:" + "c" * 64,
        "profile": "waffle",
        "seed": 200801,
        "plugin_sha256": "d" * 64,
        "vendor_urdf_sha256": "e" * 64,
        "mission_timeout_sec": 900,
        "coverage_preset": "perimeter_stateless",
        "repair_strategy": "pose_aware_robust",
        "target_xy": [0.7, 0.7],
        "dwell_sim_sec": 20,
        "introduce_after_cleaning_sim_sec": 10,
        "port": 22300,
        "domain": 181,
        "model_provider": "actual_provider",
        "model": "actual_model",
    }


def test_episode_protocol_retains_exact_source_scene_model_and_budget(launcher):
    spec = specification()
    assert launcher.validate_episode_spec(spec) == spec


@pytest.mark.parametrize(
    "key,value",
    [
        ("case", "D4"),
        ("profile", []),
        ("source_commit", "main"),
        ("p0_merge_commit", None),
        ("image", "x; touch /tmp/injected"),
        ("image_id", "latest"),
        ("plugin_sha256", "short"),
        ("vendor_urdf_sha256", "short"),
        ("mission_timeout_sec", True),
        ("mission_timeout_sec", 1801),
        ("coverage_preset", []),
        ("repair_strategy", "guessed"),
        ("target_xy", [float("nan"), 0]),
        ("target_xy", [True, 0]),
        ("dwell_sim_sec", 31),
        ("port", 65536),
        ("domain", 233),
        ("seed", False),
        ("model", ""),
        ("model_provider", "x\nsecret"),
    ],
)
def test_invalid_or_unimplemented_episode_refuses_before_dispatch(launcher, key, value):
    spec = specification()
    spec[key] = value
    with pytest.raises(ValueError):
        launcher.validate_episode_spec(spec)


@pytest.fixture
def inputs(tmp_path):
    plugin, urdf, protocol = (
        tmp_path / name for name in ("plugin.so", "vendor.urdf", "protocol.json")
    )
    plugin.write_bytes(b"\x7fELFsynthetic_offline_fixture")
    urdf.write_bytes(b'<robot name="synthetic"/>')
    spec = specification()
    spec.update(
        plugin_sha256=hashlib.sha256(plugin.read_bytes()).hexdigest(),
        vendor_urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
    )
    protocol.write_text(json.dumps(spec))
    home = tmp_path / "seed-home"
    (home / "agent").mkdir(parents=True)
    for name in ("settings.json", "models.json", "auth.json"):
        (home / "agent" / name).write_text("{}")
    return tmp_path / "episode", protocol, plugin, urdf, home, spec


@pytest.mark.parametrize(
    "fault", ["source", "dirty", "image", "unmerged", "wrong_merge", "live", "plugin"]
)
def test_actual_entry_checks_prior_merge_inputs_and_serial_execution_without_launch(
    launcher, inputs, monkeypatch, fault
):
    directory, protocol, plugin, urdf, home, spec = inputs
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["git", "rev-parse"]:
            return "foreign" if fault == "source" else spec["source_commit"]
        if argv[:2] == ["git", "status"]:
            return "dirty" if fault == "dirty" else ""
        if argv[:3] == ["docker", "image", "inspect"]:
            return "wrong" if fault == "image" else spec["image_id"]
        if argv[0] == "gh":
            return json.dumps(
                {
                    "merged": fault != "unmerged",
                    "merge_commit_sha": "wrong"
                    if fault == "wrong_merge"
                    else spec["p0_merge_commit"],
                }
            )
        if argv[:2] == ["docker", "ps"]:
            return "reh-n02-existing" if fault == "live" else ""
        pytest.fail("unexpected launch: " + repr(argv))

    monkeypatch.setattr(launcher, "command", command)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(
        launcher.subprocess, "Popen", lambda *a, **kw: pytest.fail("unexpected process")
    )
    if fault == "plugin":
        plugin.write_bytes(b"changed")
    with pytest.raises(ValueError):
        launcher.run_episode(directory, protocol, plugin, urdf, home)
    assert not directory.exists()
    assert not any(argv[:2] == ["docker", "run"] for argv in calls)


def test_failed_fixture_is_retained_and_owned_simulator_is_stopped(launcher, inputs, monkeypatch):
    directory, protocol, plugin, urdf, home, spec = inputs
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["git", "rev-parse"]:
            return spec["source_commit"]
        if argv[:2] == ["git", "status"]:
            return ""
        if argv[:3] == ["docker", "image", "inspect"]:
            return spec["image_id"]
        if argv[0] == "gh":
            return json.dumps({"merged": True, "merge_commit_sha": spec["p0_merge_commit"]})
        if argv[:2] == ["docker", "ps"]:
            return ""
        if argv[:2] == ["docker", "run"]:
            return "synthetic_container"
        if argv[:2] == ["docker", "stop"]:
            return "stopped"
        if argv[:2] == ["docker", "inspect"]:
            return "false"
        pytest.fail(repr(argv))

    def bootstrap(path, **kwargs):
        path.mkdir()
        data = {"binding": {"run_id": "synthetic", "obstacle_names": ["blocker"]}}
        (path / "physics.json").write_text(json.dumps(data))
        return data

    monkeypatch.setattr(launcher, "command", command)
    monkeypatch.setattr(launcher, "prepare_bindings", bootstrap)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
    monkeypatch.setattr(
        launcher, "wait_ready", lambda *a: (_ for _ in ()).throw(RuntimeError("incomplete source"))
    )
    result = launcher.run_episode(directory, protocol, plugin, urdf, home)
    assert result["status"] == "FAIL" and result["physical_acceptance"] == "NOT_VERIFIED"
    assert not result["autonomous_llm"] and not result["task_kernel_succeeded"]
    assert (directory / "dynamic-native-result.json").is_file()
    assert (directory / "physics.json").read_bytes() == (
        directory / "bootstrap/physics.json"
    ).read_bytes()
    assert sum(argv[:2] == ["docker", "stop"] for argv in calls) == 1


@pytest.mark.parametrize("fault", [None, "bootstrap", "launch", "sdk", "replay", "child", "world"])
def test_full_sequence_closes_world_before_replay_and_retains_failures(
    launcher, inputs, monkeypatch, fault
):
    directory, protocol, plugin, urdf, home, spec = inputs
    events = []

    def command(argv, **kwargs):
        if argv[:2] == ["git", "rev-parse"]:
            return spec["source_commit"]
        if argv[:2] == ["git", "status"]:
            return ""
        if argv[:3] == ["docker", "image", "inspect"]:
            return spec["image_id"]
        if argv[0] == "gh":
            return json.dumps({"merged": True, "merge_commit_sha": spec["p0_merge_commit"]})
        if argv[:2] == ["docker", "ps"]:
            return ""
        if argv[:2] == ["docker", "run"]:
            events.append("launch")
            if fault == "launch":
                raise RuntimeError("run response lost after creation")
            return "synthetic"
        if argv[:2] == ["docker", "stop"]:
            events.append("stop")
            if fault == "world":
                raise RuntimeError("shutdown fault")
            return "stopped"
        if argv[:2] == ["docker", "inspect"]:
            events.append("stopped_verified")
            return "false"
        pytest.fail(repr(argv))

    def bootstrap(path, **kwargs):
        path.mkdir()
        if fault == "bootstrap":
            raise RuntimeError("bootstrap fault")
        data = {"binding": {"run_id": "synthetic", "obstacle_names": ["blocker"]}}
        (path / "physics.json").write_text(json.dumps(data))
        (directory / "physics_binding.json").write_text("{}")
        return data

    class Process:
        next_pid = 10000

        def __init__(self, argv, **kwargs):
            Process.next_pid += 1
            self.pid = Process.next_pid
            self.argv = argv
            if argv[:2] == ["docker", "exec"]:
                assert argv[2:4] == ["-e", "PYTHONPATH=/workspace/src"]
            self.returncode = None
            self.observer = "pose_observer.py" in " ".join(argv)
            self.scenario = "dynamic_scenario.py" in " ".join(argv)
            if self.scenario:
                (directory / "dynamic-scenario-events.jsonl").touch()

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            if self.observer and fault == "child":
                raise RuntimeError("child cleanup fault")
            if any(a.endswith("configure_native.py") for a in self.argv):
                (directory / "home/agent").mkdir(parents=True)
            if any(Path(a).name == "native.py" for a in self.argv):
                events.append("native")
                if fault != "sdk":
                    (directory / "sdk-usage.json").write_text(
                        json.dumps([{"provider": spec["model_provider"], "model": spec["model"]}])
                    )
                (directory / "actions").mkdir()
                (directory / "actions/rosevidence_fixture.json").write_text("{}")
                (directory / "plan-events-fixture.jsonl").touch()
            if any(a.endswith("cleaning_acceptance.py") for a in self.argv):
                events.append("canonical")
            self.returncode = 0
            return 0

    def replay(*args):
        events.append("replay")
        assert events.index("canonical") < events.index("stop") < events.index("stopped_verified")
        if fault == "replay":
            raise ValueError("closed source fault")
        return {"status": "PASS_OFFLINE_MOCK_ONLY"}

    monkeypatch.setattr(launcher, "command", command)
    monkeypatch.setattr(launcher, "prepare_bindings", bootstrap)
    monkeypatch.setattr(launcher, "wait_ready", lambda *a: None)
    monkeypatch.setattr(launcher, "collect_stop_geometry", lambda *a: {"status": "MOCK"})
    monkeypatch.setattr(launcher, "replay_component_occupancy", replay)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    monkeypatch.setattr(launcher.os, "killpg", lambda *a: None)
    result = launcher.run_episode(directory, protocol, plugin, urdf, home)
    assert (directory / "dynamic-native-result.json").is_file()
    assert result["status"] == ("PASS" if fault is None else "FAIL")
    if fault == "bootstrap":
        assert "launch" not in events
    else:
        assert events.count("stop") == 1
    if fault in {"replay", "child", "world"}:
        assert result["task_kernel_succeeded"] is True
        assert result["physical_acceptance"] == "NOT_VERIFIED"
    if fault in {"child", "world"}:
        assert "replay" not in events
        assert result["cleanup_errors"]
