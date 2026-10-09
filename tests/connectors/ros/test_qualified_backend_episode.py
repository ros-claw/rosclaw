"""Qualified host source/closure refusals; no ROS, World, service or model."""

import hashlib
import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.connectors.ros.test_dynamic_native_episode import specification


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return (
        importlib.import_module("qualified_backend_episode"),
        importlib.import_module("dynamic_native_episode"),
        importlib.import_module("backend_stack"),
    )


def protocol():
    spec = specification()
    spec["schema_version"] = "rosclaw.dynamic_native_episode.v2"
    spec["backend_source"] = {
        "contact_plugin_sha256": "f" * 64,
        "instrument_service_binary_sha256": "a" * 64,
        "source_mode": "ALL_STEP_SPATIAL_ORIGINAL_SERVICE_WIRE_REQUIRED",
    }
    return spec


def test_qualified_protocol_keeps_model_body_budget_and_base_merge_gate(modules):
    qualified, episode, _ = modules
    base, backend = qualified.validate_qualified_spec(protocol(), episode.validate_episode_spec)
    assert base == specification()
    assert backend == protocol()["backend_source"]


@pytest.mark.parametrize(
    "fault",
    [
        "missing_wire",
        "short_hash",
        "downgrade",
        "extra_backend",
        "extra_base",
        "bad_model",
        "wrong_schema",
    ],
)
def test_incomplete_or_downgraded_source_protocol_is_refused(modules, fault):
    qualified, episode, _ = modules
    spec = protocol()
    if fault == "missing_wire":
        spec["backend_source"].pop("instrument_service_binary_sha256")
    elif fault == "short_hash":
        spec["backend_source"]["contact_plugin_sha256"] = "bad"
    elif fault == "downgrade":
        spec["backend_source"]["source_mode"] = "LEGACY_SAMPLED"
    elif fault == "extra_backend":
        spec["backend_source"]["admitted"] = True
    elif fault == "extra_base":
        spec["ignore_fault"] = True
    elif fault == "bad_model":
        spec["model"] = ""
    else:
        spec["schema_version"] = "v1"
    with pytest.raises(ValueError):
        qualified.validate_qualified_spec(spec, episode.validate_episode_spec)


def test_original_binary_paths_missing_or_hash_changed_cannot_dispatch(modules, tmp_path):
    qualified, _, _ = modules
    lib, worker = tmp_path / "contact.so", tmp_path / "worker"
    # Explicit synthetic ELF bytes are never executable SDK/World evidence.
    lib.write_bytes(b"\x7fELFsynthetic_not_loadable")
    worker.write_bytes(b"\x7fELFsynthetic_not_executable")
    source = {
        "contact_plugin_sha256": hashlib.sha256(lib.read_bytes()).hexdigest(),
        "instrument_service_binary_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
    }
    assert qualified.frozen_backend_files(source, lib, worker) == {
        lib: lib.read_bytes(),
        worker: worker.read_bytes(),
    }
    with pytest.raises(ValueError):
        qualified.frozen_backend_files(source, None, worker)
    worker.write_bytes(worker.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        qualified.frozen_backend_files(source, lib, worker)


@pytest.fixture
def synthetic_projection(tmp_path):
    plan = {
        "run_id": "synthetic_not_world",
        "body_snapshot_hash": "a" * 64,
        "constraint_policy_hash": "b" * 64,
        "bundle_manifest_sha256": "c" * 64,
    }
    owner = {"loaded_source_correspondence": True, "bundle_manifest_sha256": "c" * 64}
    snapshot = {
        "sampled_monotonic_sec": time.monotonic(),
        "constraint_policy_hash": "b" * 64,
        "source_fault": None,
        "live_source_constraint_satisfied": True,
        "probe_completed_cache_cycles": 1,
        "robot_collision_count": 0,
        "probe_phase": "READY_FOR_LIFT",
        "probe_lift_transaction_pending": False,
    }
    projection = {
        "snapshot": snapshot,
        "actor_envelope": {"live_source_constraint_satisfied": True},
    }
    (tmp_path / "backend-observer").mkdir()
    (tmp_path / "backend-stack-source-plan.json").write_text(json.dumps(plan))
    (tmp_path / "backend-world-source-process.json").write_text(json.dumps(owner))
    latest = tmp_path / "backend-observer/backend-observation-latest.json"
    latest.write_text(json.dumps(projection))
    return tmp_path, plan, owner, projection, latest


@pytest.mark.parametrize(
    "fault",
    [
        "stale",
        "unknown_cycle",
        "collision",
        "fault_latched",
        "missing_mapped_world",
        "wrong_bundle",
        "wrong_policy",
        "actor_closed",
    ],
)
def test_unknown_or_faulted_original_source_cannot_qualify_native(
    modules, synthetic_projection, fault
):
    qualified, _, _ = modules
    root, _, owner, projection, latest = synthetic_projection
    snapshot = projection["snapshot"]
    if fault == "stale":
        snapshot["sampled_monotonic_sec"] -= 1
    elif fault == "unknown_cycle":
        snapshot["probe_completed_cache_cycles"] = 0
    elif fault == "collision":
        snapshot["robot_collision_count"] = 1
    elif fault == "fault_latched":
        (root / "backend-stack-fault.json").write_text("{}")
    elif fault == "missing_mapped_world":
        owner["loaded_source_correspondence"] = False
    elif fault == "wrong_bundle":
        owner["bundle_manifest_sha256"] = "d" * 64
    elif fault == "wrong_policy":
        snapshot["constraint_policy_hash"] = "e" * 64
    else:
        projection["actor_envelope"]["live_source_constraint_satisfied"] = False
    latest.write_text(json.dumps(projection))
    (root / "backend-world-source-process.json").write_text(json.dumps(owner))
    with pytest.raises(ValueError):
        qualified.qualified_projection(root)


def test_source_close_cannot_use_wrong_run_or_unfinished_actual_cycle(
    modules, synthetic_projection
):
    _, _, stack = modules
    root, plan, _, projection, _ = synthetic_projection
    children = SimpleNamespace(children=[], deadline=time.monotonic() + 10)
    closure = stack.SourceClosure(root, children, plan)
    request = {key: plan[key] for key in ("run_id", "body_snapshot_hash", "constraint_policy_hash")}
    request["run_id"] = "other"
    (root / "backend-close-source-request.json").write_text(json.dumps(request))
    with pytest.raises(ValueError, match="differs"):
        closure.requested()
    projection["snapshot"]["probe_lift_transaction_pending"] = True
    assert closure.close(json.dumps(projection).encode()) is False
    assert closure.closed is False and not (root / "backend-source-closed.json").exists()


def test_closed_qualification_refuses_live_writer_or_wrong_policy(modules, tmp_path):
    qualified, _, _ = modules
    (tmp_path / "backend-stack-source-plan.json").write_text(
        json.dumps({"constraint_policy_hash": "a" * 64})
    )
    (tmp_path / "backend-source-closed.json").write_text(
        json.dumps({"source_closed_for_replay": False})
    )
    with pytest.raises(ValueError, match="must close"):
        qualified.replay_closed_qualified(tmp_path)
    (tmp_path / "backend-source-closed.json").write_text(
        json.dumps(
            {
                "source_closed_for_replay": True,
                "authorization": False,
                "constraint_policy_hash": "b" * 64,
            }
        )
    )
    with pytest.raises(ValueError, match="policy differs"):
        qualified.replay_closed_qualified(tmp_path)


def test_original_source_writers_close_before_world_without_claiming_stop(
    modules, synthetic_projection, monkeypatch
):
    import os
    import signal

    _, _, stack = modules
    root, plan, _, projection, _ = synthetic_projection
    events = []

    class Child:
        returncode = None

        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = 0
            events.append(("writer_flushed", self.pid))

    monkeypatch.setattr(os, "killpg", lambda pid, sig: events.append((sig, pid)))
    world, controller, observer = Child(1), Child(2), Child(3)
    children = SimpleNamespace(
        children=[
            ("backend-gazebo", world),
            ("backend-owned-instrument", controller),
            ("backend-independent-observer", observer),
        ],
        deadline=time.monotonic() + 10,
    )
    closure = stack.SourceClosure(root, children, plan)
    assert closure.close(json.dumps(projection).encode()) is True
    assert events == [
        (signal.SIGINT, 2),
        ("writer_flushed", 2),
        (signal.SIGINT, 3),
        ("writer_flushed", 3),
    ]
    assert world.poll() is None
    record = json.loads((root / "backend-source-closed.json").read_text())
    assert record["physical_stop_proof"] == "NOT_MEASURED"
    assert record["physical_acceptance"] == "NOT_VERIFIED" and record["authorization"] is False


@pytest.mark.parametrize("case", ["D2", "D3", "D5", "D6"])
def test_qualified_v2_keeps_actual_prior_merge_check_before_any_process(
    modules, tmp_path, monkeypatch, case
):
    from tests.connectors.ros import test_dynamic_native_episode as original

    _, episode, _ = modules
    directory, protocol_path, plugin, urdf, home, spec = original.inputs.__wrapped__(tmp_path)
    spec["schema_version"] = "rosclaw.dynamic_native_episode.v2"
    if case == "D3":
        spec.update(
            schema_version="rosclaw.dynamic_native_episode.v3",
            case="D3",
            scenario_source={
                "second_target_xy": [-0.7, -0.7],
                "second_dwell_sim_sec": 20,
                "gap_sim_sec": 5,
            },
        )
    if case == "D5":
        spec.update(
            schema_version="rosclaw.dynamic_native_episode.v4",
            case="D5",
            scenario_source={"fault_kind": "OWNED_COLLECTION_PAUSE", "pause_wall_sec": 2},
        )
    if case == "D6":
        spec.update(
            schema_version="rosclaw.dynamic_native_episode.v5",
            case="D6",
            scenario_source={
                "exposure_policy": "PENDING_BLOCKED_ENABLED_BRUSH_AND_SAME_CELL_FREE_REVISIT"
            },
        )
    contact, worker = tmp_path / "contact.so", tmp_path / "worker"
    contact.write_bytes(b"\x7fELFsynthetic_not_loadable")
    worker.write_bytes(b"\x7fELFsynthetic_not_executable")
    spec["backend_source"] = {
        "source_mode": "ALL_STEP_SPATIAL_ORIGINAL_SERVICE_WIRE_REQUIRED",
        "contact_plugin_sha256": hashlib.sha256(contact.read_bytes()).hexdigest(),
        "instrument_service_binary_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
    }
    protocol_path.write_text(json.dumps(spec))
    called = []

    def command(argv, **kwargs):
        called.append(argv)
        if argv[:2] == ["git", "rev-parse"]:
            return spec["source_commit"]
        if argv[:2] == ["git", "status"]:
            return ""
        if argv[:3] == ["docker", "image", "inspect"]:
            return spec["image_id"]
        if argv[0] == "gh":
            return json.dumps({"merged": False, "merge_commit_sha": None})
        pytest.fail("unexpected dependency/process before actual merge check")

    monkeypatch.setattr(episode, "command", command)
    monkeypatch.setattr(episode.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        episode.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("unexpected process")
    )
    with pytest.raises(ValueError, match="actual reviewed P0 PR merge"):
        episode.run_episode(
            directory,
            protocol_path,
            plugin,
            urdf,
            home,
            contact_plugin=contact,
            instrument_service_binary=worker,
        )
    assert not directory.exists()
    assert any(argv[0] == "gh" for argv in called)


def test_d3_requires_qualified_original_wire_and_an_explicit_second_blocker(modules):
    qualified, episode, _ = modules
    spec = protocol()
    spec.update(
        schema_version="rosclaw.dynamic_native_episode.v3",
        case="D3",
        scenario_source={
            "second_target_xy": [-0.7, -0.7],
            "second_dwell_sim_sec": 20,
            "gap_sim_sec": 5,
        },
    )
    base, backend = qualified.validate_qualified_spec(spec, episode.validate_episode_spec)
    assert base["case"] == "D3" and base["scenario_source"] == spec["scenario_source"]
    assert backend["source_mode"] == "ALL_STEP_SPATIAL_ORIGINAL_SERVICE_WIRE_REQUIRED"
    with pytest.raises(ValueError):
        episode.validate_episode_spec(
            {**spec, "schema_version": "rosclaw.dynamic_native_episode.v1"}
        )


@pytest.mark.parametrize(
    "fault", ["case", "extra", "target", "nan", "dwell", "gap", "implicit", "downgrade"]
)
def test_unregistered_or_concurrent_d3_protocol_refuses_before_any_process(modules, fault):
    qualified, episode, _ = modules
    spec = protocol()
    spec.update(
        schema_version="rosclaw.dynamic_native_episode.v3",
        case="D3",
        scenario_source={
            "second_target_xy": [-0.7, -0.7],
            "second_dwell_sim_sec": 20,
            "gap_sim_sec": 5,
        },
    )
    if fault == "case":
        spec["case"] = "D1"
    elif fault == "extra":
        spec["scenario_source"]["reset_deadline"] = True
    elif fault == "target":
        spec["scenario_source"]["second_target_xy"] = spec["target_xy"]
    elif fault == "nan":
        spec["scenario_source"]["second_target_xy"] = [float("nan"), 0]
    elif fault == "dwell":
        spec["scenario_source"]["second_dwell_sim_sec"] = 9
    elif fault == "gap":
        spec["scenario_source"]["gap_sim_sec"] = 0
    elif fault == "implicit":
        spec.pop("scenario_source")
    else:
        spec["schema_version"] = "rosclaw.dynamic_native_episode.v2"
    with pytest.raises(ValueError):
        qualified.validate_qualified_spec(spec, episode.validate_episode_spec)


def test_d5_preregisters_only_owned_collector_pause_without_relaxing_backend(modules):
    qualified, episode, _ = modules
    spec = protocol()
    spec.update(
        schema_version="rosclaw.dynamic_native_episode.v4",
        case="D5",
        scenario_source={"fault_kind": "OWNED_COLLECTION_PAUSE", "pause_wall_sec": 2},
    )
    base, backend = qualified.validate_qualified_spec(spec, episode.validate_episode_spec)
    assert base["case"] == "D5" and base["scenario_source"] == spec["scenario_source"]
    assert base["mission_timeout_sec"] == specification()["mission_timeout_sec"]
    assert backend == protocol()["backend_source"]
    # Old public entry remains fail closed until the full D5 host is wired.
    with pytest.raises(ValueError):
        episode.validate_episode_spec(spec)


@pytest.mark.parametrize(
    "mutation",
    [
        {"pause_wall_sec": True},
        {"pause_wall_sec": 0},
        {"pause_wall_sec": 11},
        {"fault_kind": "WORLD_PAUSE"},
        {"pid": 1234},
        {"reset_deadline": True},
    ],
)
def test_d5_closed_policy_rejects_other_processes_and_deadline_changes(modules, mutation):
    qualified, episode, _ = modules
    spec = protocol()
    spec.update(
        schema_version="rosclaw.dynamic_native_episode.v4",
        case="D5",
        scenario_source={"fault_kind": "OWNED_COLLECTION_PAUSE", "pause_wall_sec": 2},
    )
    spec["scenario_source"].update(mutation)
    with pytest.raises(ValueError, match="closed qualified D5"):
        qualified.validate_qualified_spec(spec, episode.validate_episode_spec)
