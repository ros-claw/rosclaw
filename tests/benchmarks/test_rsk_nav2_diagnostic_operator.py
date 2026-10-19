"""Synthetic closed-owner failures: all Docker calls are intercepted."""

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.ros_expert_ablation import rsk_nav2_diagnostic_operator as owner
from benchmarks.ros_expert_ablation.rsk_episode_runner import OriginalRef


def reference(root, path):
    return OriginalRef(
        path=str(path.relative_to(root)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )


def recipe():
    return {
        "source_revision": "a" * 40,
        "assets_directory": "assets",
        "observer_directory": "observer",
        "assets_inventory": {"path": "assets.json", "sha256": "b" * 64},
        "observer_inventory": {"path": "observer.json", "sha256": "c" * 64},
        "source_urdf_sha256": "d" * 64,
        "role": "readonly_localized_observation",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("role", "rsk_tf_future_stamp"),
        ("command", "ros2 topic pub"),
        ("script", "untrusted.py"),
        ("max_goals", 1),
        ("max_new_physics_worlds", 2),
        ("max_model_calls", 1),
        ("max_goals", False),
        ("max_new_physics_worlds", True),
        ("seed", True),
    ],
)
def test_recipe_cannot_accept_commands_fault_labels_or_larger_budgets(field, value):
    data = recipe()
    data[field] = value
    with pytest.raises(ValueError):
        owner.Nav2ObservationRecipeV1.model_validate(data)


@pytest.mark.parametrize(
    "mutation", ["bytes", "size", "bool-size", "extra", "missing", "symlink", "fifo"]
)
def test_complete_inventory_rejects_changed_or_special_originals(tmp_path, mutation):
    directory = tmp_path / "assets"
    directory.mkdir()
    path = directory / "data"
    path.write_bytes(b"original")
    rows = {
        "data": {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    }
    inventory = tmp_path / "inventory.json"
    if mutation == "size":
        rows["data"]["bytes"] += 1
    if mutation == "bool-size":
        rows["data"]["bytes"] = True
    inventory.write_text(json.dumps(rows))
    if mutation == "bytes":
        path.write_bytes(b"modified")
    elif mutation == "extra":
        (directory / "unregistered").write_bytes(b"extra")
    elif mutation == "missing":
        path.unlink()
    elif mutation == "symlink":
        (directory / "link").symlink_to(path)
    elif mutation == "fifo":
        import os

        os.mkfifo(directory / "fifo")
    with pytest.raises(ValueError):
        owner._inventory(tmp_path, directory, reference(tmp_path, inventory))


def test_complete_original_inventory_and_relative_directory_accepted(tmp_path):
    directory = tmp_path / "assets"
    directory.mkdir()
    (directory / "data").write_bytes(b"original")
    rows = {"data": {"sha256": hashlib.sha256(b"original").hexdigest(), "bytes": 8}}
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(rows))
    assert owner._inventory(tmp_path, directory, reference(tmp_path, inventory)) == rows
    assert owner._directory(tmp_path, "assets") == directory
    for value in ["../assets", str(directory)]:
        with pytest.raises(ValueError):
            owner._directory(tmp_path, value)
    (tmp_path / "link").symlink_to(directory)
    with pytest.raises(ValueError, match="symlink"):
        owner._directory(tmp_path, "link")


def operator(tmp_path, monkeypatch):
    directory = tmp_path / "assets"
    directory.mkdir()
    (directory / "navigation-policy.json").write_text(
        '{"body_id":"synthetic","navigation_limits":{}}'
    )
    original = tmp_path / "recipe.json"
    original.write_text(json.dumps(recipe()))
    monkeypatch.setattr(owner.Nav2SourceOnlyDiagnosticOperator, "check_ready", lambda self: None)
    op = owner.Nav2SourceOnlyDiagnosticOperator(
        recipe=reference(tmp_path, original),
        worktree=Path(owner.__file__).resolve().parents[2],
        evidence_root=tmp_path,
        output_parent=tmp_path,
    )
    op.assets, op.observer = directory, tmp_path
    monkeypatch.setattr(op, "_git", lambda *args: str(tmp_path / "common.git"))
    monkeypatch.setattr(op, "_prepare_assets", lambda *args: None)
    selected = SimpleNamespace(
        request=SimpleNamespace(
            provider="EXISTING_NATIVE_READ_ONLY_PROBE", operation="CLOCK_TF_WINDOW", id="synthetic"
        )
    )
    plan = SimpleNamespace(
        probes=[selected], model_dump=lambda **kwargs: {"scope": "SYNTHETIC_NO_ROS"}
    )
    return op, plan, selected


@pytest.mark.parametrize("budget", [None, True, float("nan"), float("inf"), 59, 301])
def test_invalid_budget_consumes_no_attempt_and_starts_nothing(tmp_path, monkeypatch, budget):
    op, plan, selected = operator(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        op.collect(plan, selected, remaining_sec=budget)
    assert op.calls == 0 and op.last_world is None and op.started_containers == 0


def test_missing_recipe_inventory_fails_before_any_slot(tmp_path, monkeypatch):
    original = tmp_path / "recipe.json"
    original.write_text(json.dumps(recipe()))
    monkeypatch.setattr(
        owner.Nav2SourceOnlyDiagnosticOperator,
        "_git",
        lambda _, *args: "a" * 40 if args[0] == "rev-parse" else "",
    )
    with pytest.raises(FileNotFoundError):
        owner.Nav2SourceOnlyDiagnosticOperator(
            recipe=reference(tmp_path, original),
            worktree=Path(owner.__file__).resolve().parents[2],
            evidence_root=tmp_path,
            output_parent=tmp_path,
        )


def test_failed_create_preserves_failure_and_consumed_attempt_without_retry(tmp_path, monkeypatch):
    op, plan, selected = operator(tmp_path, monkeypatch)
    calls = []

    def reject(argv, **kwargs):
        calls.append(argv)
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(owner.subprocess, "check_output", reject)
    with pytest.raises(subprocess.CalledProcessError):
        op.collect(plan, selected, remaining_sec=90)
    assert op.calls == 1 and op.started_containers == 0
    assert (op.last_world / "original-failure.json").is_file()
    assert (
        len(calls) == 2
        and calls[0][:2] == ["docker", "create"]
        and calls[1][:2] == ["docker", "inspect"]
    )
    with pytest.raises(ValueError):
        op.collect(plan, selected, remaining_sec=90)
    assert len(calls) == 2


@pytest.mark.parametrize("policy_present", [False, True])
def test_source_only_capture_does_not_invent_missing_body_or_navigation_policy(
    tmp_path, monkeypatch, policy_present
):
    op, plan, selected = operator(tmp_path, monkeypatch)
    path = op.assets / "navigation-policy.json"
    original = json.loads(path.read_bytes())
    if not policy_present:
        path.unlink()

    def reject(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(owner.subprocess, "check_output", reject)
    with pytest.raises(subprocess.CalledProcessError):
        op.collect(plan, selected, remaining_sec=90)
    value = json.loads((op.last_world / "settings.json").read_bytes())
    if policy_present:
        assert value["observed_body_id"] == original["body_id"]
        assert value["navigation_fixture_policy"] == original
    else:
        assert "observed_body_id" not in value and "navigation_fixture_policy" not in value
    assert value["engineering_capture"]["case"] == op.recipe.role
    assert op.recipe.max_goals == op.recipe.max_model_calls == op.started_containers == 0
    assert op.calls == 1 and (op.last_world / "original-failure.json").is_file()


def test_start_failure_stops_only_its_owned_container(tmp_path, monkeypatch):
    op, plan, selected = operator(tmp_path, monkeypatch)
    state = {"Id": "c" * 64, "Image": owner.SDK_IMAGE, "State": {"Running": True, "Pid": 123}}
    stops = []

    def create(argv, **kwargs):
        assert argv[:2] == ["docker", "create"]
        state["Name"] = "/" + argv[argv.index("--name") + 1]
        return state["Id"]

    def stop(argv, **kwargs):
        assert argv == ["docker", "stop", "--timeout", "3", state["Id"]]
        stops.append(argv)
        state["State"]["Running"] = False
        state["State"]["Pid"] = 0

    monkeypatch.setattr(owner.subprocess, "check_output", create)
    monkeypatch.setattr(op, "_inspect", lambda _: state.copy())
    monkeypatch.setattr(owner.subprocess, "run", stop)
    monkeypatch.setattr(
        owner.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("synthetic start failed")),
    )
    with pytest.raises(OSError, match="synthetic start"):
        op.collect(plan, selected, remaining_sec=90)
    assert len(stops) == 1 and op.calls == 1
    assert (
        json.loads((op.last_world / "container-after-stop.json").read_text())["State"]["Running"]
        is False
    )
    assert (op.last_world / "original-failure.json").is_file()


@pytest.mark.parametrize("mutation", ["outside", "symlink"])
def test_changed_output_parent_is_rejected_before_any_git_or_sdk_call(
    tmp_path, monkeypatch, mutation
):
    op = object.__new__(owner.Nav2SourceOnlyDiagnosticOperator)
    op.worktree = Path(owner.__file__).resolve().parents[2]
    op.root = tmp_path / "evidence"
    op.root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    op.calls = 0
    if mutation == "outside":
        op.output_parent = outside
    else:
        op.output_parent = tmp_path / "link"
        op.output_parent.symlink_to(outside)
    monkeypatch.setattr(
        op, "_git", lambda *args: pytest.fail("output path must fail before any process")
    )
    with pytest.raises(ValueError):
        op.check_ready()
    assert op.calls == 0
