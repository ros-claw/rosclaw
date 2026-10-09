"""Only fixed log paths from stopped owned containers; no real Docker or ROS."""

import hashlib
import importlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def fixture(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("generic_container_ownership")
    container, owner, image = "a" * 64, "b" * 32, "sha256:" + "c" * 64
    state = {"Running": False, "Pid": 0, "ExitCode": 143}
    row = {
        "Id": container,
        "Image": image,
        "State": state,
        "Config": {"Labels": {module.OWNER_LABEL: owner, module.KIND_LABEL: module.KIND}},
    }
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        assert kwargs["timeout"] <= 5
        if argv[:4] == ["docker", "inspect", "--type", "container"]:
            return SimpleNamespace(stdout=json.dumps([row]).encode())
        assert argv[:2] == ["docker", "cp"] and argv[2].startswith(container + ":")
        source = argv[2].split(":", 1)[1]
        assert source in module.SDK_LOG_PATHS.values()
        target = Path(argv[3])
        if target.name == "sim-logs":
            target.mkdir()
            (target / "server_console.log").write_bytes(b"original server bytes\n")
        else:
            target.write_bytes(b"original SDK bytes\n")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(module.subprocess, "run", run)
    return module, container, owner, image, row, calls


def test_original_log_bytes_and_hashes_retained_without_restarting_container(fixture, tmp_path):
    module, container, owner, image, _, calls = fixture
    output = tmp_path / "logs"
    review = module.copy_owned_sdk_logs(container, owner, output, expected_image=image)
    assert review["status"] == "CAPTURED"
    assert review["container_stopped_before_and_after"] is True
    assert review["new_World_runs"] == 0 and review["Body_admitted"] is False
    assert len(review["files"]) == 3
    for path, metadata in review["files"].items():
        raw = (output / path).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == metadata["sha256"]
        assert len(raw) == metadata["bytes"]
    assert len([argv for argv in calls if argv[1] == "inspect"]) == 5
    assert len([argv for argv in calls if argv[1] == "cp"]) == 3
    assert json.loads((output / "sdk-log-recovery.json").read_text()) == review


@pytest.mark.parametrize("bad", ["owner", "running", "image", "pid"])
def test_foreign_live_or_wrong_image_container_refused_before_copy(fixture, tmp_path, bad):
    module, container, owner, image, row, calls = fixture
    if bad == "owner":
        row["Config"]["Labels"][module.OWNER_LABEL] = "d" * 32
    elif bad == "running":
        row["State"]["Running"] = True
    elif bad == "pid":
        row["State"]["Pid"] = 42
    else:
        row["Image"] = "sha256:" + "d" * 64
    with pytest.raises(ValueError):
        module.copy_owned_sdk_logs(container, owner, tmp_path / "logs", expected_image=image)
    assert all(argv[1] == "inspect" for argv in calls)
    assert not list(tmp_path.iterdir())


def test_restart_between_initial_inspection_and_copy_is_refused(fixture, monkeypatch, tmp_path):
    module, container, owner, image, row, calls = fixture
    original = module.inspect_owned_container
    count = 0

    def inspect(*args, **kwargs):
        nonlocal count
        result = original(*args, **kwargs)
        count += 1
        if count > 1:
            return {**result, "State": {**result["State"], "Running": True, "Pid": 43}}
        return result

    monkeypatch.setattr(module, "inspect_owned_container", inspect)
    with pytest.raises(ValueError, match="changed"):
        module.copy_owned_sdk_logs(container, owner, tmp_path / "logs", expected_image=image)
    assert all(argv[1] == "inspect" for argv in calls)


@pytest.mark.parametrize("failure", ["missing", "timeout", "empty_success"])
def test_failed_or_empty_copy_cannot_become_captured_evidence(
    fixture, monkeypatch, tmp_path, failure
):
    module, container, owner, image, _, _ = fixture
    original = module.subprocess.run

    def run(argv, **kwargs):
        if argv[1] == "cp":
            if failure == "timeout":
                raise subprocess.TimeoutExpired(argv, 5)
            return SimpleNamespace(
                returncode=0 if failure == "empty_success" else 1,
                stdout=b"",
                stderr=b"not captured",
            )
        return original(argv, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", run)
    review = module.copy_owned_sdk_logs(container, owner, tmp_path / "logs", expected_image=image)
    assert review["status"] == "PARTIAL_OR_NOT_CAPTURED"
    assert review["files"] == {}


def test_container_log_symlink_is_rejected_before_reading_host_files(
    fixture, monkeypatch, tmp_path
):
    module, container, owner, image, _, _ = fixture
    original = module.subprocess.run

    def run(argv, **kwargs):
        if argv[1] == "cp":
            Path(argv[3]).symlink_to(tmp_path / "private-absent")
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        return original(argv, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(ValueError, match="symlinks"):
        module.copy_owned_sdk_logs(container, owner, tmp_path / "logs", expected_image=image)
    assert not (tmp_path / "logs/sdk-log-recovery.json").exists()


def test_existing_recovery_directory_cannot_be_overwritten(fixture, tmp_path):
    module, container, owner, image, _, calls = fixture
    output = tmp_path / "logs"
    output.mkdir()
    retained = output / "original.txt"
    retained.write_text("preserved")
    with pytest.raises(FileExistsError):
        module.copy_owned_sdk_logs(container, owner, output, expected_image=image)
    assert retained.read_text() == "preserved"
    assert all(argv[1] == "inspect" for argv in calls)
