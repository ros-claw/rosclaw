"""Real harmless subprocess cleanup and inactive-bootstrap refusal boundaries."""

import importlib
import json
import sys
import time
from pathlib import Path

import pytest


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("generic_bootstrap_runtime")


@pytest.mark.parametrize("seconds", [True, 0, -1, 1801, float("nan"), float("inf"), "10"])
def test_invalid_deadline_refused(runtime, seconds):
    with pytest.raises(ValueError):
        runtime.validate_deadline_seconds(seconds)


def test_actual_writable_mount_and_wrong_path_refused(runtime, tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        runtime.require_read_only_workspace(tmp_path)
    monkeypatch.setattr(
        runtime.os, "statvfs", lambda p: runtime.os.statvfs_result([1, 1, 1, 1, 1, 1, 1, 1, 0, 1])
    )
    with pytest.raises(ValueError, match="read-only"):
        runtime.require_read_only_workspace("/evidence")


def test_shared_partition_and_kernel_port_collision_refused(runtime, monkeypatch):
    monkeypatch.delenv("GZ_PARTITION", raising=False)
    with pytest.raises(ValueError, match="partition"):
        runtime.require_isolated_environment()
    monkeypatch.setenv("GZ_PARTITION", "rosclaw_generic_" + "a" * 32)
    monkeypatch.setenv("ROS_DOMAIN_ID", "110")

    def collision(domain):
        assert domain == 110
        raise ValueError("actual kernel port collision")

    monkeypatch.setattr(runtime, "validate_ros_domain", collision)
    with pytest.raises(ValueError, match="kernel port"):
        runtime.require_isolated_environment()


def test_expired_runtime_reaps_harmless_child_without_admission(runtime, tmp_path):
    out = tmp_path / "run"
    start = time.monotonic()
    outcome = runtime.supervise_owned_launch(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        out,
        seconds=1,
        check_source=lambda: None,
    )
    assert outcome == "IMMUTABLE_DEADLINE_REACHED"
    result = json.loads((out / "runtime-result.json").read_text())
    assert result["owned_launch_reaped"] is True
    assert result["controller_activation"] is False
    assert result["live_body_admitted"] is False
    assert result["authorization"] is False
    assert time.monotonic() - start < 7


def test_unexpected_even_successful_child_exit_is_failure(runtime, tmp_path):
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="exited"):
        runtime.supervise_owned_launch(
            [sys.executable, "-c", "pass"],
            out,
            seconds=5,
            check_source=lambda: None,
        )
    result = json.loads((out / "runtime-result.json").read_text())
    assert result["outcome"] == "UNEXPECTED_LAUNCH_EXIT"
    assert result["owned_launch_reaped"] and result["returncode"] == 0


def test_source_preflight_failure_never_starts_child(runtime, tmp_path):
    out = tmp_path / "run"

    def refuse():
        raise ValueError("changed source")

    with pytest.raises(ValueError, match="changed"):
        runtime.supervise_owned_launch(
            ["/must/not/be/executed"],
            out,
            seconds=1,
            check_source=refuse,
        )
    result = json.loads((out / "runtime-result.json").read_text())
    assert result["owned_launch_pid"] is None


def test_midrun_source_failure_reaps_child(runtime, tmp_path):
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ValueError("changed source")

    out = tmp_path / "run"
    with pytest.raises(ValueError, match="changed"):
        runtime.supervise_owned_launch(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            out,
            seconds=5,
            check_source=check,
        )
    result = json.loads((out / "runtime-result.json").read_text())
    assert result["outcome"] == "SOURCE_OR_SUPERVISOR_FAILURE"
    assert result["owned_launch_reaped"] is True
