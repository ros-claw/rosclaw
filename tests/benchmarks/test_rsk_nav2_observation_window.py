"""Synthetic immutable-file contracts; no SDK/ROS/World/model evidence."""

import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest

from benchmarks.ros_expert_ablation import rsk_nav2_observation_window as window

NOW = datetime(2026, 10, 18, tzinfo=UTC)
RUN = "a" * 32
SOURCE = "b" * 40


def fixture(tmp_path, monkeypatch, fault=None):
    execution = tmp_path / "execution"
    execution.mkdir()
    output = tmp_path / "observed"
    sink = {
        "run_id": RUN,
        "body_snapshot_hash": None,
        "terminal": False,
        "action_id": None,
        "daemon_instance_id": None,
        "lease_expires_ns": 0,
        "lease_updates": 0,
        "received_commands": 0,
        "published_velocity_messages": 0,
    }

    def save_sink():
        (execution / "governed-sink-status.json").write_text(json.dumps(sink))

    class Clock:
        now = 100_000_000_000

        @classmethod
        def monotonic_ns(cls):
            return cls.now

        @classmethod
        def sleep(cls, seconds):
            cls.now += round(seconds * 1_000_000_000)
            if fault == "same_frame":
                return
            if fault == "command_after":
                sink["received_commands"] = 1
                save_sink()
            frame(1, seconds if fault != "short_native_span" else 1.0)

    def frame(index, seconds):
        stamp = (NOW + timedelta(seconds=seconds)).isoformat()
        native = {
            "captured_at": stamp,
            "environment": {"rmw": "rmw_cyclonedds_cpp", "ros_generation": "ros2"},
            "graph": {"nodes": [], "topics": [], "actions": [], "services": []},
            "completeness": {"graph": True},
        }
        if fault == "wrong_rmw":
            native["environment"]["rmw"] = "different_rmw"
        name = f"original-native-{index:03d}.json"
        raw = json.dumps(native).encode()
        (execution / name).write_bytes(raw)
        pointer = {
            "file": name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "captured_at": stamp,
            "received_monotonic_ns": Clock.now,
        }
        if fault == "stale_pointer":
            pointer["received_monotonic_ns"] -= 500_000_001
        if fault == "bool_clock":
            pointer["received_monotonic_ns"] = True
        if fault == "path_escape":
            pointer["file"] = "../" + name
        if fault == "wrong_sha":
            pointer["sha256"] = "0" * 64
        (execution / "live-native-pointer.json").write_text(json.dumps(pointer))

    if fault == "wrong_run":
        sink["run_id"] = "c" * 32
    if fault == "bool_counter":
        sink["received_commands"] = False
    if fault == "missing_counter":
        sink.pop("lease_updates")
    if fault == "admission_symlink":
        control = execution / "execution-control"
        control.mkdir()
        (control / "admission.json").symlink_to(tmp_path / "missing-admission")
    save_sink()
    frame(0, 0)
    monkeypatch.setattr(window, "time", Clock)
    return execution, output


def capture(execution, output):
    return window.capture_owned_unadmitted_window(
        execution,
        output,
        run_id=RUN,
        body_snapshot_hash=None,
        source_revision=SOURCE,
        operation="CLOCK_TF_WINDOW",
    )


def test_exact_original_bytes_preserved_and_no_admission_or_action_claim(tmp_path, monkeypatch):
    execution, output = fixture(tmp_path, monkeypatch)
    before = (execution / "original-native-000.json").read_bytes()
    result = capture(execution, output)
    assert (output / "native-before-original.json").read_bytes() == before
    assert (output / "native-after-original.json").read_bytes() == (
        execution / "original-native-001.json"
    ).read_bytes()
    assert result["native_capture_span_sec"] == result["host_window_elapsed_sec"] == 5.6
    assert result["source_only_unenrolled"] is True
    assert result["source_body_snapshot_hash"] is None
    assert result["Body_or_L1_admission"] == "NOT_GRANTED"
    assert result["canonical_navigation_success"] is result["case_rebound"] is False
    assert result["control_commands_sent"] == result["read_only_collector_starts_Worlds"] == 0
    assert not (execution / "execution-control").exists()
    assert json.loads((output / "window-original-manifest.json").read_text()) == result


@pytest.mark.parametrize(
    "fault",
    [
        "wrong_run",
        "bool_counter",
        "missing_counter",
        "admission_symlink",
        "stale_pointer",
        "bool_clock",
        "path_escape",
        "wrong_sha",
        "wrong_rmw",
        "same_frame",
        "command_after",
        "short_native_span",
    ],
)
def test_missing_changed_or_admitted_originals_cannot_pass(tmp_path, monkeypatch, fault):
    execution, output = fixture(tmp_path, monkeypatch, fault)
    with pytest.raises(ValueError):
        capture(execution, output)
    assert not (output / "window-original-manifest.json").exists()
    if fault in {"same_frame", "command_after", "short_native_span"}:
        assert (output / "native-before-original.json").is_file()
        assert (output / "pointer-before-original.json").is_file()
        assert (output / "sink-before-original.json").is_file()


def test_unregistered_operation_stops_before_creating_output(tmp_path, monkeypatch):
    execution, output = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="closed diagnostic operation"):
        window.capture_owned_unadmitted_window(
            execution,
            output,
            run_id=RUN,
            body_snapshot_hash=None,
            source_revision=SOURCE,
            operation="SET_PARAMETER",
        )
    assert not output.exists()
