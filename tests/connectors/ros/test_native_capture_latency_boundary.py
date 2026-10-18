"""Original bytes/clocks precede derived reviews; raw diagnostic omission is explicit."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def module(relative):
    spec = importlib.util.spec_from_file_location("sdk_capture_component", ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_native_pointer_preserves_original_bytes_and_receive_clock(tmp_path):
    capture = module("integrations/ros_probe/acceptance/simulation_native_capture.py")
    raw = b'{"captured_at":"2026-10-18T07:00:00Z", "untouched": [1, 2]}'
    result = capture.persist_native_capture(
        tmp_path, raw, index=3, received_monotonic_ns=12345, publish_pointer=True
    )
    pointer = json.loads((tmp_path / "live-native-pointer.json").read_text())
    assert (tmp_path / pointer["file"]).read_bytes() == raw
    assert pointer["sha256"] == hashlib.sha256(raw).hexdigest()
    assert pointer["received_monotonic_ns"] == 12345
    assert pointer["captured_at"] == result["captured_at"] == "2026-10-18T07:00:00Z"
    with pytest.raises(FileExistsError):
        capture.persist_native_capture(
            tmp_path, raw, index=3, received_monotonic_ns=99999, publish_pointer=True
        )
    assert json.loads((tmp_path / "live-native-pointer.json").read_text()) == pointer


def test_invalid_or_naive_native_does_not_replace_current_pointer(tmp_path):
    capture = module("integrations/ros_probe/acceptance/simulation_native_capture.py")
    pointer = tmp_path / "live-native-pointer.json"
    pointer.write_text("original pointer")
    with pytest.raises(ValueError):
        capture.persist_native_capture(
            tmp_path,
            b'{"captured_at":"2026-10-18T07:00:00"}',
            index=0,
            received_monotonic_ns=1,
            publish_pointer=True,
        )
    assert pointer.read_text() == "original pointer"
    assert not (tmp_path / "original-native-000.json").exists()


def test_tf_bulk_export_preserves_times_unknown_identity_and_omission():
    capture = module("integrations/ros_probe/ros2/tf_window.py")
    rows = [
        {
            "received_monotonic": 99 + i / 1000,
            "original_receive": f"original-{i}",
            "publisher_gid": None,
        }
        for i in range(300)
    ]
    rows.extend([{"received_monotonic": 90}, {"received_monotonic": 110}])
    result = capture.raw_tf_observation_window(rows, now=100, cache_drops=7, message_window_drops=9)
    assert result["retained_current_messages"] == 300
    assert result["omitted_current_messages"] + len(result["messages"]) == 300
    assert result["messages"] == rows[172:300]
    assert result["messages"][0] is not rows[172]
    assert result["cache_drops"] == 7 and result["message_window_drops"] == 9
    assert result["completeness"] == "NOT_ESTABLISHED"
    expired = capture.raw_tf_observation_window(
        rows, now=120, cache_drops=7, message_window_drops=9
    )
    assert expired["messages"] == [] and expired["retained_current_messages"] == 0
