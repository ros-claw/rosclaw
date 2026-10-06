"""Live append races must not repair corrupt completed evidence."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "acceptance_observations", ROOT / "integrations/ros_probe/acceptance/observations.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("tail", [b"", b'{"captured_at":', b'{"sequence":2}'])
def test_unfinished_append_retains_last_completed_snapshot(tmp_path, tail):
    path = tmp_path / "witness.jsonl"
    path.write_bytes(b'{"sequence":0}\n{"sequence":1}\n' + tail)
    assert module.latest_completed_observation(path) == {"sequence": 1}


@pytest.mark.parametrize("data", [b'{"sequence":', b"", b"not JSON\n", b"{}\ncorrupt\n", b"[]\n"])
def test_missing_or_corrupt_completed_snapshot_fails(tmp_path, data):
    path = tmp_path / "witness.jsonl"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        module.latest_completed_observation(path)


def test_bounded_tail_discards_only_truncated_old_record(tmp_path):
    path = tmp_path / "witness.jsonl"
    path.write_bytes(json.dumps({"old": "x" * 100000}).encode() + b'\n{"sequence":1}\n')
    assert module.latest_completed_observation(path) == {"sequence": 1}


def test_reader_does_not_retimestamp_stale_snapshot(tmp_path):
    path = tmp_path / "witness.jsonl"
    stale = {"captured_at": "2000-01-01T00:00:00+00:00", "observation_complete": False}
    path.write_text(json.dumps(stale) + "\n" + '{"captured_at":')
    assert module.latest_completed_observation(path) == stale


def test_auxiliary_window_preserves_every_completed_row(tmp_path):
    path = tmp_path / "independent-stop.jsonl"
    path.write_bytes(b'{"sequence":0}\n{"sequence":1}\n{"sequence":')
    assert module.completed_observations(path) == [{"sequence": 0}, {"sequence": 1}]


def test_auxiliary_window_does_not_skip_corrupt_middle_row(tmp_path):
    path = tmp_path / "independent-stop.jsonl"
    path.write_bytes(b'{"sequence":0}\ncorrupt\n{"sequence":1}\n')
    with pytest.raises(ValueError):
        module.completed_observations(path)
