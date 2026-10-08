"""Fresh mission acceptance cannot borrow an older successful Practice episode."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance"


def load(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "cleaning_acceptance_identity", ROOT / "cleaning_acceptance.py"
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def episode(root, name, mission, outcome="SUCCESS"):
    p = root / "practice/sessions" / name
    p.mkdir(parents=True)
    (p / "episode.json").write_text(
        json.dumps({"practice_id": mission, "session_id": name, "outcome": outcome})
    )
    (p / "manifest.yaml").write_text(json.dumps({"practice_id": mission, "session_id": name}))
    return p / "episode.json"


def test_fresh_mission_uses_matching_practice_and_preserves_legacy(tmp_path, monkeypatch):
    m = load(monkeypatch)
    old = episode(tmp_path, "gazebo-room-cleaning", "gazebo-room-cleaning")
    fresh = episode(tmp_path, "native-d2-fresh", "native-d2-fresh")
    assert m.mission_practice_episode(tmp_path, "native-d2-fresh")[0] == fresh.resolve()
    assert m.mission_practice_episode(tmp_path, "gazebo-room-cleaning")[0] == old.resolve()


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "failed",
        "duplicate",
        "manifest",
        "outside",
        "oversize",
        "wrong_manifest",
        "bad_record",
    ],
)
def test_old_success_or_invalid_fresh_record_cannot_satisfy_mission(tmp_path, monkeypatch, fault):
    m = load(monkeypatch)
    episode(tmp_path, "gazebo-room-cleaning", "gazebo-room-cleaning")
    if fault != "missing":
        fresh = episode(
            tmp_path, "fresh", "fresh", outcome="FAILED" if fault == "failed" else "SUCCESS"
        )
        if fault == "duplicate":
            episode(tmp_path, "duplicate", "fresh")
        elif fault == "manifest":
            fresh.with_name("manifest.yaml").unlink()
        elif fault == "outside":
            fresh.unlink()
            outside = tmp_path / "foreign.json"
            outside.write_text('{"practice_id":"fresh","outcome":"SUCCESS"}')
            fresh.symlink_to(outside)
        elif fault == "oversize":
            fresh.write_bytes(b" " * 2_000_001)
        elif fault == "wrong_manifest":
            fresh.with_name("manifest.yaml").write_text(
                '{"practice_id":"old","session_id":"fresh"}'
            )
        elif fault == "bad_record":
            fresh.write_text("[]")
    with pytest.raises(ValueError):
        m.mission_practice_episode(tmp_path, "fresh")
