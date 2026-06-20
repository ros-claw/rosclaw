"""End-to-end P0 body workflow integration test.

Covers the full spec P0 chain:
  link-eurdf → inspect → update-state → diff → note →
  EMBODIMENT.md refresh → skill compatibility recheck.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from rosclaw.body.resolver import BodyResolver
from rosclaw.cli import main as rosclaw_main

FIXTURES = Path(__file__).parent / "fixtures" / "skills"


@pytest.fixture
def linked_body_with_skills(tmp_path, monkeypatch):
    """A linked body with sample skill manifests copied into the workspace."""
    monkeypatch.setenv("HOME", str(tmp_path))
    skills_dir = tmp_path / ".rosclaw" / "skills"
    skills_dir.mkdir(parents=True)
    for path in FIXTURES.glob("*.skill.yaml"):
        (skills_dir / path.name).write_text(path.read_text(), encoding="utf-8")

    with patch.object(sys, "argv", ["rosclaw", "body", "link-eurdf", "unitree-g1"]):
        assert rosclaw_main() == 0
    yield tmp_path


def test_full_p0_workflow(linked_body_with_skills, capsys):
    """Run the full P0 workflow and verify each artifact refreshes correctly."""
    resolver = BodyResolver()
    initial_hash = resolver.get_effective_body_hash()
    initial_md = resolver.embodiment_md_path.read_text(encoding="utf-8")
    assert "unitree-g1" in initial_md

    # 1. inspect (text)
    with patch.object(sys, "argv", ["rosclaw", "body", "inspect"]):
        assert rosclaw_main() == 0
    out = capsys.readouterr().out
    assert "ROSClaw Body Inspect" in out
    assert "unitree-g1" in out

    # 2. update-state: mark head camera unavailable
    with patch.object(sys, "argv", [
        "rosclaw", "body", "update-state",
        "--set", "installed_components.sensors.head_rgb_camera.status=unavailable",
        "--reason", "test camera disconnected",
    ]):
        assert rosclaw_main() == 0

    # 3. diff
    with patch.object(sys, "argv", ["rosclaw", "body", "diff"]):
        assert rosclaw_main() == 0
    out = capsys.readouterr().out
    assert "head_rgb_camera" in out or "camera" in out.lower() or "capability" in out.lower()

    # 4. note: right arm incident
    with patch.object(sys, "argv", [
        "rosclaw", "body", "note",
        "--type", "incident",
        "--severity", "warning",
        "--affects", "right_arm,dual_arm_manipulation",
        "Right arm overheated during test.",
    ]):
        assert rosclaw_main() == 0

    # 5. Verify effective body recompiled and hash changed.
    new_hash = resolver.get_effective_body_hash()
    assert new_hash != initial_hash

    # 6. EMBODIMENT.md refreshed.
    refreshed_md = resolver.embodiment_md_path.read_text(encoding="utf-8")
    assert "unitree-g1" in refreshed_md
    assert refreshed_md != initial_md

    # 7. skill check --all
    with patch.object(sys, "argv", ["rosclaw", "skill", "check", "--all"]):
        assert rosclaw_main() == 0
    out = capsys.readouterr().out
    assert "Skill compatibility" in out
    assert "camera_nav@1.0.0" in out
    assert "walk_forward@1.0.0" in out
    assert "dual_arm_lift@1.0.0" in out

    # 8. Verify the persisted report matches the CLI output.
    report = resolver.get_skill_compatibility()
    assert report.effective_body_hash == new_hash
    assert report.skills["camera_nav@1.0.0"].status == "blocked"
    assert report.skills["walk_forward@1.0.0"].status == "compatible"
    assert report.skills["dual_arm_lift@1.0.0"].status == "blocked"


def test_skill_check_single_skill(linked_body_with_skills, capsys):
    """rosclaw skill check <skill_id> reports status for one skill."""
    with patch.object(sys, "argv", [
        "rosclaw", "body", "update-state",
        "--set", "installed_components.sensors.head_rgb_camera.status=unavailable",
        "--reason", "test",
    ]):
        assert rosclaw_main() == 0

    with patch.object(sys, "argv", ["rosclaw", "skill", "check", "camera_nav"]):
        assert rosclaw_main() == 0
    out = capsys.readouterr().out
    assert "camera_nav@1.0.0" in out
    assert "blocked" in out


def test_skill_check_json_output(linked_body_with_skills, capsys):
    """rosclaw skill check --all --json emits valid JSON."""
    import json

    with patch.object(sys, "argv", ["rosclaw", "skill", "check", "--all", "--json"]):
        assert rosclaw_main() == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["effective_body_hash"]
    assert "skills" in data
    assert "walk_forward@1.0.0" in data["skills"]


def test_skill_check_without_body(capsys):
    """rosclaw skill check fails gracefully when no body is linked."""
    import tempfile

    with (
        tempfile.TemporaryDirectory() as tmp,
        patch.dict("os.environ", {"HOME": tmp}),
        patch.object(sys, "argv", ["rosclaw", "skill", "check", "--all"]),
    ):
        rc = rosclaw_main()
    assert rc == 1
    out = capsys.readouterr().out
    assert "No body linked" in out


def test_body_update_state_and_note_write_memory_events(linked_body_with_skills, monkeypatch):
    """body update-state and note must persist body_change and skill_compatibility_change memory events."""
    from rosclaw.memory.interface import MemoryInterface
    from rosclaw.memory.seekdb_client import SeekDBMemoryClient

    shared_client = SeekDBMemoryClient()
    shared_client.connect()
    monkeypatch.setattr("rosclaw.memory.interface.SeekDBMemoryClient", lambda: shared_client)

    resolver = BodyResolver()
    robot_id = resolver.get_current_body_yaml().body_instance.get("id") or resolver.body_id

    with patch.object(sys, "argv", [
        "rosclaw", "body", "update-state",
        "--set", "installed_components.sensors.head_rgb_camera.status=unavailable",
        "--reason", "test camera disconnected",
    ]):
        assert rosclaw_main() == 0

    with patch.object(sys, "argv", [
        "rosclaw", "body", "note",
        "--type", "incident",
        "--severity", "warning",
        "--affects", "right_arm,dual_arm_manipulation",
        "Right arm overheated during test.",
    ]):
        assert rosclaw_main() == 0

    body_changes = shared_client.query(
        "experience_graph", filters={"event_type": "body_change", "robot_id": robot_id}
    )
    skill_changes = shared_client.query(
        "experience_graph", filters={"event_type": "skill_compatibility_change", "robot_id": robot_id}
    )
    assert len(body_changes) == 2
    assert len(skill_changes) == 2

    hashes = set()
    for event in body_changes + skill_changes:
        meta = event.get("metadata", {})
        hashes.add(meta.get("effective_body_hash_after"))
        assert meta.get("effective_body_hash_before") != meta.get("effective_body_hash_after")
    assert len(hashes) == 2

    memory = MemoryInterface(robot_id=robot_id, seekdb_client=shared_client)
    by_type = memory.seekdb_client.query(
        "experience_graph", filters={"event_type": "skill_compatibility_change", "robot_id": robot_id}
    )
    assert any("camera_nav@1.0.0" in (rec.get("metadata", {}).get("skills", {}) or {}) for rec in by_type)
    assert any("dual_arm_lift@1.0.0" in (rec.get("metadata", {}).get("skills", {}) or {}) for rec in by_type)
