"""Extended tests for eurdf_loader validation and edge cases."""

import pytest

from rosclaw.runtime.eurdf_loader import EURDFLoader, RobotRegistry


def test_loader_list_robots_no_zoo():
    loader = EURDFLoader(zoo_path="/nonexistent/zoo")
    assert loader.list_robots() == []


def test_loader_validate_missing_files(tmp_path):
    loader = EURDFLoader(zoo_path=tmp_path)
    robot_dir = tmp_path / "bad_bot"
    robot_dir.mkdir()
    (robot_dir / "robot.eurdf.yaml").write_text("robot_id: bad_bot\nname: Bad\nvendor: test\ndof: 6\nlinks: []\njoints: []")

    result = loader.validate("bad_bot")
    assert result["valid"] is False
    assert len(result["files_missing"]) > 0


def test_loader_validate_missing_required_fields(tmp_path):
    loader = EURDFLoader(zoo_path=tmp_path)
    robot_dir = tmp_path / "bad_bot"
    robot_dir.mkdir()
    (robot_dir / "robot.eurdf.yaml").write_text("robot_id: bad_bot\nname: Bad")
    for fname in ["safety.yaml", "semantic.yaml", "capabilities.yaml", "benchmark.yaml"]:
        (robot_dir / fname).write_text("{}")

    result = loader.validate("bad_bot")
    assert result["valid"] is False
    assert any("missing field" in e for e in result["errors"])


def test_loader_validate_warnings(tmp_path):
    loader = EURDFLoader(zoo_path=tmp_path)
    robot_dir = tmp_path / "warn_bot"
    robot_dir.mkdir()
    (robot_dir / "robot.eurdf.yaml").write_text(
        "robot_id: warn_bot\nname: Warn\nvendor: test\ndof: 6\nlinks: []\njoints: []"
    )
    for fname in ["safety.yaml", "semantic.yaml", "capabilities.yaml", "benchmark.yaml"]:
        (robot_dir / fname).write_text("{}")

    result = loader.validate("warn_bot")
    assert any("No sensors" in w for w in result["warnings"])
    assert any("No actuators" in w for w in result["warnings"])
    assert any("No capabilities" in w for w in result["warnings"])
    assert any("No simulation_backends" in w for w in result["warnings"])


def test_loader_validate_parse_error(tmp_path):
    loader = EURDFLoader(zoo_path=tmp_path)
    robot_dir = tmp_path / "bad_bot"
    robot_dir.mkdir()
    (robot_dir / "robot.eurdf.yaml").write_text("not: valid: yaml: [")
    for fname in ["safety.yaml", "semantic.yaml", "capabilities.yaml", "benchmark.yaml"]:
        (robot_dir / fname).write_text("{}")

    result = loader.validate("bad_bot")
    assert result["valid"] is False
    assert any("Failed to parse" in e for e in result["errors"])


def test_registry_inspect_not_found():
    reg = RobotRegistry()
    with pytest.raises(FileNotFoundError):
        reg.inspect("nonexistent")
