"""Test LeRobot dataset skeleton exporter."""

from __future__ import annotations

import json
from pathlib import Path

from rosclaw.integrations.lerobot.dataset_exporter import LeRobotDatasetExporter
from rosclaw.practice.exporters import LeRobotSkeletonExporter


def test_export_from_episode_dir(tmp_path: Path):
    """Skeleton export should create expected files."""
    episode_dir = tmp_path / "minimal_episode"
    episode_dir.mkdir()
    (episode_dir / "metadata.json").write_text(
        json.dumps(
            {
                "robot_id": "ur5e_lab_01",
                "task_name": "test task",
                "sampling_rate": 30.0,
                "dof_names": ["j0"],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"

    exporter = LeRobotDatasetExporter()
    result = exporter.export_from_episode_dir(episode_dir, output_dir)

    assert result == output_dir
    assert (output_dir / "README.md").exists()
    assert (output_dir / "meta" / "info.json").exists()
    assert (output_dir / "meta" / "episodes.jsonl").exists()
    assert (output_dir / "meta" / "tasks.jsonl").exists()
    assert (output_dir / "meta" / "rosclaw_mapping.json").exists()
    assert (output_dir / "data" / "placeholder.jsonl").exists()
    assert (output_dir / "videos" / "README.md").exists()

    info = json.loads((output_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["robot_id"] == "ur5e_lab_01"
    assert info["task"] == "test task"
    placeholder_rows = [
        json.loads(line)
        for line in (output_dir / "data" / "placeholder.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert placeholder_rows == [{"placeholder": True, "frames": 0}]


def test_skeleton_exporter_wrapper(tmp_path: Path):
    """The wrapper should resolve episode dir under data_root."""
    data_root = tmp_path / "practice"
    episode_dir = data_root / "minimal"
    episode_dir.mkdir(parents=True)
    (episode_dir / "metadata.json").write_text(
        json.dumps({"robot_id": "r1"}), encoding="utf-8"
    )
    exporter = LeRobotSkeletonExporter(data_root)
    out = exporter.export("minimal")
    assert out.exists()
    assert (out / "meta" / "info.json").exists()


def test_skeleton_exporter_accepts_existing_relative_path(
    tmp_path: Path, monkeypatch
):
    episode_dir = tmp_path / "relative_episode"
    episode_dir.mkdir()
    (episode_dir / "metadata.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    out = LeRobotSkeletonExporter(tmp_path / "practice").export(
        "relative_episode",
        output_path=tmp_path / "output",
    )

    assert (out / "meta" / "info.json").exists()
