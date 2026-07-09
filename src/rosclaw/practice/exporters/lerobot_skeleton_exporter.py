"""Skeleton LeRobot dataset exporter wrapper for episode directories."""

from __future__ import annotations

from pathlib import Path

from rosclaw.integrations.lerobot.dataset_exporter import (
    LeRobotDatasetExporter as _LeRobotDatasetExporter,
)


class LeRobotSkeletonExporter:
    """Export a ROSClaw episode directory to a LeRobot dataset skeleton."""

    def __init__(self, data_root: Path | str):
        self._data_root = Path(data_root)
        self._exporter = _LeRobotDatasetExporter()

    def export(
        self,
        practice_id: str,
        output_path: Path | str | None = None,
        *,
        task: str | None = None,
    ) -> Path:
        """Export ``practice_id`` episode directory to a LeRobot skeleton.

        Args:
            practice_id: The practice/episode identifier, or a relative/absolute
                path to the episode directory. Relative identifiers are resolved
                under ``data_root``.
            output_path: Destination directory. Defaults to
                ``<data_root>/datasets/lerobot_skeleton/<practice_id>``.
            task: Optional language instruction / task description.

        Returns:
            Path to the exported dataset directory.
        """
        episode_dir = Path(practice_id)
        if not episode_dir.is_absolute() and not episode_dir.exists():
            episode_dir = self._data_root / practice_id
        if not episode_dir.exists():
            raise ValueError(f"Episode directory not found: {episode_dir}")

        if output_path is None:
            output_path = self._data_root / "datasets" / "lerobot_skeleton" / Path(practice_id).name
        output_path = Path(output_path)

        return self._exporter.export_from_episode_dir(
            episode_dir,
            output_path,
            task=task,
        )
