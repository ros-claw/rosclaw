"""LeRobot dataset skeleton exporter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LeRobotDatasetExporter:
    """Export a ROSClaw episode directory to a LeRobot dataset skeleton."""

    def __init__(self) -> None:
        pass

    def export_from_episode_dir(
        self,
        episode_dir: Path | str,
        output_dir: Path | str,
        *,
        task: str | None = None,
        robot_id: str | None = None,
    ) -> Path:
        """Create a LeRobot-compatible dataset skeleton from an episode directory.

        The skeleton follows LeRobotDataset v3 layout:

            <output_dir>/
              README.md
              meta/
                info.json
                episodes.jsonl
                tasks.jsonl
                rosclaw_mapping.json
              data/
                placeholder.jsonl
              videos/
                README.md
        """
        episode_dir = Path(episode_dir)
        output_dir = Path(output_dir)
        meta_dir = output_dir / "meta"
        data_dir = output_dir / "data"
        videos_dir = output_dir / "videos"

        output_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        videos_dir.mkdir(parents=True, exist_ok=True)

        metadata = self._load_episode_metadata(episode_dir)
        task = task or metadata.get("task_name") or metadata.get("task") or "rosclaw practice"
        robot_id = robot_id or metadata.get("robot_id") or metadata.get("robot") or "unknown"

        # Skeleton info.json.
        info = {
            "codebase_version": "v3.0",
            "repo_id": f"rosclaw/{episode_dir.name}",
            "total_episodes": 1,
            "total_frames": 0,
            "sampling_rate": metadata.get("sampling_rate", 30.0),
            "robot_id": robot_id,
            "task": task,
            "video": True,
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [-1],
                    "names": metadata.get("dof_names", []),
                },
                "action": {
                    "dtype": "float32",
                    "shape": [-1],
                    "names": metadata.get("dof_names", []),
                },
                "episode_index": {"dtype": "int64", "shape": [1]},
                "index_in_episode": {"dtype": "int64", "shape": [1]},
                "timestamp": {"dtype": "float32", "shape": [1]},
                "next.reward": {"dtype": "float32", "shape": [1]},
                "next.done": {"dtype": "bool", "shape": [1]},
                "next.success": {"dtype": "bool", "shape": [1]},
            },
            "rosclaw": {
                "source_episode": str(episode_dir),
                "export_tool": "rosclaw.practice.exporters.lerobot_skeleton",
                "skeleton": True,
            },
        }
        with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        # Episodes file.
        with open(meta_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "episode_index": 0,
                        "task_index": 0,
                        "length": 0,
                        "rosclaw_source": str(episode_dir),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        # Tasks file.
        with open(meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
            f.write(
                json.dumps({"task_index": 0, "task": task}, ensure_ascii=False) + "\n"
            )

        # ROSClaw-to-LeRobot mapping.
        mapping = {
            "observation": {
                "state": "observation.state",
                "images": "observation.images.{camera_name}",
                "ee_pose": "observation.ee_pose",
            },
            "action": "action",
            "task": "task",
            "episode_index": "episode_index",
            "timestamp": "timestamp",
        }
        with open(meta_dir / "rosclaw_mapping.json", "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)

        # Placeholder data file.
        with open(data_dir / "placeholder.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"placeholder": True, "frames": 0}) + "\n")

        # README files.
        (output_dir / "README.md").write_text(
            f"# LeRobot Dataset Skeleton\n\n"
            f"Exported from ROSClaw episode `{episode_dir.name}`.\n\n"
            "This is a P0 skeleton: it contains metadata and mapping files only. "
            "Populate `data/` and `videos/` with real frames before training.\n",
            encoding="utf-8",
        )
        (videos_dir / "README.md").write_text(
            "# Videos\n\n"
            "Place per-episode MP4 files here (e.g., `episode_0.mp4`).\n",
            encoding="utf-8",
        )

        return output_dir

    @staticmethod
    def _load_episode_metadata(episode_dir: Path) -> dict[str, Any]:
        """Load metadata.json from the episode directory if present."""
        meta_path = episode_dir / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass
        return {}
