"""Round trace logging and practice export."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List

from ..gesture_schema import RPSRound, CameraFrame


class RoundLogger:
    """Write rounds, frames, and telemetry to a run directory."""

    def __init__(self, root_dir: Path, save_frames: bool = True, save_telemetry: bool = True):
        self.root = Path(root_dir).expanduser()
        self.save_frames = save_frames
        self.save_telemetry = save_telemetry
        self._rounds: List[RPSRound] = []
        self._ensure_dirs()

    def _ensure_dirs(self):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "frames").mkdir(exist_ok=True)
        (self.root / "telemetry").mkdir(exist_ok=True)

    def log_round(self, round_obj: RPSRound, capture_frame: CameraFrame = None) -> None:
        self._rounds.append(round_obj)
        with open(self.root / "rounds.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(round_obj.to_dict(), ensure_ascii=False) + "\n")

        if self.save_telemetry and round_obj.robot_gesture_result.telemetry:
            tel_path = self.root / "telemetry" / f"{round_obj.round_id}_hand.json"
            tel_path.write_text(
                json.dumps(round_obj.robot_gesture_result.telemetry.__dict__, default=str, ensure_ascii=False),
                encoding="utf-8",
            )

        if self.save_frames and capture_frame is not None:
            try:
                import cv2

                frame_path = self.root / "frames" / f"{round_obj.round_id}_capture.jpg"
                cv2.imwrite(str(frame_path), capture_frame.color)
            except Exception:
                pass

    def write_summary(self, summary: dict) -> None:
        summary_path = self.root / "summary.json"
        summary["run_dir"] = str(self.root)
        summary["recorded_at"] = datetime.now().isoformat()
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    @property
    def rounds(self) -> List[RPSRound]:
        return list(self._rounds)

    @property
    def current_run_dir(self) -> Path:
        return self.root
