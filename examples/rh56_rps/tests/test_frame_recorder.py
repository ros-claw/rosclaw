"""Tests for the continuous FrameRecorder (30 Hz frame_event tap)."""

from __future__ import annotations

import time

import numpy as np

from rosclaw_rps.gesture_schema import CameraFrame, GesturePrediction
from rosclaw_rps.vision.frame_recorder import FrameRecorder


class FakeWorker:
    """Minimal RecognitionWorker stand-in."""

    def __init__(self) -> None:
        self._frame = None
        self._pred = GesturePrediction(label="rock", confidence=0.9)
        self.frame_age_s = 0.0

    def set_frame(self, ts: float) -> None:
        self._frame = CameraFrame(
            color=np.zeros((4, 6, 3), dtype=np.uint8),
            timestamp=ts,
        )

    def get_latest(self):
        return self._frame, self._pred

    def health(self) -> dict:
        return {
            "alive": True,
            "frames_read": 1,
            "last_frame_age_s": self.frame_age_s,
            "empty_streak": 0,
        }


class EmitCollector:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict, str, list[str]]] = []

    def __call__(
        self, event_type: str, payload: dict, source: str, tags: list[str]
    ) -> None:
        self.events.append((event_type, payload, source, tags))

    def of_type(self, event_type: str) -> list[dict]:
        return [p for t, p, _, _ in self.events if t == event_type]


def test_records_fresh_frames_and_skips_duplicates(tmp_path):
    worker = FakeWorker()
    collector = EmitCollector()
    rec = FrameRecorder(
        worker,
        emit=collector,
        frame_hz=120.0,
        keyframe_hz=0.0,  # metadata-only
        keyframe_dir=tmp_path / "keyframes",
    )
    rec.set_round("round-1")
    rec.start()

    ts = 1000.0
    worker.set_frame(ts)
    deadline = time.time() + 2.0
    while time.time() < deadline and rec.stats["frame_events"] < 5:
        ts += 0.05
        worker.set_frame(ts)
        time.sleep(0.05)
    rec.stop()

    events = collector.of_type("frame_event")
    assert len(events) >= 3
    assert rec.stats["duplicates_skipped"] > 0
    first = events[0]
    assert first["human_label"] == "rock"
    assert first["confidence"] == 0.9
    assert first["round_id"] == "round-1"
    assert first["width"] == 6 and first["height"] == 4
    assert first["keyframe"] is False


def test_fuses_on_camera_loss():
    worker = FakeWorker()
    worker.frame_age_s = 5.0  # camera stopped delivering frames
    collector = EmitCollector()
    rec = FrameRecorder(worker, emit=collector, frame_hz=120.0, keyframe_hz=0.0)
    rec.start()
    deadline = time.time() + 2.0
    while time.time() < deadline and rec.is_alive():
        time.sleep(0.02)
    rec.stop()

    assert rec.fused
    errors = collector.of_type("hardware_transport_error")
    assert len(errors) == 1
    assert errors[0]["health_state"] == "no_frames"
    assert rec.stats["frame_events"] == 0
