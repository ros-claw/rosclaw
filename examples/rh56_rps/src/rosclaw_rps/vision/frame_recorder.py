"""Continuously record the live camera stream as Practice ``frame_event``s.

The RPS full mode already streams 30 Hz frames through ``RecognitionWorker``
for gesture recognition, but none of them were recorded — the Practice
session only saw per-round game events.  ``FrameRecorder`` taps that same
stream so the session timeline contains the full 30 Hz RGB-D context (with
the live recognition label attached), plus keyframe PNGs at a lower rate.

Fuse protocol (learned from the D435i UVC wedge incidents): when the camera
stops delivering frames, emit ONE ``hardware_transport_error`` event and stop
recording.  Never block or kill the game loop — the UI already surfaces
camera loss to the operator.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .recognition_worker import RecognitionWorker

logger = logging.getLogger("rosclaw_rps.frame_recorder")

try:  # Keyframe saving needs OpenCV; metadata-only recording works without.
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

EmitFn = Callable[[str, dict[str, Any], str, list[str]], None]


class FrameRecorder(threading.Thread):
    """Sample ``RecognitionWorker`` frames into Practice frame_events.

    Args:
        worker: The running recognition worker to tap.
        emit: Callable ``(event_type, payload, source, tags)`` that injects a
            practice event (``RpsSkillHandler._emit_practice``).
        frame_hz: Target frame_event rate.
        keyframe_hz: Rate at which keyframe PNGs are saved (0 disables).
        keyframe_dir: Directory for keyframe PNGs (``<session>/keyframes``).
        camera_lost_after_s: Fuse threshold — stop recording when no fresh
            frame has arrived for this long.
    """

    def __init__(
        self,
        worker: RecognitionWorker,
        emit: EmitFn,
        *,
        frame_hz: float = 30.0,
        keyframe_hz: float = 1.0,
        keyframe_dir: Optional[Path] = None,
        camera_lost_after_s: float = 2.0,
        name: str = "rps-frame-recorder",
    ) -> None:
        super().__init__(name=name, daemon=True)
        self._worker = worker
        self._emit = emit
        self._interval = 1.0 / frame_hz if frame_hz > 0 else 1.0 / 30.0
        self._keyframe_interval = 1.0 / keyframe_hz if keyframe_hz > 0 else 0.0
        self._keyframe_dir = Path(keyframe_dir) if keyframe_dir else None
        self._camera_lost_after_s = camera_lost_after_s
        self._stop_event = threading.Event()
        self._round_lock = threading.Lock()
        self._round_id: Optional[str] = None
        self._fused = False
        self.stats: dict[str, Any] = {
            "frame_events": 0,
            "keyframes": 0,
            "duplicates_skipped": 0,
            "fused": False,
        }

    def set_round(self, round_id: Optional[str]) -> None:
        """Tag subsequently recorded frames with the current game round."""
        with self._round_lock:
            self._round_id = round_id

    @property
    def fused(self) -> bool:
        return self._fused

    def _save_keyframe(self, frame) -> Optional[str]:
        if cv2 is None or self._keyframe_dir is None:
            return None
        try:
            self._keyframe_dir.mkdir(parents=True, exist_ok=True)
            path = self._keyframe_dir / f"color_{self.stats['frame_events']:06d}.png"
            if cv2.imwrite(str(path), frame.color):
                return str(path)
        except Exception:
            logger.exception("Failed to save keyframe")
        return None

    def run(self) -> None:
        last_frame_ts: Optional[float] = None
        last_keyframe = 0.0
        while not self._stop_event.wait(self._interval):
            try:
                frame, pred = self._worker.get_latest()
            except Exception:
                logger.exception("Frame sampler failed to read worker state")
                continue

            # Fuse: camera stopped delivering frames entirely.
            health = self._worker.health()
            age = health.get("last_frame_age_s")
            if age is not None and age > self._camera_lost_after_s:
                self._fused = True
                self.stats["fused"] = True
                logger.error(
                    "Camera lost (no frame for %.1fs) — frame recorder fusing", age
                )
                try:
                    self._emit(
                        "hardware_transport_error",
                        {
                            "transport": "usb",
                            "device": "realsense_d435i",
                            "health_state": "no_frames",
                            "last_frame_age_s": age,
                            "action": "recording_stopped_game_continues",
                        },
                        "system",
                        ["realsense", "fuse"],
                    )
                except Exception:
                    logger.exception("Failed to emit hardware_transport_error")
                break

            if frame is None:
                continue
            if last_frame_ts is not None and frame.timestamp == last_frame_ts:
                self.stats["duplicates_skipped"] += 1
                continue
            last_frame_ts = frame.timestamp

            now = time.time()
            keyframe_path = None
            if (
                self._keyframe_interval
                and now - last_keyframe >= self._keyframe_interval
            ):
                last_keyframe = now
                keyframe_path = self._save_keyframe(frame)
                if keyframe_path:
                    self.stats["keyframes"] += 1

            with self._round_lock:
                round_id = self._round_id

            self.stats["frame_events"] += 1
            try:
                self._emit(
                    "frame_event",
                    {
                        "frame_number": self.stats["frame_events"],
                        "host_ts_ns": time.time_ns(),
                        "camera_frame_ts": frame.timestamp,
                        "has_depth": frame.depth is not None,
                        "width": int(frame.color.shape[1]),
                        "height": int(frame.color.shape[0]),
                        "human_label": pred.label if pred else "unknown",
                        "confidence": round(pred.confidence, 4) if pred else 0.0,
                        "round_id": round_id,
                        "keyframe": keyframe_path is not None,
                        "keyframe_path": keyframe_path,
                    },
                    "camera",
                    ["realsense", "rgbd", "rps_stream"],
                )
            except Exception:
                logger.exception("Failed to emit frame_event")

        logger.info(
            "Frame recorder stopped: %d frame_events, %d keyframes, fused=%s",
            self.stats["frame_events"],
            self.stats["keyframes"],
            self._fused,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=2.0)
