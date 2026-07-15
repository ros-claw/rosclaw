"""Background threads that read camera frames and run gesture recognition."""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple

from ..gesture_schema import CameraFrame, GesturePrediction
from .camera_source import CameraSource
from .hand_gesture_recognizer import HumanGestureRecognizer
from .majority_vote import MajorityVoteBuffer

logger = logging.getLogger("rosclaw_rps.recognition_worker")


class RecognitionWorker:
    """Continuously capture frames and update the vote buffer in the background.

    Capture and inference run on *separate* threads: the capture thread does
    nothing but ``camera.read()`` and store the latest frame, so the displayed
    video stays at full camera rate even when MediaPipe inference spikes.
    The inference thread pulls the newest stored frame at its own pace and
    only updates the predicted label / vote buffer.
    """

    def __init__(
        self,
        camera: CameraSource,
        recognizer: HumanGestureRecognizer,
        vote_buffer: MajorityVoteBuffer,
        process_every_n: int = 2,
        name: str = "recognition-worker",
    ):
        self._camera = camera
        self._recognizer = recognizer
        self._vote = vote_buffer
        self._process_every_n = max(1, process_every_n)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame: Optional[CameraFrame] = None
        self._latest_pred: GesturePrediction = GesturePrediction(
            label="unknown", confidence=0.0
        )
        self._frame_counter = 0
        self._frames_read = 0
        self._last_frame_time: float = 0.0
        self._empty_streak: int = 0
        self._inference_count = 0
        self._last_inference_ms: float = 0.0
        self._max_inference_ms: float = 0.0
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name=name, daemon=True
        )
        self._inference_thread = threading.Thread(
            target=self._inference_loop, name=f"{name}-inference", daemon=True
        )

    # ------------------------------------------------------------------ loops

    def _capture_loop(self) -> None:
        warn_threshold = 100  # ~1 s of None frames at 10 ms sleep
        while not self._stop_event.is_set():
            try:
                frame = self._camera.read()
            except Exception:
                logger.exception("Camera source read failed")
                time.sleep(0.1)
                continue

            if frame is None:
                with self._lock:
                    self._empty_streak += 1
                    streak = self._empty_streak
                if streak == warn_threshold:
                    logger.warning(
                        "Camera source has returned no frames for ~1 second; "
                        "check the RealSense node / USB connection."
                    )
                time.sleep(0.01)
                continue

            with self._lock:
                self._empty_streak = 0
                self._frames_read += 1
                self._frame_counter += 1
                self._last_frame_time = time.time()
                self._latest_frame = frame

    def _inference_loop(self) -> None:
        last_processed = -1
        while not self._stop_event.is_set():
            with self._lock:
                frame = self._latest_frame
                counter = self._frame_counter
            if (
                frame is None
                or counter == last_processed
                or counter % self._process_every_n != 0
            ):
                time.sleep(0.005)
                continue
            last_processed = counter

            t0 = time.perf_counter()
            try:
                pred = self._recognizer.predict(frame)
            except Exception:
                logger.exception("Gesture recognizer failed on a frame")
                pred = GesturePrediction(label="unknown", confidence=0.0)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            with self._lock:
                self._latest_pred = pred
                self._vote.update(pred)
                self._inference_count += 1
                self._last_inference_ms = elapsed_ms
                if elapsed_ms > self._max_inference_ms:
                    self._max_inference_ms = elapsed_ms

    # --------------------------------------------------------------- lifecycle

    def start(self) -> None:
        self._capture_thread.start()
        self._inference_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        for t in (self._capture_thread, self._inference_thread):
            if t.is_alive():
                t.join(timeout=2.0)

    def is_alive(self) -> bool:
        return self._capture_thread.is_alive()

    # -------------------------------------------------------------------- API

    def get_latest(self) -> Tuple[Optional[CameraFrame], GesturePrediction]:
        with self._lock:
            return self._latest_frame, self._latest_pred

    def health(self) -> dict:
        """Return coarse health metrics for diagnostics."""
        with self._lock:
            return {
                "alive": self.is_alive(),
                "frames_read": self._frames_read,
                "inferences": self._inference_count,
                "last_inference_ms": round(self._last_inference_ms, 1),
                "max_inference_ms": round(self._max_inference_ms, 1),
                "last_frame_age_s": round(time.time() - self._last_frame_time, 2)
                if self._last_frame_time
                else None,
                "empty_streak": self._empty_streak,
            }

    def reset_vote(self) -> None:
        with self._lock:
            self._vote.reset()

    def final_vote(self) -> GesturePrediction:
        with self._lock:
            return self._vote.final()
