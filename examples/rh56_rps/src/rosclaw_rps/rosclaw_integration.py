"""ROSClaw-native integration layer for the RH56 RPS demo.

This module wires the existing RPS game logic into the ROSClaw runtime:

- EventBus / RuntimeBus for all key game events.
- PracticeCoordinator + PracticeRecorder for the data flywheel.
- SkillRegistry + SkillExecutor so the demo is executed as a ROSClaw skill.
- Optional SeekDB/SQLite backend for knowledge-plane persistence.

The legacy ``cli.py`` paths are left untouched; this is an additive ROSClaw
version of the demo.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ROSClaw runtime / practice / skill / memory
from rosclaw.body.resolver import BodyResolver
from rosclaw.core.event_bus import EventBus
from rosclaw.memory.seekdb_client import SeekDBMemoryClient, SeekDBSQLiteClient
from rosclaw.practice.config import (
    PracticeConfig,
    RecorderConfig,
    SeekDBConfig,
    SourceConfig,
)
from rosclaw.practice.coordinator import PracticeCoordinator
from rosclaw.practice.recorder import PracticeRecorder
from rosclaw.practice.schemas import PracticeEventEnvelope
from rosclaw.runtime.bus import RuntimeBus
from rosclaw.runtime.event import RuntimeEvent
from rosclaw.runtime.plugin import get_runtime_plugin
from rosclaw.skill_manager.executor import SkillExecutor
from rosclaw.skill_manager.loader import SkillLoader
from rosclaw.skill_manager.registry import SkillRegistry

# RPS demo modules
from .cli import _referee_enabled, load_configs
from .game_engine import GameEngine
from .gesture_schema import GestureConfig, GestureExecutionResult, RPSRound
from .hand.gesture_executor import GestureExecutor, GestureVerifier
from .hand.rh56_controller import MockHandController, build_hand_controller

logger = logging.getLogger("rosclaw_rps.integration")


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _telemetry_summary(telemetry) -> dict[str, Any]:
    if telemetry is None:
        return {}
    currents = [v for v in telemetry.current_ma.values() if v is not None]
    forces = [abs(v) for v in telemetry.force_act.values() if v is not None]
    temps = [v for v in telemetry.temperature_c.values() if v is not None]
    return {
        "current_peak": max(currents) if currents else 0,
        "force_peak": max(forces) if forces else 0,
        "temperature_max": max(temps) if temps else 0,
    }


def _hand_telemetry_to_dict(telemetry) -> dict[str, Any]:
    """Convert a HandTelemetry snapshot to a JSON-serializable dict."""
    if telemetry is None:
        return {}
    return {
        "timestamp": telemetry.timestamp,
        "angle_actual": telemetry.angle_actual,
        "angle_set": telemetry.angle_set,
        "force_act": telemetry.force_act,
        "current_ma": telemetry.current_ma,
        "temperature_c": telemetry.temperature_c,
        "error": telemetry.error,
        "status": telemetry.status,
        "summary": _telemetry_summary(telemetry),
    }


def _gesture_summary(result: GestureExecutionResult) -> dict[str, Any]:
    return {
        "gesture_name": result.gesture_name,
        "command_success": result.command_success,
        "verified": result.verified,
        "failure_reason": result.failure_reason,
        "duration_s": round(result.duration_s, 3),
        "telemetry_summary": _telemetry_summary(result.telemetry),
    }


def _build_executor(
    config_section: dict, gestures: Dict[str, GestureConfig]
) -> GestureExecutor:
    hand = build_hand_controller(config_section)
    return GestureExecutor(hand, gestures, GestureVerifier())


class RosclawRpsSession:
    """Owns the ROSClaw runtime objects for one RPS demo run."""

    def __init__(self, config_dir: Path, rosclaw_config: dict[str, Any]):
        self.config_dir = Path(config_dir)
        self.rosclaw_config = rosclaw_config
        self.gestures: Dict[str, GestureConfig] = {}
        self.referee_gestures: Dict[str, GestureConfig] = {}
        self.demo_config: dict[str, Any] = {}
        self._load_demo_configs()

        self.event_bus: Optional[EventBus] = None
        self.runtime_bus: Optional[RuntimeBus] = None
        self.recorder: Optional[PracticeRecorder] = None
        self.coordinator: Optional[PracticeCoordinator] = None
        self.registry: Optional[SkillRegistry] = None
        self.skill_executor: Optional[SkillExecutor] = None
        self.seekdb_client: Optional[Any] = None
        self._skill_handler: Optional[RpsSkillHandler] = None

    def _load_demo_configs(self) -> None:
        self.gestures, self.referee_gestures, self.demo_config = load_configs(
            self.config_dir
        )

    @property
    def robot_id(self) -> str:
        return self.rosclaw_config.get("practice", {}).get("robot_id", "rh56_rps_robot")

    @property
    def skill_id(self) -> str:
        return self.rosclaw_config.get("practice", {}).get("skill_id", "rh56_rps")

    @property
    def task_id(self) -> Optional[str]:
        return self.rosclaw_config.get("practice", {}).get("task_id")

    def initialize(self) -> None:
        """Initialize the full ROSClaw runtime stack."""
        mem_cfg = self.rosclaw_config.get("memory", {})
        seekdb_cfg = self.rosclaw_config.get("seekdb", {})

        # SeekDB backend -----------------------------------------------------
        backend = mem_cfg.get("backend", "memory")
        if seekdb_cfg.get("enabled") or backend == "sqlite":
            db_path = _expand(
                mem_cfg.get(
                    "db_path", "~/.rosclaw/practice/runs/rh56_rps/seekdb.sqlite"
                )
            )
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.seekdb_client = SeekDBSQLiteClient(str(db_path))
        else:
            self.seekdb_client = SeekDBMemoryClient()
        self.seekdb_client.connect()

        # Buses --------------------------------------------------------------
        self.event_bus = EventBus()
        self.runtime_bus = RuntimeBus(event_bus=self.event_bus)

        # Recorder -----------------------------------------------------------
        data_root = self.rosclaw_config.get("practice", {}).get(
            "data_root",
            str(Path.home() / ".rosclaw" / "practice" / "runs" / "rh56_rps"),
        )
        recorder_cfg = RecorderConfig(
            **self.rosclaw_config.get("practice", {}).get("recorder", {})
        )
        self.recorder = PracticeRecorder(
            self.runtime_bus,
            data_root=data_root,
            publish_to_event_bus=True,
            auto_start_on_skill=True,
            config=recorder_cfg,
        )
        self.recorder.initialize()

        # Coordinator --------------------------------------------------------
        pc = self.rosclaw_config.get("practice", {})
        practice_config = PracticeConfig(
            robot_id=pc.get("robot_id", "rh56_rps_robot"),
            robot_type=pc.get("robot_type", "dual_rh56"),
            task_id=pc.get("task_id", "rh56_rps"),
            task_name=pc.get("task_name", "RH56 Rock-Paper-Scissors"),
            skill_id=pc.get("skill_id", "rh56_rps"),
            session_name=pc.get("session_name"),
            data_root=data_root,
            sources=SourceConfig(**pc.get("sources", {})),
            recorder=RecorderConfig(**pc.get("recorder", {})),
            seekdb=SeekDBConfig(**pc.get("seekdb", {})),
            publish_to_event_bus=True,
            event_bus=self.event_bus,
        )
        # The demo hardware is not yet modeled as a full rosclaw body, so we
        # explicitly tag sessions with the robot_id as body_id for SeekDB
        # queryability without relying on BodyResolver finding a linked body.
        practice_config.body_id = pc.get("body_id") or pc.get(
            "robot_id", "rh56_rps_robot"
        )
        self.coordinator = PracticeCoordinator(
            config=practice_config,
            runtime_bus=self.runtime_bus,
            recorder=self.recorder,
        )
        self.coordinator.initialize()

        # Skill registry -----------------------------------------------------
        self.registry = SkillRegistry(event_bus=self.event_bus)
        self.registry.initialize()

        manifest_path = self.config_dir / self.rosclaw_config.get(
            "skill_manifest", "skills/rh56_rps.skill.yaml"
        )
        loader = SkillLoader(self.registry)
        if manifest_path.exists():
            loader.load_skill_manifest(manifest_path)
        else:
            loader.create_programmed_skill(
                self.skill_id,
                "RH56 Rock-Paper-Scissors demo skill",
                handler=None,
            )

        # Runtime handler ----------------------------------------------------
        self._skill_handler = RpsSkillHandler(self)
        get_runtime_plugin().register(self.skill_id, self._skill_handler.run)

        # Skill executor -----------------------------------------------------
        # Point the resolver at the config dir, which has no linked body.  This
        # keeps SkillExecutor in backward-compatible fail-open mode so the demo
        # can run before a full RH56 e-URDF body model is registered.
        body_resolver = BodyResolver(workspace=self.config_dir)
        self.skill_executor = SkillExecutor(
            event_bus=self.event_bus,
            registry=self.registry,
            seekdb_client=self.seekdb_client,
            body_resolver=body_resolver,
        )
        self.skill_executor.initialize()

        logger.info("RosclawRpsSession initialized for skill %s", self.skill_id)

    def start(self) -> None:
        if self.recorder is not None:
            self.recorder.start()
        if self.coordinator is not None:
            self.coordinator.start()

    def stop(self) -> None:
        if self.coordinator is not None:
            try:
                self.coordinator.stop()
            except Exception:
                logger.exception("Coordinator stop failed")
        if self.recorder is not None:
            try:
                self.recorder.stop()
            except Exception:
                logger.exception("Recorder stop failed")
        if self.skill_executor is not None:
            try:
                self.skill_executor.stop()
            except Exception:
                logger.exception("SkillExecutor stop failed")
        if self.registry is not None:
            try:
                self.registry.stop()
            except Exception:
                logger.exception("Registry stop failed")
        if self.seekdb_client is not None:
            try:
                self.seekdb_client.disconnect()
            except Exception:
                logger.exception("SeekDB disconnect failed")

    @property
    def active_practice_id(self) -> Optional[str]:
        if self.coordinator is not None and self.coordinator.session is not None:
            return self.coordinator.session.practice_id
        return None

    def run_skill(self, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.skill_executor is None:
            raise RuntimeError("Session not initialized")
        return self.skill_executor.execute(self.skill_id, parameters=parameters)


class RpsSkillHandler:
    """Runtime handler for the ``rh56_rps`` skill.

    Implements the same game modes as the legacy CLI, but emits ROSClaw
    RuntimeEvents and PracticeEventEnvelopes at every important milestone.
    """

    def __init__(self, session: RosclawRpsSession):
        self.session = session
        self._telemetry_thread: Optional[threading.Thread] = None
        self._telemetry_stop: Optional[threading.Event] = None
        self._telemetry_lock = threading.Lock()
        self._telemetry_executors: tuple[Optional[Any], Optional[Any]] = (None, None)

    @property
    def runtime_bus(self) -> RuntimeBus:
        return self.session.runtime_bus  # type: ignore[return-value]

    @property
    def coordinator(self) -> PracticeCoordinator:
        return self.session.coordinator  # type: ignore[return-value]

    def _telemetry_config(self) -> tuple[bool, float]:
        recorder = self.session.recorder
        if recorder is None:
            return False, 0.0
        cfg = recorder.config
        return cfg.telemetry_enabled, cfg.telemetry_hz

    def _start_telemetry(self, executor: Any, ref_executor: Optional[Any]) -> None:
        enabled, hz = self._telemetry_config()
        if not enabled or hz <= 0:
            return
        self._stop_telemetry()
        with self._telemetry_lock:
            self._telemetry_executors = (executor, ref_executor)
        self._telemetry_stop = threading.Event()
        self._telemetry_thread = threading.Thread(
            target=self._telemetry_loop,
            args=(hz,),
            name="rps_telemetry",
            daemon=True,
        )
        self._telemetry_thread.start()
        logger.debug("Started RPS telemetry sampler at %.1f Hz", hz)

    def _stop_telemetry(self) -> None:
        if self._telemetry_stop is not None:
            self._telemetry_stop.set()
        if self._telemetry_thread is not None and self._telemetry_thread.is_alive():
            self._telemetry_thread.join(timeout=2.0)
            if self._telemetry_thread.is_alive():
                logger.warning("Telemetry sampler thread did not stop cleanly")
        with self._telemetry_lock:
            self._telemetry_executors = (None, None)
        self._telemetry_thread = None
        self._telemetry_stop = None

    def _telemetry_loop(self, hz: float) -> None:
        interval = 1.0 / hz
        stop_event = self._telemetry_stop
        assert stop_event is not None
        while not stop_event.wait(interval):
            try:
                with self._telemetry_lock:
                    executor, ref_executor = self._telemetry_executors
                iter_start = time.time()
                right = self._read_hand_telemetry(executor, "right")
                left = self._read_hand_telemetry(ref_executor, "left")
                payload: dict[str, Any] = {
                    "timestamp": time.time(),
                    "practice_id": self._practice_id(),
                }
                if right:
                    payload["right"] = right
                if left:
                    payload["left"] = left
                if right or left:
                    self._emit_runtime(
                        "rps.telemetry",
                        payload,
                        source="rps_demo",
                        tags=["rps", "telemetry", "joint_state"],
                    )
                # Pace to the configured rate: the wait at the top of the
                # loop alone would add the ~0.2 s of serial work on top of
                # the interval (5 Hz configured -> 2.4 Hz effective).
                elapsed = time.time() - iter_start
                if elapsed < interval:
                    stop_event.wait(interval - elapsed)
            except Exception:
                logger.exception("Telemetry sampler iteration failed")

    @staticmethod
    def _read_hand_telemetry(
        executor: Optional[Any], label: str
    ) -> Optional[dict[str, Any]]:
        if executor is None or not hasattr(executor, "hand"):
            return None
        try:
            telemetry = executor.hand.read_telemetry()
            return _hand_telemetry_to_dict(telemetry)
        except Exception as exc:
            logger.debug("Failed to read %s hand telemetry: %s", label, exc)
            return None

    def _practice_id(self) -> str:
        return self.session.active_practice_id or "unknown"

    def _emit_runtime(
        self,
        event_type: str,
        payload: dict[str, Any],
        source: str = "rps_demo",
        tags: Optional[List[str]] = None,
    ) -> None:
        self.runtime_bus.publish(
            RuntimeEvent(
                type=event_type,
                source=source,
                robot=self.session.robot_id,
                payload=payload,
                metadata={
                    "source": "runtime",
                    "tags": tags or [event_type],
                    "trace_id": self._practice_id(),
                    "task_id": self.session.task_id,
                    "skill_id": self.session.skill_id,
                },
            )
        )

    def _emit_practice(
        self,
        event_type: str,
        payload: dict[str, Any],
        source: str = "agent",
        tags: Optional[List[str]] = None,
    ) -> None:
        practice_id = self._practice_id()
        self.coordinator.emit_event(
            PracticeEventEnvelope(
                practice_id=practice_id,
                session_id=practice_id,
                robot_id=self.session.robot_id,
                source=source,
                event_type=event_type,
                payload=payload,
                tags=tags or [event_type],
                trace_id=practice_id,
                task_id=self.session.task_id,
                skill_id=self.session.skill_id,
            )
        )

    def run(self, parameters: dict[str, Any]) -> dict[str, Any]:
        mode = parameters.get("mode", "mock")
        rounds = parameters.get("rounds")
        auto = bool(parameters.get("auto", False))
        headless = bool(parameters.get("headless", False))

        self._emit_runtime(
            "rps.run.started",
            {"mode": mode, "rounds": rounds, "auto": auto, "headless": headless},
            source="rps_demo",
            tags=["rps", "run", "started"],
        )

        summary: dict[str, Any] = {}
        try:
            if mode == "mock":
                summary = self._run_mock(rounds=rounds, auto=auto)
            elif mode == "hand-test":
                summary = self._run_hand_test()
            elif mode == "full":
                summary = self._run_full(rounds=rounds, auto=auto, headless=headless)
            else:
                raise ValueError(f"Unknown mode: {mode}")
        except Exception as exc:
            logger.exception("RPS skill run failed")
            self._emit_runtime(
                "rps.run.failed",
                {"mode": mode, "error": str(exc)},
                source="rps_demo",
                tags=["rps", "run", "failed"],
            )
            return {"status": "error", "mode": mode, "error": str(exc)}
        finally:
            self._stop_telemetry()

        self._emit_practice(
            "rps.run.summary",
            {"mode": mode, "summary": summary},
            source="agent",
            tags=["rps", "run", "summary"],
        )
        self._emit_runtime(
            "rps.run.completed",
            {"mode": mode, "summary": summary},
            source="rps_demo",
            tags=["rps", "run", "completed"],
        )
        return {"status": "success", "mode": mode, "summary": summary}

    # ------------------------------------------------------------------
    # Mock mode
    # ------------------------------------------------------------------
    def _run_mock(self, rounds: Optional[int], auto: bool) -> dict[str, Any]:
        demo = self.session.demo_config
        gestures = self.session.gestures
        referee_gestures = self.session.referee_gestures
        total = rounds if rounds is not None else int(demo["demo"]["rounds"])

        hand = MockHandController()
        executor = GestureExecutor(hand, gestures, GestureVerifier())
        ref_executor: Optional[GestureExecutor] = None
        if _referee_enabled(demo):
            ref_section = demo.get("referee", {}).copy()
            ref_section["controller"] = "mock"
            ref_executor = _build_executor(ref_section, referee_gestures)

        self._start_telemetry(executor, ref_executor)

        from .vision.hand_gesture_recognizer import KeyboardRecognizer, MockRecognizer

        recognizer = MockRecognizer() if auto else KeyboardRecognizer()
        engine = GameEngine()
        rounds_played: List[RPSRound] = []

        self._execute_and_emit(executor, "ready")
        if ref_executor:
            self._execute_and_emit(ref_executor, "left_ready")

        for i in range(total):
            round_id = engine.new_round_id()
            self._emit_practice(
                "rps.round.started",
                {"round_id": round_id, "sequence": i + 1, "total": total},
                tags=["rps", "round"],
            )

            if not self._wait_for_start(auto):
                break

            commit = engine.commit_robot_choice(round_id)
            self._emit_runtime(
                "rps.robot_choice_committed",
                {
                    "round_id": round_id,
                    "robot_choice": commit.robot_choice,
                    "commit_hash": commit.commit_hash,
                },
                tags=["rps", "robot", "commit"],
            )

            pred = recognizer.predict(None)
            self._emit_runtime(
                "rps.human.gesture_detected",
                {
                    "round_id": round_id,
                    "label": pred.label,
                    "confidence": pred.confidence,
                },
                tags=["rps", "human"],
            )

            robot_result = self._execute_and_emit(executor, commit.robot_choice)
            round_obj = engine.resolve_round(round_id, commit, pred, robot_result)

            result_gesture = engine.result_gesture_map.get(round_obj.result, "error")
            self._execute_and_emit(executor, result_gesture)
            if ref_executor:
                self._execute_and_emit(
                    ref_executor,
                    round_obj.referee_gesture,
                    event_type="rps.referee.gesture_executed",
                )

            rounds_played.append(round_obj)
            self._emit_practice(
                "rps.round.resolved",
                {"round": round_obj.to_dict()},
                tags=["rps", "round", "resolved"],
            )

        self._execute_and_emit(executor, "error")
        if ref_executor:
            self._execute_and_emit(ref_executor, "left_error")

        self._stop_telemetry()

        summary = engine.summary(rounds_played)
        return summary

    def _wait_for_start(self, auto: bool) -> bool:
        if auto:
            return True
        try:
            input("Press Enter to start next round (Ctrl-C to quit)...")
            return True
        except (EOFError, KeyboardInterrupt):
            return False

    def _execute_and_emit(
        self,
        executor: GestureExecutor,
        name: str,
        event_type: str = "rps.gesture.executed",
    ) -> GestureExecutionResult:
        result = executor.execute(name)
        self._emit_runtime(
            event_type,
            _gesture_summary(result),
            source="rps_demo",
            tags=["rps", "gesture", name],
        )
        return result

    # ------------------------------------------------------------------
    # Hand-test mode
    # ------------------------------------------------------------------
    def _run_hand_test(self) -> dict[str, Any]:
        demo = self.session.demo_config
        gestures = self.session.gestures
        referee_gestures = self.session.referee_gestures

        executor = _build_executor(demo["hand"], gestures)
        ref_executor: Optional[GestureExecutor] = None
        if _referee_enabled(demo):
            ref_executor = _build_executor(demo.get("referee", {}), referee_gestures)

        self._start_telemetry(executor, ref_executor)

        right_sequence = [
            "ready",
            "rock",
            "paper",
            "scissors",
            "win",
            "lose",
            "draw",
            "error",
        ]
        left_sequence = [
            "left_ready",
            "left_thumb_up",
            "left_pinkie",
            "left_orchid",
            "left_error",
        ]

        self._execute_and_emit(executor, "ready")
        if ref_executor:
            self._execute_and_emit(ref_executor, "left_ready")

        for name in right_sequence:
            self._execute_and_emit(executor, name)
            if ref_executor and name == "ready":
                for left_name in left_sequence:
                    self._execute_and_emit(
                        ref_executor,
                        left_name,
                        event_type="rps.referee.gesture_executed",
                    )
                    time.sleep(0.5)
            time.sleep(0.5)

        self._execute_and_emit(executor, "error")
        if ref_executor:
            self._execute_and_emit(ref_executor, "left_error")
            ref_executor.hand.close()
        executor.hand.close()

        self._stop_telemetry()

        return {
            "mode": "hand-test",
            "gestures_tested": right_sequence + (left_sequence if ref_executor else []),
        }

    # ------------------------------------------------------------------
    # Full mode
    # ------------------------------------------------------------------
    def _run_full(
        self,
        rounds: Optional[int],
        auto: bool,
        headless: bool,
    ) -> dict[str, Any]:
        try:
            from .ui.simple_opencv_ui import SimpleOpenCVUI
            from .vision.camera_source import build_camera_source
            from .vision.hand_gesture_recognizer import build_recognizer
            from .vision.majority_vote import MajorityVoteBuffer
            from .vision.recognition_worker import RecognitionWorker
        except Exception as exc:
            raise RuntimeError(
                f"Vision/UI imports failed (camera/OpenCV unavailable): {exc}"
            ) from exc

        demo = self.session.demo_config
        gestures = self.session.gestures
        referee_gestures = self.session.referee_gestures

        executor = _build_executor(demo["hand"], gestures)
        ref_executor: Optional[GestureExecutor] = None
        if _referee_enabled(demo):
            ref_executor = _build_executor(demo.get("referee", {}), referee_gestures)

        self._start_telemetry(executor, ref_executor)

        engine = GameEngine()
        camera = build_camera_source(demo["camera"])
        recognizer = build_recognizer(demo["camera"])
        vote = MajorityVoteBuffer(
            window_size=int(demo["capture"]["vote_window_size"]),
            min_confidence=float(demo["capture"]["min_confidence"]),
            majority_ratio=float(demo["capture"]["majority_ratio"]),
            tail_size=int(demo["capture"].get("vote_tail_size", 0)),
        )

        # Create the UI *before* starting the recognition worker.  OpenCV's
        # highgui/Qt backend must be initialized in the main thread; if a worker
        # thread calls cv2 first, the UI window can open but stay black.
        ui = None if headless else SimpleOpenCVUI()
        if ui is not None:
            import cv2
            import numpy as np

            cv2.namedWindow(ui.window_name, cv2.WINDOW_NORMAL)
            startup = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                startup,
                "Starting camera...",
                (160, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            cv2.imshow(ui.window_name, startup)
            cv2.waitKey(1)

        worker = RecognitionWorker(
            camera,
            recognizer,
            vote,
            process_every_n=int(demo["capture"].get("process_every_n", 1)),
        )
        worker.start()

        # Continuous frame recording: tap the worker's 30 Hz stream into the
        # Practice session as frame_events (+ keyframes) when enabled.
        frame_recorder = None
        rec_cfg = (
            self.session.recorder.config if self.session.recorder is not None else None
        )
        if rec_cfg is not None and rec_cfg.frames_enabled:
            from .vision.frame_recorder import FrameRecorder

            session_dir = None
            if self.coordinator is not None and self.coordinator.session is not None:
                session_dir = Path(self.coordinator.session.session_dir)
            frame_recorder = FrameRecorder(
                worker,
                emit=self._emit_practice,
                frame_hz=rec_cfg.frame_hz,
                keyframe_hz=rec_cfg.keyframe_hz,
                keyframe_dir=(session_dir / "keyframes") if session_dir else None,
            )
            frame_recorder.start()
            logger.info(
                "Frame recorder started at %.0f Hz (keyframes %.1f Hz) -> %s",
                rec_cfg.frame_hz,
                rec_cfg.keyframe_hz,
                session_dir,
            )

        countdown_labels = demo["countdown"]["labels"]
        countdown_gestures = demo["countdown"].get("gestures", [])
        step_s = float(demo["countdown"]["step_s"])
        total = rounds if rounds is not None else int(demo["demo"]["rounds"])
        run_auto = auto or headless
        result_display_s = float(demo.get("ui", {}).get("result_display_s", 2.5))
        # Keep the live video flowing during result display (30 fps refresh);
        # the old 500 ms tick turned the result phase into a 2 fps slideshow.
        result_refresh_ms = int(demo.get("ui", {}).get("result_refresh_ms", 33))
        result_iterations = max(
            1, int(round(result_display_s * 1000 / result_refresh_ms))
        )
        # Time for the human to settle into their final gesture after the
        # reveal before the vote is latched (the serial speedup collapsed
        # the old multi-second window to ~2 s and latched mid-motion).
        settle_s = float(demo["capture"].get("settle_s", 1.0))

        rounds_played: List[RPSRound] = []

        self._execute_and_emit(executor, "ready")
        if ref_executor:
            self._execute_and_emit(ref_executor, "left_ready")

        def render_show(frame, waiting: bool = False, **kwargs):
            if ui is None:
                return
            camera_lost = _camera_lost()
            canvas = ui.render(
                frame, waiting_for_start=waiting, camera_lost=camera_lost, **kwargs
            )
            ui.show(canvas)

        def wait_or_sleep(delay_ms: int) -> int:
            if ui is None:
                time.sleep(delay_ms / 1000.0)
                return 0xFF
            return ui.wait_key(delay_ms)

        def latest_state():
            frame, pred = worker.get_latest()
            return frame, pred.label if pred else "unknown"

        def _camera_lost() -> bool:
            health = worker.health()
            last_age = health.get("last_frame_age_s")
            return last_age is None or last_age > 2.0

        _exec_result: List[Optional[GestureExecutionResult]] = [None]
        _exec_thread: List[Optional[threading.Thread]] = [None]
        _exec_lock = threading.Lock()

        def run_gesture_async(name: str) -> None:
            if _exec_thread[0] is not None and _exec_thread[0].is_alive():
                _exec_thread[0].join()

            def target():
                res = executor.execute(name)
                with _exec_lock:
                    _exec_result[0] = res

            t = threading.Thread(target=target)
            t.start()
            _exec_thread[0] = t

        def gesture_async_done() -> bool:
            return _exec_thread[0] is None or not _exec_thread[0].is_alive()

        def last_gesture_result():
            with _exec_lock:
                return _exec_result[0]

        def wait_gesture_async(
            min_wait_s: float = 0.0,
        ) -> Optional[GestureExecutionResult]:
            deadline = time.time() + min_wait_s
            while True:
                if gesture_async_done() and time.time() >= deadline:
                    break
                frame, human_label = latest_state()
                render_show(
                    frame,
                    round_id=round_id,
                    robot_committed=True,
                    human_label=human_label,
                )
                if wait_or_sleep(20) == ord("q"):
                    raise KeyboardInterrupt
            if _exec_thread[0] is not None:
                _exec_thread[0].join()
            return last_gesture_result()

        ref_thread: Optional[threading.Thread] = None
        ref_result: List[Optional[GestureExecutionResult]] = [None]

        def run_referee_async(name: str) -> None:
            nonlocal ref_thread

            def target():
                try:
                    res = ref_executor.execute(name) if ref_executor else None
                    ref_result[0] = res
                except Exception as exc:
                    logger.warning("Referee gesture %s failed: %s", name, exc)

            if ref_thread is not None and ref_thread.is_alive():
                ref_thread.join()
            ref_thread = threading.Thread(target=target)
            ref_thread.start()

        try:
            for _ in range(total):
                round_id = engine.new_round_id()
                if frame_recorder is not None:
                    frame_recorder.set_round(round_id)
                self._emit_practice(
                    "rps.round.started",
                    {"round_id": round_id},
                    tags=["rps", "round"],
                )

                frame, human_label = latest_state()
                render_show(
                    frame,
                    round_id=round_id,
                    robot_committed=False,
                    waiting=True,
                    human_label=human_label,
                )

                started = False
                while not started:
                    frame, human_label = latest_state()
                    render_show(
                        frame,
                        round_id=round_id,
                        robot_committed=False,
                        waiting=True,
                        human_label=human_label,
                    )
                    key = wait_or_sleep(16)
                    if key == ord("q"):
                        raise KeyboardInterrupt
                    if key == ord(" ") or run_auto:
                        started = True
                    if key == ord("o"):
                        run_gesture_async("error")
                        wait_gesture_async(min_wait_s=0.0)

                commit = engine.commit_robot_choice(round_id)
                self._emit_runtime(
                    "rps.robot_choice_committed",
                    {
                        "round_id": round_id,
                        "robot_choice": commit.robot_choice,
                        "commit_hash": commit.commit_hash,
                    },
                    tags=["rps", "robot", "commit"],
                )

                worker.reset_vote()
                for label, gesture_name in zip(countdown_labels, countdown_gestures):
                    run_gesture_async(gesture_name)
                    robot_result = wait_gesture_async(min_wait_s=step_s)
                    self._emit_runtime(
                        "rps.countdown.gesture_executed",
                        {
                            "round_id": round_id,
                            "countdown_label": label,
                            "gesture_name": gesture_name,
                            "verified": robot_result.verified
                            if robot_result
                            else False,
                        },
                        tags=["rps", "countdown"],
                    )

                run_gesture_async(commit.robot_choice)
                robot_result = wait_gesture_async(min_wait_s=0.0)
                self._emit_runtime(
                    "rps.robot.revealed",
                    {
                        "round_id": round_id,
                        "robot_choice": commit.robot_choice,
                        **_gesture_summary(robot_result),
                    },
                    tags=["rps", "robot", "reveal"],
                )

                # Settle window: let the human freeze their final gesture
                # while the live feed keeps running; voting continues.
                settle_deadline = time.time() + settle_s
                while time.time() < settle_deadline:
                    frame, human_label = latest_state()
                    render_show(
                        frame,
                        round_id=round_id,
                        robot_committed=True,
                        robot_choice=commit.robot_choice,
                        human_label=human_label,
                    )
                    if wait_or_sleep(33) == ord("q"):
                        raise KeyboardInterrupt

                human_pred = worker.final_vote()
                self._emit_runtime(
                    "rps.human.gesture_detected",
                    {
                        "round_id": round_id,
                        "label": human_pred.label,
                        "confidence": human_pred.confidence,
                    },
                    tags=["rps", "human"],
                )

                round_obj = engine.resolve_round(
                    round_id,
                    commit,
                    human_pred,
                    robot_result or GestureExecutionResult("error", False, False),
                )
                if ref_executor:
                    run_referee_async(round_obj.referee_gesture)

                rounds_played.append(round_obj)
                self._emit_practice(
                    "rps.round.resolved",
                    {"round": round_obj.to_dict()},
                    tags=["rps", "round", "resolved"],
                )

                for _ in range(result_iterations):
                    frame, human_label = latest_state()
                    render_show(
                        frame,
                        round_id=round_id,
                        robot_committed=True,
                        robot_choice=commit.robot_choice,
                        human_label=human_pred.label,
                        result=round_obj.result,
                        bodysense="verified"
                        if robot_result and robot_result.verified
                        else (robot_result.failure_reason if robot_result else None),
                        referee=round_obj.referee_gesture if ref_executor else None,
                    )
                    if wait_or_sleep(result_refresh_ms) == ord("q"):
                        raise KeyboardInterrupt

                run_gesture_async("ready")
                wait_gesture_async(min_wait_s=0.0)
                if ref_executor:
                    # Async: a synchronous referee execute froze the UI feed
                    # at every round boundary.
                    run_referee_async("left_ready")

        except KeyboardInterrupt:
            pass
        finally:
            self._stop_telemetry()

            if frame_recorder is not None:
                frame_recorder.stop()
                try:
                    self._emit_practice(
                        "frame_stream_summary",
                        frame_recorder.stats,
                        source="camera",
                        tags=["realsense", "summary"],
                    )
                except Exception:
                    logger.exception("Failed to emit frame_stream_summary")

            if _exec_thread[0] is not None and _exec_thread[0].is_alive():
                try:
                    _exec_thread[0].join()
                except Exception:
                    logger.exception("Failed to join gesture thread")
            if ref_thread is not None and ref_thread.is_alive():
                try:
                    ref_thread.join()
                except Exception:
                    logger.exception("Failed to join referee thread")

            # Return the hands to a safe error pose before tearing down.
            try:
                self._execute_and_emit(executor, "error")
            except Exception:
                logger.exception("Failed to emit error gesture")
            if ref_executor:
                try:
                    self._execute_and_emit(ref_executor, "left_error")
                except Exception:
                    logger.exception("Failed to emit left_error gesture")

            for closeable, label in (
                (executor.hand, "right hand"),
                (ref_executor.hand if ref_executor else None, "left hand"),
                (worker, "recognition worker"),
                (camera, "camera"),
                (ui, "UI"),
            ):
                if closeable is None:
                    continue
                try:
                    if hasattr(closeable, "close"):
                        closeable.close()
                    elif hasattr(closeable, "stop"):
                        closeable.stop()
                    elif hasattr(closeable, "release"):
                        closeable.release()
                except Exception:
                    logger.exception("Failed to close %s", label)

        summary = engine.summary(rounds_played)
        return summary
